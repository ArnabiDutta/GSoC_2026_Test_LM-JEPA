import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import random

class JEPATrainer:
    def __init__(self, model: torch.nn.Module, vocab, train_loader: DataLoader, 
                 val_loader: DataLoader, device: str, config: dict):
        self.model = model.to(device)
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = AdamW(self.model.parameters(), lr=config.get('learning_rate', 1e-4), weight_decay=config.get('weight_decay', 0.01))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.get('epochs', 50), eta_min=1e-6)
        
        self.best_val_loss = float('inf')
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.patience_counter = 0
        self.patience = config.get('early_stopping_patience', 30)
        
        wandb.init(project=config.get('wandb_project', 'SYMBA-LLM-JEPA'), config=config)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_lm_loss, total_jepa_loss, total_loss = 0.0, 0.0, 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            t_sizes = batch['t_size'].to(self.device)
            c_sizes = batch['c_size'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, jepa_loss = self.model(input_ids, attention_mask, t_sizes, c_sizes)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            vocab_size = shift_logits.size(-1)
            lm_loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
            
            loss = lm_loss + jepa_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n NaN loss detected at batch {step}. Skipping.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_jepa_loss += jepa_loss.item()
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LM': f"{lm_loss.item():.4f}", 'JEPA': f"{jepa_loss.item():.4f}"})
            
            if step % 10 == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/step_lm_loss": lm_loss.item(),
                    "train/step_jepa_loss": jepa_loss.item(),
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })
                
        num_batches = max(len(self.train_loader), 1)
        return total_loss / num_batches, total_lm_loss / num_batches, total_jepa_loss / num_batches

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        self.model.eval()
        total_lm_loss, total_jepa_loss, total_loss = 0.0, 0.0, 0.0
        total_val_tok_acc, total_val_seq_acc = 0.0, 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        sample_inputs, sample_targets, sample_preds = [], [], []
        logged_samples = False
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            t_sizes = batch['t_size'].to(self.device)
            c_sizes = batch['c_size'].to(self.device)
            
            logits, jepa_loss = self.model(input_ids, attention_mask, t_sizes, c_sizes)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            vocab_size = shift_logits.size(-1)
            lm_loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
            loss = lm_loss + jepa_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            
            preds = torch.argmax(shift_logits, dim=-1)

            batch_tok_acc_sum = 0.0
            batch_seq_acc_sum = 0.0

            for i in range(input_ids.size(0)):
                t_size = t_sizes[i].item()
                c_size = c_sizes[i].item()


                math_start = t_size          
                math_end = min(t_size + c_size - 2, preds.size(1))  

                if math_end <= math_start:
                    continue

                seq_preds = preds[i, math_start:math_end]
                seq_labels = shift_labels[i, math_start:math_end]

                seq_valid_mask = (seq_labels != -100)
                valid_count = seq_valid_mask.sum()

                if valid_count > 0:
                    seq_correct = (seq_preds == seq_labels) & seq_valid_mask
                    batch_tok_acc_sum += (seq_correct.sum().float() / valid_count.float()).item()
                    if seq_correct.sum() == valid_count:
                        batch_seq_acc_sum += 1.0

            batch_tok_acc = batch_tok_acc_sum / max(input_ids.size(0), 1)
            batch_seq_acc = batch_seq_acc_sum / max(input_ids.size(0), 1)

            total_val_tok_acc += batch_tok_acc
            total_val_seq_acc += batch_seq_acc

            total_lm_loss += lm_loss.item()
            total_jepa_loss += jepa_loss.item()
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'TokAcc': f"{batch_tok_acc:.2f}", 'SeqAcc': f"{batch_seq_acc:.2f}"})
            
            if not logged_samples:
                logged_samples = True
                for i in range(min(3, input_ids.size(0))):
                    t_len = t_sizes[i].item()
                    prompt_ids = input_ids[i, :t_len].unsqueeze(0)
                    
                    sep_tensor = torch.tensor([[self.vocab.sep_idx]], device=self.device)
                    prompt_ids = torch.cat([prompt_ids, sep_tensor], dim=1)
                    
                    generated_ids = self.model.generate(input_ids=prompt_ids, max_len=self.config.get('max_length', 512), eos_idx=self.vocab.eos_idx)
                    amp_text = self.vocab.decode(prompt_ids[0].cpu().tolist(), include_special_tokens=False)
                    target_text = self.vocab.decode(input_ids[i, t_len:t_len+c_sizes[i]].cpu().tolist(), include_special_tokens=False)
                    
                    pred_slice = generated_ids[0, prompt_ids.size(1):].cpu().tolist()
                    pred_text = self.vocab.decode(pred_slice, include_special_tokens=False)
                    
                    sample_inputs.append(amp_text)
                    sample_targets.append(target_text)
                    sample_preds.append(pred_text)

        if logged_samples and sample_inputs:
            combined_samples = list(zip(sample_inputs, sample_targets, sample_preds))
            samples_to_show = min(3, len(combined_samples))
            random_samples = random.sample(combined_samples, samples_to_show)
            
            sample_table = wandb.Table(columns=["Amplitude (Input)", "Ground Truth SqAmp", "Predicted SqAmp"])
            for inp, tgt, pred in random_samples:
                sample_table.add_data(inp, tgt, pred)
                
            wandb.log({"val/predictions": sample_table})

        num_batches = max(len(self.val_loader), 1)
        return (total_loss / num_batches, total_lm_loss / num_batches, total_jepa_loss / num_batches, 
                total_val_tok_acc / num_batches, total_val_seq_acc / num_batches)

    def fit(self, resume_from: str = None):
        start_epoch = 1
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.best_val_loss = ckpt['best_val_loss']
            start_epoch = ckpt['epoch'] + 1
            self.patience_counter = 0  
            print(f"Resumed at epoch {start_epoch} | Best val loss so far: {self.best_val_loss:.6f}")

        epochs = self.config.get('epochs', 50)
        for epoch in range(start_epoch, start_epoch + epochs):

            train_loss, train_lm, train_jepa = self.train_epoch(epoch)
            val_loss, val_lm, val_jepa, val_tok_acc, val_seq_acc = self.validate_epoch(epoch)            
            self.scheduler.step()
            
            wandb.log({
                "epoch": epoch, 
                "train/epoch_loss": train_loss, 
                "train/epoch_lm_loss": train_lm,
                "train/epoch_jepa_loss": train_jepa, 
                "val/epoch_loss": val_loss,
                "val/epoch_lm_loss": val_lm, 
                "val/epoch_jepa_loss": val_jepa,
                "val/epoch_tok_acc": val_tok_acc,
                "val/epoch_seq_acc": val_seq_acc
            })
            print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val TokAcc: {val_tok_acc:.4f} | Val SeqAcc: {val_seq_acc:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                phys = self.config.get('physics_model', 'unknown')
                mode = self.config.get('mode', 'unknown')
                notat = self.config.get('notation', 'infix')
                mult = self.config.get('augment_multiplier', 1)
                
                file_name = f"best_model_{phys}_{mode}_{notat}_mult{mult}.pt"
                ckpt_path = os.path.join(self.checkpoint_dir, file_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, ckpt_path)
                print(f"New best model saved: {ckpt_path}")
            else:
                self.patience_counter += 1
                print(f" No improvement for {self.patience_counter} epochs.")
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Model has converged.")
                    break
                    
        wandb.finish()