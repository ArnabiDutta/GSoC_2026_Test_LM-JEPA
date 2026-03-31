import os
import pickle
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from functools import partial
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import JEPADataset, jepa_collate_fn
from models.llm_jepa import LLM_JEPA
from training.trainer import JEPATrainer

def main(args):
    is_jepa = args.mode == 'jepa'
    lambda_jepa = 1.0 if is_jepa else 0.0
    
    config = {
            'epochs': 150,               
            'batch_size': 32,
            'learning_rate': 1e-4,       
            'weight_decay': 0.01,
            'max_length': 2048 if args.physics_model == 'QCD' else 512,
            'lambda_jepa': lambda_jepa, 
            'emb_size': 512,             
            'nhead': 8,
            'num_layers': 3, 
            'augment_multiplier': args.augment_multiplier,
            'physics_model': args.physics_model,
            'mode': args.mode,
            'notation': args.notation,
            'early_stopping_patience': 30,
            'wandb_project': 'SYMBA-LLM-JEPA',
            'checkpoint_dir': os.path.join(PROJECT_ROOT, "logs", args.mode, "checkpoints", args.physics_model)
        }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} | Mode: {args.mode.upper()} | Physics: {args.physics_model} | Aug: {args.augment_multiplier}x")

    data_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{args.augment_multiplier}x")
    train_df = pd.read_csv(os.path.join(data_dir, f"{args.physics_model}_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, f"{args.physics_model}_val.csv"))
    
    with open(os.path.join(data_dir, f"{args.physics_model}_{args.notation}_tokenizer.pkl"), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(data_dir, f"{args.physics_model}_{args.notation}_vocab.pkl"), 'rb') as f:
        vocab = pickle.load(f)

    train_dataset = JEPADataset(train_df, tokenizer, vocab,
                                max_length=config['max_length'], skip_too_long=True)
    val_dataset   = JEPADataset(val_df,   tokenizer, vocab,
                                max_length=config['max_length'], skip_too_long=True)
    
    collate_fn = partial(jepa_collate_fn, pad_idx=vocab.pad_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    print(f"Initializing LLM-JEPA Model (lambda_jepa={config['lambda_jepa']})...")
    model = LLM_JEPA(
        vocab_size=len(vocab),
        emb_size=config['emb_size'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        lambda_jepa=config['lambda_jepa']
    )

    trainer = JEPATrainer(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    trainer.fit(resume_from=args.resume_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline or JEPA Model")
    parser.add_argument('--physics_model', type=str, required=True, choices=['QED', 'QCD'], help="QED or QCD dataset")
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'jepa'], help="baseline (lambda=0) or jepa (lambda=1)")
    parser.add_argument('--augment_multiplier', type=int, default=1, help="Multiplier used in preprocessing (for naming the weights)")
    parser.add_argument('--notation', type=str, default='infix', choices=['infix', 'prefix'])
    parser.add_argument('--resume_from', type=str, default=None,help="Path to a .pt checkpoint to resume training from")

    args = parser.parse_args()
    main(args)