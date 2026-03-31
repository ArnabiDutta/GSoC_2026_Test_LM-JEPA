import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional
import warnings

class JEPADataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, vocab, max_length: int = 512, k_preds: int = 1,
                 skip_too_long: bool = True):
        """
        Args:
            skip_too_long: If True, samples where code_seq alone exceeds max_length are
                           dropped entirely rather than truncated. This prevents the model
                           from learning from half-chopped target expressions.
                           Set False only if you want to keep everything at the cost of
                           truncated targets.
        """
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.k_preds = k_preds
        self.skip_too_long = skip_too_long

        df = df.reset_index(drop=True)
        if skip_too_long:
            kept, skipped = [], 0
            for idx in range(len(df)):
                row = df.iloc[idx]
                sq_ids = self.vocab.encode(
                    self.tokenizer.tokenize(row['squared_amplitude'], is_source=False)
                )
                code_len = len(sq_ids) + 2
                min_text_len = 1 + k_preds
                if code_len + min_text_len > max_length:
                    skipped += 1
                else:
                    kept.append(idx)
            if skipped > 0:
                warnings.warn(
                    f"Skipped {skipped}/{len(df)} samples where code_seq alone "
                    f"exceeds max_length={max_length}. "
                    f"Consider increasing max_length to retain them."
                )
            self.df = df.iloc[kept].reset_index(drop=True)
        else:
            self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        amp_tokens = self.tokenizer.tokenize(row['amplitude'], is_source=True)
        sq_amp_tokens = self.tokenizer.tokenize(row['squared_amplitude'], is_source=False)
        
        amp_ids = self.vocab.encode(amp_tokens)
        sq_amp_ids = self.vocab.encode(sq_amp_tokens)
        
        text_seq = [self.vocab.bos_idx] + amp_ids + [self.vocab.pred_idx] * self.k_preds
        code_seq = [self.vocab.sep_idx] + sq_amp_ids + [self.vocab.eos_idx]
        
        if len(text_seq) + len(code_seq) > self.max_length:

            max_text_len = self.max_length - len(code_seq)
            text_seq = (
                [self.vocab.bos_idx]
                + amp_ids[:max_text_len - 1 - self.k_preds]
                + [self.vocab.pred_idx] * self.k_preds
            )
            warnings.warn(
                f"Amplitude truncated to fit max_length={self.max_length} "
                f"(amp_ids trimmed from {len(amp_ids)} to {max_text_len - 1 - self.k_preds} tokens). "
                f"Target (code_seq) is preserved intact."
            )

        input_ids = text_seq + code_seq
        labels = [-100] * len(text_seq) + code_seq
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            't_start': 0,
            't_size': len(text_seq),
            'c_start': len(text_seq),
            'c_size': len(code_seq)
        }


def jepa_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_idx: int) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.full((batch_size, 1, max_len, max_len), float('-inf'), dtype=torch.float32)
    
    def additive_causal_mask(size: int):
        mask = torch.zeros((size, size), dtype=torch.float32)
        mask[torch.triu(torch.ones(size, size), diagonal=1) == 1] = float('-inf')
        return mask

    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        t_start, t_size = item['t_start'], item['t_size']
        c_start, c_size = item['c_start'], item['c_size']
        
        input_ids[i, :seq_len] = item['input_ids']
        labels[i, :seq_len] = item['labels']
        
        attention_mask[i, 0, t_start:t_start+t_size, t_start:t_start+t_size] = additive_causal_mask(t_size)
        attention_mask[i, 0, c_start:c_start+c_size, c_start:c_start+c_size] = additive_causal_mask(c_size)

        if seq_len < max_len:
            attention_mask[i, 0, seq_len:, :] = 0.0

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        't_size': torch.tensor([item['t_size'] for item in batch], dtype=torch.long),
        'c_size': torch.tensor([item['c_size'] for item in batch], dtype=torch.long)
    }