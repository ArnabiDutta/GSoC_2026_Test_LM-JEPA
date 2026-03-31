import os
import re
import glob
import random
import pandas as pd
import argparse
import sys
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def normalize_indices(amp, sq_amp, seed):
    """ Randomly permutes generic indices while protecting kinematic variables. """
    random.seed(seed)
    

    protected_pattern = re.compile(r'\b([sp]_\d{1,2}|m_[a-zA-Z]+)\b')
    protected_vars = list(set(protected_pattern.findall(amp) + protected_pattern.findall(sq_amp)))
    
    for i, var in enumerate(protected_vars):
        amp = re.sub(rf'\b{var}\b', f"__PROT{i}__", amp)
        sq_amp = re.sub(rf'\b{var}\b', f"__PROT{i}__", sq_amp)
        
    amp_indices = re.findall(r'_(\d+)', amp)
    sq_amp_indices = re.findall(r'_(\d+)', sq_amp)
    unique_indices = list(set(amp_indices + sq_amp_indices))
    
    shuffled_targets = list(range(len(unique_indices)))
    random.shuffle(shuffled_targets)
    mapping = {old_idx: str(new_idx) for old_idx, new_idx in zip(unique_indices, shuffled_targets)}
    
    def replacer(match): return f"_{mapping[match.group(1)]}"
        
    norm_amp = re.sub(r'_(\d+)', replacer, amp)
    norm_sq_amp = re.sub(r'_(\d+)', replacer, sq_amp)
    
    for i, var in enumerate(protected_vars):
        norm_amp = norm_amp.replace(f"__PROT{i}__", var)
        norm_sq_amp = norm_sq_amp.replace(f"__PROT{i}__", var)
        
    return norm_amp, norm_sq_amp

def parse_and_split_data(raw_data_dir, output_dir, augment_multiplier=10):
    if not os.path.exists(raw_data_dir):
        print(f" ERROR: The directory '{raw_data_dir}' does not exist")
        return

    base_data = []
    files = glob.glob(os.path.join(raw_data_dir, "*.txt"))
    if len(files) == 0: return
    
    for filepath in files:
        physics_model = "QCD" if "QCD" in filepath else "QED"
        
        with open(filepath, 'r') as f:
            for line in tqdm(f, desc=f"Parsing {os.path.basename(filepath)}"):
                if not line.startswith("Interaction:"): continue
                
                parts = line.strip().split(' : ')
                if len(parts) >= 3:
                    amp = parts[-2].strip()
                    sq_amp = parts[-1].strip()
                    
                    base_data.append({
                        'physics_model': physics_model,
                        'amplitude': amp,
                        'squared_amplitude': sq_amp
                    })
                    
    df_base = pd.DataFrame(base_data).drop_duplicates().reset_index(drop=True)
    print(f"\nTotal Unique Base Interactions Found: {len(df_base)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for phys_mod in ['QED', 'QCD']:
        df_sub = df_base[df_base['physics_model'] == phys_mod]
        if df_sub.empty: continue
            
        print(f"\nProcessing {phys_mod} data ({len(df_sub)} base samples)...")
        df_sub = df_sub.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_train = int(0.8 * len(df_sub))
        n_val = int(0.1 * len(df_sub))
        
        train_base = df_sub.iloc[:n_train]
        val = df_sub.iloc[n_train:n_train+n_val]
        test = df_sub.iloc[n_train+n_val:]
        
        train_augmented_data = []
        
        seen_variations = set()
        
        for _, row in tqdm(train_base.iterrows(), total=len(train_base), desc=f"Augmenting {phys_mod} Train Set"):
            amp = row['amplitude']
            sq_amp = row['squared_amplitude']
            
            for seed in range(augment_multiplier):
                norm_amp, norm_sq_amp = normalize_indices(amp, sq_amp, seed)
                pair_hash = f"{norm_amp}::{norm_sq_amp}"
                
                if pair_hash not in seen_variations:
                    seen_variations.add(pair_hash)
                    train_augmented_data.append({
                        'physics_model': phys_mod,
                        'amplitude': norm_amp,
                        'squared_amplitude': norm_sq_amp
                    })
                    
        train_df = pd.DataFrame(train_augmented_data)
        
        train_df.to_csv(os.path.join(output_dir, f'{phys_mod}_train.csv'), index=False)
        val.to_csv(os.path.join(output_dir, f'{phys_mod}_val.csv'), index=False)
        test.to_csv(os.path.join(output_dir, f'{phys_mod}_test.csv'), index=False)
        
        print(f"Saved {phys_mod}: Train={len(train_df)} (Augmented), Val={len(val)} (Base), Test={len(test)} (Base)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment and Split Dataset")
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, "data", "processed_15x"))
    parser.add_argument('--augment_multiplier', type=int, default=1, help="Set to 1 for NO augmentation, >1 for augmentation.")
    
    args = parser.parse_args()
    parse_and_split_data(args.raw_data_dir, args.output_dir, args.augment_multiplier)