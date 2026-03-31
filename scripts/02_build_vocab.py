import os
import pickle
import pandas as pd
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from data.tokenizer import PhysicsAwareTokenizer, SymbolicVocab

def build_and_save_vocab(physics_model: str, output_dir: str, notation: str):
    train_csv_path = os.path.join(output_dir, f"{physics_model}_train.csv")
    print(f"Loading data from {train_csv_path}")
    
    df = pd.read_csv(train_csv_path)
    print(f"Verified: Loaded {len(df)} training samples")
    
    tokenizer = PhysicsAwareTokenizer(df=df, notation=notation)
    shared_tokens = tokenizer.build_shared_vocab()
    vocab = SymbolicVocab(tokens=shared_tokens, special_symbols=tokenizer.special_symbols)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tok_path = os.path.join(output_dir, f'{physics_model}_{notation}_tokenizer.pkl')
    voc_path = os.path.join(output_dir, f'{physics_model}_{notation}_vocab.pkl')
    
    with open(tok_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(voc_path, 'wb') as f:
        pickle.dump(vocab, f)
        
    print(f"Tokenizer and Vocab saved to {output_dir} for {physics_model} ({notation} mode)")
    print(f"Total Vocabulary Size: {len(vocab)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Vocab for QED or QCD")
    parser.add_argument('--physics_model', type=str, required=True, choices=['QED', 'QCD'])
    parser.add_argument('--notation', type=str, default='infix', choices=['infix', 'prefix'])
    parser.add_argument('--augment_multiplier', type=int, default=1)
    args = parser.parse_args()
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", f"processed_{args.augment_multiplier}x")
    build_and_save_vocab(args.physics_model, OUTPUT_DIR, args.notation)