# quick_check.py — run from project root
import pickle, pandas as pd, warnings

DATA_DIR = "data/processed_15x"
for phys in ['QED', 'QCD']:
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(f"{DATA_DIR}/{phys}_{split}.csv")
        with open(f"{DATA_DIR}/{phys}_prefix_tokenizer.pkl", 'rb') as f:
            tok = pickle.load(f)
        with open(f"{DATA_DIR}/{phys}_prefix_vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        
        truncated = 0
        amp_lengths, sq_lengths = [], []
        for _, row in df.iterrows():
            a = [vocab.bos_idx] + vocab.encode(tok.tokenize(row['amplitude'], True)) + [vocab.pred_idx]
            c = [vocab.sep_idx] + vocab.encode(tok.tokenize(row['squared_amplitude'], False)) + [vocab.eos_idx]
            amp_lengths.append(len(a))
            sq_lengths.append(len(c))
            if len(a) + len(c) > 512:
                truncated += 1
        
        print(f"{phys} {split}: {truncated}/{len(df)} truncated "
              f"| amp p95={sorted(amp_lengths)[int(0.95*len(amp_lengths))]} "
              f"| sq p95={sorted(sq_lengths)[int(0.95*len(sq_lengths))]}")