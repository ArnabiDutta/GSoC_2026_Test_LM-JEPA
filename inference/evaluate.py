import os
import sys
import time
import pickle
import argparse
import random
import pandas as pd
import numpy as np
import torch
import sympy
import signal
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.llm_jepa import LLM_JEPA

def timeout_handler(signum, frame):
    raise TimeoutError("SymPy Timeout")

def levenshtein_accuracy(y_true_str, y_pred_str):
    """ Calculates true Levenshtein (Edit) Accuracy for syntax trees """
    true_tokens = y_true_str.split()
    pred_tokens = y_pred_str.split()
    
    if not true_tokens or not pred_tokens:
        return 0.0
        
    n, m = len(true_tokens), len(pred_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if true_tokens[i - 1] == pred_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution
                           
    edit_distance = dp[n][m]
    max_len = max(n, m)
    
    return max(0.0, (1.0 - edit_distance / max_len)) * 100.0


def token_accuracy(y_true_str, y_pred_str):
    """ Calculates positional token-by-token accuracy """
    true_tokens = y_true_str.split()
    pred_tokens = y_pred_str.split()
    
    if not true_tokens or not pred_tokens:
        return 0.0
        
    matches = sum(1 for t, p in zip(true_tokens, pred_tokens) if t == p)
    
    max_len = max(len(true_tokens), len(pred_tokens))
    
    return (matches / max_len) * 100.0


def prefix_to_infix(tokens):
    """ Converts Prefix AST back to human-readable Infix for SymPy """
    stack = []
    ops = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
    
    try:
        for token in reversed(tokens):
            if token in ops:
                if len(stack) < 2: return "" 
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append(f"({op1} {ops[token]} {op2})")
            else:
                stack.append(token)
        return stack[0] if stack else ""
    except:
        return ""

def is_algebraically_equivalent(pred_str, gt_str, notation='infix'):
    """ Uses SymPy with a strict 2-second timeout to prevent infinite hangs """
    if not pred_str or not gt_str: return False
    
    try:
        if notation == 'prefix':
            pred_infix = prefix_to_infix(pred_str.split())
            gt_infix = prefix_to_infix(gt_str.split())
        else:
            # Reconstruct infix string naturally for sympy
            pred_infix = pred_str.replace('^', '**').replace('INT_NEG', '-')
            gt_infix = gt_str.replace('^', '**').replace('INT_NEG', '-')

        if not pred_infix or not gt_infix: return False

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)
        
        pred_expr = parse_expr(pred_infix, evaluate=False)
        gt_expr = parse_expr(gt_infix, evaluate=False)
        
        diff = sympy.simplify(pred_expr - gt_expr)
        
        signal.alarm(0)
        return diff == 0
    except Exception:
        signal.alarm(0)
        return False

def evaluate_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluation for {args.physics_model} on {device}")

    data_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{args.augment_multiplier}x")
    test_data_path = os.path.join(data_dir, f"{args.physics_model}_test.csv")
    
    tok_path = os.path.join(data_dir, f"{args.physics_model}_{args.notation}_tokenizer.pkl")
    voc_path = os.path.join(data_dir, f"{args.physics_model}_{args.notation}_vocab.pkl")
    output_dir = os.path.join(PROJECT_ROOT, "inference", args.mode, args.physics_model)

    print("Loading test data, tokenizer, and vocabulary")
    test_df = pd.read_csv(test_data_path)
    
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(voc_path, 'rb') as f:
        vocab = pickle.load(f)

    lambda_jepa = 1.0 if args.mode == 'jepa' else 0.0
    model = LLM_JEPA(
        vocab_size=len(vocab),
        emb_size=512,        
        nhead=8,             
        num_layers=3,        
        lambda_jepa=lambda_jepa
    ).to(device)
    
    if args.wt_path:
        ckpt_path = args.wt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Provided weights path not found: {ckpt_path}")
    else:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "logs", args.mode, "checkpoints", args.physics_model)
        ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(checkpoint_dir, 'best_model_aug.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No weights found in {checkpoint_dir}. Please use --weights_path.")
            
    print(f"Loading model weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
            f"| val_loss={ckpt.get('best_val_loss', '?'):.6f}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    total_samples = len(test_df)
    exact_matches = 0
    algebraic_matches = 0
    levenshtein_scores = []
    token_acc_scores = []
    total_tokens_generated = 0
    total_inference_time = 0.0
    results = []

    print(f"\nEvaluating {total_samples} samples")
    
    for idx, row in tqdm(test_df.iterrows(), total=total_samples, desc="Inference"):
        raw_amplitude = row['amplitude']
        raw_ground_truth = row['squared_amplitude']
        
        amp_prefix_tokens = tokenizer.tokenize(raw_amplitude, is_source=True)

        amp_tokens = (["<BOS>"] + amp_prefix_tokens
                      + ["[PRED]"] * args.k_preds
                      + ["<SEP>"])
        amp_ids = vocab.encode(amp_tokens)
        input_tensor = torch.tensor([amp_ids], dtype=torch.long, device=device)
        prompt_len = input_tensor.size(1)
        
        start_time = time.time()
        with torch.no_grad():
            generated_tensor = model.generate(
                input_ids=input_tensor, 
                max_len=args.max_len, 
                eos_idx=vocab.eos_idx,
                repetition_penalty=1.0,
                temperature=0.1
            )
        end_time = time.time()
        
        generated_ids = generated_tensor[0, prompt_len:].cpu().tolist()
        predicted_sq_amp = vocab.decode(generated_ids, include_special_tokens=False).strip()
        
        gt_prefix_tokens = tokenizer.tokenize(raw_ground_truth, is_source=False)
        gt_ids = vocab.encode(gt_prefix_tokens)
        clean_ground_truth = vocab.decode(gt_ids, include_special_tokens=False).strip()
        
        total_inference_time += (end_time - start_time)
        total_tokens_generated += len(generated_ids)
        
        is_exact = (predicted_sq_amp == clean_ground_truth)
        if is_exact: exact_matches += 1
            
        lev_score = levenshtein_accuracy(clean_ground_truth, predicted_sq_amp)
        levenshtein_scores.append(lev_score)

        tok_acc = token_accuracy(clean_ground_truth, predicted_sq_amp)
        token_acc_scores.append(tok_acc)
        
        is_algebraic = True if is_exact else is_algebraically_equivalent(predicted_sq_amp, clean_ground_truth, args.notation)
        if is_algebraic: algebraic_matches += 1

        results.append({
            'amplitude': raw_amplitude,
            'gt_prefix': clean_ground_truth,
            'pred_prefix': predicted_sq_amp,
            'levenshtein_acc': lev_score,
            'token_acc': tok_acc,
            'exact_match': is_exact,
            'algebraic_match': is_algebraic
        })

    exact_acc = (exact_matches / total_samples) * 100
    alg_acc = (algebraic_matches / total_samples) * 100
    avg_lev = np.mean(levenshtein_scores)
    avg_tok = np.mean(token_acc_scores)
    tokens_per_sec = total_tokens_generated / total_inference_time if total_inference_time > 0 else 0

    print("\n" + "="*70)
    print(f"{args.physics_model} {args.notation} EVALUATION METRICS ({args.mode.upper()}) ({args.augment_multiplier}) ")
    print("="*70)
    print(f"Total Samples:       {total_samples}")
    print(f"1. Sequence Acc:     {exact_acc:.2f}% ({exact_matches}/{total_samples})  <-- Exact Match")
    print(f"2. Token Accuracy:   {avg_tok:.2f}%")
    print(f"3. Algebraic Match:  {alg_acc:.2f}% ({algebraic_matches}/{total_samples})  <-- SymPy Verified")
    print(f"4. Structural (Lev): {avg_lev:.2f}%")
    print(f"5. Inference Speed:  {tokens_per_sec:.2f} tokens/sec")
    print("="*70)
    
    print("\n[ RANDOM SAMPLES ]")
    samples_to_show = min(3, len(results))
    random_samples = random.sample(results, samples_to_show)
    
    for i, sample in enumerate(random_samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"AMP  (Input) : {sample['amplitude']}")
        print(f"GT   (SqAmp) : {sample['gt_prefix']}")
        print(f"PRED (SqAmp) : {sample['pred_prefix']}")
        print(f"Matches      : Seq={sample['exact_match']} | TokenAcc={sample['token_acc']:.1f}% | Alg={sample['algebraic_match']}")
    
    print("\n" + "="*70)

    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f'evaluation_results_{args.notation}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Evaluate SYMBA LLM-JEPA")
    parser.add_argument('--physics_model', type=str, required=True, choices=['QED', 'QCD'])
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'jepa'])
    parser.add_argument('--k_preds', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512) 
    parser.add_argument('--wt_path', type=str, default=None, help="Explicit path to the .pt weights file")
    parser.add_argument('--augment_multiplier', type=int, default=1, help="Multiplier used in preprocessing")
    parser.add_argument('--notation', type=str, default='infix', choices=['infix', 'prefix'])

    args = parser.parse_args()
    evaluate_model(args)