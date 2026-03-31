import re
import warnings
from typing import List, Set
import time
from tqdm import tqdm

class PhysicsAwareTokenizer:
    def __init__(self, df=None, special_symbols=None, unk_idx=1, notation='infix'):
        self.amps = df['amplitude'].tolist() if df is not None else None
        self.sqamps = df['squared_amplitude'].tolist() if df is not None else None
        
        self.special_symbols = special_symbols or ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "[PRED]"]
        self.unk_idx = unk_idx
        self.notation = notation.lower()
        
        # Precedence for Shunting-Yard (Prefix generation)
        self.precedence = {'NEG': 5, '^': 4, '*': 3, '/': 3, '+': 2, '-': 2, '(': 1, ')': 1, '[': 1, ']': 1, ',': 1}
        self.operator_map = {'NEG': 'neg', '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '^': 'pow'}
        
        identifier_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*(?:_{[^}]+})?(?:\([^)]+\))?(?:_[a-zA-Z0-9_]+)?(?:\^[a-zA-Z0-9_]+)?'
        number_pattern = r'[0-9]+(?:\.[0-9]+)?'
        operator_pattern = r'[-+*/^()\[\],]'
        
        self.lex_pattern = re.compile(f'{identifier_pattern}|{number_pattern}|{operator_pattern}')

    def handle_unary_minus(self, expr: str) -> str:
        """ Replaces unary minus with 'NEG ' (WITH A SPACE) so Shunting-Yard sees it as an operator """
        expr = re.sub(r'(?<=[^\w)\]])\-(?=[\w(\[])', 'NEG ', expr)
        expr = re.sub(r'^-', 'NEG ', expr)
        expr = expr.replace('INT_NEG', 'NEG ')
        return expr

    def preprocess_expression(self, expr: str) -> str:
        expr = expr.replace('**', '^')
        
        expr = expr.replace('^(*)', '^CONJ').replace('(*)', '^CONJ')
        
        expr = re.sub(r'\bi\b(?!\w)', 'I_UNIT', expr)
        expr = re.sub(r'\be\b(?=\^|[+\-*/()| ])', 'E_CHARGE', expr)
        expr = expr.replace('reg_prop', 'REG_PROP')
        
        expr = re.sub(r'\bs_(\d{2,})\b', r'MANDELSTAM_\1', expr)
        expr = re.sub(r'\bs_(\d+)\b(?!\d)', r'S_\1', expr)
        expr = re.sub(r'\bp_(\d+)\b', r'P_\1', expr)
        
        expr = self.handle_unary_minus(expr)
        
        return expr.strip()

    def tokenize_expression(self, expr: str) -> List[str]:
        """ Uses the clean compiler regex to perfectly extract tokens without shattering them """
        expr = expr.replace('\\\\', '\\')
        return self.lex_pattern.findall(expr)

    def to_prefix(self, tokens: List[str]) -> List[str]:
        """ Correctly converts Infix to Prefix (Polish) Notation using Shunting-Yard """
        reversed_tokens = []
        bracket_map = {'(': ')', ')': '(', '[': ']', ']': '['}
        for token in reversed(tokens):
            reversed_tokens.append(bracket_map.get(token, token))
            
        stack = []
        output = []
        
        for token in reversed_tokens:
            if token not in self.precedence:
                output.append(token)
            elif token in ['(', '[']:
                stack.append(token)
            elif token in [')', ']']:
                target = '(' if token == ')' else '['
                while stack and stack[-1] != target:
                    output.append(stack.pop())
                if stack:
                    stack.pop() # Pop '(' or '['
            else:
                while (stack and stack[-1] not in ['(', '['] and 
                       self.precedence.get(stack[-1], 0) >= self.precedence[token]):
                    output.append(stack.pop())
                stack.append(token)
                
        while stack:
            output.append(stack.pop())        
            
        output = [self.operator_map.get(tok, tok) for tok in output]
        return list(reversed(output))

    def tokenize(self, expr: str, is_source: bool = True) -> List[str]:
        try:
            expr = self.preprocess_expression(expr)
            tokens = self.tokenize_expression(expr)
            if self.notation == 'prefix':
                return self.to_prefix(tokens)
            return tokens
        except Exception as e:
            warnings.warn(f"Tokenization failed for '{expr}': {e}")
            return [self.special_symbols[self.unk_idx]]

    def build_shared_vocab(self) -> Set[str]:
        if self.amps is None or self.sqamps is None: return set()
            
        vocab_set = set()
        start_time = time.time()
        for expr in tqdm(self.amps, desc="Processing amplitude vocab"):
            vocab_set.update(self.tokenize(expr, is_source=True))
        for expr in tqdm(self.sqamps, desc="Processing squared amplitude vocab"):
            vocab_set.update(self.tokenize(expr, is_source=False))
            
        print(f"Shared vocab built in {time.time() - start_time:.2f}s, unique tokens: {len(vocab_set)}")
        return vocab_set


class SymbolicVocab:
    def __init__(self, tokens: set, special_symbols: list):
        self.token_list = special_symbols + sorted(list(tokens - set(special_symbols)))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.token_list)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        self.pad_idx = self.token_to_idx.get("<PAD>", 0)
        self.unk_idx = self.token_to_idx.get("<UNK>", 1)
        self.bos_idx = self.token_to_idx.get("<BOS>", 2)
        self.eos_idx = self.token_to_idx.get("<EOS>", 3)
        self.sep_idx = self.token_to_idx.get("<SEP>", 4)
        self.pred_idx = self.token_to_idx.get("[PRED]", 5) 
        
        self.pad_tok, self.unk_tok = "<PAD>", "<UNK>"

    def encode(self, tokens: list) -> list:
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list, include_special_tokens: bool = True) -> str:
        if include_special_tokens:
            tokens = [self.idx_to_token.get(idx, self.unk_tok) for idx in indices]
        else:
            tokens = [self.idx_to_token.get(idx, self.unk_tok) for idx in indices 
                     if idx not in {self.pad_idx, self.bos_idx, self.eos_idx, self.sep_idx, self.pred_idx}]
        return ' '.join(tokens)

    def __len__(self) -> int: return len(self.token_list)