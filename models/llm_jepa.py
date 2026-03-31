import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        seq_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:, :seq_len, :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class LLM_JEPA(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 512, 
                 dropout: float = 0.1, lambda_jepa: float = 1.0):
        super(LLM_JEPA, self).__init__()
        
        self.tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.pos_encoding = PositionalEncoding(emb_size, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.generator = nn.Linear(emb_size, vocab_size)
        self.lambda_jepa = lambda_jepa

    def forward(self, input_ids: Tensor, attention_mask: Tensor, t_sizes: Tensor, c_sizes: Tensor):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        emb = self.pos_encoding(self.tok_emb(input_ids))
        
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
        hidden_states_gen = self.transformer(emb, mask=causal_mask)
        logits = self.generator(hidden_states_gen)

        heads = self.transformer.layers[0].self_attn.num_heads
        mask_3d = attention_mask.squeeze(1).repeat_interleave(heads, dim=0) 
        
        hidden_states_jepa = self.transformer(emb, mask=mask_3d)
        
        jepa_loss = 0.0
        for i in range(batch_size):
            t_end_idx = min(int(t_sizes[i].item()) - 1, seq_len - 1)
            c_end_idx = min(int(t_sizes[i].item() + c_sizes[i].item()) - 1, seq_len - 1)
            
            if t_end_idx < 0 or c_end_idx < 0:
                continue
                
            text_rep = hidden_states_jepa[i, t_end_idx, :]
            code_rep = hidden_states_jepa[i, c_end_idx, :]
            
            text_rep = F.normalize(text_rep, p=2, dim=0, eps=1e-8)
            code_rep = F.normalize(code_rep, p=2, dim=0, eps=1e-8)
            
            cos_sim = F.cosine_similarity(text_rep.unsqueeze(0), code_rep.unsqueeze(0)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            jepa_loss += (1.0 - cos_sim.squeeze())
            
        jepa_loss = (jepa_loss / max(batch_size, 1)) * self.lambda_jepa
        return logits, jepa_loss
        
    def generate(self, input_ids: Tensor, max_len: int, eos_idx: int, repetition_penalty: float = 1.0, temperature: float = 0.8):

        self.eval()
        prompt_len = input_ids.size(1)  # <--- RECORD PROMPT LENGTH HERE

        with torch.no_grad():
            for _ in range(max_len):
                seq_len = input_ids.size(1)
                
                # Standard causal mask for generation
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
                
                emb = self.pos_encoding(self.tok_emb(input_ids))
                hidden_states = self.transformer(emb, mask=causal_mask)
                
                next_token_logits = self.generator(hidden_states[:, -1, :])
                
                if repetition_penalty != 1.0:
                    for i in range(input_ids.size(0)):
                        if seq_len > prompt_len:
                            generated_tokens = set(input_ids[i, prompt_len:].tolist())
                            for token_idx in generated_tokens:
                                if next_token_logits[i, token_idx] < 0:
                                    next_token_logits[i, token_idx] *= repetition_penalty
                                else:
                                    next_token_logits[i, token_idx] /= repetition_penalty
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == eos_idx:
                    break
                    
        return input_ids