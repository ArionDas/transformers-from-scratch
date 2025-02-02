import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 32
    n_heads: int = 32 ## number of heads for query
    n_kv_heads = Optional[int] = None ## number of heads for key and value
    vocab_size: int = -1 ## will be set when we load tokenizer
    multiple_of = 256
    ffn_dim_multiplier = Optional[int] = None
    norm_eps: float = 1e-5
    
    ## Needed for KV cache
    max_batch_size: int = 1
    max_seq_len: int = 1024
    
    device: str = None
    

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, "Vocab size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
            
        self.norm = RMSNorm(args.dim, eps * args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len, self.args.device)
        
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        