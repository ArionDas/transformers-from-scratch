import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        ## creating a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        ## creating a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) ## log space to provide numerical stability
        
        ## applying sine to even position and cosine to odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) ## (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape(1), :]).requires_grad_(False)
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        

class FeedForwardNetwork(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, d_ff) ## w1 & b1
        self.linear2 = nn.Linear(d_ff, d_model) ## w2 & b2
        
    def forward(self, x):
        
        ## (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"
        
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, d_model) ## Wq
        self.w_k = nn.Linear(d_model, d_model) ## Wk
        self.w_v = nn.Linear(d_model, d_model) ## Wv
        
        self.w_o = nn.Linear(d_model, d_model) ## Wo
    
    @staticmethod 
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        ## (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) ## (batch, n_heads, seq_len, seq_len) 
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = torch.softmax(attention_scores, dim=-1) ## (batch, n_heads, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores
        
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) ## (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)   ## (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) ## (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        
        ## (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, value, key, mask, self.dropout)
        
        ## (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        ## (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
    
    
class ResidualConnection(nn.Module):
    
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(ResidualConnection(self_attention_block.d_model, dropout) for _ in range(2))
        
    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
    
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        
        
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)