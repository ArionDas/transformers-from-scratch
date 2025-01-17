import torch
import torch.nn as nn
import torch.utils.data as Dataset
from typing import Any


class BilingualDataset(nn.Module):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.Tensor(tokenizer_src.token_to_id("[SOS]"), dtype=torch.long)
        self.eos_token = torch.Tensor(tokenizer_src.token_to_id("[EOS]"), dtype=torch.long)
        self.pad_token = torch.Tensor(tokenizer_src.token_to_id("[PAD]"), dtype=torch.long)
        
        
    def __len__(self):
        return len(self.ds)
    
    
    def __getitem__(self, index: Any) -> Any:
        
        ## extracting the text from the dataset
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        
        ## tokenizing the text (in one pass)
        enc_input_tokens = self.tokenizer_src.encoder(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encoder(tgt_text).ids
        
        ## adding special tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 ## 2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 ## 1 for [SOS]
        
        assert enc_num_padding_tokens >= 0 and dec_num_padding_tokens >= 0, "Sequence length is too short"
        
        ## SOS & EOS tokens for encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        ## EOS token for decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_tokens] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), ## (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), ## (1, 1, seq_len) & (seq_len, seq_len)
            "label": label, ## (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
        