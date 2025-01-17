import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import BilingualDataset, causal_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_tokenizer(config, ds, lang):
    
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_tokens=["[UNK]"]))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(ds[lang], trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer


def get_ds(config):
    
    ds_raw = load_dataset("Helsinki-NLP/opus_books", f"{config["lang_src"]}-{config["lang_tgt"]}", split="train")
    
    ## build tokenizer
    tokenizer_src = get_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config["lang_tgt"])
    
    ## train-validation split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))
    
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_raw])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["seq_len"])
    valid_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=False)
    
    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt