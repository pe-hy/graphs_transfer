"""
Tokenizer creation and vocabulary building module.
"""
import json
import os
from typing import Dict, List, Set

import hydra
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


SPECIAL_TOKENS = ["[BOS]", "[PAD]", "[MASK]", "[UNK]", "[EOS]"]


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for tokenizer creation.
    
    Args:
        cfg: Configuration object from Hydra.
    """
    vocab = get_vocab(cfg)
    tokenizer = create_tokenizer(vocab, cfg)
    print(f"Tokenizer saved to: {cfg.data.tokenizer_path}")


def create_tokenizer(vocab: Set[str], cfg: DictConfig) -> Tokenizer:
    """
    Create and save a WordLevel tokenizer with the given vocabulary.
    
    Args:
        vocab: Set of vocabulary tokens.
        cfg: Configuration object from Hydra.
        
    Returns:
        The created tokenizer.
    """
    # Calculate number of regular tokens (excluding special tokens)
    regular_token_count = len(vocab)
    
    # Create vocabulary dictionary with indices
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}
    
    # Initialize tokenizer with complete vocabulary
    tokenizer = Tokenizer(WordLevel(vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    # Calculate total vocabulary size including special tokens
    total_token_count = regular_token_count + len(SPECIAL_TOKENS)
    
    # Get and print special token IDs
    bos_id = tokenizer.encode("[BOS]").ids[0]
    eos_id = tokenizer.encode("[EOS]").ids[0]
    out_id = tokenizer.encode(f"{cfg.data.split_str}").ids[0]
    
    print(f"Regular vocabulary size (excluding special tokens): {regular_token_count}")
    print(f"Total vocabulary size (including special tokens): {total_token_count}")
    print(f"BOS token ID: {bos_id}")
    print(f"EOS token ID: {eos_id}")
    print(f"OUT token ID: {out_id}")
    
    # Save tokenizer
    save_path = cfg.data.tokenizer_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    return tokenizer


def get_vocab(cfg: DictConfig) -> Set[str]:
    """
    Build vocabulary from training and validation files.
    
    Args:
        cfg: Configuration object containing file paths.
        
    Returns:
        Set of unique tokens for vocabulary.
    """
    train_data = _load_json_file(cfg.data.train_file)
    val_data = _load_json_file(cfg.data.test_file)
    
    combined_data = train_data + val_data
    
    # Extract both input and output text from the new data format
    all_text = []
    for item in combined_data:
        all_text.append(item["input"])
        all_text.append(item["output"])
    
    # Join all text and split into tokens
    combined_text = " ".join(all_text)
    vocab = set(combined_text.split())

    vocab.add(cfg.data.split_str)
    
    return vocab


def _load_json_file(file_path: str) -> List[Dict]:
    """
    Helper function to load and parse JSON files.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Parsed JSON content.
    """
    with open(file_path, "rb") as f:
        return json.load(f)


if __name__ == "__main__":
    main()