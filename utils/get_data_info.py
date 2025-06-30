import os
import sys

from data import *
import math
import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

@hydra.main(
    config_path="../config",
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig):

    tokenizer = get_tokenizer(cfg)
    tokenized_datasets = get_data(cfg, tokenizer, for_info=True)
    
    # Get the IDs of special tokens
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Analyze token counts for each split
    for split in tokenized_datasets.keys():
        # Count tokens up to the EOS token or count non-padding tokens if no EOS token
        token_counts = []
        for sample in tokenized_datasets[split]:
            input_ids = sample["input_ids"]
            
            # Find the position of the EOS token
            if eos_token_id in input_ids:
                # Count up to and including the EOS token
                length = input_ids.index(eos_token_id) + 1
            else:
                # Count all non-padding tokens if the EOS token was truncated
                length = sum(1 for token_id in input_ids if token_id != pad_token_id)
            
            token_counts.append(length)
        
        # Calculate statistics
        avg_tokens = np.mean(token_counts)
        max_tokens = np.max(token_counts)
        min_tokens = np.min(token_counts)
        
        # Find the sequences with max and min tokens
        max_idx = int(np.argmax(token_counts))  # Convert numpy.int64 to Python int
        min_idx = int(np.argmin(token_counts))  # Convert numpy.int64 to Python int
        
        # Get the original text if needed (before tokenization)
        max_sequence_tokens = tokenized_datasets[split][max_idx]["input_ids"]
        min_sequence_tokens = tokenized_datasets[split][min_idx]["input_ids"]
        
        # Get the sequences up to the EOS token or without padding
        if eos_token_id in max_sequence_tokens:
            eos_pos = max_sequence_tokens.index(eos_token_id)
            max_sequence_tokens_actual = max_sequence_tokens[:eos_pos+1]
        else:
            max_sequence_tokens_actual = [t for t in max_sequence_tokens if t != pad_token_id]
            
        if eos_token_id in min_sequence_tokens:
            eos_pos = min_sequence_tokens.index(eos_token_id)
            min_sequence_tokens_actual = min_sequence_tokens[:eos_pos+1]
        else:
            min_sequence_tokens_actual = [t for t in min_sequence_tokens if t != pad_token_id]
        
        # Print the statistics
        print(f"\n--- Token Statistics for {split} set (up to EOS token) ---")
        print(f"Number of samples: {len(token_counts)}")
        print(f"Average token count: {avg_tokens:.2f}")
        print(f"Maximum token count: {max_tokens} (sample index: {max_idx})")
        print(f"Minimum token count: {min_tokens} (sample index: {min_idx})")
        
        # Print token distribution
        print("\nToken count distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(token_counts, p):.1f}")
            
        # Optional: Decode and print the max/min sequences (first 100 tokens)
        print("\nMax token sequence:")
        print(tokenizer.decode(max_sequence_tokens_actual))
        
        print("\nMin token sequence:")
        print(tokenizer.decode(min_sequence_tokens_actual))

if __name__ == "__main__":
    main()