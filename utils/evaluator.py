import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from transformers import TrainerCallback
from lightning.pytorch.callbacks import Callback
import os
import json
import torch
import numpy as np
import wandb
import pandas as pd
from datetime import datetime
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
import torch
from pathlib import Path
from datetime import datetime


def convert_litgpt_to_hf(cfg):

    out_dir = Path(cfg.convert_hf.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.convert_hf.in_path)
    model_path = out_dir / "pytorch_model.bin"
    model_path = Path(model_path)

    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)

    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        # state_dict=state_dict,
        attn_implementation="flash_attention_2",
    )
    return hf_model


class Evaluator:
    def __init__(self, config, test_set, tokenizer, split_str, step=None, model=None):
        self.config = config
        self.num_examples = config.eval.num_examples
        self.batch_size = config.eval.batch_size
        self.global_step = step
        self.tokenizer = tokenizer
        self.results_dir = config.eval.results_dir
        self.model = model
        self.hf_model = convert_litgpt_to_hf(config)
        self.test_set = test_set
        self.step = step
        self.split_str = split_str
        os.makedirs(self.results_dir, exist_ok=True)

        self.prompts, self.gts = self.get_prompts()
        self.full_predictions = None
        self.predictions_after_delimiter = None

    def get_prompts(self):
        search_token_id = self.tokenizer.encode(self.split_str, add_special_tokens=False)[0]

        gts = []
        prompts = []
        for sample in self.test_set:
            input_ids = sample["input_ids"]
            try:
                split_index = input_ids.index(search_token_id)
                # Find the EOS token
                end_index = input_ids.index(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id in input_ids else len(input_ids)
            except:
                print("ERROR")
                print(input_ids)
                print(sample)
                print(self.tokenizer.decode(input_ids, skip_special_tokens=True))
                continue
                
            # Take everything up to Search: token
            prompt_ids = input_ids[: split_index + 1]

            # Decode to text, add BOS token at start
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_prompt = self.tokenizer.bos_token + " " + prompt_text
            
            # Ground truth is everything AFTER the delimiter up to EOS
            gt_ids = input_ids[split_index+1:end_index]
            gt = self.tokenizer.decode(gt_ids, skip_special_tokens=True)
            # print("FULL_PROMPT: ", full_prompt, "\n\n")
            # print("GT: ", gt, "\n\n")

            # Re-encode with BOS token
            prompt_with_bos = self.tokenizer.encode(
                full_prompt, add_special_tokens=False
            )
            prompts.append(prompt_with_bos)
            gts.append(gt)

        return prompts, gts

    def get_preds(self):
        batch_size = self.batch_size
        data = self.prompts
        tokenizer = self.tokenizer
        output_texts_concat = []
        predictions_after_delimiter = []
        
        search_token_id = self.tokenizer.encode(self.split_str, add_special_tokens=False)[0]

        self.hf_model.cuda()
        self.hf_model.eval()

        for b in trange(0, len(data), batch_size):
            batch = data[b : min(b + batch_size, len(data))]
            batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
            input_prompt = inputs["input_ids"]
            # print("INPUT PROMPT: ", input_prompt, "\n\n")

            outputs = self.hf_model.generate(
                input_ids=input_prompt,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"].to("cuda"),
                max_length=self.config.model.block_size,
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Process each generated sequence
            for output_ids in outputs.tolist():
                # Find the delimiter token in the output
                try:
                    split_index = output_ids.index(search_token_id)
                    # Find the EOS token
                    end_index = output_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in output_ids else len(output_ids)
                    
                    # Get full generated text
                    full_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    
                    # Get just the part after the delimiter
                    after_delimiter = tokenizer.decode(output_ids[split_index+1:end_index], skip_special_tokens=True)
                    
                    # print("AFTER DELIMITER: ", after_delimiter, "\n\n")
                except ValueError:
                    # print(f"Warning: Could not find delimiter or EOS in generated output")
                    full_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    after_delimiter = ""
                
                output_texts_concat.append(full_text)
                predictions_after_delimiter.append(after_delimiter)

        return output_texts_concat, predictions_after_delimiter

    def calculate_metrics(self, predictions, gts):
        """Calculate token-level and exact match accuracy"""
        token_accuracies = []
        exact_matches = []
        
        for pred, gt in zip(predictions, gts):
            # Tokenize prediction and ground truth
            pred_tokens = self.tokenizer.encode(pred, add_special_tokens=False)
            gt_tokens = self.tokenizer.encode(gt, add_special_tokens=False)
            
            # Calculate token-by-token accuracy
            min_len = min(len(pred_tokens), len(gt_tokens))
            matches = 0
            for i in range(min_len):
                if pred_tokens[i] == gt_tokens[i]:
                    matches += 1
                    
            token_acc = matches / max(len(pred_tokens), len(gt_tokens)) if max(len(pred_tokens), len(gt_tokens)) > 0 else 1.0
            token_accuracies.append(token_acc)
            
            # Calculate exact match
            exact_match = 1.0 if pred == gt else 0.0
            exact_matches.append(exact_match)
        
        metrics = {
            "token_full_accuracy": np.mean(token_accuracies),
            "exact_match_accuracy": np.mean(exact_matches)
        }
        
        return metrics

    def save(self, full_predictions, predictions_after_delimiter, gts, metrics=None):
        eval_dir = os.path.join(self.config.eval.results_dir, f"step_{self.step}")
        os.makedirs(eval_dir, exist_ok=True)
        results_file = os.path.join(eval_dir, f"results_{self.num_examples}.json")
        
        results = {
            "full_predictions": full_predictions,
            "predictions_after_delimiter": predictions_after_delimiter,
            "ground_truths": gts
        }
        
        if metrics:
            results["metrics"] = metrics
            
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    def evaluate(self):
        # Get predictions
        full_preds, preds_after_delimiter = self.get_preds()
        
        # Save predictions as attributes so they can be accessed later
        self.full_predictions = full_preds
        self.predictions_after_delimiter = preds_after_delimiter
        
        # Calculate metrics
        metrics = self.calculate_metrics(preds_after_delimiter, self.gts)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        self.save(full_preds, preds_after_delimiter, self.gts, metrics)
        
        # Clean up model to free memory
        del self.hf_model
        torch.cuda.empty_cache()

        return metrics