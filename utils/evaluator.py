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
import networkx as nx
import re
from datetime import datetime
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional


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


class PathParser:
    """Utility class to parse model outputs based on data format."""
    
    def __init__(self, data_format: str):
        self.data_format = data_format.lower()
        
    def parse_input(self, input_text: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse input to extract start and end nodes."""
        try:
            # Input format: "ST : start_node , end_node"
            match = re.search(r'ST\s*:\s*(\d+)\s*,\s*(\d+)', input_text)
            if match:
                start_node = int(match.group(1))
                end_node = int(match.group(2))
                
                # Convert back from indices format if needed
                if self.data_format == "indices":
                    start_node -= 10000
                    end_node -= 10000
                    
                return start_node, end_node
            return None, None
        except:
            return None, None
    
    def parse_output(self, output_text: str) -> Optional[List[int]]:
        """Parse model output to extract the path."""
        try:
            # Remove " : END" suffix if present
            output_text = output_text.strip()
            if output_text.endswith(": END"):
                output_text = output_text[:-6].strip()
            elif output_text.endswith(":END"):
                output_text = output_text[:-4].strip()
            
            if self.data_format == "standard" or self.data_format == "indices":
                return self._parse_standard_output(output_text)
            elif self.data_format == "grammar":
                return self._parse_grammar_output(output_text)
            else:
                raise ValueError(f"Unknown data format: {self.data_format}")
                
        except Exception as e:
            print(f"Error parsing output '{output_text}': {e}")
            return None
    
    def _parse_standard_output(self, output_text: str) -> Optional[List[int]]:
        """Parse standard/indices format: "node1 , node2 , node3" """
        try:
            if not output_text.strip():
                return []
            
            # Split by comma and extract numbers
            parts = output_text.split(',')
            path = []
            
            for part in parts:
                part = part.strip()
                if part:
                    # Extract number from the part
                    numbers = re.findall(r'\d+', part)
                    if numbers:
                        node = int(numbers[0])
                        # Convert back from indices format if needed
                        if self.data_format == "indices":
                            node -= 10000
                        path.append(node)
            
            return path if path else None
        except:
            return None
    
    def _parse_grammar_output(self, output_text: str) -> Optional[List[int]]:
        """Parse grammar format: "( node1 , node2 ) , ( node2 , node3 )" """
        try:
            if not output_text.strip():
                return []
            
            # Find all pairs in parentheses
            pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', output_text)
            
            if not pairs:
                return None
            
            # Reconstruct path from pairs
            path = [int(pairs[0][0])]  # Start with first node of first pair
            
            for i, (node1, node2) in enumerate(pairs):
                node1, node2 = int(node1), int(node2)
                
                # Check consistency: each pair should connect
                if i > 0 and node1 != path[-1]:
                    print(f"Inconsistent grammar path: expected {path[-1]}, got {node1}")
                    return None
                
                path.append(node2)
            
            return path
        except Exception as e:
            print(f"Error parsing grammar output: {e}")
            return None


class PathValidator:
    """Utility class to validate paths using the graph."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def validate_path(self, start_node: int, end_node: int, path: List[int]) -> Dict[str, Any]:
        """
        Validate a path and return detailed validation results.
        
        Returns:
            Dict with validation metrics:
            - is_valid: bool - overall validity
            - correct_endpoints: bool - path starts and ends correctly
            - valid_edges: bool - all edges in path exist in graph
            - is_shortest: bool - path is a shortest path
            - path_length: int - length of the path
            - shortest_length: int - length of shortest path
        """
        result = {
            "is_valid": False,
            "correct_endpoints": False,
            "valid_edges": False,
            "is_shortest": False,
            "path_length": len(path) if path else 0,
            "shortest_length": 0,
            "error": None
        }
        
        if not path:
            result["error"] = "Empty path"
            return result
        
        try:
            # Check if all nodes are in the graph
            for node in path:
                if node not in self.graph:
                    result["error"] = f"Node {node} not in graph"
                    return result
            
            # Check endpoints
            if path[0] == start_node and path[-1] == end_node:
                result["correct_endpoints"] = True
            else:
                result["error"] = f"Wrong endpoints: path goes from {path[0]} to {path[-1]}, expected {start_node} to {end_node}"
                return result
            
            # Check if all edges exist
            valid_edges = True
            for i in range(len(path) - 1):
                if not self.graph.has_edge(path[i], path[i + 1]):
                    result["error"] = f"Edge ({path[i]}, {path[i + 1]}) does not exist in graph"
                    valid_edges = False
                    break
            
            result["valid_edges"] = valid_edges
            if not valid_edges:
                return result
            
            # Check if it's a shortest path
            try:
                shortest_path = nx.shortest_path(self.graph, start_node, end_node)
                result["shortest_length"] = len(shortest_path)
                result["is_shortest"] = len(path) == len(shortest_path)
                
                if result["correct_endpoints"] and result["valid_edges"]:
                    result["is_valid"] = True
                    
            except nx.NetworkXNoPath:
                result["error"] = f"No path exists between {start_node} and {end_node}"
                return result
                
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
        
        return result


class Evaluator:
    def __init__(self, config, test_set, tokenizer, split_str, step=None, model=None, graph=None, data_format="standard"):
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
        self.graph = graph
        self.data_format = data_format
        
        # Initialize parser and validator
        self.parser = PathParser(data_format)
        self.validator = PathValidator(graph) if graph is not None else None
        
        os.makedirs(self.results_dir, exist_ok=True)

        self.prompts, self.gts = self.get_prompts()
        self.full_predictions = None
        self.predictions_after_delimiter = None
        self.path_validities = None

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
                    
                except ValueError:
                    full_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    after_delimiter = ""
                
                output_texts_concat.append(full_text)
                predictions_after_delimiter.append(after_delimiter)

        return output_texts_concat, predictions_after_delimiter

    def validate_predictions(self, predictions: List[str]) -> Tuple[List[bool], Dict[str, float]]:
        """Validate all predictions and return validity flags and metrics."""
        if self.validator is None:
            print("Warning: No graph provided, cannot validate paths")
            return [False] * len(predictions), {}
        
        validities = []
        validation_results = []
        
        for i, (prediction, prompt_ids) in enumerate(zip(predictions, self.prompts)):
            # Parse the input prompt to get start and end nodes
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            start_node, end_node = self.parser.parse_input(prompt_text)
            
            if start_node is None or end_node is None:
                validities.append(False)
                validation_results.append({"error": "Could not parse input"})
                continue
            
            # Parse the predicted path
            predicted_path = self.parser.parse_output(prediction)
            
            if predicted_path is None:
                validities.append(False)
                validation_results.append({"error": "Could not parse prediction"})
                continue
            
            # Validate the path
            validation_result = self.validator.validate_path(start_node, end_node, predicted_path)
            validities.append(validation_result["is_valid"])
            validation_results.append(validation_result)
        
        # Calculate aggregate metrics
        if validation_results:
            total_valid = sum(validities)
            total_predictions = len(validities)
            
            correct_endpoints = sum(1 for r in validation_results if r.get("correct_endpoints", False))
            valid_edges = sum(1 for r in validation_results if r.get("valid_edges", False))
            shortest_paths = sum(1 for r in validation_results if r.get("is_shortest", False))
            
            path_lengths = [r.get("path_length", 0) for r in validation_results if r.get("path_length", 0) > 0]
            shortest_lengths = [r.get("shortest_length", 0) for r in validation_results if r.get("shortest_length", 0) > 0]
            
            validation_metrics = {
                "path_validity_rate": total_valid / total_predictions if total_predictions > 0 else 0.0,
                "correct_endpoints_rate": correct_endpoints / total_predictions if total_predictions > 0 else 0.0,
                "valid_edges_rate": valid_edges / total_predictions if total_predictions > 0 else 0.0,
                "shortest_path_rate": shortest_paths / total_predictions if total_predictions > 0 else 0.0,
                "avg_predicted_path_length": np.mean(path_lengths) if path_lengths else 0.0,
                "avg_shortest_path_length": np.mean(shortest_lengths) if shortest_lengths else 0.0,
            }
        else:
            validation_metrics = {}
        
        return validities, validation_metrics

    def calculate_metrics(self, predictions, gts):
        """Calculate token-level and exact match accuracy plus path validation metrics"""
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
        
        base_metrics = {
            "token_full_accuracy": np.mean(token_accuracies),
            "exact_match_accuracy": np.mean(exact_matches)
        }
        
        # Add path validation metrics
        validities, validation_metrics = self.validate_predictions(predictions)
        self.path_validities = validities  # Store for later use
        
        # Combine all metrics
        all_metrics = {**base_metrics, **validation_metrics}
        
        return all_metrics

    def save(self, full_predictions, predictions_after_delimiter, gts, metrics=None):
        eval_dir = os.path.join(self.config.eval.results_dir, f"step_{self.step}")
        os.makedirs(eval_dir, exist_ok=True)
        results_file = os.path.join(eval_dir, f"results_{self.num_examples}.json")
        
        results = {
            "full_predictions": full_predictions,
            "predictions_after_delimiter": predictions_after_delimiter,
            "ground_truths": gts,
            "data_format": self.data_format
        }
        
        if metrics:
            results["metrics"] = metrics
            
        if hasattr(self, 'path_validities') and self.path_validities:
            results["path_validities"] = self.path_validities
            
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    def evaluate(self):
        # Get predictions
        full_preds, preds_after_delimiter = self.get_preds()
        
        # Save predictions as attributes so they can be accessed later
        self.full_predictions = full_preds
        self.predictions_after_delimiter = preds_after_delimiter
        
        # Calculate metrics (including path validation)
        metrics = self.calculate_metrics(preds_after_delimiter, self.gts)
        
        # Print metrics
        print(f"\nEvaluation Metrics (Data Format: {self.data_format}):")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        self.save(full_preds, preds_after_delimiter, self.gts, metrics)
        
        # Clean up model to free memory
        del self.hf_model
        torch.cuda.empty_cache()

        return metrics