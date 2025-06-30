import os
import glob
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM
import hydra
from omegaconf import DictConfig
from data import get_data_for_inference, get_tokenizer, Datamodule

def calculate_metrics(results_dict, tokenizer, delimiter_str):
    """
    Calculate token-level and exact match accuracy after the delimiter.
    
    Args:
        results_dict: Dictionary containing predictions and ground truth
        tokenizer: The tokenizer used
        delimiter_str: The delimiter string from config (e.g., "OUT")
    """
    overall_metrics = {
        'token_full_accuracy': [],
        'exact_match_accuracy': []
    }
    
    dataset_metrics = {}
    
    # Get token ID for the delimiter
    delimiter_token_id = tokenizer.encode(delimiter_str, add_special_tokens=False)[0]
    
    for datapath, data in results_dict.items():
        # Initialize metrics for this dataset
        dataset_metrics[datapath] = {
            'token_full_accuracy': [],
            'exact_match_accuracy': []
        }
        
        gt_solutions_ids = data['gt_solutions_ids']
        predictions_ids = data['predictions_ids']
        
        for i in range(len(gt_solutions_ids)):
            gt_ids = gt_solutions_ids[i]
            pred_ids = predictions_ids[i]
            
            gt_post_delimiter = gt_ids
            pred_post_delimiter = pred_ids
            # Calculate token-by-token accuracy after delimiter
            min_len = min(len(gt_post_delimiter), len(pred_post_delimiter))
            matches = 0
            for j in range(min_len):
                if gt_post_delimiter[j] == pred_post_delimiter[j]:
                    matches += 1
            token_acc = matches / max(len(gt_post_delimiter), len(pred_post_delimiter)) if max(len(gt_post_delimiter), len(pred_post_delimiter)) > 0 else 1.0
            
            # Calculate exact match after delimiter
            exact_match = 1.0 if gt_post_delimiter == pred_post_delimiter else 0.0
            
            # Add metrics for this example
            dataset_metrics[datapath]['token_full_accuracy'].append(token_acc)
            dataset_metrics[datapath]['exact_match_accuracy'].append(exact_match)
            
            # Add to overall metrics too
            overall_metrics['token_full_accuracy'].append(token_acc)
            overall_metrics['exact_match_accuracy'].append(exact_match)
    
    # Calculate averages for each dataset
    for datapath in dataset_metrics:
        for metric in dataset_metrics[datapath]:
            dataset_metrics[datapath][metric] = np.mean(dataset_metrics[datapath][metric]) if dataset_metrics[datapath][metric] else 0
    
    # Calculate overall averages
    for metric in overall_metrics:
        overall_metrics[metric] = np.mean(overall_metrics[metric]) if overall_metrics[metric] else 0
    
    return overall_metrics, dataset_metrics

@hydra.main(
    config_path="../config",
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig):
    batch_size = cfg.inference.batch_size
    num_workers = cfg.data.num_workers
    delimiter_str = cfg.data.split_str
    
    # Get HF model for batch inference
    model_dir = Path(f"{cfg.inference.modelpath}")
    state_dict = torch.load(model_dir / "model.pth")
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        # state_dict=state_dict,
        attn_implementation="flash_attention_2",
    )
    
    hf_model.cuda()
    hf_model.eval()
    
    tokenizer = get_tokenizer(cfg)
    
    # Load the data from a directory
    datapaths = glob.glob(f"{cfg.inference.datapath}/*.json")
    
    # Tokenize it
    tokenized_datasets = get_data_for_inference(cfg, datapaths, tokenizer)
    
    results_dict = {}
    
    # For all tokenized datasets
    for i, tok_dataset in tqdm(enumerate(tokenized_datasets)):
        # Get the corresponding datapath
        current_path = datapaths[i]
        
        data = Datamodule(tok_dataset, batch_size, num_workers, tokenizer)
        data.connect(max_seq_length=cfg.model.block_size)
        data.setup()
        
        test_set = data.test_dataset
        
        # Use ":" as the delimiter
        delimiter_token_id = tokenizer.encode(delimiter_str, add_special_tokens=False)[0]
        
        # Initialize lists for this dataset
        solutions_text = []
        solutions_ids = []
        prompts_text = []
        prompts_ids = []
        
        for sample in tqdm(test_set, desc=f"Processing {os.path.basename(current_path)}"):
            input_ids = sample["input_ids"]
            
            try:
                # Find the colon delimiter
                split_index = input_ids.index(delimiter_token_id)
                # Find the EOS token
                end_index = input_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in input_ids else len(input_ids)
            except ValueError:
                print(f"Warning: Could not find delimiter {delimiter_str} or EOS token in sample")
                print(tokenizer.decode(input_ids, skip_special_tokens=True))
                continue
                
            # Take everything up to ":" token (including it)
            prompt_ids = input_ids[:split_index + 1]
            
            # Decode to text, add BOS token at start
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_prompt = tokenizer.bos_token + " " + prompt_text
            # Solution is everything after ":" up to EOS
            solution_text = tokenizer.decode(input_ids[split_index+1:end_index], skip_special_tokens=True)
            solution_ids = input_ids[split_index+1:end_index]
            
            # Re-encode prompt with BOS token
            prompt_with_bos = tokenizer.encode(
                full_prompt, add_special_tokens=False
            )
            
            prompts_ids.append(prompt_with_bos)
            prompts_text.append(full_prompt)
            solutions_text.append(solution_text)
            solutions_ids.append(solution_ids)
        
        # Store the lists in the dictionary for this datapath
        results_dict[current_path] = {
            'prompts_ids': prompts_ids,
            'prompts_text': prompts_text,
            'gt_solutions_text': solutions_text,
            'gt_solutions_ids': solutions_ids,
            'predictions_text': [],
            'predictions_ids': []
        }
        
        # Process in batches for generation
        for b in trange(0, len(prompts_ids), batch_size, desc=f"Generating predictions for {os.path.basename(current_path)}"):
            batch = prompts_ids[b:min(b + batch_size, len(prompts_ids))]
            batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
            
            with torch.no_grad():
                outputs = hf_model.generate(
                    input_ids=inputs["input_ids"],
                    pad_token_id=tokenizer.pad_token_id,
                    attention_mask=inputs["attention_mask"],
                    max_length=cfg.model.block_size,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Process each generated sequence
            batch_outputs = outputs.tolist()
            for j, output_ids in enumerate(batch_outputs):
                # Find the delimiter token in the output
                try:
                    split_index = output_ids.index(delimiter_token_id)
                    # Find the EOS token
                    end_index = output_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in output_ids else len(output_ids)
                    
                    # Extract everything after the delim up to EOS
                    generated_ids = output_ids[split_index+1:end_index]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                except ValueError:
                    print(f"Warning: Could not find delimiter or EOS in generated output")
                    # If no delimiter found, use empty list/string as prediction
                    generated_ids = []
                    generated_text = ""
                
                results_dict[current_path]['predictions_text'].append(generated_text)
                results_dict[current_path]['predictions_ids'].append(generated_ids)
    
    # Calculate metrics
    overall_metrics, dataset_metrics = calculate_metrics(results_dict, tokenizer, cfg.data.split_str)
    
    # Add metrics to results_dict
    results_dict['overall_metrics'] = overall_metrics
    results_dict['dataset_metrics'] = dataset_metrics
    
    # Print metrics
    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nDataset Metrics:")
    for datapath, metrics in dataset_metrics.items():
        print(f"\n{os.path.basename(datapath)}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Create output directory
    output_dir = Path("./temp/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the results dictionary as pickle
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results saved to {output_dir / 'results.pkl'}")

if __name__ == "__main__":
    main()