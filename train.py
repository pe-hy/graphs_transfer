# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data import *
import hydra
from config import hf_config
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.evaluator import Evaluator, DualEvaluator
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor
import json
import os
import wandb
import numpy as np
import pickle
import networkx as nx
from transformers import get_cosine_schedule_with_warmup
from typing import Optional, List

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting training...")


def load_graph(graph_path: str) -> nx.Graph:
    """Load the NetworkX graph from pickle file."""
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        logging.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    except Exception as e:
        logging.error(f"Failed to load graph from {graph_path}: {e}")
        raise


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, train_batches, delimiter_token_id, graph, data_formats, trainer_ckpt_path=None):
        super().__init__()

        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.trainer_ckpt_path = trainer_ckpt_path
        self.train_batches = train_batches
        self.delimiter_token_id = delimiter_token_id
        self.graph = graph
        self.data_formats = data_formats  # List of formats for test sets
        _, self.hf_conf = hf_config.get_configs(cfg)

    def setup(self, stage):

        self.hf_conf["bos_token_id"] = self.preprocessor.tokenizer.convert_tokens_to_ids("[BOS]")
        self.hf_conf["eos_token_id"] = self.preprocessor.tokenizer.convert_tokens_to_ids("[EOS]")
        self.hf_conf["vocab_size"] = len(self.preprocessor.tokenizer.get_vocab())

        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def mask_targets(self, input_ids, target_ids):
        # Find positions with delimiter tokens
        delimiter_positions = (input_ids == self.delimiter_token_id)
        
        # Create a shifted version where positions after delimiter are marked
        # This will include the delimiter itself as True
        first_search_pos = torch.zeros_like(input_ids, dtype=torch.bool)
        first_search_pos[:, 1:] = delimiter_positions.cumsum(dim=1)[:, :-1].bool()
        
        # Create the mask - True for delimiter and positions before it
        mask = ~first_search_pos.cumsum(dim=1).bool()
        
        # Apply the mask to targets, setting masked positions to -100
        return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets_no_mask, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets_no_mask)
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets)
        out, loss = self(idx, targets)
        
        # Compute accuracy only on non-masked tokens
        predictions = torch.argmax(out, dim=-1)
        
        # Create mask for non-masked tokens in the shifted targets
        valid_mask = (targets[:, 1:] != -100)
        
        # Compare predictions with targets only on non-masked positions
        correct = (predictions[:, :-1] == targets[:, 1:]) * valid_mask
        
        # Count total correct predictions and total valid tokens
        total_correct = correct.sum()
        total_tokens = valid_mask.sum()
        
        # Calculate accuracy (avoid division by zero)
        accuracy = total_correct.float() / total_tokens if total_tokens > 0 else torch.tensor(0.0)

        self.log(f"acc", accuracy, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f"loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return {f"loss": loss}
    
    def on_validation_epoch_end(self):
        # Check if we have dual test sets
        has_dual_test = (hasattr(self.trainer.datamodule, 'test_main_dataset') and 
                        hasattr(self.trainer.datamodule, 'test_second_dataset'))
        
        save_path = self.cfg.convert_hf.in_path
        self.llm.model.to(self.llm.preprocessor.device)
        self.llm.save(save_path)
        self.llm.model.to(self.device)

        if has_dual_test:
            # Use dual evaluator for mixed training
            test_main = self.trainer.datamodule.test_main_dataset
            test_second = self.trainer.datamodule.test_second_dataset
            
            evaluator = DualEvaluator(
                self.cfg,
                [test_main, test_second],
                self.data_formats,  # Should be [format1, format2]
                self.preprocessor.tokenizer,
                self.cfg.data.split_str,
                self.global_step,
                self.llm.model,
                self.graph
            )
            
            # Get metrics dictionary from dual evaluator
            metrics = evaluator.evaluate()
            
            # Log all metrics to wandb/Lightning
            for metric_name, value in metrics.items():
                self.log(
                    f"Evaluation/{metric_name}",
                    value,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            
            # Log examples for both test sets to wandb
            if wandb.run is not None:
                for i, result in enumerate(evaluator.results):
                    test_name = result['test_set_name']
                    examples_table = wandb.Table(columns=["Prompt", "Prediction", "Ground Truth", "Exact Match", "Valid Path"])
                    
                    # Select a few random indices to log
                    num_examples = min(self.cfg.wandb.num_examples_reported, len(result['prompts']))
                    indices = np.random.choice(len(result['prompts']), num_examples, replace=False)
                    
                    for idx in indices:
                        prompt = self.preprocessor.tokenizer.decode(result['prompts'][idx], skip_special_tokens=True)
                        pred = result['predictions_after_delimiter'][idx]
                        gt = result['gts'][idx]
                        exact_match = pred == gt
                        
                        # Get path validity if available
                        valid_path = False
                        if result['path_validities'] and idx < len(result['path_validities']):
                            valid_path = result['path_validities'][idx]
                        
                        examples_table.add_data(prompt, pred, gt, exact_match, valid_path)
                    
                    wandb.log({f"evaluation_examples_{test_name}": examples_table}, step=self.global_step)
        else:
            # Fallback to single test set evaluation
            test = self.trainer.datamodule.dataset["test"]
            
            # Use the first format if available, otherwise default to "standard"
            data_format = self.data_formats[0] if self.data_formats else "standard"
            
            evaluator = Evaluator(
                self.cfg,
                test,
                self.preprocessor.tokenizer,
                self.cfg.data.split_str,
                self.global_step,
                self.llm.model,
                self.graph,
                data_format
            )
            
            # Get metrics dictionary from evaluator
            metrics = evaluator.evaluate()
            
            # Log all metrics to wandb/Lightning
            for metric_name, value in metrics.items():
                self.log(
                    f"Evaluation/{metric_name}",
                    value,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            
            if wandb.run is not None and hasattr(evaluator, 'predictions_after_delimiter'):
                # Create example table
                examples_table = wandb.Table(columns=["Prompt", "Prediction", "Ground Truth", "Exact Match", "Valid Path"])
                
                # Select a few random indices to log
                num_examples = min(self.cfg.wandb.num_examples_reported, len(evaluator.prompts))
                indices = np.random.choice(len(evaluator.prompts), num_examples, replace=False)
                
                for i in indices:
                    prompt = self.preprocessor.tokenizer.decode(evaluator.prompts[i], skip_special_tokens=True)
                    pred = evaluator.predictions_after_delimiter[i]
                    gt = evaluator.gts[i]
                    exact_match = pred == gt
                    
                    # Get path validity if available
                    valid_path = False
                    if hasattr(evaluator, 'path_validities') and i < len(evaluator.path_validities):
                        valid_path = evaluator.path_validities[i]
                    
                    examples_table.add_data(prompt, pred, gt, exact_match, valid_path)
                
                wandb.log({"evaluation_examples": examples_table}, step=self.global_step)

    def configure_optimizers(self):

        if self.cfg.optim.lr_type == "linear":
            warmup_steps = self.cfg.optim.warmup_steps
            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: (step + 1) / warmup_steps
            )
            return [optimizer], [scheduler]
        elif self.cfg.optim.lr_type == "linear-reg":
            warmup_steps = self.cfg.optim.warmup_steps
            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: (step + 1) / warmup_steps
            )
            return [optimizer], [scheduler]
        else:
            n_steps = self.cfg.model.epochs * self.train_batches
            warmup_steps = self.cfg.optim.warmup_steps

            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay, betas=(0.9, 0.95)
            )
            
            # Warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, total_iters=warmup_steps
            )
            
            # Cosine scheduler with minimum at 0.5 * initial_lr
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_steps - warmup_steps, eta_min=self.cfg.optim.lr * 0.3
            )
            
            # Combine them
            combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            scheduler = {
                "scheduler": combined_scheduler,
                "interval": "step",
            }
            return [optimizer], [scheduler]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.llm(idx, targets)


@hydra.main(
    config_path="config",
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig):
    L.seed_everything(42, workers=True)
    conf, _ = hf_config.get_configs(cfg)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print("Current model configuration:")
    print(f"n_layer: {cfg.model.n_layer}")
    print(f"n_head: {cfg.model.n_head}")
    print(f"n_embd: {cfg.model.n_embd}")
    print(f"Model name: {cfg.model.name}")

    batch_size = cfg.model.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers

    # Load the graph
    graph_path = cfg.data.get("graph_path", "data/graph.pkl")
    graph = load_graph(graph_path)

    tokenizer = get_tokenizer(cfg)
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    conf.padded_vocab_size = len(tokenizer.get_vocab())
    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)

    # Handle test files - check if we have dual test files from command line args
    test_files = None
    data_formats = ["standard"]  # Default
    
    # Check if running with mixed data (dual test files)
    train_file_path = cfg.data.train_file
    if "mixed_" in train_file_path:
        # Extract formats from path like "data/256/mixed_standard_indices/train.json"
        import re
        match = re.search(r'mixed_([^_]+)_([^/]+)', train_file_path)
        if match:
            format1, format2 = match.groups()
            data_formats = [format1, format2]
            
            # Construct test file paths
            base_dir = os.path.dirname(train_file_path)
            test_main_path = os.path.join(base_dir, "test_main.json")
            test_second_path = os.path.join(base_dir, "test_second.json")
            test_files = [test_main_path, test_second_path]
            
            print(f"Detected mixed training with formats: {format1} + {format2}")
            print(f"Test files: {test_main_path}, {test_second_path}")

    datasets = get_data(cfg, tokenizer, test_files=test_files)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)
    data.connect(max_seq_length=cfg.model.block_size)
    data.setup()

    train_size = len(data.train_dataloader())
    trace_start_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=True)[0]

    lit_model = LitLLM(
        model=model, 
        cfg=cfg, 
        train_batches=train_size, 
        preprocessor=preprocessor,
        delimiter_token_id=trace_start_token_id,
        graph=graph,
        data_formats=data_formats
    )

    logger = WandbLogger(
        project=cfg.wandb.proj_name, name=f"{cfg.model.name}", config=wandb_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="acc",  # what metric to track
        dirpath=f"temp/checkpoints/{cfg.model.name}",  # where to save checkpoints
        filename="{epoch:02d}-{acc:.4f}",  # how to name checkpoints
        save_top_k=2,  # save top 3 models
        mode="max",  # lower val_loss is better
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of params:", total_params)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()