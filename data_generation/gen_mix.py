#!/usr/bin/env python3
"""
Mixed Language Data Generator for Clean Framework

This script generates training data that mixes two of the existing data formats
(standard, indices, grammar) and creates separate test sets for each format.
"""

import json
import os
import random
import pickle
import argparse
import networkx as nx
from typing import List, Dict, Any, Tuple


def load_graph(graph_path: str) -> nx.Graph:
    """Load the NetworkX graph from pickle file."""
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    except FileNotFoundError:
        print(f"Graph file {graph_path} not found. Please run gen_graphs.py first to create the graph.")
        raise


def format_standard(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in standard variant."""
    input_str = f"ST S : {start_node} , {end_node}"
    path_str = " , ".join(map(str, path))
    output_str = f"{path_str} : END"
    return {"input": input_str, "output": output_str}


def format_indices(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in indices variant (all indices +10000)."""
    offset_start = start_node + 10000
    offset_end = end_node + 10000
    offset_path = [node + 10000 for node in path]
    
    input_str = f"ST I : {offset_start} , {offset_end}"
    path_str = " , ".join(map(str, offset_path))
    output_str = f"{path_str} : END"
    return {"input": input_str, "output": output_str}


def format_grammar(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in grammar variant (consecutive node pairs)."""
    input_str = f"ST G : {start_node} , {end_node}"
    
    # Create pairs of consecutive nodes in the path
    pairs = []
    for i in range(len(path) - 1):
        pairs.append(f"( {path[i]} , {path[i+1]} )")
    
    pairs_str = " , ".join(pairs)
    output_str = f"{pairs_str} : END"
    return {"input": input_str, "output": output_str}


def format_grammar_indices(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in grammar_indices variant (consecutive node pairs with +10000 offset)."""
    offset_start = start_node + 10000
    offset_end = end_node + 10000
    offset_path = [node + 10000 for node in path]
    
    input_str = f"ST GI : {offset_start} , {offset_end}"
    
    # Create pairs of consecutive nodes in the path (with offset)
    pairs = []
    for i in range(len(offset_path) - 1):
        pairs.append(f"( {offset_path[i]} , {offset_path[i+1]} )")
    
    pairs_str = " , ".join(pairs)
    output_str = f"{pairs_str} : END"
    return {"input": input_str, "output": output_str}


def generate_all_shortest_paths(graph: nx.Graph) -> List[Tuple[int, int, List[int]]]:
    """Generate all shortest paths in the graph."""
    paths = []
    n_nodes = graph.number_of_nodes()
    
    print(f"Computing shortest paths for {n_nodes * (n_nodes - 1)} node pairs...")
    
    for start_node in range(n_nodes):
        for end_node in range(n_nodes):
            if start_node != end_node:
                try:
                    shortest_path = nx.shortest_path(graph, start_node, end_node)
                    paths.append((start_node, end_node, shortest_path))
                except nx.NetworkXNoPath:
                    continue
    
    return paths


def generate_mixed_language_data(
    graph_path: str,
    format1: str,
    format2: str,
    train_format1_count: int,
    train_format2_count: int,
    test_format1_count: int,
    test_format2_count: int,
    output_dir: str = "data/512",
    seed: int = 42
):
    """Generate mixed language training data."""
    
    # Set random seed
    random.seed(seed)
    
    # Load graph and generate all paths
    graph = load_graph(graph_path)
    all_paths = generate_all_shortest_paths(graph)
    
    print(f"Generated {len(all_paths)} total shortest path examples")
    
    # Get formatters
    formatters = {
        "standard": format_standard,
        "indices": format_indices,
        "grammar": format_grammar,
        "grammarindices": format_grammar_indices
    }
    
    if format1 not in formatters or format2 not in formatters:
        raise ValueError(f"Invalid format. Choose from: {list(formatters.keys())}")
    
    formatter1 = formatters[format1]
    formatter2 = formatters[format2]
    
    # Calculate total examples needed
    total_needed = train_format1_count + train_format2_count + test_format1_count + test_format2_count
    
    if total_needed > len(all_paths):
        print(f"Warning: Requested {total_needed} examples but only {len(all_paths)} available")
        print("Will sample with replacement")
        # Sample with replacement
        all_paths = random.choices(all_paths, k=total_needed)
    else:
        # Shuffle and take what we need
        random.shuffle(all_paths)
        all_paths = all_paths[:total_needed]
    
    # Split paths for each purpose
    idx = 0
    train_format1_paths = all_paths[idx:idx + train_format1_count]
    idx += train_format1_count
    
    train_format2_paths = all_paths[idx:idx + train_format2_count]
    idx += train_format2_count
    
    test_format1_paths = all_paths[idx:idx + test_format1_count]
    idx += test_format1_count
    
    test_format2_paths = all_paths[idx:idx + test_format2_count]
    
    # Format the data
    print(f"\nGenerating unique examples:")
    print(f"- Training {format1}: {len(train_format1_paths)} unique examples")
    print(f"- Training {format2}: {len(train_format2_paths)} unique examples")
    print(f"- Test {format1}: {len(test_format1_paths)} examples")
    print(f"- Test {format2}: {len(test_format2_paths)} examples")
    
    # Create unique training data
    train_format1_data = []
    for start_node, end_node, path in train_format1_paths:
        train_format1_data.append(formatter1(start_node, end_node, path))
    
    train_format2_data = []
    for start_node, end_node, path in train_format2_paths:
        train_format2_data.append(formatter2(start_node, end_node, path))
    
    # Calculate ratio for oversampling to 10000 total samples
    total_unique_train = train_format1_count + train_format2_count
    ratio_format1 = train_format1_count / total_unique_train
    ratio_format2 = train_format2_count / total_unique_train
    
    target_total = 10000
    target_format1_count = int(ratio_format1 * target_total)
    target_format2_count = target_total - target_format1_count  # Ensure exact total
    
    print(f"\nOversampling to {target_total} total training examples:")
    print(f"- Format1 ratio: {ratio_format1:.3f} -> {target_format1_count} samples")
    print(f"- Format2 ratio: {ratio_format2:.3f} -> {target_format2_count} samples")
    
    # Oversample format1 data
    if target_format1_count > len(train_format1_data):
        # Need to repeat examples
        oversampled_format1 = []
        while len(oversampled_format1) < target_format1_count:
            oversampled_format1.extend(train_format1_data)
        oversampled_format1 = oversampled_format1[:target_format1_count]
        random.shuffle(oversampled_format1)
    else:
        oversampled_format1 = random.sample(train_format1_data, target_format1_count)
    
    # Oversample format2 data
    if target_format2_count > len(train_format2_data):
        # Need to repeat examples
        oversampled_format2 = []
        while len(oversampled_format2) < target_format2_count:
            oversampled_format2.extend(train_format2_data)
        oversampled_format2 = oversampled_format2[:target_format2_count]
        random.shuffle(oversampled_format2)
    else:
        oversampled_format2 = random.sample(train_format2_data, target_format2_count)
    
    # Combine and shuffle final training data
    train_data = oversampled_format1 + oversampled_format2
    random.shuffle(train_data)
    
    # Create test data
    test_format1_data = [formatter1(s, e, p) for s, e, p in test_format1_paths]
    test_format2_data = [formatter2(s, e, p) for s, e, p in test_format2_paths]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    with open(os.path.join(output_dir, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, "test_main.json"), 'w') as f:
        json.dump(test_format1_data, f, indent=2)
    
    with open(os.path.join(output_dir, "test_second.json"), 'w') as f:
        json.dump(test_format2_data, f, indent=2)
    
    print(f"\nFiles saved to {output_dir}:")
    print(f"- train.json: {len(train_data)} mixed examples (oversampled from {total_unique_train} unique)")
    print(f"- test_main.json: {len(test_format1_data)} {format1} examples")
    print(f"- test_second.json: {len(test_format2_data)} {format2} examples")


def main():
    parser = argparse.ArgumentParser(description="Generate mixed format training data")
    
    parser.add_argument("--graph_path", default="data/graph.pkl", 
                       help="Path to the graph pickle file")
    parser.add_argument("--format1", default="standard",
                       choices=["standard", "indices", "grammar", "grammarindices"],
                       help="First format")
    parser.add_argument("--format2", default="indices", 
                       choices=["standard", "indices", "grammar", "grammarindices"],
                       help="Second format")
    parser.add_argument("--train_format1", type=int, default=1000,
                       help="Number of format1 examples in training")
    parser.add_argument("--train_format2", type=int, default=1000,
                       help="Number of format2 examples in training")
    parser.add_argument("--test_format1", type=int, default=500,
                       help="Number of format1 examples in test")
    parser.add_argument("--test_format2", type=int, default=500,
                       help="Number of format2 examples in test")
    parser.add_argument("--output_dir", default="data/mixed_format",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("Mixed Format Data Generator")
    print("=" * 40)
    print(f"Format 1: {args.format1}")
    print(f"Format 2: {args.format2}")
    print(f"Unique training examples: {args.train_format1} + {args.train_format2} = {args.train_format1 + args.train_format2}")
    print(f"Will be oversampled to: 10000 total (maintaining ratio)")
    print(f"Test: {args.test_format1} + {args.test_format2}")
    print(f"Output: {args.output_dir}")
    print()
    
    generate_mixed_language_data(
        graph_path=args.graph_path,
        format1=args.format1,
        format2=args.format2,
        train_format1_count=args.train_format1,
        train_format2_count=args.train_format2,
        test_format1_count=args.test_format1,
        test_format2_count=args.test_format2,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("Done!")


if __name__ == "__main__":
    main()