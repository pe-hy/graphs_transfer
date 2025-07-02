import networkx as nx
import json
import itertools
import random
import os
import pickle
from collections import Counter
from typing import List, Dict, Any, Tuple

def generate_graph(n_nodes: int, connectivity: str = "medium") -> nx.Graph:
    """
    Generate a connected random graph with specified connectivity level.
    Creates sparser graphs to ensure variable path lengths (1-12).
    
    Args:
        n_nodes: Number of nodes
        connectivity: "low", "medium", or "high" - affects edge density
    
    Returns:
        Connected NetworkX graph
    """
    
    # Much lower edge probabilities to create longer paths
    if connectivity == "low":
        p = max(0.01, 2.0 / n_nodes)  # Very sparse
    elif connectivity == "medium":
        p = max(0.02, 3.0 / n_nodes)  # Sparse
    else:  # high
        p = max(0.03, 4.0 / n_nodes)  # Still sparse but more connected
    
    # Try multiple times to get a good graph with variable path lengths
    best_G = None
    best_avg_path_length = 0
    
    for attempt in range(10):
        # Generate random graph
        G = nx.erdos_renyi_graph(n_nodes, p)
        
        # Ensure graph is connected
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = next(iter(components[i]))
                node2 = next(iter(components[i+1]))
                G.add_edge(node1, node2)
        
        # Check average path length - we want it to be reasonable (2-6)
        try:
            avg_path_length = nx.average_shortest_path_length(G)
            if 2.0 <= avg_path_length <= 8.0:  # Good range for variable paths
                if avg_path_length > best_avg_path_length:
                    best_G = G
                    best_avg_path_length = avg_path_length
        except:
            continue
    
    # If no good graph found, use the last one
    if best_G is None:
        G = nx.erdos_renyi_graph(n_nodes, p)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = next(iter(components[i]))
                node2 = next(iter(components[i+1]))
                G.add_edge(node1, node2)
        best_G = G
        best_avg_path_length = nx.average_shortest_path_length(G)
    
    print(f"Generated random graph with {best_G.number_of_nodes()} nodes and {best_G.number_of_edges()} edges")
    print(f"Average shortest path length: {best_avg_path_length:.2f} (connectivity: {connectivity})")
    
    return best_G

def save_graph(G: nx.Graph, output_base_dir: str):
    """Save the graph for later use in training and evaluation."""
    graph_path = os.path.join(output_base_dir, "graph.pkl")
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {graph_path}")
    return graph_path

def load_graph(input_base_dir: str):
    """Load graph from path."""
    graph_path = os.path.join(input_base_dir, "graph.pkl")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {graph_path}")
    return graph


def format_standard(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in standard variant."""
    input_str = f"ST : {start_node} , {end_node}"
    path_str = " , ".join(map(str, path))
    output_str = f"{path_str} : END"
    
    return {
        "input": input_str,
        "output": output_str
    }

def format_indices(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in indices variant (all indices +10000)."""
    offset_start = start_node + 10000
    offset_end = end_node + 10000
    offset_path = [node + 10000 for node in path]
    
    input_str = f"ST : {offset_start} , {offset_end}"
    path_str = " , ".join(map(str, offset_path))
    output_str = f"{path_str} : END"
    
    return {
        "input": input_str,
        "output": output_str
    }

def format_grammar(start_node: int, end_node: int, path: List[int]) -> Dict[str, str]:
    """Format data in grammar variant (consecutive node pairs)."""
    input_str = f"ST : {start_node} , {end_node}"
    
    # Create pairs of consecutive nodes in the path
    pairs = []
    for i in range(len(path) - 1):
        pairs.append(f"( {path[i]} , {path[i+1]} )")
    
    pairs_str = " , ".join(pairs)
    output_str = f"{pairs_str} : END"
    
    return {
        "input": input_str,
        "output": output_str
    }

def generate_all_shortest_paths(G: nx.Graph, n_nodes: int) -> Dict[Tuple[int, int], List[int]]:
    """Generate all shortest paths in the graph and return as dictionary."""
    paths_dict = {}
    total_pairs = n_nodes * (n_nodes - 1)
    
    print(f"Computing shortest paths for {total_pairs} node pairs...")
    
    for count, (start_node, end_node) in enumerate(itertools.permutations(range(n_nodes), 2)):
        if count % 1000 == 0:
            print(f"Processed {count}/{total_pairs} pairs...")
        
        try:
            shortest_path = nx.shortest_path(G, start_node, end_node)
            paths_dict[(start_node, end_node)] = shortest_path
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between {start_node} and {end_node}")
            continue
    
    return paths_dict

def select_structured_paths(paths_dict: Dict[Tuple[int, int], List[int]], n_nodes: int, total_samples: int) -> List[Tuple[int, int, List[int]]]:
    """
    Select paths using the structured approach:
    - 100 paths: one for each starting vertex (different random end for each)
    - 100 paths: one for each ending vertex (different random start for each)  
    - Remaining: random selection from all possible pairs
    """
    selected_paths = []
    used_pairs = set()
    
    # Phase 1: One path for each starting vertex (100 paths)
    print("Phase 1: Selecting paths with each vertex as start...")
    available_pairs = list(paths_dict.keys())
    
    for start_vertex in range(n_nodes):
        # Find all pairs with this start vertex
        candidates = [(s, e) for (s, e) in available_pairs if s == start_vertex and (s, e) not in used_pairs]
        if candidates:
            chosen_pair = random.choice(candidates)
            start_node, end_node = chosen_pair
            path = paths_dict[chosen_pair]
            selected_paths.append((start_node, end_node, path))
            used_pairs.add(chosen_pair)
        else:
            print(f"Warning: No available path found with start vertex {start_vertex}")
    
    print(f"Phase 1 complete: {len(selected_paths)} paths selected")
    
    # Phase 2: One path for each ending vertex (100 paths)
    print("Phase 2: Selecting paths with each vertex as end...")
    
    for end_vertex in range(n_nodes):
        # Find all pairs with this end vertex that haven't been used
        candidates = [(s, e) for (s, e) in available_pairs if e == end_vertex and (s, e) not in used_pairs]
        if candidates:
            chosen_pair = random.choice(candidates)
            start_node, end_node = chosen_pair
            path = paths_dict[chosen_pair]
            selected_paths.append((start_node, end_node, path))
            used_pairs.add(chosen_pair)
        else:
            print(f"Warning: No available path found with end vertex {end_vertex}")
    
    print(f"Phase 2 complete: {len(selected_paths)} paths selected")
    
    # Phase 3: Random selection for remaining samples
    remaining_needed = total_samples - len(selected_paths)
    if remaining_needed > 0:
        print(f"Phase 3: Randomly selecting {remaining_needed} more paths...")
        
        # Get unused pairs
        unused_pairs = [(s, e) for (s, e) in available_pairs if (s, e) not in used_pairs]
        
        if len(unused_pairs) < remaining_needed:
            print(f"Warning: Only {len(unused_pairs)} unused pairs available, needed {remaining_needed}")
            remaining_needed = len(unused_pairs)
        
        # Randomly select from unused pairs
        random_pairs = random.sample(unused_pairs, remaining_needed)
        
        for start_node, end_node in random_pairs:
            path = paths_dict[(start_node, end_node)]
            selected_paths.append((start_node, end_node, path))
    
    print(f"Total paths selected: {len(selected_paths)}")
    return selected_paths

def oversample_data(data: List[Any], target_size: int) -> List[Any]:
    """
    Oversample data by repeating examples to reach target_size.
    
    Args:
        data: List of data examples
        target_size: Desired total number of examples after oversampling
    
    Returns:
        Oversampled data list
    """
    if len(data) == 0:
        return data
    
    if target_size <= len(data):
        return data[:target_size]
    
    # Calculate how many times we need to repeat the data
    oversampled_data = []
    
    # Repeat the entire dataset multiple times
    full_repeats = target_size // len(data)
    for _ in range(full_repeats):
        oversampled_data.extend(data)
    
    # Add partial data to reach exact target size
    remaining = target_size - len(oversampled_data)
    if remaining > 0:
        # Shuffle the data and take the first 'remaining' examples
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        oversampled_data.extend(shuffled_data[:remaining])
    
    # Final shuffle to mix repeated examples
    random.shuffle(oversampled_data)
    
    print(f"Oversampled from {len(data)} unique examples to {len(oversampled_data)} total examples")
    return oversampled_data

def split_train_test(data: List[Any], train_samples: int, test_samples: int) -> Tuple[List[Any], List[Any]]:
    """Split data into train and test sets."""
    total_requested = train_samples + test_samples
    
    if total_requested > len(data):
        print(f"Warning: Requested {total_requested} samples but only {len(data)} available")
        train_samples = min(train_samples, len(data))
        test_samples = min(test_samples, len(data) - train_samples)
    
    # Shuffle data for random split
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    train_data = shuffled_data[:train_samples]
    test_data = shuffled_data[train_samples:train_samples + test_samples]
    
    return train_data, test_data

def generate_shortest_paths_dataset(
    n_nodes: int,
    train_samples: int,
    test_samples: int,
    connectivity: str = "medium",
    output_base_dir: str = "data",
    input_base_dir: str = "data",
    oversample_train: bool = True
):
    """
    Generate shortest paths dataset in all 3 variants with train/test splits.
    Uses structured selection and saves the graph for later use.
    
    Args:
        n_nodes: Number of nodes in the graph
        train_samples: Number of unique training samples to select
        test_samples: Number of test samples
        connectivity: Graph connectivity level
        output_base_dir: Base directory for output
        oversample_train: If True, oversample training data to n_nodes * n_nodes examples
    """
    
    if n_nodes != 100:
        print(f"Warning: Algorithm designed for 100 nodes, got {n_nodes}")
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate graph
    G = generate_graph(n_nodes, connectivity)
    # G = load_graph(input_base_dir)
    
    # Save the graph for later use
    graph_path = save_graph(G, output_base_dir)
    
    # Generate all shortest paths
    all_paths_dict = generate_all_shortest_paths(G, n_nodes)
    
    print(f"Generated {len(all_paths_dict)} total shortest path examples")
    
    # Select paths using structured approach
    total_samples_needed = train_samples + test_samples
    selected_path_tuples = select_structured_paths(all_paths_dict, n_nodes, total_samples_needed)
    
    # Split paths into train/test
    train_paths, test_paths = split_train_test(selected_path_tuples, train_samples, test_samples)
    
    print(f"Split data: {len(train_paths)} unique train paths, {len(test_paths)} test paths")
    
    # Calculate oversampling target for training data
    train_target_size = n_nodes * n_nodes if oversample_train else len(train_paths)
    
    if oversample_train:
        print(f"Will oversample training data to {train_target_size} examples (N×N = {n_nodes}×{n_nodes})")
    
    # Define formatters for each variant
    formatters = {
        "standard": format_standard,
        "indices": format_indices,
        "grammar": format_grammar
    }
    
    # Generate datasets for each variant using the same train/test split
    for variant_name, formatter in formatters.items():
        print(f"\nProcessing {variant_name} variant...")
        
        # Create directory for this variant
        if (train_samples // 1000) > 0:
            variant_dir = f"{output_base_dir}/{variant_name}_{train_samples // 1000}k"
        else:
            variant_dir = f"{output_base_dir}/{variant_name}_{train_samples}_samples"
        os.makedirs(variant_dir, exist_ok=True)
        
        # Format train data (unique examples)
        train_formatted = []
        for start_node, end_node, path in train_paths:
            formatted_entry = formatter(start_node, end_node, path)
            train_formatted.append(formatted_entry)
        
        # Oversample training data if requested
        if oversample_train:
            train_formatted_oversampled = oversample_data(train_formatted, train_target_size)
        else:
            train_formatted_oversampled = train_formatted
        
        # Format test data
        test_formatted = []
        for start_node, end_node, path in test_paths:
            formatted_entry = formatter(start_node, end_node, path)
            test_formatted.append(formatted_entry)
        
        # Save train data
        train_filename = f"{variant_dir}/train.json"
        with open(train_filename, 'w') as f:
            json.dump(train_formatted_oversampled, f, indent=2)
        
        # Save test data
        test_filename = f"{variant_dir}/test.json"
        with open(test_filename, 'w') as f:
            json.dump(test_formatted, f, indent=2)
        
        print(f"Saved {len(train_formatted_oversampled)} training samples to {train_filename}")
        print(f"  ({len(train_formatted)} unique examples" + (f", oversampled to {len(train_formatted_oversampled)})" if oversample_train else ")"))
        print(f"Saved {len(test_formatted)} testing samples to {test_filename}")
        
        # Show sample entries
        print(f"Sample {variant_name} entries:")
        for i in range(min(3, len(train_formatted))):
            print(f"  Input: {train_formatted[i]['input']}")
            print(f"  Output: {train_formatted[i]['output']}")

# Example usage
if __name__ == "__main__":
    # Generate datasets for 100 nodes with structured selection and oversampling
    print("Generating shortest paths datasets with structured selection and oversampling...")
    print("=" * 60)
    
    # Parameters
    N_NODES = 100
    TRAIN_SAMPLES = 32  # Number of unique training samples per variant # 512 1024 2048 4096 8192
    TEST_SAMPLES = 1024   # Number of testing samples per variant
    CONNECTIVITY = "medium"  # Options: "low", "medium", "high"
    OVERSAMPLE_TRAIN = True  # Oversample training data to N×N examples
    
    print(f"Configuration:")
    print(f"  Nodes: {N_NODES}")
    print(f"  Unique training samples per variant: {TRAIN_SAMPLES}")
    print(f"  Testing samples per variant: {TEST_SAMPLES}")
    print(f"  Graph connectivity: {CONNECTIVITY}")
    print(f"  Oversample training data: {OVERSAMPLE_TRAIN}")
    if OVERSAMPLE_TRAIN:
        print(f"  Training data will be oversampled to: {N_NODES * N_NODES} examples")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate datasets
    generate_shortest_paths_dataset(
        n_nodes=N_NODES,
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        connectivity=CONNECTIVITY,
        oversample_train=OVERSAMPLE_TRAIN
    )
    
    print(f"\n{'='*60}")
    print("FILES GENERATED:")
    print(f"- data/graph.pkl (NetworkX graph)")
    
    suffix = f"_oversampled_{N_NODES * N_NODES}" if OVERSAMPLE_TRAIN else ""
    for variant in ["standard", "indices", "grammar"]:
        print(f"- data/{variant}_{TRAIN_SAMPLES // 1000}k{suffix}/train.json")
        print(f"- data/{variant}_{TRAIN_SAMPLES // 1000}k{suffix}/test.json")
    print(f"{'='*60}")