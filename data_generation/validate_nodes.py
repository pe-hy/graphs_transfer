import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

def extract_nodes_from_input(input_str: str, variant: str) -> Tuple[int, int]:
    """
    Extract start and end nodes from input string.
    
    Args:
        input_str: Input string in format "ST : start , end"
        variant: "standard", "grammar", or "indices"
    
    Returns:
        Tuple of (start_node, end_node)
    """
    # Extract numbers from the input string
    numbers = [int(x) for x in re.findall(r'\d+', input_str)]
    
    if len(numbers) != 2:
        raise ValueError(f"Expected 2 numbers in input, found {len(numbers)}: {input_str}")
    
    start_node, end_node = numbers
    
    # For indices variant, subtract the offset
    if variant == "indices":
        start_node -= 10000
        end_node -= 10000
    
    return start_node, end_node

def validate_node_coverage(
    data_dir: str = "data", 
    train_samples: int = 9000,
    n_nodes: int = 100,
    check_test: bool = False
) -> Dict[str, Dict]:
    """
    Validate that all nodes appear as both start and end vertices in the training data.
    
    Args:
        data_dir: Base directory containing the datasets
        train_samples: Number of training samples (used for directory naming)
        n_nodes: Total number of nodes expected (default 100)
        check_test: Whether to also check test set coverage
    
    Returns:
        Dictionary with validation results for each variant
    """
    
    variants = ["standard", "indices", "grammar"]
    results = {}
    
    print(f"{'='*70}")
    print("NODE COVERAGE VALIDATION")
    print(f"{'='*70}")
    print(f"Expected nodes: 0 to {n_nodes - 1}")
    print(f"Expected: All {n_nodes} nodes should appear as both start and end vertices")
    
    for variant in variants:
        print(f"\n{variant.upper()} VARIANT:")
        print("-" * 40)
        
        variant_dir = f"{data_dir}/{variant}_{train_samples // 1000}k_oversampled_10000"
        variant_results = {}
        
        for split in (["train", "test"] if check_test else ["train"]):
            filename = f"{variant_dir}/{split}.json"
            
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                print(f"\n{split.upper()} SET ({len(data)} samples):")
                
                # Track nodes that appear as start/end
                start_nodes = set()
                end_nodes = set()
                start_counts = Counter()
                end_counts = Counter()
                
                # Process each entry
                for i, entry in enumerate(data):
                    try:
                        start_node, end_node = extract_nodes_from_input(entry["input"], variant)
                        
                        # Validate node range
                        if not (0 <= start_node < n_nodes):
                            print(f"  Warning: Start node {start_node} out of range [0, {n_nodes-1}] at entry {i}")
                        if not (0 <= end_node < n_nodes):
                            print(f"  Warning: End node {end_node} out of range [0, {n_nodes-1}] at entry {i}")
                        
                        start_nodes.add(start_node)
                        end_nodes.add(end_node)
                        start_counts[start_node] += 1
                        end_counts[end_node] += 1
                        
                    except Exception as e:
                        print(f"  Error processing entry {i}: {e}")
                        print(f"    Input: {entry['input']}")
                
                # Check coverage
                expected_nodes = set(range(n_nodes))
                missing_start = expected_nodes - start_nodes
                missing_end = expected_nodes - end_nodes
                
                # Results summary
                print(f"  Start nodes coverage: {len(start_nodes)}/{n_nodes} nodes")
                print(f"  End nodes coverage: {len(end_nodes)}/{n_nodes} nodes")
                
                if missing_start:
                    print(f"  ❌ Missing START nodes: {sorted(missing_start)}")
                else:
                    print(f"  ✅ All nodes appear as START vertices")
                
                if missing_end:
                    print(f"  ❌ Missing END nodes: {sorted(missing_end)}")  
                else:
                    print(f"  ✅ All nodes appear as END vertices")
                
                # Frequency analysis
                if start_counts:
                    start_freq = list(start_counts.values())
                    print(f"  Start frequency - Min: {min(start_freq)}, Max: {max(start_freq)}, Avg: {sum(start_freq)/len(start_freq):.1f}")
                
                if end_counts:
                    end_freq = list(end_counts.values())
                    print(f"  End frequency - Min: {min(end_freq)}, Max: {max(end_freq)}, Avg: {sum(end_freq)/len(end_freq):.1f}")
                
                # Store results
                variant_results[split] = {
                    "total_samples": len(data),
                    "start_nodes_found": len(start_nodes),
                    "end_nodes_found": len(end_nodes),
                    "missing_start_nodes": sorted(missing_start),
                    "missing_end_nodes": sorted(missing_end),
                    "start_counts": dict(start_counts),
                    "end_counts": dict(end_counts),
                    "complete_start_coverage": len(missing_start) == 0,
                    "complete_end_coverage": len(missing_end) == 0,
                    "complete_coverage": len(missing_start) == 0 and len(missing_end) == 0
                }
                
            except FileNotFoundError:
                print(f"  ❌ File not found: {filename}")
                variant_results[split] = {"error": "File not found"}
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                variant_results[split] = {"error": str(e)}
        
        results[variant] = variant_results
    
    return results

def print_detailed_coverage_report(results: Dict[str, Dict], n_nodes: int = 100):
    """Print a detailed summary of coverage results."""
    
    print(f"\n{'='*70}")
    print("DETAILED COVERAGE SUMMARY")
    print(f"{'='*70}")
    
    # Overall summary
    all_variants_complete = True
    
    for variant, variant_data in results.items():
        if "train" in variant_data and "complete_coverage" in variant_data["train"]:
            if not variant_data["train"]["complete_coverage"]:
                all_variants_complete = False
                break
    
    if all_variants_complete:
        print("🎉 SUCCESS: All variants have complete node coverage in training data!")
    else:
        print("⚠️  WARNING: Some variants have incomplete node coverage")
    
    # Per-variant summary
    for variant, variant_data in results.items():
        print(f"\n{variant.upper()}:")
        
        if "train" in variant_data:
            train_data = variant_data["train"]
            if "error" in train_data:
                print(f"  ❌ Error: {train_data['error']}")
            else:
                status = "✅ COMPLETE" if train_data["complete_coverage"] else "❌ INCOMPLETE"
                print(f"  Training set: {status}")
                print(f"    Start coverage: {train_data['start_nodes_found']}/{n_nodes}")
                print(f"    End coverage: {train_data['end_nodes_found']}/{n_nodes}")
                
                if train_data["missing_start_nodes"]:
                    print(f"    Missing start nodes: {train_data['missing_start_nodes']}")
                if train_data["missing_end_nodes"]:
                    print(f"    Missing end nodes: {train_data['missing_end_nodes']}")

def find_nodes_with_low_frequency(results: Dict[str, Dict], threshold: int = 2, split: str = "train") -> Dict[str, Dict]:
    """Find nodes that appear infrequently as start or end vertices."""
    
    print(f"\n{'='*70}")
    print(f"LOW FREQUENCY NODES ANALYSIS (threshold: {threshold})")
    print(f"{'='*70}")
    
    low_freq_summary = {}
    
    for variant, variant_data in results.items():
        if split not in variant_data or "error" in variant_data[split]:
            continue
            
        data = variant_data[split]
        start_counts = data.get("start_counts", {})
        end_counts = data.get("end_counts", {})
        
        # Find low frequency nodes
        low_freq_start = {node: count for node, count in start_counts.items() if count <= threshold}
        low_freq_end = {node: count for node, count in end_counts.items() if count <= threshold}
        
        # Find nodes that never appear (count = 0)
        all_nodes = set(range(100))  # Assuming 100 nodes
        zero_freq_start = all_nodes - set(start_counts.keys())
        zero_freq_end = all_nodes - set(end_counts.keys())
        
        # Add zero frequency nodes to low frequency dict
        for node in zero_freq_start:
            low_freq_start[node] = 0
        for node in zero_freq_end:
            low_freq_end[node] = 0
        
        print(f"\n{variant.upper()} - {split.upper()}:")
        print(f"  Low frequency START nodes (≤{threshold}): {len(low_freq_start)}")
        if low_freq_start:
            sorted_start = sorted(low_freq_start.items())
            print(f"    {dict(sorted_start)}")
        
        print(f"  Low frequency END nodes (≤{threshold}): {len(low_freq_end)}")
        if low_freq_end:
            sorted_end = sorted(low_freq_end.items())
            print(f"    {dict(sorted_end)}")
        
        low_freq_summary[variant] = {
            "low_freq_start": low_freq_start,
            "low_freq_end": low_freq_end,
            "zero_freq_start": list(zero_freq_start),
            "zero_freq_end": list(zero_freq_end)
        }
    
    return low_freq_summary

def validate_structured_selection_hypothesis(results: Dict[str, Dict], n_nodes: int = 100) -> None:
    """
    Validate that the structured selection worked as expected:
    - At least 100 paths should have unique start vertices
    - At least 100 paths should have unique end vertices
    """
    
    print(f"\n{'='*70}")
    print("STRUCTURED SELECTION VALIDATION")
    print(f"{'='*70}")
    print("Expected: At least 100 unique start vertices and 100 unique end vertices")
    print("(Since first 200 samples should cover all vertices)")
    
    for variant, variant_data in results.items():
        if "train" not in variant_data or "error" in variant_data["train"]:
            continue
            
        train_data = variant_data["train"]
        
        print(f"\n{variant.upper()}:")
        
        # Check if we have the expected coverage
        start_coverage = train_data["start_nodes_found"]
        end_coverage = train_data["end_nodes_found"]
        
        start_status = "✅ PASS" if start_coverage >= n_nodes else f"❌ FAIL ({start_coverage}/{n_nodes})"
        end_status = "✅ PASS" if end_coverage >= n_nodes else f"❌ FAIL ({end_coverage}/{n_nodes})"
        
        print(f"  Unique start vertices: {start_coverage}/{n_nodes} - {start_status}")
        print(f"  Unique end vertices: {end_coverage}/{n_nodes} - {end_status}")
        
        # Additional analysis: check if any vertex appears very frequently
        start_counts = train_data.get("start_counts", {})
        end_counts = train_data.get("end_counts", {})
        
        if start_counts:
            max_start_freq = max(start_counts.values())
            max_start_nodes = [node for node, count in start_counts.items() if count == max_start_freq]
            print(f"  Most frequent start vertex(es): {max_start_nodes} (appears {max_start_freq} times)")
        
        if end_counts:
            max_end_freq = max(end_counts.values())
            max_end_nodes = [node for node, count in end_counts.items() if count == max_end_freq]
            print(f"  Most frequent end vertex(es): {max_end_nodes} (appears {max_end_freq} times)")

def main():
    """Main function to run all validations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate node coverage in shortest paths dataset")
    parser.add_argument("--data_dir", default="data", help="Base data directory")
    parser.add_argument("--train_samples", type=int, default=9000, help="Number of training samples (for directory naming)")
    parser.add_argument("--n_nodes", type=int, default=100, help="Number of nodes in graph")
    parser.add_argument("--check_test", action="store_true", help="Also validate test set")
    parser.add_argument("--threshold", type=int, default=2, help="Threshold for low frequency analysis")
    
    args = parser.parse_args()
    
    # Run main validation
    results = validate_node_coverage(
        data_dir=args.data_dir,
        train_samples=args.train_samples,
        n_nodes=args.n_nodes,
        check_test=args.check_test
    )
    
    # Print detailed summary
    print_detailed_coverage_report(results, args.n_nodes)
    
    # Run structured selection validation
    validate_structured_selection_hypothesis(results, args.n_nodes)
    
    # Find low frequency nodes
    find_nodes_with_low_frequency(results, args.threshold, "train")
    
    if args.check_test:
        find_nodes_with_low_frequency(results, args.threshold, "test")
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")

def quick_validate(data_dir: str = "data", train_samples: int = 9000):
    """Quick validation function for simple usage."""
    print("Running quick node coverage validation...")
    
    results = validate_node_coverage(data_dir, train_samples, 100, False)
    print_detailed_coverage_report(results, 100)
    validate_structured_selection_hypothesis(results, 100)
    
    # Return simple boolean result
    all_complete = all(
        variant_data.get("train", {}).get("complete_coverage", False) 
        for variant_data in results.values()
    )
    
    return all_complete, results

# Example usage functions
def example_usage():
    """Show example usage of the validation functions."""
    print("EXAMPLE USAGE:")
    print("=" * 50)
    print("1. Quick validation (default settings):")
    print("   success, results = quick_validate()")
    print("")
    print("2. Detailed validation with custom parameters:")
    print("   results = validate_node_coverage(")
    print("       data_dir='my_data',")
    print("       train_samples=5000,") 
    print("       n_nodes=100,")
    print("       check_test=True")
    print("   )")
    print("")
    print("3. Command line usage:")
    print("   python validate_nodes.py --data_dir data --train_samples 9000 --check_test")
    print("")

if __name__ == "__main__":
    # If no command line arguments provided, run quick validation
    import sys
    
    if len(sys.argv) == 1:
        print("No arguments provided, running quick validation...")
        print("Use --help for more options\n")
        
        success, results = quick_validate()
        
        if success:
            print("\n🎉 OVERALL RESULT: All datasets have complete node coverage!")
        else:
            print("\n⚠️  OVERALL RESULT: Some datasets have incomplete node coverage.")
            print("Check the detailed output above for specific issues.")
        
        print("\nFor more options, run: python validate_nodes.py --help")
        
    else:
        main()