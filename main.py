# integrated_pipeline.py
import json
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.iteration import iterate_main
from src.model_utils import get_qwen_model, get_summary_model
from src.docstring_utils import generate_and_insert_docstrings, narrow_top_files
from src.bug_localization import localize_and_generate_patches

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the complete pipeline for bug localization and patch generation")
    parser.add_argument("--k", type=int, default=60, help="Number of top files to initially consider")
    parser.add_argument("--p", type=int, default=10, help="Number of top files to narrow down to")
    parser.add_argument("--n", type=int, default=1, help="Number of top functions to consider for bug fixing")
    parser.add_argument("--bm25", type=float, default=0.8, help="Weight for BM25 ranking in initial file selection")
    parser.add_argument("--semantic", type=float, default=0.2, help="Weight for semantic ranking in initial file selection")
    parser.add_argument("--sem-weight", type=float, default=0.7, help="Weight for semantic ranking in narrowing files")
    parser.add_argument("--output", type=str, default=None, help="Output directory for patches")
    parser.add_argument("--enable-logging", action="store_true", help="Enable logging of LLM inputs and outputs")
    parser.add_argument("--skip-docstrings", action="store_true", help="Skip generating docstrings (use with caution)")
    parser.add_argument("--dataset", type=str, default="data/swe_bench_lite_test.json", help="Path to the dataset")
    
    args = parser.parse_args()
    
    # Configuration
    k = args.k  # Number of top files to initially consider
    p = args.p  # Number of files to narrow down to
    n = args.n  # Number of top functions to consider for bug fixing
    weightBM25 = args.bm25
    weightSemantic = args.semantic
    sem_weight = args.sem_weight
    
    # Set output directory
    if args.output is None:
        output_dir = f"patches/{k}_{p}_{n}/{weightBM25}_{weightSemantic}_{sem_weight}"
    else:
        output_dir = args.output
    
    # Print configuration
    print("Running pipeline with the following parameters:")
    print(f"Number of initial files (k): {k}")
    print(f"Number of narrowed files (p): {p}")
    print(f"Number of top functions for bug analysis (n): {n}")
    print(f"BM25 weight for initial selection: {weightBM25}")
    print(f"Semantic weight for initial selection: {weightSemantic}")
    print(f"Semantic weight for narrowing files: {sem_weight}")
    print(f"Output directory: {output_dir}")
    print(f"Enable logging: {args.enable_logging}")
    print(f"Skip docstrings: {args.skip_docstrings}")
    print(f"Dataset: {args.dataset}")
    
    # Load the dataset
    try:
        with open(args.dataset) as f:
            data_lite = json.load(f)
        
        # Create a mapping from row index to instance_id
        instance_id_map = {str(i): entry["instance_id"] for i, entry in enumerate(data_lite)}
        print(f"Loaded dataset with {len(data_lite)} entries")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Step 1: Run initial file ranking to get top k files (By default, k=60)
    top_files_path = f"top_files/{k}/{weightBM25}_{weightSemantic}.json"
    
    try:
        with open(top_files_path, "r") as f:
            top_files_data = json.load(f)
            print(f"Loaded existing top files data from {top_files_path}")
    except FileNotFoundError:
        # If top files haven't been identified yet, run the first phase
        print("Top files data not found. Running file ranking phase...")
        file_hits = iterate_main(data_lite, k, weightBM25, weightSemantic)
        
        # Load the results that were saved by iterate_main
        with open(top_files_path, "r") as f:
            top_files_data = json.load(f)
    
    # Step 2: Generate and insert docstrings for functions and modules
    try: 
        enhanced_data_path = f"enhanced_data/{k}/{weightBM25}_{weightSemantic}.json"
        with open(enhanced_data_path, "r") as f:
            enhanced_data = json.load(f)
        print("\nSkipping docstring generation...")
        print(f"Loaded existing enhanced data from {enhanced_data_path}")
    except FileNotFoundError:
        if args.skip_docstrings:
            print("Skipping docstring generation as per user request.")
            enhanced_data = top_files_data
        else:
            # Generate and insert docstrings
            print("Generating docstrings for functions and modules...")
            enhanced_data = generate_and_insert_docstrings(
                top_files_data, 
                k, 
                weightBM25, 
                weightSemantic, 
                enable_logging=args.enable_logging
            )
    
    # Step 3: Narrow down top k files to top p files using docstrings
    try:
        narrowed_data_path = f"narrowed_data/{k}_{p}/{(1-sem_weight)}_{sem_weight}.json"
        with open(narrowed_data_path, "r") as f:
            narrowed_data = json.load(f)
        print(f"\nLoaded existing narrowed data from {narrowed_data_path}")
    except FileNotFoundError:
        print("Narrowed data not found. Running narrowing phase...")
        narrowed_data = narrow_top_files(
            enhanced_data,
            k,
            p,
            data_lite[0]["problem_statement"],  # Use first issue description as example
            weightBM25 = weightBM25,
            weightSemantic = weightSemantic,
            sem_weight = sem_weight,
            enable_logging=args.enable_logging
        )
        # Save narrowed data
        os.makedirs(os.path.dirname(narrowed_data_path), exist_ok=True)
        with open(narrowed_data_path, "w") as f:
            json.dump(narrowed_data, f, indent=4)
        print(f"Saved narrowed data to {narrowed_data_path}")

    
    # Step 4: Update the top files in narrowed data
        
    for row_idx, entry in narrowed_data.items():
        if "top_p_files" in entry:
            narrowed_data[row_idx]["top_n_files"] = entry["top_p_files"]
    
    # Step 5: Run bug localization and patch generation
    try:
        with open(f"{output_dir}/all_patches.json", "r") as f:
            patch_results = json.load(f)
        print(f"\nLoaded existing patch results from {output_dir}/all_patches.json")
    except FileNotFoundError: 
        print("Patch results not found. Running bug localization and patch generation...")
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Run bug localization and patch generation
        patch_results = localize_and_generate_patches(
            narrowed_data, 
            n, 
            output_dir, 
            enable_logging=args.enable_logging
        )
        # Save patch results 
        with open(f"{output_dir}/all_patches.json", "w") as f:
            json.dump(patch_results, f, indent=4)
    
    # Create a SWEBench-compatible evaluation file
    swe_bench_patches = []
    for row_idx, result in patch_results.items():
        instance_id = instance_id_map.get(str(row_idx), f"row_{row_idx}")  # fallback if not found
        print("RowIdx is: ", row_idx)
        model_patch = ""

        if "swe_bench_patch_file" in result:
            print("swe_bench_patch_file is present in results")
            with open(result["swe_bench_patch_file"], "r") as f:

                #---------------------
                raw_patch = json.load(f).get("patch", "")
                repo_with_underscore = result.get("repo_with_underscore", "")

                cleaned_lines = []
                for line in raw_patch.splitlines():
                    if line.startswith(("--- ", "+++ ")):
                        if repo_with_underscore in line:
                            idx = line.find(repo_with_underscore)
                            relative_path = line[idx + len(repo_with_underscore) + 1:]
                            cleaned_line = line[:4] + f"{'a' if line.startswith('---') else 'b'}/{relative_path}"
                            cleaned_lines.append(cleaned_line)
                        else:
                            cleaned_lines.append(line)
                    else:
                        if line.strip():
                            cleaned_lines.append(line)

                model_patch = "\n".join(cleaned_lines)
                #----------------------------------

        swe_bench_patches.append({
            "instance_id": instance_id,
            "model_name_or_path": "Qwen2.5-7B-Instruct",
            "model_patch": ""
        })
    
    print(len(swe_bench_patches))
    
    # Save the SWEBench-compatible evaluation file
    swe_bench_eval_file = os.path.join(output_dir, "swe_bench_evaluation.jsonl")
    with open(swe_bench_eval_file, "w") as f:
        print("Dumping")
        for entry in swe_bench_patches:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nComplete pipeline finished. Results saved to {output_dir}")
    print(f"Generated SWEBench-compatible evaluation file: {swe_bench_eval_file}")
    print(f"You can now run the SWEBench-lite evaluation pipeline using this file.")

if __name__ == "__main__":
    main()