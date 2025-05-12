# src/bug_localization.py
import ast
import re
import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from src.docstring_utils import release_vllm_model

def get_bug_localization_model(num_gpus=None):
    """Initialize the vLLM model for bug localization on multiple GPUs."""
    # Get the number of available GPUs if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print(f"Initializing bug localization model with {num_gpus} GPUs")
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    # Use vLLM with tensor parallelism across multiple GPUs
    model = LLM(
        model=model_id,
        tensor_parallel_size=num_gpus,  # Use all available GPUs
        trust_remote_code=True,  # Required for some models like Qwen
        dtype="half",  # Use half precision for efficiency
        gpu_memory_utilization=0.8,  # Prevent OOM errors
        # Enable swap space for handling larger models
        swap_space=4,  # GB, adjust based on available RAM
    )
    return model

def extract_functions_from_file(file_path):
    """
    Extract all functions from a Python file as AST nodes.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        list: List of function nodes with additional metadata
    """
    functions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            
        tree = ast.parse(source)
        lines = source.splitlines()
        
        # Helper function to add parent references to AST nodes
        def add_parents(node, parent=None):
            for child in ast.iter_child_nodes(node):
                child.parent = node
                add_parents(child, node)
        
        add_parents(tree)
        
        # Extract all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                func_source = "\n".join(lines[start_line:end_line])
                
                # Get function complexity metrics
                arg_count = len(node.args.args)
                body_lines = end_line - start_line
                
                # Determine if function is in a class
                class_name = None
                parent = getattr(node, "parent", None)
                while parent is not None:
                    if isinstance(parent, ast.ClassDef):
                        class_name = parent.name
                        break
                    parent = getattr(parent, "parent", None)
                
                functions.append({
                    "name": node.name,
                    "source": func_source,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                    "class_name": class_name,
                    "arg_count": arg_count,
                    "body_lines": body_lines,
                    "file_path": file_path
                })
                
    except Exception as e:
        print(f"Error extracting functions from {file_path}: {e}")
    
    return functions

def rank_functions(functions, issue_description):
    """
    Rank functions based on relevance to the issue description using 
    a combination of BM25 and semantic similarity.
    
    Args:
        functions (list): List of function dictionaries
        issue_description (str): Description of the issue
        
    Returns:
        list: Ranked list of functions with relevance scores
    """
    if not functions:
        return []
    
    # Prepare documents for ranking
    docs = []
    for func in functions:
        # Create a document that includes function name and source
        func_doc = f"{func['name']} {func['source']}"
        docs.append(func_doc)
    
    # BM25 ranking
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = issue_description.lower().split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    
    # Normalize BM25 scores
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    
    # Semantic similarity ranking
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    issue_embedding = semantic_model.encode(issue_description, convert_to_tensor=True)
    doc_embeddings = semantic_model.encode(docs, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(issue_embedding, doc_embeddings)[0].cpu().numpy()
    
    # Normalize semantic scores
    if np.max(semantic_scores) > 0:
        semantic_scores = semantic_scores / np.max(semantic_scores)
    
    # Combine scores with weights (0.7 for BM25, 0.3 for semantic)
    combined_scores = 0.3 * bm25_scores + 0.7 * semantic_scores
    
    # Create ranked function list
    ranked_functions = []
    for i, func in enumerate(functions):
        func_copy = func.copy()
        func_copy["relevance_score"] = float(combined_scores[i])
        ranked_functions.append(func_copy)
    
    # Sort by relevance score
    ranked_functions.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return ranked_functions

def create_bug_fix_prompt(function, issue_description):
    """
    Create a prompt for the LLM to generate a bug fix.
    
    Args:
        function (dict): Function information
        issue_description (str): Description of the issue
        
    Returns:
        str: Prompt for the LLM
    """
    prompt = f"""You are a Python expert tasked with fixing a bug in a function. 
    
Issue Description:
{issue_description}

Function to fix:
```python
{function['source']}
```

Please analyze the code carefully to identify the bug based on the issue description. 
Then provide the fixed version of the function with your changes clearly explained.

First, explain what the bug is, then provide the fixed function:

Bug Analysis:
"""
    
    return prompt

def extract_fixed_function(llm_output):
    """
    Extract the fixed function code from the LLM output.
    
    Args:
        llm_output (str): LLM output text
        
    Returns:
        tuple: (fixed_code, explanation)
    """
    # Try to extract code between ```python and ``` markers
    code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
    code_match = code_pattern.search(llm_output)
    
    # Try to extract the explanation
    explanation = ""
    if "Bug Analysis:" in llm_output:
        parts = llm_output.split("Bug Analysis:", 1)
        if len(parts) > 1:
            explanation_text = parts[1]
            if "```python" in explanation_text:
                explanation = explanation_text.split("```python")[0].strip()
            else:
                explanation = explanation_text.strip()
    
    if code_match:
        fixed_code = code_match.group(1).strip()
        return fixed_code, explanation
    
    # If no code block found, try to find the function definition
    function_pattern = re.compile(r'def\s+\w+\s*\(.*?\).*?:', re.DOTALL)
    start_match = function_pattern.search(llm_output)
    
    if start_match:
        # Extract from function definition to the end
        start_pos = start_match.start()
        fixed_code = llm_output[start_pos:].strip()
        return fixed_code, explanation
    
    # If all else fails, return the whole output
    return llm_output.strip(), explanation

def generate_patch(file_path, function, fixed_code):
    """
    Generate a unified diff patch for the fixed function.
    
    Args:
        file_path (str): Path to the file
        function (dict): Function information
        fixed_code (str): Fixed function code
        
    Returns:
        str: Unified diff patch
    """
    try:
        # Read the original file
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
        
        start_line = function["start_line"] - 1  # 0-indexed
        end_line = function["end_line"]
        patched_lines = original_lines.copy()
        
        fixed_lines = [line if line.endswith("\n") else line + "\n" for line in fixed_code.splitlines()]
        print("\nFixed Lines length:", len(fixed_lines))
        print("\nFixed lines: ", fixed_lines)
        
        expected_old = function["end_line"] - (function["start_line"] - 1)
        
        got_old = len(original_lines[start_line : end_line + 1])
        
        print(f"expected_old={expected_old}, got_old={got_old}")

        if got_old > expected_old:
            suffix = patched_lines[end_line:]
        else:
            suffix = patched_lines[end_line + 1:]

        full_patched_code = patched_lines[:start_line]
        full_patched_code += fixed_lines
        full_patched_code += suffix

        print("\nOriginal Hunk: ", original_lines)
        print("\nPatched Hunk: ", full_patched_code)

        # with open("/home/madhav/madhav/LMZ/SLM_Debugger_kondor/SLM-Debugger/Testing/original.py","w",encoding="utf-8") as f1:
        #     f1.writelines(original_lines)
        # with open("/home/madhav/madhav/LMZ/SLM_Debugger_kondor/SLM-Debugger/Testing/patched.py","w",encoding="utf-8") as f1:
        #     f1.writelines(full_patched_code)
        
        # Generate unified diff
        import difflib
        file_path_rel = os.path.normpath(file_path)
        if file_path_rel.startswith(os.path.sep):
            file_path_rel = file_path_rel[1:]
            
        diff_lines = difflib.unified_diff(
            original_lines,
            full_patched_code,
            fromfile=f"a/{file_path_rel}",
            tofile=f"b/{file_path_rel}",
            # lineterm='\n',
            # n=5
        )

        diff_text = ''.join(diff_lines)
        if not diff_text.endswith('\n'):
            diff_text += '\n'
        return diff_text

    except Exception as e:
        print(f"Error generating patch for {file_path}: {e}")
        return ""

def batch_generate_fixes(functions, issue_descriptions, model, log_dir=None):
    """
    Generate fixes for a batch of functions using the LLM.
    
    Args:
        functions (list): List of function dictionaries
        issue_descriptions (list): List of issue descriptions
        model: vLLM model
        log_dir (str, optional): Directory to log LLM outputs, None disables logging
        
    Returns:
        list: List of (function, fixed_code, explanation) tuples
    """
    # Create prompts for all functions
    prompts = []
    for func, issue in zip(functions, issue_descriptions):
        prompts.append(create_bug_fix_prompt(func, issue))
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,  # Lower temperature for more deterministic responses
        max_tokens=1000,
        top_k=10,
    )
    
    # Generate fixes using vLLM
    outputs = model.generate(prompts, sampling_params)
    
    # Process the results
    results = []
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        fixed_code, explanation = extract_fixed_function(generated)
        
        # Log LLM output if logging is enabled
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            func_name = functions[i]['name']
            log_file = os.path.join(log_dir, f"{func_name}_llm_output.txt")
            
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("PROMPT:\n")
                f.write("-------\n")
                f.write(prompts[i])
                f.write("\n\nRESPONSE:\n")
                f.write("---------\n")
                f.write(generated)
                f.write("\n\nEXTRACTED FIX:\n")
                f.write("--------------\n")
                f.write(fixed_code)
                f.write("\n\nEXPLANATION:\n")
                f.write("------------\n")
                f.write(explanation)
        
        results.append((functions[i], fixed_code, explanation))
    
    return results

def select_most_buggy_function(functions, issue_description, model):
    """
    Ask the LLM to identify the most buggy function among a list of candidates.

    Args:
        functions (list): List of function dicts (each with 'name', 'source', etc.)
        issue_description (str): The issue/problem description
        model: vLLM model instance

    Returns:
        dict: The single most buggy function selected by the LLM
    """
    print("In the function passing 10 functions to LLM and expecting 1 buggy in return")
    prompt = f"""You are a Python expert analyzing multiple functions to identify the most buggy one based on the issue description.

Issue Description:
{issue_description}

Below are 10 candidate functions from different parts of the codebase:

"""

    for idx, func in enumerate(functions):
        prompt += f"\nFunction {idx+1} ({func['name']}):\n```python\n{func['source']}\n```\n"

    prompt += "\nWhich one function is most likely to contain the bug described above? Reply with only the number (e.g., '3'). Do not provide any explanation."

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=256,
        stop=["\n"],
    )

    output = model.generate([prompt], sampling_params)[0].outputs[0].text.strip()

    # Extract function number (e.g., "Function 3")
    match = re.search(r'\b(\d+)\b', output)
    if match:
        index = int(match.group(1)) - 1
        print(f"Selected function number {index}")
        return functions[index]
    else:
        print("Warning: Could not parse LLM response. Defaulting to first function.")
        return functions[0]


def localize_and_generate_patches(top_files_data, n=5, output_dir="patches", enable_logging=False):
    """
    Main function to localize bugs and generate patches for the top files.
    
    Args:
        top_files_data (dict): Dictionary containing top files data
        n (int): Number of top functions to consider
        output_dir (str): Directory to store patches and results
        enable_logging (bool): Whether to log LLM inputs and outputs
        
    Returns:
        dict: Results of bug patches
    """
    # Initialize the model
    model = get_bug_localization_model()
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Process each entry in the top files data
    for row_idx, entry in top_files_data.items():
        repo_with_underscore = entry["repo_with_underscore"]
        repo = entry["repo"]
        top_files = entry["top_n_files"]
        issue_description = entry["issue_description"]
        
        print(f"\n{'-'*80}")
        print(f"Processing row {row_idx}: {repo}")
        print(f"Issue description: {issue_description[:100]}...")
        
        # Create output directory for this repo
        repo_output_dir = os.path.join(output_dir, f"row_{row_idx}")
        os.makedirs(repo_output_dir, exist_ok=True)
        
        # Create log directory if logging is enabled
        log_dir = None
        if enable_logging:
            log_dir = os.path.join(repo_output_dir, "llm_logs")
            os.makedirs(log_dir, exist_ok=True)
            print(f"LLM logging enabled. Logs will be saved to: {log_dir}")
        
        # Extract and rank functions from all top files
        all_functions = []
        for file_path in top_files:
            # The path format depends on the structure of your decoupled directory
            full_path = f"decoupled/60/0.8_0.2/{row_idx}/{repo_with_underscore}/{file_path}"
            if not os.path.exists(full_path):
                print(f"Warning: File not found: {full_path}")
                continue
            
            file_functions = extract_functions_from_file(full_path)
            for func in file_functions:
                # Store the relative path for patch generation
                func["rel_path"] = file_path
            all_functions.extend(file_functions)
        
        print(f"Extracted {len(all_functions)} functions from {len(top_files)} files")
        
        # Rank functions by relevance to issue description
        ranked_functions = rank_functions(all_functions, issue_description)
        
        # Take top n functions for bug fixing
        top_n_functions = ranked_functions[:n]
        print(f"Top {n} potentially buggy functions:")
        for i, func in enumerate(top_n_functions):
            print(f"{i+1}. {func['name']} (score: {func['relevance_score']:.4f}) in {func['rel_path']}")
        
        # Generate bug fixes for top functions
        issue_descriptions = [issue_description] * len(top_n_functions)
        fixes = batch_generate_fixes(top_n_functions, issue_descriptions, model, log_dir)
        
        # Generate patches and store results
        patch_results = []
        combined_patch = ""
        
        for func, fixed_code, explanation in fixes:
            # Generate patch
            file_path = func["file_path"]
            patch = generate_patch(file_path, func, fixed_code)
            
            # Save individual patch
            patch_file = os.path.join(repo_output_dir, f"{func['name']}_patch.diff")
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write(patch)
            
            # Add to combined patch
            combined_patch += patch
            
            patch_results.append({
                "function_name": func["name"],
                "class_name": func["class_name"],
                "file_path": func["rel_path"],
                "patch_file": patch_file,
                "explanation": explanation,
                "relevance_score": func["relevance_score"]
            })
            
            print(f"Generated patch for function: {func['name']} in {func['rel_path']}")
        
        # Save combined patch in SWEBench format
        combined_patch_file = os.path.join(repo_output_dir, "combined_patch.diff")
        with open(combined_patch_file, "w", encoding="utf-8") as f:
            f.write(combined_patch)
        
        # Create a JSON patch file with complete metadata for SWEBench evaluation
        swe_bench_patch = {
            "repo": repo,
            "base_commit": entry.get("base_commit", ""),
            "problem_statement": issue_description,
            "patch": combined_patch
        }
        
        swe_bench_patch_file = os.path.join(repo_output_dir, "swe_bench_patch.json")
        with open(swe_bench_patch_file, "w", encoding="utf-8") as f:
            json.dump(swe_bench_patch, f, indent=4)
        
        # Store results for this repo
        results[row_idx] = {
            "repo": repo,
            "repo_with_underscore": repo_with_underscore,
            "issue_description": issue_description,
            "patches": patch_results,
            "combined_patch_file": combined_patch_file,
            "swe_bench_patch_file": swe_bench_patch_file
        }
        
        # Save intermediate results
        with open(os.path.join(repo_output_dir, "patch_results.json"), "w") as f:
            json.dump(results[row_idx], f, indent=4)
    
    # Save overall results
    with open(os.path.join(output_dir, "all_patches.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Release model resources
    release_vllm_model(model)
    
    return results

def evaluate_fixes(fix_results, data_lite):
    """
    Evaluate the quality of bug fixes by comparing with ground truth patches.
    
    Args:
        fix_results (dict): Results of bug fixes
        data_lite (dict): Original data with ground truth patches
        
    Returns:
        dict: Evaluation metrics
    """
    metrics = {
        "total_issues": len(fix_results),
        "files_correctly_identified": 0,
        "functions_correctly_identified": 0,
        "successful_fixes": 0
    }
    
    # Create a mapping from row index to ground truth patch
    ground_truth = {}
    for i, entry in enumerate(data_lite):
        ground_truth[str(i)] = entry.get("patch", "")
    
    # Evaluate each fix
    for row_idx, result in fix_results.items():
        # Extract modified files from ground truth patch
        from src.utils import extract_changed_file_from_patch
        gt_files = extract_changed_file_from_patch(ground_truth.get(row_idx, ""))
        
        # Check if any of the fixed files match ground truth
        fixed_files = [fix["file_path"] for fix in result["fixes"]]
        files_match = any(gt_file in fixed_files for gt_file in gt_files)
        
        if files_match:
            metrics["files_correctly_identified"] += 1
        
        # TODO: More sophisticated evaluation of function-level matches and fix quality
        # This would require parsing the ground truth patch to extract changed functions
        
    return metrics