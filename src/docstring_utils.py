# src/enhanced_docstring_utils.py
import json
import os
import torch
from vllm import LLM, SamplingParams
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import gc
import ray
from src.ranking import rank_files_by_content
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt', quiet=True)

try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


def batch_generate_summaries(functions, model):
    """Generate summaries for a batch of functions using vLLM."""
    # Validate and filter functions
    valid_functions = functions
    
    # # Create prompts for all valid functions in this batch
    prompts = []
    for func in valid_functions:
        prompts.append(create_summary_prompt(func))
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=100,
        stop=['"""'],
        top_k=50,
    )
    
    # Use vLLM to generate all outputs at once
    outputs = model.generate(prompts, sampling_params)
    
    # Process the results
    results = []
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        extracted = extract_docstring_block(generated)
        results.append((valid_functions[i], extracted))
    
    return results


def extract_module_docstring(llm_output):
    """Extract the docstring from the LLM output."""
    # Clean up and extract the relevant part of the output
    if '"""' in llm_output:
        # If triple quotes are in the output, use them as delimiters
        start_idx = llm_output.find('"""')
        if start_idx >= 0:
            content = llm_output[start_idx+3:]
            end_idx = content.find('"""')
            if end_idx >= 0:
                return '"""' + content[:end_idx] + '"""'
            else:
                return '"""' + content.strip() + '"""'
    
    # If we reached here, no proper docstring format was found
    # Format it properly as a docstring
    cleaned_output = llm_output.strip()
    if "### Summary:" in cleaned_output:
        # Extract just the summary part
        cleaned_output = cleaned_output.split("### Summary:", 1)[1].strip()
    
    # Format as proper docstring
    return f'"""{cleaned_output}"""'


def filter_global_missing_functions_docstrings(global_missing_func_docstring_list):
    filtered_functions = [
        func for func in global_missing_func_docstring_list
        if (func["end_lineno"] - func["start_lineno"]) >= 3 and (func["end_lineno"] - func["start_lineno"]) <= 100
    ]
    func_count = {}
    print("Length of Filtered Functions: ", len(filtered_functions))
    for func in filtered_functions:
        diff = func["file"]
        func_count[diff] = func_count.get(diff, 0) + 1
    function_file_count = defaultdict(list)
    for file_name, count in func_count.items():
        function_file_count[count].append(file_name)
    files_to_remove = set()
    for key, files in function_file_count.items():
        if key > 40:
            files_to_remove.update(files)
    final_filtered_list = [func for func in filtered_functions if func["file"] not in files_to_remove]
    print("After filtering files with too many functions: ", len(final_filtered_list))
    return final_filtered_list


def extract_functions_from_files(file_list, repo_with_underscore, row_idx, n, weightBM25, weightSemantic):
    all_functions = []
    for file in file_list:
        if file.endswith(".py"):
            funcs = extract_functions_without_docstrings(file, repo_with_underscore, row_idx, n, weightBM25, weightSemantic)
            all_functions.extend(funcs)
    return all_functions


def get_files_missing_module_docstrings(file_list, repo_with_underscore, row_idx=None, n=10, weightBM25=0.5, weightSemantic=0.5):
    missing_docstring_files = []
    for file in file_list:
        full_path = os.path.join(f"decoupled/{n}/{weightBM25}_{weightSemantic}/{row_idx}/{repo_with_underscore}/", file)
        if file.endswith(".py") and not has_module_docstring(full_path):
            missing_docstring_files.append(file)
    return missing_docstring_files


def get_summary_model(num_gpus=None):
    """Initialize the vLLM model for generating summaries on multiple GPUs."""
    # Get the number of available GPUs if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print(f"Initializing model with {num_gpus} GPUs")
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    # Use vLLM with tensor parallelism across multiple GPUs
    model = LLM(
        model=model_id,
        tensor_parallel_size=num_gpus,  # Use all available GPUs
        dtype="half",  # Use half precision for efficiency
        trust_remote_code=True,
        gpu_memory_utilization=0.8,  # Prevent OOM errors
        # Enable swap space for handling larger models
        swap_space=4,  # GB, adjust based on available RAM
    )
    return model


def release_vllm_model(model):
    """
    Tear down a vLLM model or engine instance and free all GPU memory.
    Handles both vllm.LLM wrappers and raw LLMEngine objects.
    """
    # 1) Clear model-parallel state
    destroy_model_parallel()
    
    # 2) Determine the underlying engine
    engine = getattr(model, "llm_engine", model)
    
    # 3) Shutdown executor threads/processes
    try:
        engine.model_executor.shutdown()
    except Exception:
        pass
    
    # 4) Delete executor and model references
    if hasattr(engine, "model_executor"):
        del engine.model_executor
    del model
    
    # 5) Force Python GC + empty CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    
    # 6) Shutdown Ray to release any actors
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

def validate_function(func):
    """Validate that a function object has all required fields."""
    if not isinstance(func, dict):
        return False
    
    required_fields = ['source', 'file', 'name', 'class_name']
    return all(field in func for field in required_fields)

def generate_and_insert_docstrings(top_files_data, n, weightBM25, weightSemantic, enable_logging=False):
    """
    Generate and insert docstrings for functions and modules in the top files.
    
    Args:
        top_files_data (dict): Dictionary containing top files data
        n (int): Number of top files
        weightBM25 (float): Weight for BM25 ranking
        weightSemantic (float): Weight for semantic ranking
        enable_logging (bool): Whether to log the process
        
    Returns:
        dict: Enhanced top files data with docstring information
    """
    # Initialize docstring summary model
    model = get_summary_model()
    
    # Store enhanced data
    enhanced_data = {}
    
    # Process each entry in the top files data
    for row_idx, entry in top_files_data.items():
        repo_with_underscore = entry["repo_with_underscore"]
        top_files = entry["top_n_files"]
        issue_description = entry["issue_description"]
        
        print(f"\n{'-'*80}")
        print(f"Processing row {row_idx}: {repo_with_underscore}")
        print(f"Issue description: {issue_description[:100]}...")
        
        # Base path for files
        base_path = f"decoupled/{n}/{weightBM25}_{weightSemantic}/{row_idx}/{repo_with_underscore}"
        
        # Create output directory for docstrings
        docstrings_dir = f"docstrings/{n}/{weightBM25}_{weightSemantic}/{row_idx}"
        os.makedirs(docstrings_dir, exist_ok=True)
        
        # Create log directory if logging is enabled
        log_dir = None
        if enable_logging:
            log_dir = os.path.join(docstrings_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            print(f"Logging enabled. Logs will be saved to: {log_dir}")
        
        # Step 1: Find files missing module docstrings
        files_missing_module = get_files_missing_module_docstrings(
            top_files, repo_with_underscore, row_idx, n, weightBM25, weightSemantic
        )
        print(f"Files missing module-level docstrings: {len(files_missing_module)}")
        
        # Step 2: Extract functions missing docstrings
        functions_missing = extract_functions_from_files(
            files_missing_module, repo_with_underscore, row_idx, n, weightBM25, weightSemantic
        )
        print(f"Number of functions missing docstrings: {len(functions_missing)}")
        
        # Step 3: Filter functions based on size and complexity
        filtered_funcs = filter_global_missing_functions_docstrings(functions_missing)
        print(f"Filtered functions missing docstrings: {len(filtered_funcs)}")
        
        # Step 4: Generate docstrings for functions - with proper batching
        batch_size = 16
        num_gpus_available = torch.cuda.device_count()
        adjusted_batch_size = batch_size * max(1, num_gpus_available)
        print(f"Using adjusted batch size: {adjusted_batch_size} based on GPU count: {num_gpus_available}")
        
        # Split functions into batches
        batches = [filtered_funcs[i:i+adjusted_batch_size] for i in range(0, len(filtered_funcs), adjusted_batch_size)]
        
        # Process each batch with vLLM
        all_function_results = []
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating function docstrings")):
            print(f"\nProcessing batch {batch_idx+1}/{len(batches)} with {len(batch)} functions")
            
            try:
                # Validate the batch before processing
                valid_batch = [func for func in batch if validate_function(func)]
                if not valid_batch:
                    print(f"Skipping batch {batch_idx+1} - no valid functions found")
                    continue
                    
                batch_results = batch_generate_summaries(valid_batch, model)
                all_function_results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch {batch_idx+1}: {str(e)}")
                # Try processing one by one for this batch
                for func_idx, func in enumerate(batch):
                    try:
                        if validate_function(func):
                            print(f"Processing function {func_idx+1}/{len(batch)} individually")
                            result = process_single_function(func, model)
                            if result:
                                all_function_results.append(result)
                    except Exception as e2:
                        print(f"Failed to process function {func_idx+1} individually: {str(e2)}")
        
        # Step 5: Generate module summaries using the function docstrings - with proper batching
        module_batch_size = max(1, adjusted_batch_size // 4)  # Module summaries are longer, use smaller batch
        
        # Prepare module data
        module_data = []
        for file_path in files_missing_module:
            full_path = os.path.join(base_path, file_path)
            
            if not os.path.exists(full_path):
                print(f"Warning: File not found: {full_path}")
                continue
                
            try:
                # Read file content
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Find relevant function docstrings for this file
                file_functions = [
                    (func, docstring) for func, docstring in all_function_results
                    if func["file"].endswith(file_path)
                ]
                
                if file_functions:
                    # Create module info
                    module_info = {
                        "path": file_path,
                        "full_path": full_path,
                        "content": content,
                        "file_functions": file_functions
                    }
                    module_data.append(module_info)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Split modules into batches
        module_batches = [module_data[i:i+module_batch_size] for i in range(0, len(module_data), module_batch_size)]
        
        # Process each batch
        all_module_results = []
        for batch_idx, batch in enumerate(tqdm(module_batches, desc="Generating module docstrings")):
            print(f"\nProcessing module batch {batch_idx+1}/{len(module_batches)} with {len(batch)} modules")
            
            try:
                # Process batch
                batch_prompts = []
                for module_info in batch:
                    prompt = create_module_prompt_with_docstrings(
                        module_info["path"], 
                        module_info["content"], 
                        module_info["file_functions"]
                    )
                    batch_prompts.append(prompt)
                
                # Configure sampling parameters
                sampling_params = SamplingParams(
                    temperature=0.2,
                    max_tokens=300,
                    top_k=50,
                )
                
                # Use vLLM to generate all outputs at once
                outputs = model.generate(batch_prompts, sampling_params)
                
                # Process results
                for i, output in enumerate(outputs):
                    generated = output.outputs[0].text
                    extracted = extract_module_docstring(generated)
                    
                    module_result = (batch[i], extracted)
                    all_module_results.append(module_result)
                    
                    # Log if enabled
                    if log_dir:
                        log_file = os.path.join(log_dir, f"{os.path.basename(batch[i]['path'])}_module_summary.txt")
                        with open(log_file, "w", encoding="utf-8") as f:
                            f.write("PROMPT:\n")
                            f.write("-------\n")
                            f.write(batch_prompts[i])
                            f.write("\n\nRESPONSE:\n")
                            f.write("---------\n")
                            f.write(generated)
                            f.write("\n\nEXTRACTED DOCSTRING:\n")
                            f.write("-------------------\n")
                            f.write(extracted)
            except Exception as e:
                print(f"Error processing module batch {batch_idx+1}: {str(e)}")
                # Handle individual modules if batch processing fails
                for module_idx, module_info in enumerate(batch):
                    try:
                        prompt = create_module_prompt_with_docstrings(
                            module_info["path"], 
                            module_info["content"], 
                            module_info["file_functions"]
                        )
                        
                        # Use vLLM for single generation
                        single_output = model.generate([prompt], SamplingParams(
                            temperature=0.2,
                            max_tokens=300,
                            top_k=50,
                        ))
                        
                        generated = single_output[0].outputs[0].text
                        extracted = extract_module_docstring(generated)
                        
                        module_result = (module_info, extracted)
                        all_module_results.append(module_result)
                        
                        # Log if enabled
                        if log_dir:
                            log_file = os.path.join(log_dir, f"{os.path.basename(module_info['path'])}_module_summary.txt")
                            with open(log_file, "w", encoding="utf-8") as f:
                                f.write("PROMPT:\n")
                                f.write("-------\n")
                                f.write(prompt)
                                f.write("\n\nRESPONSE:\n")
                                f.write("---------\n")
                                f.write(generated)
                                f.write("\n\nEXTRACTED DOCSTRING:\n")
                                f.write("-------------------\n")
                                f.write(extracted)
                    except Exception as e2:
                        print(f"Failed to process module {module_idx+1} individually: {str(e2)}")
        
        # Save the results
        with open(os.path.join(docstrings_dir, "function_docstrings.json"), "w") as f:
            # Convert function objects to serializable format
            serializable_function_results = [
                {
                    "function": {
                        "name": func["name"],
                        "file": func["file"],
                        "class_name": func["class_name"]
                    },
                    "docstring": docstring
                }
                for func, docstring in all_function_results
            ]
            json.dump(serializable_function_results, f, indent=4)
        
        with open(os.path.join(docstrings_dir, "module_docstrings.json"), "w") as f:
            # Convert module objects to serializable format
            serializable_module_results = [
                {
                    "file": {
                        "path": module_info["path"],
                        "full_path": module_info["full_path"]
                    },
                    "docstring": docstring
                }
                for module_info, docstring in all_module_results
            ]
            json.dump(serializable_module_results, f, indent=4)
        
        # Store enhanced data
        enhanced_data[row_idx] = {
            **entry,
            "function_docstrings": os.path.join(docstrings_dir, "function_docstrings.json"),
            "module_docstrings": os.path.join(docstrings_dir, "module_docstrings.json")
        }
    
    # Release model resources
    release_vllm_model(model)
    
    return enhanced_data


def create_module_prompt_with_docstrings(file_path, content, function_docstrings):
    """
    Create a prompt for module summary generation that includes function docstrings.
    
    Args:
        file_path (str): Path to the file
        content (str): File content
        function_docstrings (list): List of (function, docstring) tuples
        
    Returns:
        str: Prompt for the LLM
    """
    docstrings_section = "\n\n### Function Docstrings:\n"
    for func, docstring in function_docstrings:
        docstrings_section += f"\nFunction: {func['name']}\n"
        if func['class_name']:
            docstrings_section += f"Class: {func['class_name']}\n"
        docstrings_section += f"Docstring: {docstring}\n"
    
    prompt = f"""Below is the complete source code of a Python file, along with docstrings for some of its functions.

Your task is to read the file and generate a concise and informative module-level docstring that describes:
- The overall purpose of the file
- The main classes and functions it defines
- Key responsibilities or logic handled in this file
- How the classes and functions interact (if applicable)

Be concise, but include technical details where useful.

### Python File: {file_path}
{content}

{docstrings_section}

### Module-Level Docstring:
"""
    
    return prompt

import ast
import textwrap

def insert_inline_docstrings(code: str, docstring_data: list) -> str:
    """
    Insert function docstrings inline into the source code string using AST.
    """
    lines = code.splitlines()
    tree = ast.parse(code)

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    doc_map = {}
    for item in docstring_data:
        func = item["function"]
        key = (func["class_name"], func["name"])
        doc_map[key] = item["docstring"]

    new_lines = lines[:]
    offset = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            class_name = None

            parent = getattr(node, 'parent', None)
            while parent:
                if isinstance(parent, ast.ClassDef):
                    class_name = parent.name
                    break
                parent = getattr(parent, 'parent', None)

            key = (class_name, func_name)
            if key in doc_map:
                raw_docstring = doc_map[key]
                docstring = f'"""{raw_docstring}"""\n'

                indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
                docstring_lines = textwrap.indent(docstring, ' ' * (indent + 4)).splitlines()

                insert_at = node.body[0].lineno if node.body else node.lineno + 1
                new_lines[insert_at - 1 + offset:insert_at - 1 + offset] = docstring_lines
                offset += len(docstring_lines)

    return "\n".join(new_lines)


def narrow_top_files(top_files_data, n, p, issue_description, weightBM25=0.8, weightSemantic=0.2, sem_weight=0.7, enable_logging=False):
    """
    Narrow down the top k files to top p files that are most relevant to the issue description,
    using generated docstrings and module summaries.
    
    Args:
        top_files_data (dict): Enhanced top files data with docstring information
        n (int): Number of top files
        p (int): Number of files to narrow down to
        issue_description (str): Description of the issue
        weightBM25 (float): Weight for BM25 ranking in initial selection
        weightSemantic (float): Weight for semantic ranking in initial selection
        sem_weight (float): Weight for semantic ranking in narrowing phase (default: 0.7)
        enable_logging (bool): Whether to log the process
        
    Returns:
        dict: Narrowed top files data
    """
    # Store narrowed data
    narrowed_data = {}
    
    # Initialize the semantic model directly (NOT through vLLM)
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"\nNarrowing down top {n} files to top {p} files...")
    print(f"Using semantic weight: {sem_weight}, BM25 weight: {1-sem_weight}")
    
    # Process each entry in the top files data
    for row_idx, entry in top_files_data.items():
        repo_with_underscore = entry["repo_with_underscore"]
        top_k_files = entry["top_n_files"]
        
        print(f"\n{'-'*40}")
        print(f"Processing row {row_idx}: {repo_with_underscore}")
        
        # Create log directory if logging is enabled
        log_dir = None
        if enable_logging:
            log_dir = f"narrowing_logs/{n}_{p}/{weightBM25}_{weightSemantic}/{row_idx}"
            os.makedirs(log_dir, exist_ok=True)
            print(f"Logging enabled. Logs will be saved to: {log_dir}")
        
        # Load function and module docstrings
        function_docstrings = []
        if "function_docstrings" in entry and os.path.exists(entry["function_docstrings"]):
            with open(entry["function_docstrings"], "r") as f:
                function_docstrings = json.load(f)
                print(f"Loaded {len(function_docstrings)} function docstrings")
        
        module_docstrings = []
        if "module_docstrings" in entry and os.path.exists(entry["module_docstrings"]):
            with open(entry["module_docstrings"], "r") as f:
                module_docstrings = json.load(f)
                print(f"Loaded {len(module_docstrings)} module docstrings")
        
        # Create enhanced file content by adding docstrings
        enhanced_file_contents = {}
        for file_path in top_k_files:
            base_path = f"decoupled/{n}/{weightBM25}_{weightSemantic}/{row_idx}/{repo_with_underscore}"
            full_path = os.path.join(base_path, file_path)
            
            if not os.path.exists(full_path):
                print(f"Warning: File not found: {full_path}")
                continue
                
            try:
                # Read file content
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Add module docstring if available
                module_docstring = next(
                    (item["docstring"] for item in module_docstrings if item["file"]["path"] == file_path),
                    None
                )
                
                if module_docstring:
                    content = module_docstring + "\n\n" + content
                
                # Add function docstrings (informational only, not inserted in the file)
                file_function_docstrings = [
                    f"{item['function']['name']}: {item['docstring']}"
                    for item in function_docstrings
                    if item['function']['file'].endswith(file_path)
                ]
                
                #New way of adding file docstrings inline along with respective functions
                inline_docstring_items = [
                    item for item in function_docstrings
                    if item['function']['file'].endswith(file_path)
                ]
                enhanced_content = insert_inline_docstrings(content, inline_docstring_items)
                
                enhanced_file_contents[file_path] = enhanced_content
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Rank files using enhanced content
        if enhanced_file_contents:
            # Create a list of files and their contents
            file_list = list(enhanced_file_contents.keys())
            content_list = [enhanced_file_contents[file] for file in file_list]
            
            # Get issue embedding using SentenceTransformer directly
            issue_embedding = semantic_model.encode(issue_description, convert_to_tensor=True)
            
            # Get file embeddings using SentenceTransformer directly
            file_embeddings = semantic_model.encode(content_list, convert_to_tensor=True)
            
            # Calculate semantic similarity scores
            semantic_scores = util.pytorch_cos_sim(issue_embedding, file_embeddings)[0].cpu().numpy()
            
            # Tokenize content for BM25
            tokenized_content = [nltk.word_tokenize(content.lower()) for content in content_list]
            bm25 = BM25Okapi(tokenized_content)
            query_tokens = nltk.word_tokenize(issue_description.lower())
            bm25_scores = np.array(bm25.get_scores(query_tokens))
            
            # Normalize scores
            if np.max(semantic_scores) > 0:
                semantic_scores = semantic_scores / np.max(semantic_scores)
            if np.max(bm25_scores) > 0:
                bm25_scores = bm25_scores / np.max(bm25_scores)
            
            # Combine scores using the specified semantic weight
            combined_scores = sem_weight * semantic_scores + (1 - sem_weight) * bm25_scores
            
            # Rank files by combined score
            ranked_files = sorted(zip(file_list, combined_scores, semantic_scores, bm25_scores), 
                                key=lambda x: x[1], reverse=True)
            
            # Take top p files
            top_p_files = [file for file, _, _, _ in ranked_files[:p]]
            
            # Log the ranking if enabled
            if log_dir:
                log_file = os.path.join(log_dir, "ranking_details.json")
                ranking_details = {
                    "issue_description": issue_description,
                    "semantic_weight": sem_weight,
                    "bm25_weight": (1 - sem_weight),
                    "rankings": [
                        {
                            "file": file,
                            "combined_score": float(combined_score),
                            "semantic_score": float(semantic_score),
                            "bm25_score": float(bm25_score),
                            "selected": file in top_p_files
                        }
                        for file, combined_score, semantic_score, bm25_score in ranked_files
                    ]
                }
                
                with open(log_file, "w") as f:
                    json.dump(ranking_details, f, indent=4)
                print(f"Saved ranking details to {log_file}")
            
            # Store narrowed data
            narrowed_data[row_idx] = {
                **entry,
                "top_p_files": top_p_files
            }
            
            # Print top ranked files
            print(f"\nTop {p} files selected:")
            for i, (file, score, sem_score, bm25_score) in enumerate(ranked_files[:p]):
                print(f"{i+1}. {file}")
                print(f"   Combined score: {score:.4f} (Semantic: {sem_score:.4f}, BM25: {bm25_score:.4f})")
        else:
            # If no enhanced content, just take the first p files
            narrowed_data[row_idx] = {
                **entry,
                "top_p_files": top_k_files[:p]
            }
            print("\nNo enhanced content found. Taking first p files.")
    
    return narrowed_data