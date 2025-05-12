# src/ranking.py
import os
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# Download NLTK tokenizer data if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

def rank_files_by_name(file_list, issue_description, n, weightBM25, weightSemantic):
    """
    Rank files by their names based on relevance to the issue description.
    
    Args:
        file_list (list): List of file paths
        issue_description (str): Description of the issue
        n (int): Number of top files to return
        weightBM25 (float): Weight for BM25 ranking
        weightSemantic (float): Weight for semantic ranking
        
    Returns:
        list: Ranked list of files (top n)
    """
    tokenized_file_list = []
    
    # Initialize semantic model directly (NOT through vLLM)
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Truncate file paths to remove repository prefix
    file_list_truncated = [file.split("/", 1)[1] for file in file_list]

    # Tokenize file names for BM25
    for file in file_list_truncated:
        # Replace slashes with spaces for better tokenization
        tokens = file.replace("/", " ").split()
        tokenized_file_list.append(nltk.word_tokenize(" ".join(tokens).lower()))

    # BM25 scoring on file names
    bm25 = BM25Okapi(tokenized_file_list)
    query_tokens = nltk.word_tokenize(issue_description.lower())
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    # Semantic similarity scoring
    issue_embedding = semantic_model.encode(issue_description, convert_to_tensor=True)
    file_embeddings = semantic_model.encode(file_list_truncated, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(issue_embedding, file_embeddings)[0].cpu().numpy()

    # Normalize scores
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    if np.max(semantic_scores) > 0:
        semantic_scores = semantic_scores / np.max(semantic_scores)

    # Combine scores with weights
    combined_scores = weightBM25 * bm25_scores + weightSemantic * semantic_scores
    
    # Rank files by combined score
    ranked_files = sorted(zip(file_list_truncated, combined_scores), key=lambda x: x[1], reverse=True)
    
    # Return top n files
    return [file for file, _ in ranked_files[:n]]

def rank_files_by_content(row_index, repo_with_underscore, file_list, issue_description, n, weightBM25, weightSemantic, main_folder):
    """
    Rank files by their content based on relevance to the issue description.
    
    Args:
        row_index (str/int): Index of the row in the dataset
        repo_with_underscore (str): Repository name with underscores
        file_list (list): List of file paths
        issue_description (str): Description of the issue
        n (int): Number of top files to return
        weightBM25 (float): Weight for BM25 ranking
        weightSemantic (float): Weight for semantic ranking
        main_folder (str): Main folder containing the files
        
    Returns:
        list: Ranked list of files (top n)
    """
    tokenized_file_list = []
    file_content_list = []
    
    # Read content of each file
    for file in file_list:
        file_path = f"{main_folder}/{row_index}/{repo_with_underscore}/{file}"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            file_path_string = f"{repo_with_underscore}/{file}"
            file_path_with_content = file_path_string + "\n" + content
            file_content_list.append(content)
            tokenized_file_list.append(nltk.word_tokenize(file_path_with_content))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Add empty content for files that couldn't be read
            file_content_list.append("")
            tokenized_file_list.append([])
    
    # BM25 scoring on file content
    bm25 = BM25Okapi(tokenized_file_list)
    query_tokens = nltk.word_tokenize(issue_description.lower())
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    # Semantic similarity scoring
    # Initialize semantic model directly (NOT through vLLM)
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    issue_embedding = semantic_model.encode(issue_description, convert_to_tensor=True)
    file_embeddings = semantic_model.encode(file_content_list, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(issue_embedding, file_embeddings)[0].cpu().numpy()

    # Normalize scores
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    if np.max(semantic_scores) > 0:
        semantic_scores = semantic_scores / np.max(semantic_scores)

    # Combine scores with weights
    combined_scores = weightBM25 * bm25_scores + weightSemantic * semantic_scores
    
    # Process paths to match the expected format
    processed_paths = [s.replace(' ', '/').split('/', 1)[1] for s in [f"{repo_with_underscore}/{file}" for file in file_list]]
    
    # Rank files by combined score
    ranked_files = sorted(zip(processed_paths, combined_scores), key=lambda x: x[1], reverse=True)
    
    # Return top n files
    return [file for file, _ in ranked_files[:n]]

def rank_functions_by_relevance(functions, issue_description, n=5):
    """
    Rank functions based on their relevance to the issue description.
    
    Args:
        functions (list): List of function dictionaries
        issue_description (str): Description of the issue
        n (int): Number of top functions to return
        
    Returns:
        list: Ranked list of functions (top n)
    """
    if not functions:
        return []
    
    # Prepare documents for ranking (function names and source code)
    docs = []
    for func in functions:
        func_doc = f"{func['name']} {func['source']}"
        docs.append(func_doc)
    
    # BM25 ranking (70% weight)
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = issue_description.lower().split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    
    # Normalize BM25 scores
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    
    # Semantic similarity ranking (30% weight)
    # Initialize semantic model directly (NOT through vLLM)
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    issue_embedding = semantic_model.encode(issue_description, convert_to_tensor=True)
    doc_embeddings = semantic_model.encode(docs, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(issue_embedding, doc_embeddings)[0].cpu().numpy()
    
    # Normalize semantic scores
    if np.max(semantic_scores) > 0:
        semantic_scores = semantic_scores / np.max(semantic_scores)
    
    # Combine scores with weights (70% BM25, 30% semantic)
    combined_scores = 0.7 * bm25_scores + 0.3 * semantic_scores
    
    # Add scores to functions
    scored_functions = []
    for i, func in enumerate(functions):
        func_copy = func.copy()
        func_copy["relevance_score"] = float(combined_scores[i])
        scored_functions.append(func_copy)
    
    # Sort by relevance score
    scored_functions.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Return top n functions
    return scored_functions[:n]