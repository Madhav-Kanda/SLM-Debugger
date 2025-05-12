# SLM-Debugger: SWEBench Bug Fixing Pipeline using SLMs

This repository contains a pipeline for localizing and fixing bugs in software repositories using the SWEBench dataset. The pipeline combines semantic search, docstring generation, and bug fixing to automatically generate patches for bugs, and issues.

## Pipeline Overview

The pipeline consists of the following stages:

1. **Top-k File Selection**: Identifies the top k files most relevant to the issue description using a combination of BM25 and semantic similarity.
2. **Docstring Generation**: Generates docstrings for functions missing documentation in the top-k files to better understand their purpose.
3. **Module Summary Generation**: Creates file-level (module) docstrings using the generated function docstrings.
4. **Top-p File Narrowing**: Uses the enhanced files (with docstrings) to narrow down to the top p files most likely to contain the bug.
5. **Function Localization**: Identifies specific functions within these files that are most likely to contain the bug.
6. **Patch Generation**: Generates fixes for the identified functions and creates patch files.
7. **Evaluation Preparation**: Formats the results for evaluation with the SWEBench toolkit.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- vLLM
- SentenceTransformer
- NLTK
- Rank-BM25
- Matplotlib 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Madhav-Kanda/SLM-Debugger.git
cd swebench-bug-fixing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install git LFS:
```bash
sudo apt install git-lfs
```

4. Download the files pushed to LFS
```bash
git lfs install
git lfs pull
```

## Usage

### Running the Complete Pipeline

To run the complete pipeline:

```bash
python main.py --k 60 --p 10 --n 5
```

Options:
- `--k`: Number of top files to initially consider (default: 60)
- `--p`: Number of top files to narrow down to (default: 10)
- `--n`: Number of top functions to consider for bug fixing (default: 5)
- `--bm25`: Weight for BM25 ranking in initial file selection (default: 0.8)
- `--semantic`: Weight for semantic ranking in initial file selection (default: 0.2)
- `--sem-weight`: Weight for semantic ranking in narrowing files (default: 0.7)
- `--output`: Output directory for patches (default: patches/{k}_{p}_{n}/{bm25}_{semantic})
- `--enable-logging`: Enable logging of LLM inputs and outputs
- `--skip-docstrings`: Skip generating docstrings (use with caution)
- `--dataset`: Path to the dataset (default: data/swe_bench_lite_test.json)

### Testing a Single Issue

To run the patch generation for a single issue:

```bash
python test_bug_patch.py --row_idx 5 --n 3 --enable-logging
```

Options:
- `--row_idx`: Required index identifying which issue to analyze
- `--n`: Number of top functions to analyze (default: 5)
- `--top_files_path`: Path to the pre-computed top files (default: top_files/60/0.8_0.2.json)
- `--output_dir`: Directory to save patches (default: patches/test_{row_idx})
- `--dataset_path`: Path to the SWEBench dataset (default: data/swe_bench_lite_test.json)
- `--enable-logging`: Enable logging of LLM inputs and outputs

### Running SWEBench Evaluation

To prepare your patches for SWEBench evaluation:

```bash
python run_evaluation.py --patch-results patches/60_10_5/0.8_0.2/patch_results.json --swekit-path /path/to/swebench-toolkit
```

Options:
- `--patch-results`: Path to the patch results JSON file
- `--output`: Path to save the SWEBench-compatible evaluation file (optional)
- `--swekit-path`: Path to the SWEBench evaluation toolkit
- `--only-prepare`: Only prepare the evaluation file without running the evaluation

### Visualizing Results

To visualize the results of your pipeline:

```bash
python visualize_results.py --results patches/60_10_5/0.8_0.2/patch_results.json
```

Options:
- `--results`: Path to patch results JSON
- `--output`: Output directory for visualizations (default: {results_dir}/visualizations)

## Directory Structure

```
slm-debugger/
├── data/
│   └── swe_bench_lite_test.json      # SWEBench dataset
├── src/
│   ├── iteration.py                  # File ranking utilities
│   ├── ranking.py                    # Ranking utilities
│   ├── docstring_utils.py            # Docstring utilities
│   ├── bug_localization.py           # Bug localization and fixing
│   └── utils.py                      #  Utilities for various tasks
├── top_files/                        # Top files data
├── enhanced_data/                    # Enhanced data with docstrings
├── narrowed_data/                    # Narrowed top files data
├── patches/                          # Generated patches
├── decoupled/                        # Decoupled repository files
├── swe_bench_repos/                  # Cloned repositories
├── integrated_pipeline.py            # Complete pipeline script
├── test_bug_patch.py                 # Single issue testing script
├── run_evaluation.py                 # SWEBench evaluation script
├── visualize_results.py              # Results visualization script
├── main.py                           # Original main script
└── requirements.txt                  # Dependencies
```

## Outputs

The pipeline generates the following outputs:

1. **Patches**: Individual function patches, combined patches, and SWEBench-compatible patches
2. **Docstrings**: Generated function and module docstrings
3. **Logs**: LLM inputs and outputs (if logging is enabled)
4. **Visualizations**: Charts and summary report (when using visualize_results.py)

## Model Configuration

The pipeline uses the following models:
- **Bug Localization and Fixing**: Qwen/Qwen2.5-7B-Instruct
- **Docstring Generation**: Qwen/Qwen2.5-7B-Instruct
- **Semantic Search**: all-MiniLM-L6-v2 (SentenceTransformer)

You can modify the model configurations in the corresponding utility files.


## References

This project is built to work with the SWEBench benchmark:
- SWEBench: [https://github.com/SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench)