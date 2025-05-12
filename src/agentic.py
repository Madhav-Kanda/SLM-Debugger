import autogen
import ast
import json
import subprocess
import tempfile
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class LocalModelConfig:
    """Configuration for locally hosted SLM"""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load the local model and tokenizer"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device
        )
        print("Model loaded successfully!")
        
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """Generate response using the local model"""
        if self.pipeline is None:
            self.load_model()
            
        # Format prompt for Qwen model
        formatted_prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = outputs[0]["generated_text"]
        # Extract only the assistant's response
        assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        return assistant_response

class ProgramRepairSystem:
    """AutoGen-based agentic system for program repair with auto-termination"""
    
    def __init__(self, model_config: LocalModelConfig):
        self.model_config = model_config
        self.agents = self._create_agents()
        self.group_chat = self._create_group_chat()
        self.manager = self._create_manager()
        self.repair_success = False
        
    def _create_agents(self):
        """Create specialized agents for different aspects of program repair"""
        
        # Custom response function for AutoGen agents using local SLM
        def create_local_response_function(system_message: str, is_tester: bool = False):
            def response_function(recipient, messages, sender, config):
                """Custom reply function for the agent"""
                if not messages:
                    return False, None
                
                # Get the last message content
                last_message = messages[-1].get("content", "")
                
                # Combine system message with user message
                prompt = f"{system_message}\n\nUser: {last_message}\nAssistant:"
                
                # Generate response using local model
                response = self.model_config.generate_response(
                    prompt,
                    max_tokens=1000,
                    temperature=0.3
                )
                
                # Check if this is the tester and if the fix is successful
                if is_tester and "CONCLUSION:" in response:
                    conclusion = response.split("CONCLUSION:")[1].strip()
                    if "successful" in conclusion.lower():
                        self.repair_success = True
                
                return True, response
            
            return response_function
        
        # Analyzer Agent - Analyzes the bug and creates a repair plan
        analyzer_system_msg = """You are a bug analysis expert. 
Your task is to:
1. Analyze the buggy function to understand what it does
2. Identify the type of bug (syntax error, logic error, runtime error, etc.)
3. Create a detailed repair plan with specific steps
4. Suggest possible fixes without implementing them yet

Format your analysis as:
ANALYSIS:
[Your analysis here]

BUG TYPE:
[Bug type]

REPAIR PLAN:
[Numbered steps for repair]
"""
        
        # Create ConversableAgent with custom response function
        analyzer = autogen.ConversableAgent(
            name="BugAnalyzer",
            system_message=analyzer_system_msg,
            llm_config=False,  # Disable default LLM
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
        
        # Register custom reply function with correct signature
        analyzer.register_reply([autogen.ConversableAgent, None], create_local_response_function(analyzer_system_msg))
        
        # Fixer Agent - Implements the actual fix
        fixer_system_msg = """You are an expert code fixer.
Your task is to:
1. Take the repair plan from the analyzer
2. Implement the fix for the buggy function
3. Maintain the original function signature
4. Ensure the fix is minimal and doesn't break existing functionality
5. Provide only the fixed function, not the entire file

Format your response as:
FIXED_CODE:
```python
[Fixed function here]
```

EXPLANATION:
[Brief explanation of the fix]
"""
        
        fixer = autogen.ConversableAgent(
            name="CodeFixer",
            system_message=fixer_system_msg,
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
        fixer.register_reply([autogen.ConversableAgent, None], create_local_response_function(fixer_system_msg))
        
        # Validator Agent - Validates the fix
        validator_system_msg = """You are a code validation expert.
Your task is to:
1. Validate the proposed fix syntactically and logically
2. Check if the fix maintains the original function's purpose
3. Identify any potential edge cases or new bugs introduced
4. Suggest test cases to verify the fix

Format your response as:
VALIDATION:
[Your validation result]

ISSUES_FOUND:
[List of issues if any]

TEST_CASES:
[Suggested test cases]
"""
        
        validator = autogen.ConversableAgent(
            name="CodeValidator",
            system_message=validator_system_msg,
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
        validator.register_reply([autogen.ConversableAgent, None], create_local_response_function(validator_system_msg))
        
        # Tester Agent - Executes and tests the fix
        tester_system_msg = """You are a code testing expert.
Your task is to:
1. Create comprehensive test cases for the fixed function
2. Execute tests to verify the fix works correctly
3. Compare behavior with the original function
4. Report test results clearly

Format your response as:
TEST_RESULTS:
[Test execution results]

CONCLUSION:
[Whether the fix is successful]
"""
        
        tester = autogen.ConversableAgent(
            name="CodeTester",
            system_message=tester_system_msg,
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
        # Register tester with special flag to check for success
        tester.register_reply([autogen.ConversableAgent, None], create_local_response_function(tester_system_msg, is_tester=True))
        
        # User Proxy Agent - Coordinates the repair process
        user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,  # Only allow one auto-reply to check for termination
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") or self.repair_success,
            system_message="""You are the program repair coordinator.
You initiate the repair process and ensure all agents complete their tasks.
Execute code when necessary and provide the final repaired function.
""",
            code_execution_config={
                "work_dir": "repair_workspace",
                "use_docker": False,
            }
        )
        
        # Add custom response to user proxy to terminate when repair is successful
        def user_proxy_response(recipient, messages, sender, config):
            if self.repair_success:
                return True, """
Program repair has been successfully completed!
The buggy function has been fixed and tested.
                
TERMINATE
"""
            return False, None
        
        user_proxy.register_reply([autogen.ConversableAgent, None], user_proxy_response)
        
        return {
            "analyzer": analyzer,
            "fixer": fixer,
            "validator": validator,
            "tester": tester,
            "user_proxy": user_proxy
        }
    
    def _create_group_chat(self):
        """Create a group chat for agent collaboration"""
        # Custom speaker selection method for agent flow
        def custom_speaker_selection(last_speaker, groupchat):
            # Define the flow order
            agent_order = [
                "UserProxy",
                "BugAnalyzer",
                "CodeFixer",
                "CodeValidator",
                "CodeTester",
                "UserProxy"  # Back to UserProxy to check for termination
            ]
            
            # Get current speaker index
            current_speaker_name = last_speaker.name if last_speaker else "UserProxy"
            
            try:
                current_index = agent_order.index(current_speaker_name)
                next_index = (current_index + 1) % len(agent_order)
                next_speaker_name = agent_order[next_index]
            except ValueError:
                # Default to UserProxy if speaker not found
                next_speaker_name = "UserProxy"
            
            # Find and return the next speaker agent
            for agent in groupchat.agents:
                if agent.name == next_speaker_name:
                    return agent
            
            # Default to first agent
            return groupchat.agents[0]
        
        return autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["analyzer"],
                self.agents["fixer"],
                self.agents["validator"],
                self.agents["tester"]
            ],
            messages=[],
            max_round=15,
            speaker_selection_method=custom_speaker_selection
        )
    
    def _create_manager(self):
        """Create a group chat manager with disabled LLM"""
        # Create a manager that doesn't use OpenAI or any LLM
        manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=False,  # Disable LLM for the manager
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") or self.repair_success
        )
        
        return manager
    
    def extract_function_from_code(self, code: str, function_name: str) -> str:
        """Extract a specific function from code"""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the function's code
                start_line = node.lineno - 1
                end_line = node.end_lineno
                lines = code.split('\n')
                return '\n'.join(lines[start_line:end_line])
        return None
    
    def repair_function(self, buggy_function: str, function_name: str = None, 
                        error_message: str = None, expected_behavior: str = None) -> Dict:
        """
        Repair a buggy function using the agentic approach
        
        Args:
            buggy_function: The buggy function code
            function_name: Name of the function (optional)
            error_message: The error message if available
            expected_behavior: Expected behavior of the function
        
        Returns:
            Dict containing the repair results
        """
        # Reset repair success flag
        self.repair_success = False
        
        # Prepare the initial message
        initial_message = f"""
I need help fixing this buggy function:

BUGGY FUNCTION:
```python
{buggy_function}
```

{f"ERROR MESSAGE: {error_message}" if error_message else ""}
{f"EXPECTED BEHAVIOR: {expected_behavior}" if expected_behavior else ""}

Please analyze this bug and provide a fix following our systematic approach.
"""
        
        # Start the group chat
        self.agents["user_proxy"].initiate_chat(
            self.manager,
            message=initial_message
        )
        
        # Extract results from the conversation
        results = {
            "original_function": buggy_function,
            "analysis": None,
            "fix": None,
            "validation": None,
            "test_results": None,
            "success": self.repair_success
        }
        
        # Parse the conversation for results
        for message in self.group_chat.messages:
            content = message.get("content", "")
            
            if "ANALYSIS:" in content:
                results["analysis"] = content.split("ANALYSIS:")[1].split("BUG TYPE:")[0].strip()
            
            if "FIXED_CODE:" in content:
                # Extract the fixed code from between ```python and ```
                import re
                match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
                if match:
                    results["fix"] = match.group(1).strip()
            
            if "VALIDATION:" in content:
                results["validation"] = content.split("VALIDATION:")[1].split("ISSUES_FOUND:")[0].strip()
            
            if "TEST_RESULTS:" in content:
                results["test_results"] = content.split("TEST_RESULTS:")[1].split("CONCLUSION:")[0].strip()
                
                # Check if the fix was successful
                if "CONCLUSION:" in content:
                    conclusion = content.split("CONCLUSION:")[1].strip()
                    results["success"] = "successful" in conclusion.lower()
        
        return results

class CSVBugRepairProcessor:
    """Process buggy functions from CSV file and apply repairs"""
    
    def __init__(self, csv_path: str, model_config: LocalModelConfig):
        """
        Initialize the CSV processor
        
        Args:
            csv_path: Path to the CSV file containing buggy functions
            model_config: Configuration for the local SLM
        """
        self.csv_path = csv_path
        self.model_config = model_config
        self.repair_system = ProgramRepairSystem(model_config)
        self.df = None
        self.repair_results = []
        
    def load_csv(self):
        """Load the CSV file"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded CSV with {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def extract_function_info(self, row):
        """Extract function information from a CSV row"""
        return {
            "index": row.get("index", -1),
            "repo": row.get("repo", ""),
            "function_name": row.get("function_name", ""),
            "file_path": row.get("file_path", ""),
            "source": row.get("source", ""),  # The buggy function code
            "issue_description": row.get("issue_description", "")
        }
    
    def repair_function_from_row(self, row_index: int):
        """Repair a single function from the CSV"""
        if self.df is None:
            print("CSV not loaded. Call load_csv() first.")
            return None
        
        if row_index >= len(self.df):
            print(f"Row index {row_index} out of bounds")
            return None
        
        row = self.df.iloc[row_index]
        function_info = self.extract_function_info(row)
        
        print(f"\nProcessing function '{function_info['function_name']}' from {function_info['repo']}")
        print(f"File: {function_info['file_path']}")
        print(f"Issue: {function_info['issue_description']}")
        
        # Prepare the repair request
        buggy_function = function_info['source']
        function_name = function_info['function_name']
        error_message = function_info['issue_description']
        
        # Apply repair
        repair_results = self.repair_system.repair_function(
            buggy_function=buggy_function,
            function_name=function_name,
            error_message=error_message,
            expected_behavior=f"Fix the issue: {error_message}"
        )
        
        # Add original info to results
        repair_results.update({
            "original_info": function_info,
            "row_index": row_index
        })
        
        self.repair_results.append(repair_results)
        return repair_results
    
    def repair_all_functions(self):
        """Repair all functions in the CSV"""
        if self.df is None:
            print("CSV not loaded. Call load_csv() first.")
            return []
        
        print(f"\nStarting repair of {len(self.df)} functions...")
        
        for idx in range(len(self.df)):
            print(f"\n{'='*50}")
            print(f"Processing row {idx + 1}/{len(self.df)}")
            result = self.repair_function_from_row(idx)
            
            if result and result.get('success'):
                print(f"✅ Successfully repaired: {result['original_info']['function_name']}")
            else:
                print(f"❌ Failed to repair: {result['original_info']['function_name'] if result else 'Unknown'}")
        
        return self.repair_results
    
    def create_results_summary(self):
        """Create a summary of all repair results"""
        if not self.repair_results:
            return "No repair results available"
        
        summary = f"# Repair Summary\n\n"
        summary += f"Total functions processed: {len(self.repair_results)}\n"
        
        successful = [r for r in self.repair_results if r.get('success', False)]
        failed = [r for r in self.repair_results if not r.get('success', False)]
        
        summary += f"Successful repairs: {len(successful)}\n"
        summary += f"Failed repairs: {len(failed)}\n"
        summary += f"Success rate: {len(successful)/len(self.repair_results)*100:.1f}%\n\n"
        
        # Detailed results
        summary += "## Detailed Results\n\n"
        
        for i, result in enumerate(self.repair_results, 1):
            info = result.get('original_info', {})
            summary += f"### {i}. {info.get('function_name', 'Unknown Function')}\n"
            summary += f"**Repository**: {info.get('repo', 'Unknown')}\n"
            summary += f"**File**: {info.get('file_path', 'Unknown')}\n"
            summary += f"**Status**: {'✅ Success' if result.get('success') else '❌ Failed'}\n"
            
            if result.get('success'):
                summary += "\n**Original Code:**\n```python\n" + result.get('original_function', '') + "\n```\n"
                summary += "\n**Fixed Code:**\n```python\n" + result.get('fix', '') + "\n```\n"
                summary += f"\n**Issue**: {info.get('issue_description', '')}\n"
            else:
                summary += f"\n**Error**: {result.get('error', 'Unknown error')}\n"
            
            summary += "\n---\n\n"
        
        return summary
    
    def save_results_csv(self, output_path: str):
        """Save repair results back to a CSV file"""
        if self.df is None or not self.repair_results:
            print("No data to save")
            return False
        
        # Create a new DataFrame with original data plus repair results
        results_df = self.df.copy()
        
        # Add new columns for repair results
        results_df['repair_successful'] = False
        results_df['fixed_code'] = ''
        results_df['repair_analysis'] = ''
        results_df['test_results'] = ''
        
        # Fill in repair results
        for result in self.repair_results:
            idx = result.get('row_index', -1)
            if idx >= 0 and idx < len(results_df):
                results_df.loc[idx, 'repair_successful'] = result.get('success', False)
                results_df.loc[idx, 'fixed_code'] = result.get('fix', '')
                results_df.loc[idx, 'repair_analysis'] = result.get('analysis', '')
                results_df.loc[idx, 'test_results'] = result.get('test_results', '')
        
        # Save to CSV
        try:
            results_df.to_csv(output_path, index=False)
            print(f"Saved results to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

# Example usage for the hit_functions.csv file
def process_hit_functions():
    # Set up model configuration
    local_model_config = LocalModelConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="auto"
    )
    
    # Create processor
    processor = CSVBugRepairProcessor("src/hit_functions.csv", local_model_config)
    
    # Load and process CSV
    if processor.load_csv():
        # Repair all functions
        results = processor.repair_all_functions()
        
        # Create summary
        summary = processor.create_results_summary()
        print("\n\n" + summary)
        
        # Save results
        processor.save_results_csv("hit_functions_repaired.csv")
        
        # Save summary to file
        with open("repair_summary.md", "w") as f:
            f.write(summary)
        
        print("\nAll done! Check 'hit_functions_repaired.csv' and 'repair_summary.md' for results.")
    else:
        print("Failed to load CSV file")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Repair buggy functions from CSV using local SLM")
    parser.add_argument("--csv", default="src/hit_functions.csv", help="Path to CSV file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--output", default="hit_functions_repaired.csv", help="Output CSV path")
    parser.add_argument("--summary", default="repair_summary.md", help="Summary file path")
    
    args = parser.parse_args()
    
    # Set up model configuration
    local_model_config = LocalModelConfig(
        model_name=args.model,
        device="auto"
    )
    
    # Create processor
    processor = CSVBugRepairProcessor(args.csv, local_model_config)
    
    # Process CSV
    if processor.load_csv():
        # Repair all functions
        results = processor.repair_all_functions()
        
        # Create summary
        summary = processor.create_results_summary()
        
        # Save results
        processor.save_results_csv(args.output)
        
        # Save summary
        with open(args.summary, "w") as f:
            f.write(summary)
        
        print(f"\nCompleted! Results saved to '{args.output}' and '{args.summary}'")
    else:
        print("Failed to process CSV file")