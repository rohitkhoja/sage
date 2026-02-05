#!/usr/bin/env python3
"""
OTT-QA Batch Processor - Agent-Based Retrieval

For each question:
1. Send question + prompt to GPT-4o-mini
2. Parse Python code from response
3. Execute the code (which calls HNSW + BM25 + Graph APIs)
4. Save results (JSON with chunk IDs + TXT with logs)
5. Calculate Recall@20 against gold_docs
6. Retry if empty (max 2 retries)
"""

import csv
import json
import re
import subprocess
import sys
import os
import ast
import threading
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Configuration
CSV_FILE = "/shared/khoja/CogComp/output/dense_sparse_average_results (1).csv"
PROMPT_FILE = "/shared/khoja/CogComp/agent/OTT_QA_AGENT_PROMPT.md"
FEW_SHOT_FILE = "/shared/khoja/CogComp/agent/ott_qa_few_shot_examples.md"
OUTPUT_DIR = "/shared/khoja/CogComp/agent/output/ott_qa_batch_1"
OPENAI_API_KEY = ""
MODEL = "gpt-4.1-mini"
MAX_RETRIES = 1
MAX_WORKERS = 30 # Number of parallel workers

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


class OTTQAProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.prompt_template = self._load_prompt_template()
        self.few_shot_examples = self._load_few_shot_examples()
        
    def _load_prompt_template(self) -> str:
        """Load the agent prompt template"""
        with open(PROMPT_FILE, 'r') as f:
            return f.read()
    
    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples"""
        with open(FEW_SHOT_FILE, 'r') as f:
            return f.read()
    
    def _build_prompt(self, question: str, retry_context: Optional[str] = None) -> str:
        """Build the full prompt for GPT"""
        prompt = f"""{self.prompt_template}

## Few-Shot Examples

{self.few_shot_examples}

## Your Task

Generate Python code to retrieve relevant chunks for this question:

**Question**: {question}

"""
        if retry_context:
            prompt += f"""
## Previous Attempt Resulted in Empty/Low Results

The previous code generated {retry_context}. Please reconsider your approach:

**Remember**: You have access to HNSW (semantic), BM25 (keyword), and Graph neighbor expansion.

1. **Analyze the question more carefully** - Did you miss any entities or key terms?
2. **Try different query phrasings** - Use synonyms, break down complex queries
3. **Use more aggressive neighbor expansion** - Multi-hop questions need graph traversal
4. **Broaden the search** - Use union instead of intersection if needed
5. **Check for specific table/document names** in the question

6. **MANDATORY FALLBACKS**:
   - Track `last_non_empty` throughout execution
   - If final result is empty, return `last_non_empty`
   - Never return an empty list without trying fallbacks

Generate new code with a revised approach:
"""
        else:
            prompt += "\nGenerate the Python code to retrieve chunks for this question:\n"
        
        return prompt
    
    def _call_gpt(self, prompt: str) -> str:
        """Call GPT-4o-mini API"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f" GPT API error: {e}")
            return None
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from GPT response"""
        # Look for code blocks
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Also try without language tag
        pattern = r'```\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to find Python code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if 'import requests' in line or 'import json' in line or '#!/usr/bin/env python' in line:
                in_code = True
            if in_code:
                code_lines.append(line)
                if 'final_chunks' in line and '=' in line:
                    break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return None
    
    def _execute_code(self, code: str, question: str) -> Tuple[List[str], str]:
        """Execute generated Python code and capture output"""
        temp_file = f"/tmp/ott_qa_code_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
        
        try:
            # Wrap code to ensure final_chunks is captured
            wrapped_code = code + """

# Extract final_chunks if it exists
import json
if 'final_chunks' in locals() or 'final_chunks' in globals():
    try:
        chunks = list(locals().get('final_chunks', globals().get('final_chunks', [])))
        print(f"\\n=== FINAL_CHUNKS_START ===")
        print(json.dumps(chunks))
        print(f"=== FINAL_CHUNKS_END ===")
    except Exception as e:
        print(f"\\n=== FINAL_CHUNKS_ERROR === {e}")
else:
    print(f"\\n=== FINAL_CHUNKS_START ===\\n[]\\n=== FINAL_CHUNKS_END ===")
"""
            
            with open(temp_file, 'w') as f:
                f.write(wrapped_code)
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=300,
                cwd="/shared/khoja/CogComp/agent"
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Extract final_chunks from stdout
            final_chunks = self._extract_final_chunks(stdout)
            
            log_output = f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n\n=== EXIT CODE ===\n{result.returncode}"
            
            return final_chunks, log_output
        
        except subprocess.TimeoutExpired:
            return [], f" Code execution timed out after 300 seconds"
        except Exception as e:
            return [], f" Execution error: {e}"
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def _extract_final_chunks(self, stdout: str) -> List[str]:
        """Extract final_chunks from execution output"""
        pattern = r'=== FINAL_CHUNKS_START ===\s*\n(.*?)\n=== FINAL_CHUNKS_END ==='
        match = re.search(pattern, stdout, re.DOTALL)
        if match:
            try:
                chunks_json = match.group(1).strip()
                chunks = json.loads(chunks_json)
                if isinstance(chunks, list):
                    return [str(c) for c in chunks]
            except json.JSONDecodeError:
                pass
        
        return []
    
    def calculate_recall_at_k(self, retrieved: List[str], gold: List[str], k: int = 20) -> float:
        """Calculate Recall@k"""
        if not gold:
            return 0.0
        
        retrieved_set = set(retrieved[:k])
        gold_set = set(gold)
        
        found = retrieved_set & gold_set
        recall = len(found) / len(gold_set) * 100.0
        
        return recall
    
    def process_question(self, question: str, question_id: str, gold_docs: List[str]) -> Dict:
        """Process a single question with retry logic"""
        results = {
            'question_id': question_id,
            'question': question,
            'gold_docs': gold_docs,
            'attempts': [],
            'final_chunks': [],
            'recall_at_20': 0.0,
            'success': False
        }
        
        for attempt in range(MAX_RETRIES + 1):
            print(f"\n{'='*80}")
            print(f"Question {question_id} - Attempt {attempt + 1}/{MAX_RETRIES + 1}")
            print(f"Question: {question[:100]}...")
            print(f"{'='*80}")
            
            # Build prompt
            retry_context = None
            if attempt > 0:
                prev_count = len(results['attempts'][-1]['final_chunks']) if results['attempts'] else 0
                prev_recall = results['attempts'][-1].get('recall_at_20', 0) if results['attempts'] else 0
                retry_context = f"{prev_count} chunks (Recall@20: {prev_recall:.1f}%)"
            
            prompt = self._build_prompt(question, retry_context)
            
            # Call GPT
            print("\n Calling GPT-4o-mini...")
            response = self._call_gpt(prompt)
            
            if not response:
                print(" Failed to get response from GPT")
                results['attempts'].append({
                    'attempt': attempt + 1,
                    'error': 'GPT API call failed',
                    'code': None,
                    'final_chunks': [],
                    'recall_at_20': 0.0,
                    'log': ''
                })
                continue
            
            # Extract code
            print(" Extracting code from response...")
            code = self._extract_code(response)
            
            if not code:
                print(" Could not extract code from response")
                results['attempts'].append({
                    'attempt': attempt + 1,
                    'error': 'Code extraction failed',
                    'code': response[:500],
                    'final_chunks': [],
                    'recall_at_20': 0.0,
                    'log': ''
                })
                continue
            
            print(f" Extracted {len(code)} characters of code")
            
            # Execute code
            print(" Executing code...")
            final_chunks, log_output = self._execute_code(code, question)
            
            # Calculate Recall@20
            recall_at_20 = self.calculate_recall_at_k(final_chunks, gold_docs, k=20)
            
            print(f" Execution complete: {len(final_chunks)} chunks retrieved")
            print(f" Recall@20: {recall_at_20:.2f}%")
            
            # Store attempt
            attempt_result = {
                'attempt': attempt + 1,
                'code': code,
                'final_chunks': final_chunks,
                'recall_at_20': recall_at_20,
                'log': log_output,
                'error': None
            }
            results['attempts'].append(attempt_result)
            results['final_chunks'] = final_chunks
            results['recall_at_20'] = recall_at_20
            
            # Success criteria: have some results AND recall > 0
            if final_chunks and recall_at_20 > 0:
                results['success'] = True
                print(f" Success! Found {len(final_chunks)} chunks with Recall@20: {recall_at_20:.2f}%")
                break
            elif final_chunks:
                # Have results but no recall - might retry
                print(f" Got {len(final_chunks)} chunks but Recall@20 is 0%")
                if attempt < MAX_RETRIES:
                    print("Retrying...")
            else:
                print(f" No chunks retrieved, {'retrying...' if attempt < MAX_RETRIES else 'max retries reached'}")
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """Save results to JSON and TXT files"""
        question_id = results['question_id']
        
        # Save JSON file
        json_file = os.path.join(output_dir, f"question_{question_id}.json")
        json_output = {
            'question_id': question_id,
            'question': results['question'],
            'gold_docs': results['gold_docs'],
            'final_chunks': results['final_chunks'],
            'recall_at_20': results['recall_at_20'],
            'success': results['success'],
            'num_attempts': len(results['attempts']),
            'timestamp': datetime.now().isoformat()
        }
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        # Save TXT file with logs
        txt_file = os.path.join(output_dir, f"question_{question_id}.txt")
        with open(txt_file, 'w') as f:
            f.write(f"Question ID: {question_id}\n")
            f.write(f"Question: {results['question']}\n")
            f.write(f"Gold Docs: {results['gold_docs']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Final Recall@20: {results['recall_at_20']:.2f}%\n")
            f.write(f"Final Chunks Count: {len(results['final_chunks'])}\n")
            f.write(f"Number of Attempts: {len(results['attempts'])}\n")
            f.write(f"\n{'='*80}\n\n")
            
            for attempt in results['attempts']:
                f.write(f"ATTEMPT {attempt['attempt']}\n")
                f.write(f"{'='*80}\n")
                if attempt['error']:
                    f.write(f"Error: {attempt['error']}\n\n")
                f.write(f"Recall@20: {attempt['recall_at_20']:.2f}%\n")
                f.write(f"Final Chunks: {attempt['final_chunks'][:10]}...\n")
                f.write(f"Count: {len(attempt['final_chunks'])}\n\n")
                f.write(f"LOG OUTPUT:\n{'-'*80}\n{attempt['log'][:5000]}\n\n")
                f.write(f"GENERATED CODE:\n{'-'*80}\n{attempt.get('code', 'N/A')[:3000]}\n\n")
                f.write(f"{'='*80}\n\n")
        
        print(f" Results saved:")
        print(f" JSON: {json_file}")
        print(f" TXT: {txt_file}")


def calculate_existing_recall(question_id: str, output_dir: Path, gold_docs: List[str]) -> float:
    """Calculate Recall@20 for a question based on existing results"""
    json_file = output_dir / f"question_{question_id}.json"
    if not json_file.exists():
        return -1.0 # Not processed yet
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            return data.get('recall_at_20', 0.0)
    except Exception:
        return -1.0


def process_single_question(question_data: Tuple[str, str, List[str]], processor: OTTQAProcessor, 
                           output_dir: str, progress_lock: threading.Lock, 
                           stats: Dict) -> Dict:
    """Worker function to process a single question"""
    question_id, question, gold_docs = question_data
    
    try:
        results = processor.process_question(question, question_id, gold_docs)
        processor.save_results(results, output_dir)
        
        # Thread-safe progress update
        with progress_lock:
            stats['processed'] += 1
            stats['total_recall'] += results['recall_at_20']
            avg_recall = stats['total_recall'] / stats['processed']
            print(f"\n [{stats['processed']}/{stats['total']}] Q{question_id}: "
                  f"Recall@20 = {results['recall_at_20']:.2f}% | "
                  f"Running Avg = {avg_recall:.2f}%")
        
        return results
        
    except Exception as e:
        print(f" Error processing question {question_id}: {e}")
        error_results = {
            'question_id': question_id,
            'question': question,
            'gold_docs': gold_docs,
            'attempts': [],
            'final_chunks': [],
            'recall_at_20': 0.0,
            'success': False,
            'error': str(e)
        }
        processor.save_results(error_results, output_dir)
        
        with progress_lock:
            stats['processed'] += 1
        
        return error_results


def main():
    """Main execution function"""
    processor = OTTQAProcessor()
    
    # Read CSV file
    print(f" Reading CSV file: {CSV_FILE}")
    questions = []
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_id = row['question_id']
            question = row['question']
            gold_docs_str = row.get('gold_docs', '[]')
            
            try:
                gold_docs = ast.literal_eval(gold_docs_str)
            except:
                gold_docs = []
            
            questions.append((question_id, question, gold_docs))
    
    print(f" Found {len(questions)} questions")
    
    # Check existing results
    print(f"\n Checking existing results...")
    questions_to_process = []
    skipped_count = 0
    
    for question_id, question, gold_docs in questions:
        existing_recall = calculate_existing_recall(question_id, Path(OUTPUT_DIR), gold_docs)
        
        if existing_recall >= 50.0:
            print(f" Skipping Q{question_id}: Recall@20 = {existing_recall:.1f}% (>= 50%)")
            skipped_count += 1
        else:
            if existing_recall >= 0:
                print(f" Will reprocess Q{question_id}: Recall@20 = {existing_recall:.1f}% (< 50%)")
            else:
                print(f" Will process Q{question_id}: Not yet processed")
            questions_to_process.append((question_id, question, gold_docs))
    
    print(f"\n Summary: {skipped_count} skipped, {len(questions_to_process)} to process\n")
    
    # Process questions in parallel
    if not questions_to_process:
        print("No questions to process.")
        return
    
    progress_lock = threading.Lock()
    stats = {
        'processed': 0,
        'total': len(questions_to_process),
        'total_recall': 0.0
    }
    
    print(f" Starting parallel processing with {MAX_WORKERS} workers...\n")
    
    # Process questions in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_question, q_data, processor, OUTPUT_DIR, 
                          progress_lock, stats): q_data 
            for q_data in questions_to_process
        }
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result() # This will raise any exceptions that occurred
            except Exception as e:
                q_data = futures[future]
                print(f" Unexpected error with question {q_data[0]}: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(" Batch processing complete!")
    print(f" Results saved to: {OUTPUT_DIR}")
    if stats['processed'] > 0:
        print(f" Final Average Recall@20: {stats['total_recall'] / stats['processed']:.2f}%")
        print(f" Total Questions Processed: {stats['processed']}/{stats['total']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

