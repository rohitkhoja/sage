#!/usr/bin/env python3
"""
Batch process questions from CSV file using GPT-4o-mini to generate code.

The MAG Agent system uses:
- HNSW (Hierarchical Navigable Small World) for similarity search on embeddings
- Neo4j graph database for relationship traversal and graph queries

For each question:
1. Send question + prompt to GPT-4o-mini
2. Parse Python code from response
3. Execute the code (which calls HNSW search + Neo4j traversal APIs)
4. Save results (JSON with node IDs + TXT with logs)
5. Retry if empty (max 2 retries)
"""

import csv
import json
import re
import subprocess
import sys
import os
import ast
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# Configuration
CSV_FILE = "/shared/khoja/CogComp/output/BM25_stark_mag_human_rewritten.csv"
PROMPT_FILE = "/shared/khoja/CogComp/agent/AGENT_PROMPT.md"
FEW_SHOT_FILE = "/shared/khoja/CogComp/agent/few_shot_examples.md"
OUTPUT_DIR = "/shared/khoja/CogComp/agent/output/batch_questions"
OPENAI_API_KEY = ""
MODEL = "gpt-4.1-mini"
MAX_RETRIES = 2

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QuestionProcessor:
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

## 📚 Few-Shot Examples

{self.few_shot_examples}

## 🎯 Your Task

Generate Python code to answer this question:

**Question**: {question}

"""
        if retry_context:
            prompt += f"""
## ⚠️ Previous Attempt Resulted in Empty Results

The previous code generated {retry_context} results. Please reconsider your approach:

**Remember**: You have access to both HNSW similarity search AND Neo4j graph traversal.

1. **Analyze the question more carefully** - Did you miss any entities, relationships, or constraints?
2. **Try a different approach** - Perhaps search in a different order, use different search methods, or broaden the search
3. **Check your intersections** - Are you being too restrictive? Consider using unions instead of intersections in some steps
4. **Verify API calls** - Make sure all API calls are correctly formatted and handle None/empty responses
5. **Use Neo4j metadata** - The graph database contains rich metadata for filtering (year, venue, citation counts, etc.)

6. **MANDATORY FINAL-ATTEMPT FALLBACKS**
   - Track `last_non_zero` while computing intersections; if the final result is empty, return `last_non_zero`.
   - If looking for common papers between multiple authors and none are common, return the union of author result sets instead of an empty list.
   - Print which fallback you used.

Generate new code with a revised approach:
"""
        else:
            prompt += "\nGenerate the Python code to answer this question:\n"
        
        return prompt
    
    def _call_gpt(self, prompt: str) -> str:
        """Call GPT-4o-mini API"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                #max_tokens=4000,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ GPT API error: {e}")
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
        
        # If no code blocks, try to find Python code between markers
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if 'import requests' in line or 'import json' in line or '#!/usr/bin/env python' in line:
                in_code = True
            if in_code:
                code_lines.append(line)
                if 'final_nodes' in line and '=' in line:
                    break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return None
    
    def _execute_code(self, code: str, question: str) -> Tuple[List[int], str]:
        """Execute generated Python code and capture output"""
        # Create a temporary Python file
        temp_file = f"/tmp/generated_code_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
        
        try:
            # Wrap code to ensure final_nodes is captured
            wrapped_code = code + """
# Extract final_nodes if it exists
import json
if 'final_nodes' in locals() or 'final_nodes' in globals():
    try:
        nodes = list(locals().get('final_nodes', globals().get('final_nodes', [])))
        print(f"\\n=== FINAL_NODES_START ===")
        print(json.dumps(nodes))
        print(f"=== FINAL_NODES_END ===")
    except Exception as e:
        print(f"\\n=== FINAL_NODES_ERROR === {e}")
else:
    print(f"\\n=== FINAL_NODES_START ===\\n[]\\n=== FINAL_NODES_END ===")
"""
            
            with open(temp_file, 'w') as f:
                f.write(wrapped_code)
            
            # Execute the code and capture stdout/stderr
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=600,
                cwd="/shared/khoja/CogComp/agent"
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Extract final_nodes from stdout
            final_nodes = self._extract_final_nodes(stdout)
            
            log_output = f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n\n=== EXIT CODE ===\n{result.returncode}"
            
            return final_nodes, log_output
        
        except subprocess.TimeoutExpired:
            return [], f"❌ Code execution timed out after 600 seconds"
        except Exception as e:
            return [], f"❌ Execution error: {e}"
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def _extract_final_nodes(self, stdout: str) -> List[int]:
        """Extract final_nodes from execution output"""
        # Look for the marker we added
        pattern = r'=== FINAL_NODES_START ===\s*\n(.*?)\n=== FINAL_NODES_END ==='
        match = re.search(pattern, stdout, re.DOTALL)
        if match:
            try:
                nodes_json = match.group(1).strip()
                nodes = json.loads(nodes_json)
                if isinstance(nodes, list):
                    # Convert all to integers
                    final = []
                    for n in nodes:
                        if isinstance(n, int):
                            final.append(n)
                        elif isinstance(n, str) and n.isdigit():
                            final.append(int(n))
                    return final
            except json.JSONDecodeError:
                pass
        
        # Fallback: Try to parse from stdout patterns
        patterns = [
            r'Node IDs:\s*\[([^\]]+)\]',
            r'final_nodes\s*=\s*\[([^\]]+)\]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stdout)
            if matches:
                nodes = []
                for match in matches:
                    # Extract all integers from the match
                    nums = re.findall(r'\d+', match)
                    nodes.extend([int(n) for n in nums])
                if nodes:
                    return nodes
        
        return []
    
    def process_question(self, question: str, query_id: int) -> Dict:
        """Process a single question with retry logic"""
        results = {
            'query_id': query_id,
            'question': question,
            'attempts': [],
            'final_nodes': [],
            'success': False
        }
        
        for attempt in range(MAX_RETRIES + 1):
            print(f"\n{'='*80}")
            print(f"Question {query_id} - Attempt {attempt + 1}/{MAX_RETRIES + 1}")
            print(f"Question: {question}")
            print(f"{'='*80}")
            
            # Build prompt (include retry context if not first attempt)
            retry_context = None
            if attempt > 0:
                prev_result_count = len(results['attempts'][-1]['final_nodes']) if results['attempts'] else 0
                retry_context = f"{prev_result_count} (empty)"
            
            prompt = self._build_prompt(question, retry_context)
            
            # Call GPT
            print("\n🤖 Calling GPT-4o-mini...")
            response = self._call_gpt(prompt)
            
            if not response:
                print("❌ Failed to get response from GPT")
                results['attempts'].append({
                    'attempt': attempt + 1,
                    'error': 'GPT API call failed',
                    'code': None,
                    'final_nodes': [],
                    'log': ''
                })
                continue
            
            # Extract code
            print("📝 Extracting code from response...")
            code = self._extract_code(response)
            
            if not code:
                print("❌ Could not extract code from response")
                results['attempts'].append({
                    'attempt': attempt + 1,
                    'error': 'Code extraction failed',
                    'code': response[:500],  # Store first 500 chars of response
                    'final_nodes': [],
                    'log': ''
                })
                continue
            
            print(f"✅ Extracted {len(code)} characters of code")
            
            # Execute code
            print("🚀 Executing code...")
            final_nodes, log_output = self._execute_code(code, question)
            
            print(f"✅ Execution complete: {len(final_nodes)} nodes found")
            
            # Store attempt
            attempt_result = {
                'attempt': attempt + 1,
                'code': code,
                'final_nodes': final_nodes,
                'log': log_output,
                'error': None
            }
            results['attempts'].append(attempt_result)
            results['final_nodes'] = final_nodes
            
            # If we got results, we're done
            if final_nodes:
                results['success'] = True
                print(f"✅ Success! Found {len(final_nodes)} nodes")
                break
            else:
                print(f"⚠️ No results found, {'retrying...' if attempt < MAX_RETRIES else 'max retries reached'}")
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """Save results to JSON and TXT files"""
        query_id = results['query_id']
        
        # Save JSON file with node IDs
        json_file = os.path.join(output_dir, f"question_{query_id}.json")
        json_output = {
            'query_id': query_id,
            'question': results['question'],
            'final_nodes': results['final_nodes'],
            'success': results['success'],
            'num_attempts': len(results['attempts']),
            'timestamp': datetime.now().isoformat()
        }
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        # Save TXT file with logs
        txt_file = os.path.join(output_dir, f"question_{query_id}.txt")
        with open(txt_file, 'w') as f:
            f.write(f"Question ID: {query_id}\n")
            f.write(f"Question: {results['question']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Final Nodes Count: {len(results['final_nodes'])}\n")
            f.write(f"Number of Attempts: {len(results['attempts'])}\n")
            f.write(f"\n{'='*80}\n\n")
            
            for attempt in results['attempts']:
                f.write(f"ATTEMPT {attempt['attempt']}\n")
                f.write(f"{'='*80}\n")
                if attempt['error']:
                    f.write(f"Error: {attempt['error']}\n\n")
                f.write(f"Final Nodes: {attempt['final_nodes']}\n")
                f.write(f"Count: {len(attempt['final_nodes'])}\n\n")
                f.write(f"LOG OUTPUT:\n{'-'*80}\n{attempt['log']}\n\n")
                f.write(f"GENERATED CODE:\n{'-'*80}\n{attempt.get('code', 'N/A')[:2000]}\n\n")
                f.write(f"{'='*80}\n\n")
        
        print(f"📁 Results saved:")
        print(f"   JSON: {json_file}")
        print(f"   TXT:  {txt_file}")


def calculate_recall_at_20(query_id: int, output_dir: Path, gold_docs: List[int]) -> float:
    """Calculate Recall@20 for a question based on existing results"""
    if not gold_docs:
        return 0.0
    
    json_file = output_dir / f"question_{query_id}.json"
    if not json_file.exists():
        return 0.0
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            agent_nodes = data.get('final_nodes', [])
            
            # Extract paper IDs from nodes
            agent_paper_ids = set()
            for node in agent_nodes:
                if isinstance(node, dict):
                    paper_id = node.get('node_index') or node.get('paperId') or node.get('id')
                else:
                    paper_id = node
                if paper_id:
                    try:
                        agent_paper_ids.add(int(paper_id))
                    except (ValueError, TypeError):
                        pass
            
            gold_set = set(gold_docs)
            agent_list = list(agent_paper_ids)
            found_in_top20 = gold_set & set(agent_list[:20])
            
            recall_at_20 = len(found_in_top20) / len(gold_set) * 100.0
            return recall_at_20
    except Exception:
        return 0.0


def main():
    """Main execution function"""
    processor = QuestionProcessor()
    
    # Read CSV file with gold documents
    print(f"📖 Reading CSV file: {CSV_FILE}")
    questions = []
    gold_docs_map = {}
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = int(row['query_id'])
            question = row['query']
            gold_docs = ast.literal_eval(row.get('gold_docs', '[]'))
            questions.append((query_id, question))
            gold_docs_map[query_id] = gold_docs
    
    print(f"✅ Found {len(questions)} questions")
    
    # Calculate recall for existing results and filter
    print(f"\n📊 Checking existing results for recall scores...")
    questions_to_process = []
    skipped_count = 0
    
    for query_id, question in questions:
        gold_docs = gold_docs_map.get(query_id, [])
        recall_at_20 = calculate_recall_at_20(query_id, Path(OUTPUT_DIR), gold_docs)
        
        if recall_at_20 >= 50.0:
            print(f"⏭️  Skipping Q{query_id}: Recall@20 = {recall_at_20:.1f}% (>= 50%)")
            skipped_count += 1
        else:
            questions_to_process.append((query_id, question))
            print(f"✅ Will process Q{query_id}: Recall@20 = {recall_at_20:.1f}% (< 50%)")
    
    print(f"\n📊 Summary: {skipped_count} questions skipped, {len(questions_to_process)} questions to process\n")
    
    # Process each question
    for query_id, question in questions_to_process:
        try:
            results = processor.process_question(question, query_id)
            processor.save_results(results, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Error processing question {query_id}: {e}")
            # Save error result
            error_results = {
                'query_id': query_id,
                'question': question,
                'attempts': [],
                'final_nodes': [],
                'success': False,
                'error': str(e)
            }
            processor.save_results(error_results, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("✅ Batch processing complete!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

