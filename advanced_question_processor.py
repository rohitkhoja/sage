import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import Dict, List, Tuple, Set
import sys
import re

# Load environment
load_dotenv()
OPENAI_API_KEY = ""

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Data directories
DEFAULT_DATA_DIR = "/shared/khoja/CogComp/output/graph_enhancement_cache_main/csv_vs_graph_analysis/CSV10_vs_Graph10_vs_CSV20"
TABLE_CHUNKS_PATH = "/shared/khoja/CogComp/output/full_pipeline/table_chunks_with_metadata.json"
DOC_CHUNKS_DIR = "/shared/khoja/CogComp/output/full_pipeline/docs_chunks_1"
ANSWERS_PATH = "/shared/khoja/CogComp/datasets/dev.traced.json"
OUTPUT_DIR = "/shared/khoja/CogComp/output/processed_questions"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QuestionProcessor:
    def __init__(self, max_workers=5):
        self.max_workers = max_workers
        self.table_chunks = {}
        self.doc_chunks_cache = {}
        self.answers_dict = {}
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.token_usage_by_scenario: Dict[str, Dict[str, int]] = {}
        self.token_lock = threading.Lock()

    def _add_token_usage(self, scenario: str, response) -> None:
        try:
            usage = getattr(response, 'usage', None)
            if usage is None:
                return
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
            with self.token_lock:
                agg = self.token_usage_by_scenario.setdefault(scenario, {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_calls': 0
                })
                agg['prompt_tokens'] += prompt_tokens
                agg['completion_tokens'] += completion_tokens
                agg['total_calls'] += 1
        except Exception:
            pass
        
    def load_all_data(self):
        """Load all required data into memory for fast processing"""
        print("Loading table chunks...")
        with open(TABLE_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
            for chunk in table_data:
                self.table_chunks[chunk['chunk_id']] = chunk
        print(f"Loaded {len(self.table_chunks)} table chunks")
        
        print("Loading answers...")
        with open(ANSWERS_PATH, 'r', encoding='utf-8') as f:
            answers_data = json.load(f)
            for item in answers_data:
                self.answers_dict[item['question_id']] = item['answer-text']
        print(f"Loaded {len(self.answers_dict)} answers")
        
        print("Loading doc chunks...")
        self.load_all_doc_chunks()
        print(f"Loaded {len(self.doc_chunks_cache)} doc chunks")
        print("Data loading completed")
    
    def load_all_doc_chunks(self):
        """Load all document chunks and map them by their chunk_id"""
        doc_files = [f for f in os.listdir(DOC_CHUNKS_DIR) if f.endswith('.json')]
        print(f"Found {len(doc_files)} doc chunk files to load...")
        
        loaded_count = 0
        for filename in doc_files:
            try:
                file_path = os.path.join(DOC_CHUNKS_DIR, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    if 'chunk_id' in chunk_data:
                        self.doc_chunks_cache[chunk_data['chunk_id']] = chunk_data
                        loaded_count += 1
                    if loaded_count % 1000 == 0:
                        print(f"  Loaded {loaded_count} doc chunks...")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        print(f"Successfully loaded {loaded_count} doc chunks")
    
    def load_doc_chunk(self, chunk_id: str) -> Dict:
        """Load a single document chunk, using cache"""
        if chunk_id in self.doc_chunks_cache:
            return self.doc_chunks_cache[chunk_id]
        else:
            print(f"Warning: Could not find doc chunk {chunk_id}")
            return None
    
    def get_chunk_content(self, chunk_id: str) -> str:
        """Get content of any chunk (table or doc)"""
        # Try table chunks first
        if chunk_id in self.table_chunks:
            return self.table_chunks[chunk_id]['content']
        
        # Try doc chunks
        doc_chunk = self.load_doc_chunk(chunk_id)
        if doc_chunk and 'content' in doc_chunk:
            return doc_chunk['content']
        
        return ""
    
    def get_chunk_source_info(self, chunk_id: str) -> str:
        """Get source information for any chunk (table or doc)"""
        # Try table chunks first
        if chunk_id in self.table_chunks:
            return self.table_chunks[chunk_id].get('source_name', 'Unknown Table Source')
        
        # Try doc chunks
        doc_chunk = self.load_doc_chunk(chunk_id)
        if doc_chunk:
            # Try metadata.topic first, then source_info.source_name
            if 'metadata' in doc_chunk and 'topic' in doc_chunk['metadata']:
                return doc_chunk['metadata']['topic']
            elif 'source_info' in doc_chunk and 'source_name' in doc_chunk['source_info']:
                return doc_chunk['source_info']['source_name']
        
        return "Unknown Document Source"

    def get_chunk_source_details(self, chunk_id: str) -> Dict[str, str]:
        """Return a dict with source_name and topic (if available)."""
        # Tables
        if chunk_id in self.table_chunks:
            table_obj = self.table_chunks[chunk_id]
            metadata = table_obj.get('metadata', {}) or {}
            return {
                'source_name': table_obj.get('source_name', 'Unknown Table Source'),
                'topic': metadata.get('topic', '')
            }
        # Docs
        doc_chunk = self.load_doc_chunk(chunk_id)
        if doc_chunk:
            source_name = ''
            if 'source_info' in doc_chunk and 'source_name' in doc_chunk['source_info']:
                source_name = doc_chunk['source_info']['source_name']
            topic = ''
            if 'metadata' in doc_chunk and 'topic' in doc_chunk['metadata']:
                topic = doc_chunk['metadata']['topic']
            return {
                'source_name': source_name or 'Unknown Document Source',
                'topic': topic
            }
        return {'source_name': 'Unknown Source', 'topic': ''}

    def table_to_markdown(self, rows_with_headers: List[Dict]) -> str:
        """Convert a list of row dicts into a Markdown table string."""
        if not rows_with_headers or not isinstance(rows_with_headers, list):
            return ""
        headers = list(rows_with_headers[0].keys())
        def esc(cell: str) -> str:
            if cell is None:
                return ""
            if not isinstance(cell, str):
                cell = str(cell)
            return cell.replace('|', '\\|')
        header_line = "| " + " | ".join(esc(h) for h in headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = []
        for row in rows_with_headers:
            data_lines.append("| " + " | ".join(esc(row.get(h, "")) for h in headers) + " |")
        return "\n".join([header_line, sep_line] + data_lines)

    def get_table_prompt_payload(self, chunk_id: str) -> Dict[str, str]:
        """Build a prompt payload for a table including markdown and metadata descriptions."""
        table_obj = self.table_chunks.get(chunk_id, {})
        source_name = table_obj.get('source_name', 'Unknown Table Source')
        rows = table_obj.get('rows_with_headers', [])
        md = self.table_to_markdown(rows)
        metadata = table_obj.get('metadata', {}) or {}
        table_desc = metadata.get('table_description', '')
        col_desc = metadata.get('col_desc', '')
        return {
            'source_name': source_name,
            'markdown': md,
            'table_description': table_desc,
            'column_descriptions': col_desc
        }
    
    def normalize_for_accuracy(self, text: str) -> str:
        """Normalize text by lowercasing and removing non-alphanumeric characters for comparison."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return re.sub(r"[^a-z0-9]", "", text.lower())
    
    def answers_match(self, expected: str, formatted: str) -> bool:
        """Return True if normalized expected is in formatted or vice versa, else False."""
        exp_norm = self.normalize_for_accuracy(expected)
        fmt_norm = self.normalize_for_accuracy(formatted)
        if not exp_norm or not fmt_norm:
            return False
        return exp_norm in fmt_norm or fmt_norm in exp_norm
    
    def filter_relevant_doc_chunks(self, question: str, doc_chunk_ids: List[str], scenario: str) -> List[str]:
        """Use LLM to filter relevant document chunks in batches of 5"""
        if not doc_chunk_ids:
            return []
        
        relevant_chunks = []
        
        # Process in batches of 5
        for i in range(0, len(doc_chunk_ids), 5):
            batch = doc_chunk_ids[i:i+5]
            batch_content = {}
            
            # Get content and source info for this batch
            for chunk_id in batch:
                content = self.get_chunk_content(chunk_id)
                source_info = self.get_chunk_source_info(chunk_id)
                if content:
                    batch_content[chunk_id] = {
                        'content': content,
                        'source_info': source_info
                    }
            
            if not batch_content:
                continue
                
            # Create prompt for filtering
            prompt = self.create_doc_filtering_prompt(question, batch_content)
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0
                )
                self._add_token_usage(scenario, response)
                
                result = response.choices[0].message.content.strip()
                # Parse the result to get relevant chunk IDs
                relevant_batch = self.parse_filtering_result(result, batch)
                relevant_chunks.extend(relevant_batch)
                
            except Exception as e:
                print(f"Error filtering batch: {e}")
                # If filtering fails, include all chunks from batch
                relevant_chunks.extend(batch)
        
        return relevant_chunks
    
    def filter_relevant_table_chunks(self, question: str, table_chunk_ids: List[str], scenario: str) -> List[str]:
        """Use LLM to filter relevant table chunks one at a time"""
        if not table_chunk_ids:
            return []
        
        relevant_chunks = []
        
        # Process tables one at a time
        for chunk_id in table_chunk_ids:
            content = self.get_chunk_content(chunk_id)
            
            if not content:
                continue
                
            # Create prompt for filtering this single table
            table_payload = self.get_table_prompt_payload(chunk_id)
            prompt = self.create_table_filtering_prompt(question, chunk_id, table_payload)
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0
                )
                self._add_token_usage(scenario, response)
                
                result = response.choices[0].message.content.strip().lower()
                # Simple yes/no parsing
                if 'yes' in result or 'relevant' in result:
                    relevant_chunks.append(chunk_id)
                
            except Exception as e:
                print(f"Error filtering table {chunk_id}: {e}")
                # If filtering fails, include the table
                relevant_chunks.append(chunk_id)
        
        return relevant_chunks
    
    def create_doc_filtering_prompt(self, question: str, batch_content: Dict[str, Dict]) -> str:
        """Create a CoT-driven prompt for document chunk filtering with planning."""
        chunks_text = ""
        for chunk_id, chunk_data in batch_content.items():
            content = chunk_data['content']
            source_info = chunk_data['source_info']
            chunks_text += f"\nCHUNK_ID: {chunk_id}\nSOURCE/TOPIC: {source_info}\nCONTENT: {content[:800]}...\n---"

        prompt = f"""
You are a careful evidence selector. First create a brief, step-by-step plan for how you would answer the question given the available document chunks. Then, using that plan, choose which chunks are potentially relevant to answering the question.

QUESTION: {question}

DOCUMENT CHUNKS:
{chunks_text}

Reasoning steps (do not skip):
1) Brief plan to answer the question (what key facts/entities you need and why)
2) For each chunk, check if it supports any step in the plan
3) Select liberally if a chunk may contain supporting evidence

Output strictly as JSON list of strings with CHUNK_IDs. No explanation.
Example: ["chunk_id_1", "chunk_id_3"]
"""
        return prompt
    
    def create_table_filtering_prompt(self, question: str, chunk_id: str, table_payload: Dict[str, str]) -> str:
        """Create a CoT-driven prompt for single table filtering including markdown and metadata."""
        source_name = table_payload.get('source_name', '')
        markdown = table_payload.get('markdown', '')
        table_desc = table_payload.get('table_description', '')
        col_desc = table_payload.get('column_descriptions', '')

        prompt = f"""
You are a careful table evidence selector. First create a brief, step-by-step plan for how you would answer the question given the available table, then decide if this table is relevant.

QUESTION: {question}

TABLE METADATA:
- Source Name: {source_name}
- Table Description: {table_desc}
- Column Descriptions: {col_desc}

TABLE (Markdown):
{markdown}

Reasoning steps (do not skip):
1) Brief plan to answer the question (what key signals this table might provide)
2) Check if the table plausibly supports the plan

Answer with a single token: YES or NO.
"""
        return prompt
    
    def parse_filtering_result(self, result: str, original_batch: List[str]) -> List[str]:
        """Parse LLM filtering result"""
        try:
            # Clean the result
            cleaned = result.replace("```json", "").replace("```", "").strip()
            chunk_ids = json.loads(cleaned)
            
            # Validate that returned IDs are from the original batch
            valid_ids = [cid for cid in chunk_ids if cid in original_batch]
            return valid_ids
        except:
            # If parsing fails, return all chunks from batch
            return original_batch
    
    def generate_answer(self, question: str, relevant_doc_chunks: List[str], table_chunks: List[str], scenario: str) -> str:
        """Generate answer using filtered doc chunks and table chunks with a plan + CoT."""
        # Collect all content with source information
        doc_context_blocks = []
        for chunk_id in relevant_doc_chunks:
            content = self.get_chunk_content(chunk_id)
            source_details = self.get_chunk_source_details(chunk_id)
            if content:
                doc_context_blocks.append(
                    f"DOCUMENT SOURCE: {source_details.get('source_name','')}\nTOPIC: {source_details.get('topic','')}\nCONTENT: {content}"
                )

        table_context_blocks = []
        for chunk_id in table_chunks:
            payload = self.get_table_prompt_payload(chunk_id)
            table_context_blocks.append(
                f"TABLE SOURCE: {payload.get('source_name','')}\nTABLE DESCRIPTION: {payload.get('table_description','')}\nCOLUMN DESCRIPTIONS: {payload.get('column_descriptions','')}\nTABLE (Markdown):\n{payload.get('markdown','')}"
            )

        if not doc_context_blocks and not table_context_blocks:
            return "No relevant information found to answer the question."

        context_text = "\n\n".join(doc_context_blocks + table_context_blocks)

        prompt = f"""
You are a careful question answering system. First, plan how you will answer the question using the provided context. Then, follow your plan step-by-step (chain-of-thought) to derive the final answer.

QUESTION: {question}

CONTEXT:
{context_text}

Reasoning steps (do not skip):
1) Plan: identify the exact signals needed to answer the question
2) Evidence: cite which document/table snippets support each step
3) Synthesis: combine the evidence to arrive at the answer
4) Final check: ensure the answer is strictly supported by the given context

Output only the final answer on the last line, without extra commentary.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
            self._add_token_usage(scenario, response)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating answer."
    
    def format_answer(self, generated_answer: str, expected_answer: str, scenario: str) -> str:
        """Convert generated answer format to match expected format"""
        prompt = f"""
Given a generated answer and the expected answer format, reformat the generated answer to match the expected format style while keeping the factual content accurate.

GENERATED ANSWER: {generated_answer}
EXPECTED FORMAT EXAMPLE: {expected_answer}

Few examples of convertion for your understanding:
1. answer: ITA, gold answer: Italy. Reasoning- ITA is country code of Italy hence ITA and Italy are same and you can convert ITA to Italy.
    Your Output: Italy
2. answer: 17, gold answer: 17 years. Reasoning- 17 of answer is same as 17 years of the gold answer in the given context of question.
    Your Output: 17 years
3. answer : 10, gold answer: 10. Reasoning- Since, both values are already same no convertion is needed.
    Your Output: 10
4. answer : 0, gold answer: 5. Reasoning- Since, both values are semantically not same no convertion is needed for the answer.
    Your Output: 0
5. answer : The answer is not present in the table. , gold answer: 5. Reasoning- Since, both values are semantically not same no convertion is needed for the answer.
    Your Output: The answer is not present in the table.

REFORMATTED ANSWER:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            self._add_token_usage(scenario, response)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error formatting answer: {e}")
            return generated_answer
    
    def process_single_question(self, question_data: Dict, scenario: str) -> Dict:
        """Process a single question through the complete pipeline"""
        question_id = question_data['question_id']
        question_text = question_data['question_text']
        
        print(f"Processing question {question_id}: {question_text[:100]}...")
        
        # Get all retrieved chunks
        csv_chunks = question_data.get('analysis', {}).get('csv_k_retrieved', [])
        graph_chunks = question_data.get('analysis', {}).get('extended_csv_retrieved', [])
        all_chunks = list(set(csv_chunks + graph_chunks))
        
        # Separate table and doc chunks
        table_chunk_ids = []
        doc_chunk_ids = []
        
        for chunk_id in all_chunks:
            if chunk_id in self.table_chunks:
                table_chunk_ids.append(chunk_id)
            else:
                doc_chunk_ids.append(chunk_id)
        
        print(f"  Found {len(table_chunk_ids)} table chunks, {len(doc_chunk_ids)} doc chunks")
        
        # Filter relevant doc chunks using LLM
        relevant_doc_chunks = self.filter_relevant_doc_chunks(question_text, doc_chunk_ids, scenario)
        print(f"  Filtered to {len(relevant_doc_chunks)} relevant doc chunks")
        
        # Filter relevant table chunks using LLM
        relevant_table_chunks = self.filter_relevant_table_chunks(question_text, table_chunk_ids, scenario)
        print(f"  Filtered to {len(relevant_table_chunks)} relevant table chunks")
        
        # Generate answer
        generated_answer = self.generate_answer(question_text, relevant_doc_chunks, relevant_table_chunks, scenario)
        
        # Get expected answer for format reference
        expected_answer = self.answers_dict.get(question_id, "")
        
        # Format the answer
        formatted_answer = self.format_answer(generated_answer, expected_answer, scenario)
        
        # Accuracy fields
        expected_normalized = self.normalize_for_accuracy(expected_answer)
        formatted_normalized = self.normalize_for_accuracy(formatted_answer)
        is_correct = self.answers_match(expected_answer, formatted_answer)

        return {
            'question_id': question_id,
            'question_text': question_text,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer,
            'formatted_answer': formatted_answer,
            'expected_normalized': expected_normalized,
            'formatted_normalized': formatted_normalized,
            'is_correct': is_correct,
            'chunks_used': {
                'table_chunks_original': table_chunk_ids,
                'table_chunks_filtered': relevant_table_chunks,
                'doc_chunks_original': doc_chunk_ids,
                'doc_chunks_filtered': relevant_doc_chunks
            }
        }
    
    def process_questions(self, questions: List[Dict], max_questions: int = None, scenario: str = "default") -> List[Dict]:
        """Process multiple questions in parallel and return list of per-question results."""
        if max_questions:
            questions = questions[:max_questions]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_question = {
                executor.submit(self.process_single_question, q, scenario): q 
                for q in questions
            }
            
            for future in as_completed(future_to_question):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed {len(results)}/{len(questions)} questions")
                except Exception as e:
                    print(f"Error processing question: {e}")
        
        return results
    
    def load_questions_from_files(self, data_dir: str) -> List[Dict]:
        """Load all questions from the result files in the given scenario directory"""
        all_questions = []
        
        result_files = ['csv_wins.json', 'graph_wins.json', 'both_win.json', 'both_lose.json']
        
        for filename in result_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    questions = data.get('questions', [])
                    print(f"Loaded {len(questions)} questions from {filename}")
                    all_questions.extend(questions)
        
        print(f"Total questions loaded: {len(all_questions)}")
        return all_questions

def main():
    """Main function to run the question processing pipeline across multiple scenarios"""
    processor = QuestionProcessor(max_workers=200)

    print("Loading all data...")
    processor.load_all_data()

    # Scenario directories to process
    scenario_dirs = {
        'CSV1_vs_Graph1_vs_CSV2': "/shared/khoja/CogComp/output/graph_enhancement_cache_main/csv_vs_graph_analysis/CSV1_vs_Graph1_vs_CSV2",
        'CSV2_vs_Graph3_vs_CSV5': "/shared/khoja/CogComp/output/graph_enhancement_cache_main/csv_vs_graph_analysis/CSV2_vs_Graph3_vs_CSV5",
        'CSV5_vs_Graph5_vs_CSV10': "/shared/khoja/CogComp/output/graph_enhancement_cache_main/csv_vs_graph_analysis/CSV5_vs_Graph5_vs_CSV10",
        'CSV10_vs_Graph10_vs_CSV20': "/shared/khoja/CogComp/output/graph_enhancement_cache_main/csv_vs_graph_analysis/CSV10_vs_Graph10_vs_CSV20",
        'CSV20_vs_Graph30_vs_CSV50': "/shared/khoja/CogComp/output/graph_enhancement_cache_main/csv_vs_graph_analysis/CSV20_vs_Graph30_vs_CSV50",
    }

    all_summaries = {}

    for scenario, data_dir in scenario_dirs.items():
        print(f"\n=== Processing scenario: {scenario} ===")
        print("Loading questions...")
        questions = processor.load_questions_from_files(data_dir)
        if not questions:
            print(f"No questions found for scenario {scenario}, skipping.")
            continue

        # Process all questions in this scenario
        results = processor.process_questions(questions, scenario=scenario)

        # Save results per scenario
        out_path = os.path.join(OUTPUT_DIR, f"results_{scenario}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results for {scenario} to {out_path}")

        # Accuracy per scenario
        total = len(results)
        correct = sum(1 for r in results if r.get('is_correct'))
        acc = (correct / total * 100.0) if total else 0.0
        token_usage = processor.token_usage_by_scenario.get(scenario, {})
        all_summaries[scenario] = {
            'total': total,
            'correct': correct,
            'accuracy_pct': acc,
            'token_usage': token_usage
        }
        print(f"Accuracy for {scenario}: {correct}/{total} = {acc:.2f}%")
        if token_usage:
            print(f"Token usage for {scenario}: prompts={token_usage.get('prompt_tokens',0)}, completions={token_usage.get('completion_tokens',0)}, calls={token_usage.get('total_calls',0)}")

    # Save aggregated summary
    summary_path = os.path.join(OUTPUT_DIR, "summary_all_scenarios.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary across scenarios to {summary_path}")

if __name__ == "__main__":
    main()
