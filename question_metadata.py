import os
import json
import threading
from openai import OpenAI
from dotenv import load_dotenv
import sys
# Load environment
load_dotenv()
OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Prepare 5 identical clients (using same API key)


# Input and output directories
INPUT_DIR = r"C:\Users\prash\OneDrive\Desktop\links\updated_chunks\chunks_cache_gpu_0"
OUTPUT_DIR = r"C:\Users\prash\OneDrive\Desktop\links\updated_chunks\chunks_cache_gpu_1"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_metadata_from_llm(content, api_key):
    """Get metadata from OpenAI API for given content using specific API key"""
    prompt = (
          """ 
You are a entity and event extraction engine. Given a question, you must extract all the entities and events from the question and return them in a JSON object.

Example:

Question: "What were the major economic and cultural impacts of the Industrial Revolution in England during the 18th and 19th centuries?"

Output:

{
  "entities": [
    "England",
    "18th century",
    "19th century",
    ...
  ],
  "events": [
    "Industrial Revolution",
    "Economic Impact",
    "Cultural Impact",
    ...
  ]
}

Instructions
1. entities: one key per unique NAMED entity. It cannot be a noun phrase. Nicknames and abbreviations count as separate entities but original must added as well in the list.
2. events: one key per named event.

Output must be pure JSON only, no markdown or comments.
You cannot have 2 keys with the same name in the same object.

Instructions:
1. EVERY SINGLE ENTITY or EVENT NAMED DIRECTTLY OR INDIRECTLY IN THE TEXT MUST BE INCLUDED.
ENSURE YOU DO NOT MISS ANY DELIMITERS OR QUOTES.
SELF VERIFY YOUR OUTPUT IS VALID JSON.
DO NOT ADD ANY OTHER KEYS, JUST THE ONES WHICH I MENTIONED. 
Now, when given any new question, output exactly the same JSON structure, filling in all fields with the information present in that question.
This is the question you must parse and provide metadata for:
""" + content
    )
    try:
        # Create a new OpenAI client for this request
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0
        )
        raw = response.choices[0].message.content
        # Strip markdown
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error from OpenAI for client using key {api_key[:4]}...: {e}")
        return None

def process_files_for_client(client_index, file_indices, all_files):
    """Process files assigned to a specific client"""
    api_key = api_keys[client_index]
    results = []

    for idx in file_indices:
        filename = all_files[idx]
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        # Check if output file already exists
        if os.path.exists(out_path):
            print(f"Client {client_index+1}: Skipping {filename} (already exists in output)")
            results.append(f"Skipped {filename} (already exists)")
            continue

        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'metadata' in data:
                print(f"Client {client_index+1}: Skipping {filename} (metadata exists)")
                results.append(f"Skipped {filename}")
                continue

            if 'content' not in data:
                print(f"Client {client_index+1}: No content in {filename}, skipping")
                results.append(f"Skipped {filename} (no content)")
                continue

            content = data['content']
            print(f"Client {client_index+1}: Processing {filename}...")
            metadata = get_metadata_from_llm(content, api_key)
            if metadata is None:
                print(f"Client {client_index+1}: Failed metadata for {filename}")
                results.append(f"Failed {filename}")
                break

            data['metadata'] = metadata
            # Write to output directory
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Client {client_index+1}: Written {filename} to output")
            results.append(f"Success {filename}")

        except Exception as e:
            print(f"Client {client_index+1}: Error {filename}: {e}")
            results.append(f"Error {filename}")
    return results

def process_json_files():
    """Distribute JSON files across 15 threads and process"""
    # List all JSON files in input dir\
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} files in {INPUT_DIR}")

    # Round-robin distribution
    num_clients = len(api_keys)
    assignments = [[] for _ in range(num_clients)]
    for i, fname in enumerate(files):
        assignments[i % num_clients].append(i)

    for i, idxs in enumerate(assignments):
        print(f"Client {i+1} -> {len(idxs)} files")

    # Launch threads
    threads = []
    for i, idxs in enumerate(assignments):
        if not idxs:
            continue
        t = threading.Thread(target=process_files_for_client, args=(i, idxs, files))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("All processing complete")

if __name__ == "__main__":
    process_json_files()
