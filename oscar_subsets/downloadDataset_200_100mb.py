from datasets import load_dataset
from transformers import GPT2Tokenizer
import json
import os
import torch

# Define the desired subset sizes in bytes
subset_sizes = {
    "200MB": 200 * 1024 * 1024,
    "100MB": 100 * 1024 * 1024
}

# Load the OSCAR 23.01 dataset in streaming mode
print("Loading OSCAR 23.01 dataset...")
oscar_dataset = load_dataset("oscar-corpus/OSCAR-2301", "en", split='train', streaming=True,trust_remote_code=True)


# Initialize the GPT-2 tokenizer for GPT-2 small
print("Initializing GPT-2 Small tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# Directory to save the subsets
output_dir = "./oscar_subsets"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Iterate over each desired subset size
for label, size in subset_sizes.items():
    print(f"\nProcessing subset: {label} ({size / (1024 * 1024):.2f} MB)...")
    total_size = 0
    count = 0
    raw_output_path = os.path.join(output_dir, f"oscar_subset_{label}_raw.jsonl")
    tokenized_output_path = os.path.join(output_dir, f"oscar_subset_{label}_tokenized.pt")
    
    raw_data = []
    tokenized_data = []
    
    for example in oscar_dataset:
        # print(example.keys())
        # break
        # print("")
        text = example['text']
        meta=example['meta']
        #warc_headers=example['warc_headers']
        
        example_size = len(text.encode('utf-8'))+len(str(meta).encode('utf-8')) #+len(warc_headers.encode('utf-8'))
        
        # Check if adding this example exceeds the desired subset size
        if total_size + example_size > size:
            break
        
        # Save the raw example in memory
        raw_entry = {
            "text": text,
            #"warc_headers": example.get('warc_headers', {}),
            "metadata": example.get('meta', {}),
        }
        raw_data.append(raw_entry)
        
        # Tokenize text
        tokenized_text = tokenizer(text, truncation=True, max_length=1024)["input_ids"]

    # Optionally tokenize metadata or headers if present and textual
        tokenized_metadata = None
        tokenized_headers = None

        if example.get('meta'):
            tokenized_metadata = tokenizer(str(example['meta']), truncation=True, max_length=1024)["input_ids"]
        # if example.get('warc_headers'):
        #     tokenized_headers = tokenizer(str(example['warc_headers']), truncation=True, max_length=1024)["input_ids"]

        # Append to tokenized_data
        tokenized_data.append({
            "tokenized_text": tokenized_text,
            "tokenized_metadata": tokenized_metadata,
            #"tokenized_headers": tokenized_headers,
        })

        
        total_size += example_size
        count += 1

        if count%2000==0:
            print(f"collected {(total_size/(1024*1024)):.2f} MB data ({total_size/size*100:.2f}% completed)")

    # Save raw and tokenized data
    with open(raw_output_path, "w", encoding="utf-8") as raw_file:
        for entry in raw_data:
            json.dump(entry, raw_file)
            raw_file.write('\n')
    torch.save(tokenized_data, tokenized_output_path)  # Save tokenized data in .pt format

    print(f"Subset '{label}' saved: raw data at {raw_output_path}, tokenized data at {tokenized_output_path}, "
          f"total size: {total_size / (1024 * 1024):.2f} MB, total examples: {count}")
