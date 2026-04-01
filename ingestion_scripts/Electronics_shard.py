import gzip
import json
from ingestion_helper import iter_rows, extract_reviews
import os

INPUT_FILE = "Data/raw_data/Electronics.json.gz"
OUTPUT_PREFIX = "Data/shards"
SHARD_SIZE = 100000

def shard_reviews():
    shard_id = 0
    count = 0 
    total = 0

    os.makedirs(OUTPUT_PREFIX, exist_ok=True)
    output = gzip.open(f"{OUTPUT_PREFIX}/shard_{shard_id:03d}.jsonl.gz","wt")

    for row in iter_rows(INPUT_FILE):

        raw = extract_reviews(row)
        if raw is None:
            continue
        
        output.write(json.dumps(raw, ensure_ascii=False) + "\n")
        count += 1
        total += 1

        if count >= SHARD_SIZE:
            output.close()
            shard_id += 1
            count = 0
            output = gzip.open(f"{OUTPUT_PREFIX}/shard_{shard_id:03d}.jsonl.gz","wt")


    output.close()
    print(f"Total reviews extracted: {total}")


if __name__ == "__main__":
    shard_reviews()