from psycopg2.extras import execute_values
import json
from scripts.ingestion_helper import iter_rows,extract_metadata,extract_reviews
import psycopg2
import argparse
import os

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST","postgres"),
    "database": os.getenv("POSTGRES_DB","rag_db"),
    "user": os.getenv("POSTGRES_USER","rag_user"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT",5432))
}

# Products Metadata can change with time we need latest data per ASIN so doing upsert 
# Delete and insert is load heavy
PRODUCT_UPSERT_SQL = '''
    INSERT INTO products_table (
        asin, title, brand, category, price, price_raw, source_run
    )
    VALUES %s
    ON CONFLICT (asin)
    DO UPDATE SET
        title = EXCLUDED.title,
        brand = EXCLUDED.brand,
        category = EXCLUDED.category,
        price = EXCLUDED.price,
        price_raw = EXCLUDED.price_raw,
        source_run = EXCLUDED.source_run,
        updated_at = now();
'''

# Reviews are immutable
REVIEW_UPSERT_SQL = '''
    INSERT INTO reviews_table (
        review_id, asin, review_text, summary_text, source_run
        )
    VALUES %s
    ON CONFLICT (review_id)
    DO NOTHING;
'''

def load_products(cur, product_file: str, run_date: str, batch_size: int = 5000):
    buffer = []
    seen , written , skipped = 0, 0, 0

    for obj in iter_rows(product_file):
        seen += 1
        res = extract_metadata(obj)
        if res is None:
            skipped += 1
            continue

        asin, meta = res
        buffer.append((
            asin,
            meta.get("title", ""),
            meta.get("brand", ""),
            json.dumps(meta.get("category")),                     
            meta.get("price", None),
            str(meta["price_raw"]) if meta.get("price_raw") is not None else None,
            run_date
        ))

        if len(buffer) >= batch_size:
            # Template use due to execute values not supporting jsonb type directly
            execute_values(cur, PRODUCT_UPSERT_SQL, buffer, template="(%s, %s, %s, %s::jsonb, %s, %s, %s)", page_size=batch_size)
            written += len(buffer)
            buffer.clear()

    if buffer:
        execute_values(cur, PRODUCT_UPSERT_SQL, buffer, template="(%s, %s, %s, %s::jsonb, %s, %s, %s)", page_size=len(buffer))
        written += len(buffer)

    return {"seen": seen, "written": written, "skipped": skipped}


def load_reviews(cur, product_file: str, run_date: str, batch_size: int = 5000):
    buffer = []
    seen , written , skipped = 0, 0, 0

    for obj in iter_rows(product_file):
        seen += 1
        res = extract_reviews(obj)
        if res is None:
            skipped += 1
            continue

        buffer.append((
            res["review_id"],
            res["asin"],
            res.get("review_text", ""),
            res.get("summary_text", ""),
            run_date
        ))

        if len(buffer) >= batch_size:
            execute_values(cur, REVIEW_UPSERT_SQL, buffer, page_size=batch_size)
            written += len(buffer)
            buffer.clear()

    if buffer:
        execute_values(cur, REVIEW_UPSERT_SQL, buffer, page_size=len(buffer))
        written += len(buffer)

    return {"seen": seen, "written": written, "skipped": skipped}


def update_rag_ingest_state(cur, shard_idx: int):
    cur.execute('''
        INSERT INTO rag_ingest_state (id, next_shard_idx)
        VALUES (1, %s)
        ON CONFLICT (id) 
        DO UPDATE SET next_shard_idx = EXCLUDED.next_shard_idx, updated_at = now()
    ''', (shard_idx+1,))
        

def run_loader(product_file_path:str, review_file_path:str, run_date:str, start_shard:int, end_shard:int, overwrite_partition:bool = False):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            if overwrite_partition:
                cur.execute("DELETE FROM reviews_table WHERE source_run = %s;", (run_date,))
                cur.execute("DELETE FROM products_table WHERE source_run = %s;", (run_date,))

            prod_stats = load_products(cur, product_file_path, run_date)

            all_reviews_stats = {"seen": 0, "written": 0, "skipped": 0}
            for shard_idx in range(start_shard, end_shard+1):
                shard_file = os.path.join(review_file_path, f"shard_{shard_idx:03d}.jsonl.gz")
                rev_stats = load_reviews(cur, shard_file, run_date)
                update_rag_ingest_state(cur, shard_idx)
                print(f"Shard {shard_idx}: {rev_stats}")

                all_reviews_stats["seen"] += rev_stats["seen"]
                all_reviews_stats["written"] += rev_stats["written"]
                all_reviews_stats["skipped"] += rev_stats["skipped"]

            conn.commit()
        return {"products": prod_stats, "reviews": all_reviews_stats}
    
    except Exception as e:
        conn.rollback()
        print(f"Error during loading: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Loading Amazon data into Postgres")
    
    parser.add_argument(
        "--products",
        default = "Data/raw_data/meta_Electronics.json.gz",
        help="Path to metadata file"
    )
    
    parser.add_argument(
        "--reviews",
        default = "Data/shards",
        help="Path to review file"
    )
    
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date partition (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite partition for run_date"
    )

    parser.add_argument(
        "--starting-shard",
        type=int,
        required=True,
        help="Starting shard number"
    )

    parser.add_argument(
        "--ending-shard",
        type=int,
        required=True,
        help="Ending shard number "
    )

    args = parser.parse_args()

    stats = run_loader(
        product_file_path=args.products,
        review_file_path=args.reviews,
        run_date=args.run_date,
        overwrite_partition=args.overwrite,
        start_shard=args.starting_shard,
        end_shard=args.ending_shard
    )

    print(stats)