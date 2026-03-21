import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import psycopg2
import uuid

DB_CONFIG = {
    "host" : os.getenv("POSTGRES_HOST"),
    "database" : os.getenv("POSTGRES_DB"),
    "user" : os.getenv("POSTGRES_USER"),
    "password" : os.getenv("POSTGRES_PASSWORD"),
    "port" : int(os.getenv("POSTGRES_PORT"))
}

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_reviews(conn):
    with conn.cursor() as cur:
        query = "SELECT review_id,asin,review_text,summary_text FROM reviews_table WHERE review_text IS NOT NULL OR summary_text IS NOT NULL;"
        cur.execute(query)
        rows = cur.fetchall()
        
    df = pd.DataFrame(rows, columns=['review_id', 'asin', 'review_text', 'summary_text'])
    df['text'] = df['review_text'].fillna('') + ' ' + df['summary_text'].fillna('')
    df.drop(columns=['review_text', 'summary_text'], inplace=True)
    return df
    

def create_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df['text'].fillna('').tolist()
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection_name = "reviews_embeddings"

    existing_names = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )


    points = [
        PointStruct(
            id = uuid.uuid5(uuid.NAMESPACE_OID, row["review_id"]), 
            vector=emb,
            payload={
                "asin": row["asin"],
                "text": row["text"],
            },
        )
        for row, emb in zip(df.to_dict('records'), embeddings)
    ]

    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])

    print("Embeddings upserted to Qdrant")

def main():
    conn = get_connection()
    try:
        df = fetch_reviews(conn)
        print(f"Fetched {len(df)} reviews")
    finally:
        conn.close()

    if len(df) == 0:
        print("No rows to embed. Exiting.")
        return

    create_embeddings(df)

if  __name__ == "__main__":
    main()