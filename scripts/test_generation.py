from pathlib import Path
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,  
    AbstractQuerySynthesizer,           
    MultiHopAbstractQuerySynthesizer,    
)
import pandas as pd
from langchain_core.documents import Document
from ingestion_scripts.ingestion_helper import iter_rows, extract_metadata, extract_reviews
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain import LangchainLLMWrapper, LangchainEmbeddingsWrapper


REVIEW_FILE = "Data/raw_data/Electronics.json.gz"
METADATA_FILE = "Data/raw_data/meta_Electronics.json.gz"
OUTPUT_DIR = "Data/testsets"


def build_metadata_dict(metadata_file, limit=2000):
    docs, seen = [], set()
    for obj in iter_rows(metadata_file):
        if len(docs) >= limit:
            break

        res = extract_metadata(obj)
        if res is None:
            continue

        asin, meta = res
        if asin in seen:
            continue
        seen.add(asin)

        docs.append(Document(
            page_content=(
                f"Product: {meta['title']}\n"
                f"Brand: {meta['brand']}\n"
                f"Category: {meta.get('category', '')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Description: {meta['description']}"
            ),
            metadata={"asin": asin, "doc_type": "metadata"}
        ))

    return docs


def build_review_dict(review_file, limit = 5000):
    docs, seen = [], set()

    for obj in iter_rows(review_file):
        if(len(docs) >=limit):
            break
        res = extract_reviews(obj)
        if res is None:
            continue
        rid = res["review_id"]

        if rid in seen:
            continue
        seen.add(rid)

        docs.append(Document(
            page_content=(
                f"Review: {res.get('review_text', '')}\n"
                f"Summary: {res.get('summary_text', '')}"
            ),
            metadata={"review_id": res["review_id"], "asin": res["asin"], "doc_type": "review"}
        ))

    return docs

all_docs = build_metadata_dict(METADATA_FILE) + build_review_dict(REVIEW_FILE)

llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:70b", temperature=0))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

generator = TestsetGenerator(
    llm=llm,
    embedding_model=embeddings,
)

synthesizers = [
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.40),
    (AbstractQuerySynthesizer(llm=llm), 0.30),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.30),
]

dataset = generator.generate_with_langchain_docs(
    documents=all_docs,
    testset_size=100,
    query_distribution=synthesizers
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
dataset.to_pandas().to_csv(f"{OUTPUT_DIR}/testset.csv", index=False)