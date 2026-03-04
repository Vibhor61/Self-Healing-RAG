import json
import gzip
from typing import  Optional, Tuple
import hashlib
import re


def extract_shard(path:str) :
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def iter_rows(path : str) :
    with extract_shard(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue 


def stable_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x1f")  # separator
    return h.hexdigest()


WS = re.compile(r"\s+")
def norm_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = WS.sub(" ", s)
    return s

PRICE_CLEAN = re.compile(r"[^\d.]")
def norm_price(x) -> Optional[float]:
    if x is None:
        return None

    if isinstance(x, list):
        if not x:
            return None
        x = x[0]

    s = norm_text(x)

    if not s:
        return None
    
    if "-" in s:
        s = s.split("-")[0]

    s = PRICE_CLEAN.sub("", s)
    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        return None


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_metadata(obj: dict) -> Optional[Tuple[str, dict]]:
    asin = norm_text(obj.get("asin"))
    if not asin:
        return None
    
    cat = obj.get("category")
    if cat is None:
        cat = obj.get("categories") or obj.get("main_cat")

    

    keep = {
        "asin": asin,
        "title": norm_text(obj.get("title")),
        "brand": norm_text(obj.get("brand")),
        "category": cat,
        "price": norm_price(obj.get("price")),
        "price_raw" : obj.get("price")
    }

    return asin, keep


def extract_reviews(obj : dict):

    asin = norm_text(obj.get("asin"))
    if not asin:
        return None

    review_text = norm_text(obj.get("reviewText"))
    summary_text = norm_text(obj.get("summary"))
    ts = norm_text(obj.get("unixReviewTime"))

    if not review_text and not summary_text:
        return None
    
    review_id = stable_hash(asin,review_text,summary_text,ts)

    return {
        "review_id":review_id,
        "asin" : asin,
        "review_text" : review_text,
        "summary_text" : summary_text
    }
    