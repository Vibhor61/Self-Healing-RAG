CREATE TABLE IF NOT EXISTS products_table (
  asin TEXT PRIMARY KEY,
  title TEXT,
  brand TEXT,
  category JSONB,
  price DOUBLE PRECISION,
  price_raw TEXT,
  source_run TEXT,
  updated_at TIMESTAMPTZ DEFAULT now()

  search_vector tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(brand,'')), 'B') ||
    setweight(to_tsvector('english', coalesce(category::text,'')), 'C') ||
    setweight(to_tsvector('english', coalesce(price_raw,'')), 'D')
  )STORED
);

CREATE INDEX IF NOT EXISTS idx_asin_id ON products_table(asin);


CREATE TABLE IF NOT EXISTS reviews_table (
  review_id TEXT PRIMARY KEY,
  asin TEXT NOT NULL,
  review_text TEXT,
  summary_text TEXT,
  source_run TEXT,
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reviews_id ON reviews_table(review_id);


CREATE TABLE IF NOT EXISTS rag_ingest_state (
  id INT PRIMARY KEY DEFAULT 1,
  next_shard_idx INT NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT now()
);

INSERT INTO rag_ingest_state (id, next_shard_idx)
VALUES (1, 0)
ON CONFLICT (id) DO NOTHING;
