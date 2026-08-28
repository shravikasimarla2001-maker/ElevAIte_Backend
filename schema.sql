-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table 1: Ingested Learning Content Knowledge Base
CREATE TABLE IF NOT EXISTS content_items (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    url TEXT UNIQUE NOT NULL,
    source_type VARCHAR(50) DEFAULT 'web', -- 'youtube', 'blog', 'release_note'
    snippet TEXT,
    raw_content TEXT,
    embedding vector(768), -- Vertex AI text-embedding-004 produces 768-dim vectors
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: User Personalized Recommendations Store
CREATE TABLE IF NOT EXISTS user_recommendations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content_id INT REFERENCES content_items(id) ON DELETE CASCADE,
    skill_domain VARCHAR(255) NOT NULL,
    relevance_score FLOAT NOT NULL,
    why_watch_takeaways JSONB NOT NULL, -- 2-3 tailored bullet points
    target_role VARCHAR(255),
    is_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity index (HNSW for low latency cosine search)
CREATE INDEX IF NOT EXISTS content_embedding_hnsw_idx 
ON content_items USING hnsw (embedding vector_cosine_ops);