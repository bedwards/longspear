-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table for nomic-embed-text (768 dimensions)
CREATE TABLE IF NOT EXISTS documents_nomic (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    persona VARCHAR(100) NOT NULL,
    source_file VARCHAR(500),
    video_title VARCHAR(500),
    video_date DATE,
    video_url VARCHAR(500),
    chunk_index INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding vector(768)
);

-- Documents table for mxbai-embed-large (1024 dimensions)
CREATE TABLE IF NOT EXISTS documents_mxbai (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    persona VARCHAR(100) NOT NULL,
    source_file VARCHAR(500),
    video_title VARCHAR(500),
    video_date DATE,
    video_url VARCHAR(500),
    chunk_index INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding vector(1024)
);

-- HNSW indexes for cosine similarity search
CREATE INDEX IF NOT EXISTS idx_documents_nomic_embedding
    ON documents_nomic USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_documents_mxbai_embedding
    ON documents_mxbai USING hnsw (embedding vector_cosine_ops);

-- Indexes for filtering by persona
CREATE INDEX IF NOT EXISTS idx_documents_nomic_persona
    ON documents_nomic (persona);

CREATE INDEX IF NOT EXISTS idx_documents_mxbai_persona
    ON documents_mxbai (persona);
