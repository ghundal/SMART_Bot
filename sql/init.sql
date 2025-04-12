CREATE DATABASE smart;
\connect smart;

CREATE EXTENSION IF NOT EXISTS vectors;
CREATE EXTENSION IF NOT EXISTS pgroonga;

CREATE TABLE class(
    class_id TEXT primary key,
    class_name TEXT,
    authors TEXT,
    term TEXT
);

CREATE TABLE access(
    class_id TEXT NOT NULL REFERENCES class,
    "user" TEXT,
    user_email TEXT
);

CREATE TABLE document(
    document_id TEXT primary key,
    class_id TEXT NOT NULL REFERENCES class
);

CREATE TABLE chunk (
    document_id TEXT NOT NULL REFERENCES document,
    page_number INT,
    chunk_text TEXT,
    embedding VECTOR(768)
);

CREATE TABLE audit (
    audit_id SERIAL PRIMARY KEY,
    user_email TEXT,
    query TEXT,
    query_embedding VECTOR(768),
    document_ids TEXT[],          -- Document IDs retrieved
    chunk_texts TEXT[],           -- Chunks passed to the LLM
    response TEXT,
    event_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX pgroonga_chunk_text_index ON chunk USING pgroonga (chunk_text);

-- Table for storing user tokens
CREATE TABLE IF NOT EXISTS user_tokens (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    token TEXT NOT NULL,
    expires_at BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_email)
);

-- Add chat_history table to store chat conversations
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    chat_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    title TEXT DEFAULT 'Untitled Chat',
    messages JSONB NOT NULL,
    dts BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, session_id, model)
);

-- Create indexes for faster queries
CREATE INDEX idx_chat_history_chat_id ON chat_history(chat_id);
CREATE INDEX idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX idx_chat_history_model ON chat_history(model);
CREATE INDEX idx_chat_history_dts ON chat_history(dts DESC);
