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

CREATE INDEX pgroonga_chunk_text_index ON chunk USING pgroonga (chunk_text);
