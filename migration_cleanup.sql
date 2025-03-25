-- SQL migration script to clean up and reduce database size

-- Create backup of Book table with content before removing it
CREATE TABLE book_backup AS SELECT id, title, author, publication_year, file_path, gutenberg_id, centrality_score, total_clicks FROM searchengine_book;

-- Update any NULL file_path values to avoid issues
UPDATE searchengine_book SET file_path = 'books_data/unknown.txt' WHERE file_path IS NULL;

-- Run VACUUM to reclaim storage space
VACUUM;

-- Instructions:
-- 1. Run this file with: sqlite3 db.sqlite3 < migration_cleanup.sql
-- 2. This will create a backup of the book table
-- 3. Then it will ensure all file_path values are valid
-- 4. Finally it will reclaim storage space