#!/bin/bash

# Stop any running Django servers
pkill -f "python manage.py runserver" || true

# Backup the current database
# echo "Backing up current database..."
# cp db.sqlite3 db.sqlite3.bak || true

# # Remove the database and recreate it
# echo "Removing old database..."
# rm -f db.sqlite3
# rm -f db.sqlite3-journal db.sqlite3-wal db.sqlite3-shm

# # Create database schema directly to avoid migration issues
# echo "Creating database schema directly..."
# cat > dump.sql << EOF
# -- First create the tables we need for Django
# CREATE TABLE django_migrations (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     app VARCHAR(255) NOT NULL,
#     name VARCHAR(255) NOT NULL,
#     applied DATETIME NOT NULL
# );

# -- Mark all migrations as applied
# INSERT INTO django_migrations (app, name, applied) VALUES 
# ('contenttypes', '0001_initial', datetime('now')),
# ('contenttypes', '0002_remove_content_type_name', datetime('now')),
# ('auth', '0001_initial', datetime('now')),
# ('auth', '0002_alter_permission_name_max_length', datetime('now')),
# ('auth', '0003_alter_user_email_max_length', datetime('now')),
# ('auth', '0004_alter_user_username_opts', datetime('now')),
# ('auth', '0005_alter_user_last_login_null', datetime('now')),
# ('auth', '0006_require_contenttypes_0002', datetime('now')),
# ('auth', '0007_alter_validators_add_error_messages', datetime('now')),
# ('auth', '0008_alter_user_username_max_length', datetime('now')),
# ('auth', '0009_alter_user_last_name_max_length', datetime('now')),
# ('auth', '0010_alter_group_name_max_length', datetime('now')),
# ('auth', '0011_update_proxy_permissions', datetime('now')),
# ('auth', '0012_alter_user_first_name_max_length', datetime('now')),
# ('admin', '0001_initial', datetime('now')),
# ('admin', '0002_logentry_remove_auto_add', datetime('now')),
# ('admin', '0003_logentry_add_action_flag_choices', datetime('now')),
# ('sessions', '0001_initial', datetime('now')),
# ('searchengine', '0001_initial', datetime('now')),
# ('searchengine', '0002_remove_book_text_content_alter_book_file_path', datetime('now')),
# ('searchengine', '0003_booksimilarity_book_similar_books_and_more', datetime('now')),
# ('searchengine', '0004_word_wordoccurrence', datetime('now')),
# ('searchengine', '0005_remove_bookindex', datetime('now')),
# ('searchengine', '0006_remove_bookindex', datetime('now')),
# ('searchengine', '0007_remove_bookindex', datetime('now'));

# -- Create basic tables needed for Django to work
# CREATE TABLE django_content_type (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     app_label VARCHAR(100) NOT NULL,
#     model VARCHAR(100) NOT NULL,
#     UNIQUE(app_label, model)
# );

# -- Create searchengine tables
# CREATE TABLE searchengine_book (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     title VARCHAR(255) NOT NULL,
#     author VARCHAR(255) NOT NULL,
#     publication_year INTEGER NULL,
#     file_path VARCHAR(512) NOT NULL,
#     gutenberg_id INTEGER NULL UNIQUE,
#     centrality_score REAL NOT NULL DEFAULT 0.0,
#     total_clicks INTEGER NOT NULL DEFAULT 0
# );

# -- Create SearchLog model
# CREATE TABLE searchengine_searchlog (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     search_term VARCHAR(255) NOT NULL,
#     is_regex BOOLEAN NOT NULL,
#     timestamp DATETIME NOT NULL,
#     user_id INTEGER NULL REFERENCES auth_user(id)
# );

# -- Create BookClick model
# CREATE TABLE searchengine_bookclick (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     timestamp DATETIME NOT NULL,
#     search_log_id INTEGER NOT NULL REFERENCES searchengine_searchlog(id),
#     book_id INTEGER NOT NULL REFERENCES searchengine_book(id),
#     UNIQUE(search_log_id, book_id)
# );

# -- Create BookSimilarity model
# CREATE TABLE searchengine_booksimilarity (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     similarity_score REAL NOT NULL DEFAULT 0.0,
#     from_book_id INTEGER NOT NULL REFERENCES searchengine_book(id),
#     to_book_id INTEGER NOT NULL REFERENCES searchengine_book(id),
#     UNIQUE(from_book_id, to_book_id)
# );

# CREATE INDEX searchengine_booksimilarity_from_book_id ON searchengine_booksimilarity(from_book_id);
# CREATE INDEX searchengine_booksimilarity_to_book_id ON searchengine_booksimilarity(to_book_id);
# CREATE INDEX searchengine_booksimilarity_similarity_score ON searchengine_booksimilarity(similarity_score);

# -- Create the optimal_index table with proper indexes
# CREATE TABLE optimal_index (
#     word TEXT PRIMARY KEY,
#     book_ids TEXT NOT NULL,
#     book_counts TEXT NOT NULL,
#     total_occurrences INTEGER NOT NULL DEFAULT 0
# );
# CREATE INDEX optimal_index_total_idx ON optimal_index(total_occurrences DESC);

# -- Create auth tables needed by Django
# CREATE TABLE auth_user (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     password VARCHAR(128) NOT NULL,
#     last_login DATETIME NULL,
#     is_superuser BOOL NOT NULL, 
#     username VARCHAR(150) NOT NULL UNIQUE,
#     last_name VARCHAR(150) NOT NULL,
#     email VARCHAR(254) NOT NULL,
#     is_staff BOOL NOT NULL,
#     is_active BOOL NOT NULL,
#     date_joined DATETIME NOT NULL,
#     first_name VARCHAR(150) NOT NULL
# );

# -- Performance optimizations
# PRAGMA journal_mode = WAL;
# PRAGMA synchronous = NORMAL;
# PRAGMA cache_size = 10000000;
# PRAGMA temp_store = MEMORY;
# PRAGMA mmap_size = 30000000000;
# PRAGMA busy_timeout = 60000;
# PRAGMA locking_mode = NORMAL;
# PRAGMA foreign_keys = ON;
# VACUUM;
# ANALYZE;
# EOF

# Create database with our custom schema
# sqlite3 db.sqlite3 < dump.sql

# echo "Database created successfully!"

# Run the indexing with optimized settings
echo "Starting book indexing with optimized settings..."
python manage.py reindex_local_books --memory-mode --batch-size 300 --db-batch-size 100 --max-workers 4 --skip-existing --skip-centrality

echo "Indexing complete!"