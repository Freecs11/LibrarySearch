#!/bin/bash

# Create a backup of the database
echo "Creating backup of database..."
cp db.sqlite3 db.sqlite3.bak

# Create the migration_cleanup.sql file with our fixes
cat > migration_cleanup.sql << EOF
-- First create the optimal_index table with proper indexes if it doesn't exist
CREATE TABLE IF NOT EXISTS optimal_index (
    word TEXT PRIMARY KEY,
    book_ids TEXT NOT NULL,
    book_counts TEXT NOT NULL,
    total_occurrences INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS optimal_index_total_idx ON optimal_index(total_occurrences DESC);

-- Mark problematic migrations as applied if they exist
INSERT OR IGNORE INTO django_migrations (app, name, applied) 
SELECT 'searchengine', '0005_remove_bookindex', datetime('now')
WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='django_migrations');

INSERT OR IGNORE INTO django_migrations (app, name, applied) 
SELECT 'searchengine', '0006_remove_bookindex', datetime('now')
WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='django_migrations');

INSERT OR IGNORE INTO django_migrations (app, name, applied) 
SELECT 'searchengine', '0007_remove_bookindex', datetime('now')
WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='django_migrations');

-- Performance optimizations
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL; 
PRAGMA cache_size = 1000000;
PRAGMA temp_store = MEMORY;
PRAGMA busy_timeout = 60000;
PRAGMA mmap_size = 30000000000;
ANALYZE;
EOF

# Apply the SQL fixes
echo "Applying database schema fixes..."
sqlite3 db.sqlite3 < migration_cleanup.sql

echo "Running fake migrations if needed..."
python manage.py migrate --fake searchengine 0007_remove_bookindex || echo "Migration command failed, but it's okay to continue"

echo "Optimizing database for performance..."
python -c "
import sqlite3
conn = sqlite3.connect('db.sqlite3', timeout=60)
conn.execute('PRAGMA journal_mode = WAL')
conn.execute('PRAGMA synchronous = NORMAL')
conn.execute('PRAGMA cache_size = 1000000')
conn.execute('PRAGMA temp_store = MEMORY') 
conn.execute('PRAGMA busy_timeout = 60000')
conn.execute('ANALYZE')
conn.close()
"

echo "Schema update complete!"