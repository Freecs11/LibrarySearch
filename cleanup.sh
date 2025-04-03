#!/bin/bash

echo "Cleaning up unnecessary files..."

# Remove alternative/interim implementation files in searchengine
files_to_remove=(
    "searchengine/efficient_search.py"
    "searchengine/final_search.py"
    "searchengine/simple_search.py"
    "searchengine/models_eff.py"
    "searchengine/models_improved.py"
    "searchengine/utils_eff.py"
    "searchengine/utils_improved.py"
    "searchengine/views_improved.py"
)

# Remove interim management commands
commands_to_remove=(
    "searchengine/management/commands/efficient_indexing.py"
    "searchengine/management/commands/fixed_index.py"
    "searchengine/management/commands/migrate_to_improved_indexing.py"
    "searchengine/management/commands/rebuild_index.py"
    "searchengine/management/commands/reindex_improved.py"
    "searchengine/management/commands/simple_index.py"
    "searchengine/management/commands/cleanup_index.py"
)

# Root directory files to remove
root_files_to_remove=(
    "EFFICIENT_INDEXING.md"
    "EFFICIENT_INDEX_README.md"
    "IMPROVED_INDEXING.md"
    "clean_database.py"
    "clean_index.py"
    "clean_search.py"
    "direct_index.py"
    "drop_old_index.py"
    "drop_tables.py"
    "optimal_index.py"
    "optimal_search.py"
    "search_functions.py"
)

# Keep the documentation file
# "book_processing.md" - This is useful documentation

# Remove all files
echo "Removing files in searchengine directory..."
for file in "${files_to_remove[@]}" "${commands_to_remove[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file"
        rm "$file"
    else
        echo "File $file not found"
    fi
done

echo "Removing files in root directory..."
for file in "${root_files_to_remove[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file"
        rm "$file"
    else
        echo "File $file not found"
    fi
done

echo "Cleanup complete!"