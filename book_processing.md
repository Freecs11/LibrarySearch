# Book Processing Improvements

This document explains the improvements made to the book processing pipeline to address slow downloads, inefficient filtering, and text indexing optimization.

## Issues with the Original Approach

1. **Late Filtering**: Books were downloaded first, then checked for minimum word count afterward
2. **Single-threaded Processing**: Books were downloaded one at a time
3. **Inefficient File Management**: Files were kept even if books didn't meet requirements
4. **Slow Book Acquisition**: No parallel processing to speed up downloads

## Improved Approach

A new command `fetch_gutenberg_books_improved.py` has been created with the following improvements:

### 1. Parallel Downloads

- Uses `ThreadPoolExecutor` for concurrent downloads
- Configurable number of worker threads (`--max-workers`)
- Progress tracking with `tqdm`

### 2. Smarter Book Selection

- Prioritizes a curated list of known high-quality books
- Targets specific ID ranges with better candidates
- Shuffles additional book IDs for variety

### 3. Early Filtering

- Checks word count before saving to the database
- Removes files for books that don't meet length requirements
- Extracts metadata before committing to database

### 4. Cleanup Utilities

- New `--cleanup` option to remove:
  - Books that don't meet word count requirements
  - Orphaned files (files not referenced by any book)

## Usage

```bash

# Basic usage - index all downloaded books
python manage.py index_gutenberg_books

# Skip books already in the database
python manage.py index_gutenberg_books --skip-existing

# Process specific books
python manage.py index_gutenberg_books --specific-books "1342,11,84"

# Skip centrality calculation (faster)
python manage.py index_gutenberg_books --skip-centrality

# Use different ranking method
python manage.py index_gutenberg_books --ranking-method "pagerank"

# Basic usage - download all books from good_books.txt
python manage.py download_gutenberg_books

# Download with verification (skip existing books)
python manage.py download_gutenberg_books --verify

# Limit to first 100 books with custom output directory
python manage.py download_gutenberg_books --limit 100 --output-dir /path/to/books

# Increase parallelism and timeout for slow connections
python manage.py download_gutenberg_books --max-workers 50 --timeout 120

||

  # First find good books that meet requirements
  python manage.py find_good_books --count 500 --max-workers 30 --output good_books.txt

  # Then download them
  python manage.py fetch_gutenberg_books_improved --count 1664 --max-workers 30
  
# Fetch books with improved process
python manage.py fetch_gutenberg_books_improved --count 1664 --max-workers 20

# Only recalculate centrality for existing books
python manage.py fetch_gutenberg_books_improved --recalculate-centrality

# Clean up short books and orphaned files
python manage.py fetch_gutenberg_books_improved --cleanup
```

## Stemming and Word Indexing Improvements

The search engine now implements stemming and improved word filtering:

### Stemming Implementation

Words are now reduced to their root form using the Porter stemmer algorithm:

- "play", "playing", "plays", "played" → all stem to "play"
- "formal", "formally", "formality" → all stem to "formal"

Benefits:
1. Reduces index size by merging related word forms
2. Improves search quality by matching similar words
3. Makes the search more intuitive for users

### Minimum Word Length

Words shorter than 3 characters are now excluded from indexing by default, which:
1. Reduces noise in search results
2. Decreases index size significantly
3. Improves performance of searches

### Reindexing with Improved Processing

To apply these improvements to your existing book collection:

```bash
# Reindex all books with stemming and minimum word length of 3
python manage.py reindex_local_books --reindex-all

# Use a custom minimum word length
python manage.py reindex_local_books --reindex-all --min-word-length 4
```

## Results

The improved approach offers:

1. **Faster Processing**: Multiple books downloaded simultaneously
2. **Storage Efficiency**: Only compliant books are stored
3. **Better Success Rate**: Targeting better book ID ranges increases compliance
4. **Resource Conservation**: Reduced database size and disk usage
5. **Improved Search Quality**: Stemming helps find related word forms
6. **Smaller Index**: Filtering short words reduces index size significantly
7. **More Relevant Results**: Better word filtering improves search relevance

## Technical Details

### Download Process

1. Check if file already exists and meets requirements
2. Try multiple URL formats from Project Gutenberg
3. Verify word count before saving to disk
4. Extract metadata, save to disk, and add to database

### Book Selection Strategy

Book IDs are selected from:
1. Curated list of classic literature (likely to be long enough)
2. Specific ranges known to contain longer works
3. Random selection from broader ranges as a fallback

### Text Processing and Indexing

The indexing process now:
1. Applies the Porter stemmer to reduce words to their root form
2. Filters out words shorter than the minimum length (default: 3 characters)
3. Preserves multi-word phrases for exact matching
4. Properly handles search queries by stemming the query terms

When a search is performed, the system:
1. Stems the search terms
2. Matches against the stemmed index
3. Also tries to match the full query for exact phrase searches
4. Ranks results by relevance and centrality scores