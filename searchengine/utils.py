"""Utility functions for the search engine."""

import os
import re
import json
import requests
import numpy as np
import time
import sqlite3
import itertools
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Import nltk
import nltk

# Download NLTK resources first - before any imports
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Now import NLTK components after downloading
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Import remaining modules
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Django imports
from django.db.models import Count, F
from django.db import transaction, connection
from .models import Book, BookSimilarity
from .config import (
    BOOKS_STORAGE_PATH,
    GUTENBERG_MIRROR,
    MIN_WORDS_PER_BOOK,
    MAX_WORDS_TO_INDEX,
    JACCARD_SIMILARITY_THRESHOLD,
    MAX_GRAPH_WORKERS,
    GRAPH_CHUNK_SIZE,
)

# Verify NLTK resources are properly loaded
try:
    test_tokenize = nltk_word_tokenize("Test sentence.")
except Exception as e:
    # Define a fallback tokenizer
    def nltk_word_tokenize(text):
        return text.split()


# Initialize stemmer
try:
    stemmer = PorterStemmer()
    test_stem = stemmer.stem("testing")
except Exception as e:
    # Define a fallback stemmer that does nothing
    class FallbackStemmer:
        def stem(self, word):
            return word

    stemmer = FallbackStemmer()

# Initialize stopwords
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception as e:
    # Define fallback stopwords
    STOP_WORDS = set(["the", "a", "an", "and", "or", "but", "is", "are", "was", "were"])

# Minimum word length for indexing
MIN_WORD_LENGTH = 4

# Database path
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db.sqlite3"
)

# Enable timing diagnostics
ENABLE_TIMING = False


def log_time(func):
    """Decorator to log execution time of functions"""

    def wrapper(*args, **kwargs):
        if not ENABLE_TIMING:
            return func(*args, **kwargs)

        # start_time = time.time()
        result = func(*args, **kwargs)
        # elapsed_time = time.time() - start_time

        # Get the name of the first argument if it's a class instance
        # obj_name = ""
        # if args and hasattr(args[0], "__class__"):
        #     if hasattr(args[0], "title"):
        #         obj_name = f" [{args[0].title[:20]}...]"
        #     elif hasattr(args[0], "gutenberg_id"):
        #         obj_name = f" [ID: {args[0].gutenberg_id}]"

        # Print timing info
        # print(
        #     f"TIMING: {func.__name__}{obj_name} took {elapsed_time:.2f} seconds",
        #     file=sys.stderr,
        # )

        return result

    return wrapper


def clean_text(text):
    """Clean and normalize text for indexing and searching."""
    # Safety check for non-string input
    if not isinstance(text, str):
        print(f"WARNING: clean_text received non-string: {type(text)}")
        if text is None:
            return ""
        try:
            text = str(text)
        except Exception as e:
            print(f"ERROR in clean_text: {e}")
            return ""

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


@log_time
def tokenize_text(text, apply_stemming=True):
    """
    Tokenize text into words, removing stopwords and applying stemming.
    COMPLETELY REWRITTEN to avoid any NLTK issues.
    """

    # Simple but reliable tokenization function without NLTK
    def simple_tokenize(text_to_tokenize):
        # Remove punctuation and replace with spaces
        text_to_tokenize = re.sub(r"[^\w\s]", " ", text_to_tokenize)
        # Split on whitespace
        return text_to_tokenize.split()

    try:
        # Safety check for None or empty text
        if not text:
            return []

        # Make sure input is a string
        if not isinstance(text, str):
            print(f"WARNING: tokenize_text received non-string: {type(text)}")
            if text is None:
                return []
            try:
                text = str(text)
            except Exception as e:
                print(f"ERROR: Could not convert to string: {e}")
                return []

        # Clean the text first - we're already handling string conversion above
        text_lower = text.lower()
        # Remove punctuation
        text_clean = re.sub(r"[^\w\s]", " ", text_lower)
        # Remove numbers
        text_clean = re.sub(r"\d+", " ", text_clean)
        # Remove extra whitespace
        text_clean = re.sub(r"\s+", " ", text_clean).strip()

        # Use simple tokenization that won't fail
        tokens = (
            simple_tokenize(text_clean)
            if callable(simple_tokenize)
            else text_clean.split()
        )

        # Filter tokens - minimum length and stopwords
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < MIN_WORD_LENGTH:
                continue

            # Skip stopwords
            if token.lower() in STOP_WORDS:
                continue

            # Skip tokens with underscores or repeating characters
            if "_" in token or re.search(r"(.)\1{3,}", token):
                continue

            filtered_tokens.append(token)

        # Apply stemming if requested
        if apply_stemming:
            stemmed_tokens = []
            for token in filtered_tokens:
                try:
                    # Use our stemmer that was tested in initialization
                    stemmed = stemmer.stem(token)
                    if len(stemmed) >= MIN_WORD_LENGTH:
                        stemmed_tokens.append(stemmed)
                except Exception as e:
                    # If stemming fails, just use the original token
                    print(f"Stemming failed for '{token}': {e}")
                    if len(token) >= MIN_WORD_LENGTH:
                        stemmed_tokens.append(token)
            filtered_tokens = stemmed_tokens

        # Remove duplicates but preserve order
        seen = set()
        unique_tokens = []
        for token in filtered_tokens:
            if token.lower() not in seen:
                seen.add(token.lower())
                unique_tokens.append(token.lower())

        return unique_tokens

    except Exception as e:
        print(f"Critical error in tokenize_text: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Ultra simple fallback as last resort
        try:
            if isinstance(text, str):
                # Just return the lowercase words that meet length requirement
                return [w.lower() for w in text.split() if len(w) >= MIN_WORD_LENGTH]
            return []
        except:
            return []


@log_time
def index_book(book):
    """
    Index a book for efficient searching using the optimal index.
    This adds the book to the optimal_index table.
    """
    # Get content from file and tokenize it with stemming
    content = book.get_text_content()

    content_tokens = tokenize_text(content, apply_stemming=True)

    # Tokenize title and author for special weighting (with stemming)
    title_tokens = tokenize_text(book.title, apply_stemming=True)
    author_tokens = tokenize_text(book.author, apply_stemming=True)

    # Count word occurrences
    word_counts = Counter(content_tokens)

    # Apply boosting to title and author words
    for word in title_tokens:
        word_counts[word] += 20

    for word in author_tokens:
        word_counts[word] += 10

    # Add to the optimal index
    add_to_optimal_index(book.id, word_counts)

    return len(word_counts)


@log_time
def add_to_optimal_index(book_id, word_counts):
    """
    Add a book's words to the optimal index with improved batch processing.
    Uses larger batches and optimized SQLite operations.
    """
    if not word_counts:
        return

    str_book_id = str(book_id)

    # Use much larger batch size for better performance
    batch_size = 10000  # Increased from 1000
    word_items = list(word_counts.items())

    # Process in fewer, larger batches
    for i in range(0, len(word_items), batch_size):
        batch = dict(word_items[i : i + batch_size])
        _process_word_batch_bulk(str_book_id, batch)


def _process_word_batch_bulk(str_book_id, word_counts):
    """Process a batch of words using optimized bulk operations"""
    max_retries = 5
    retry_delay = 0.5

    # Use SQLite directly for better performance
    from django.db import connection

    connection.close()  # Close any existing connections

    # Prepare data
    words = list(word_counts.keys())

    # Skip if empty
    if not words:
        return

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=60)

            # Enhanced pragmas for better write performance
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 100000")  # Increased cache
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 30000000000")
            conn.execute("PRAGMA busy_timeout = 60000")  # Increased timeout

            cursor = conn.cursor()

            # Begin a single transaction for the entire batch
            conn.execute("BEGIN IMMEDIATE TRANSACTION")

            # Optimized approach: Use a temporary table for bulk operations
            cursor.execute(
                "CREATE TEMPORARY TABLE temp_words (word TEXT, book_id TEXT, count INTEGER)"
            )

            # Insert all words into the temp table in one go
            temp_data = [
                (word, str_book_id, count) for word, count in word_counts.items()
            ]
            cursor.executemany("INSERT INTO temp_words VALUES (?, ?, ?)", temp_data)

            # Update existing words using a more efficient join approach
            cursor.execute(
                """
                UPDATE optimal_index 
                SET 
                    book_ids = CASE
                        WHEN book_ids LIKE '%' || ? || '%' THEN book_ids
                        ELSE json_insert(book_ids, '$[' || json_array_length(book_ids) || ']', ?)
                    END,
                    book_counts = json_set(book_counts, '$.' || ?, temp.count),
                    total_occurrences = total_occurrences + temp.count
                FROM temp_words AS temp
                WHERE optimal_index.word = temp.word
            """,
                (str_book_id, str_book_id, str_book_id),
            )

            # Insert new words that don't exist yet
            cursor.execute(
                """
                INSERT INTO optimal_index (word, book_ids, book_counts, total_occurrences)
                SELECT 
                    temp.word,
                    json_array(temp.book_id),
                    json_object(temp.book_id, temp.count),
                    temp.count
                FROM temp_words AS temp
                WHERE NOT EXISTS (
                    SELECT 1 FROM optimal_index WHERE word = temp.word
                )
            """
            )

            # Commit all changes at once
            conn.commit()

            # Drop temporary table
            cursor.execute("DROP TABLE temp_words")

            conn.close()
            return  # Success

        except sqlite3.OperationalError as e:
            # Handle database locks
            try:
                conn.rollback()
                conn.close()
            except:
                pass

            if "database is locked" in str(e).lower():
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2**attempt)
                    print(
                        f"Database locked during batch operation. Retrying in {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries reached for batch operation: {e}")
            else:
                print(f"SQLite operational error: {e}")
                break

        except Exception as e:
            # Handle other exceptions
            try:
                conn.rollback()
                conn.close()
            except:
                pass
            print(f"Unexpected error in batch operation: {e}")
            break


@log_time
def search_books(query, is_regex=False, limit=1000):
    """
    Search for books using the optimal index.

    Args:
        query: The search query
        is_regex: Whether to treat the query as a regular expression
        limit: Maximum number of results to return

    Returns:
        List of Book objects matching the query
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()

    try:
        # Get book IDs and scores
        if is_regex:
            book_scores = search_regex(cursor, query)
        else:
            book_scores = search_keywords(cursor, query)

        if not book_scores:
            return []

        # Get top-scoring books - more efficiently
        top_book_ids = [book_id for book_id, _ in book_scores.most_common(limit)]

        if not top_book_ids:
            return []

        # Fetch Book objects in a single query (more efficient)
        books = list(Book.objects.filter(id__in=top_book_ids))

        # Add relevance scores to each book
        for book in books:
            book.relevance = book_scores[book.id]
            # Add matched terms attribute to make the frontend work
            book.matched_terms = {"query": True}

        # Sort in memory to preserve relevance order
        books.sort(key=lambda b: top_book_ids.index(b.id))

        return books
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return []
    finally:
        conn.close()


@log_time
def search_keywords(cursor, query):
    """Search for books using keywords."""
    # Tokenize the query
    tokens = tokenize_text(query, apply_stemming=True)

    if not tokens:
        return Counter()

    # Use a more efficient approach for multiple tokens
    if len(tokens) > 1:
        # Create placeholders for the SQL IN clause
        placeholders = ", ".join(["?" for _ in tokens])

        # Get book counts for all tokens at once
        cursor.execute(
            f"""
            SELECT word, book_counts 
            FROM optimal_index 
            WHERE word IN ({placeholders})
            """,
            tokens,
        )

        # Process all results at once
        book_scores = Counter()
        for word, book_counts_json in cursor.fetchall():
            book_counts = json.loads(book_counts_json)
            for book_id, count in book_counts.items():
                book_scores[int(book_id)] += int(count)

        return book_scores
    else:
        # Single token search - simpler case
        book_scores = Counter()
        if tokens:
            token = tokens[0]
            cursor.execute(
                "SELECT book_counts FROM optimal_index WHERE word = ?", (token,)
            )
            result = cursor.fetchone()

            if result:
                # Parse book counts
                book_counts = json.loads(result[0])

                # Add scores to counter
                for book_id, count in book_counts.items():
                    book_scores[int(book_id)] += int(count)

    return book_scores


@log_time
def search_regex(cursor, pattern):
    """Search for books using a regular expression pattern."""
    try:
        # Validate the regex pattern
        regex = re.compile(pattern, re.IGNORECASE)
    except Exception as e:
        print(f"Invalid regex pattern: {e}")
        return Counter()

    # Simple and efficient approach: get all words, filter with regex
    cursor.execute("SELECT word, book_counts FROM optimal_index")

    # Initialize result counter
    book_scores = Counter()

    # Find all words matching the regex
    matched_words = []
    for word, book_counts_json in cursor.fetchall():
        if regex.search(word):
            matched_words.append(word)
            book_counts = json.loads(book_counts_json)
            for book_id, count in book_counts.items():
                book_scores[int(book_id)] += int(count)

    # Debug info
    print(f"Found {len(matched_words)} words matching regex '{pattern}'")

    return book_scores


@log_time
def get_recommendations(books, limit=5):
    """
    Get book recommendations based on a list of books.

    Args:
        books: List of Book objects to base recommendations on
        limit: Maximum number of recommendations

    Returns:
        List of recommended Book objects
    """
    if not books:
        return []

    # Option 1: Use BookSimilarity model directly (faster)
    # This leverages the pre-calculated relationships
    result_ids = [book.id for book in books[:3]]

    # Check if we have similarity data
    similarity_exists = BookSimilarity.objects.filter(
        from_book_id__in=result_ids
    ).exists()

    if similarity_exists:
        # Get similar books from BookSimilarity
        similar_book_ids = list(
            BookSimilarity.objects.filter(from_book_id__in=result_ids)
            .exclude(to_book_id__in=result_ids)
            .values_list("to_book_id", flat=True)
            .distinct()[:limit]
        )

        if similar_book_ids:
            return list(Book.objects.filter(id__in=similar_book_ids))

    # Fallback to direct word-based search (slower but more reliable)
    # Get book IDs
    book_ids = [book.id for book in books[:3]]
    book_id_str_list = [f'"{book_id}"' for book_id in book_ids]

    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()

    try:
        # More efficient query that gets all the data at once
        placeholders = ", ".join(["?" for _ in book_ids])
        book_id_sql_list = ", ".join(book_id_str_list)

        # First, find all books that share words with our source books
        cursor.execute(
            f"""
            SELECT DISTINCT o1.word 
            FROM optimal_index o1
            WHERE o1.book_ids LIKE '%{book_id_sql_list}%'
            ORDER BY o1.total_occurrences DESC
            LIMIT 30
        """
        )

        top_words = [row[0] for row in cursor.fetchall()]

        if not top_words:
            return []

        # Build IN clause for words
        word_placeholders = ", ".join(["?" for _ in top_words])

        # Now find books that contain these words
        query = f"""
            SELECT book_id, SUM(occurrences) as total_count
            FROM (
                SELECT json_each.value as book_id, 
                       json_extract(o2.book_counts, '$.' || json_each.value) as occurrences
                FROM optimal_index o2, json_each(o2.book_ids)
                WHERE o2.word IN ({word_placeholders})
            )
            WHERE book_id NOT IN ({placeholders})
            GROUP BY book_id
            ORDER BY total_count DESC
            LIMIT {limit}
        """

        cursor.execute(query, top_words + [str(bid) for bid in book_ids])

        # Get top-scoring books
        top_book_ids = [int(row[0]) for row in cursor.fetchall()]

        if not top_book_ids:
            return []

        # Fetch Book objects in a single query
        return list(Book.objects.filter(id__in=top_book_ids))
    finally:
        conn.close()


def extract_metadata_from_gutenberg(text):
    """Extract title, author, and year from Project Gutenberg text."""
    title = "Unknown Title"
    author = "Unknown Author"
    year = None

    # Try to extract title
    title_match = re.search(r"Title:\s*([^\r\n]+)", text)
    if title_match:
        title = title_match.group(1).strip()

    # Try to extract author
    author_match = re.search(r"Author:\s*([^\r\n]+)", text)
    if author_match:
        author = author_match.group(1).strip()

    # Try to extract year
    year_match = re.search(r"Release date:\s*(?:.*?)(\d{4})", text)
    if year_match:
        try:
            year = int(year_match.group(1))
        except ValueError:
            pass

    return title, author, year


@log_time
def calculate_jaccard_similarity(book1, book2):
    """Calculate Jaccard similarity between two books using the optimal index."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cursor = conn.cursor()

    try:
        # Get words for book1
        cursor.execute(
            """
            SELECT word FROM optimal_index
            WHERE book_ids LIKE ?
            """,
            (f'%"{book1.id}"%',),
        )
        words1 = {row[0] for row in cursor.fetchall()}

        # Get words for book2
        cursor.execute(
            """
            SELECT word FROM optimal_index
            WHERE book_ids LIKE ?
            """,
            (f'%"{book2.id}"%',),
        )
        words2 = {row[0] for row in cursor.fetchall()}

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0

        return intersection / union
    finally:
        conn.close()


@log_time
def build_jaccard_graph():
    """Build a graph where books are nodes and edges represent Jaccard similarity."""
    start_time = time.time()
    print("Starting Jaccard graph construction...")

    # First, clear existing similarity relationships
    BookSimilarity.objects.all().delete()

    # Get all books
    all_books = list(Book.objects.all())
    book_count = len(all_books)

    # Create the graph
    G = nx.Graph()

    # Add nodes
    for book in all_books:
        G.add_node(book.id, title=book.title, author=book.author)

    # Add edges
    similarity_objects = []
    processed = 0

    # Process book pairs
    for i, book1 in enumerate(all_books):
        for book2 in all_books[i + 1 :]:
            similarity = calculate_jaccard_similarity(book1, book2)

            # Only add edges above the threshold
            if similarity >= JACCARD_SIMILARITY_THRESHOLD:
                G.add_edge(book1.id, book2.id, weight=similarity)

                # Create bidirectional edges
                similarity_objects.append(
                    BookSimilarity(
                        from_book=book1, to_book=book2, similarity_score=similarity
                    )
                )
                similarity_objects.append(
                    BookSimilarity(
                        from_book=book2, to_book=book1, similarity_score=similarity
                    )
                )

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{book_count} books...")

    # Bulk create the similarity relationships
    if similarity_objects:
        BookSimilarity.objects.bulk_create(
            similarity_objects, batch_size=1000, ignore_conflicts=True
        )

    print(f"Graph construction complete in {time.time() - start_time:.2f} seconds")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Stored {len(similarity_objects)} similarity relationships in database")

    return G


@log_time
def calculate_pagerank(G):
    """Calculate PageRank for all books in the graph."""
    pagerank = nx.pagerank(G)

    # Update book records with PageRank scores
    for book_id, score in pagerank.items():
        Book.objects.filter(id=book_id).update(centrality_score=score)

    return pagerank


@log_time
def calculate_closeness_centrality(G):
    """Calculate closeness centrality for all books in the graph."""
    closeness = nx.closeness_centrality(G)

    # Update book records with closeness centrality scores
    for book_id, score in closeness.items():
        Book.objects.filter(id=book_id).update(centrality_score=score)

    return closeness


@log_time
def calculate_betweenness_centrality(G):
    """Calculate betweenness centrality for all books in the graph."""
    betweenness = nx.betweenness_centrality(G)

    # Update book records with betweenness centrality scores
    for book_id, score in betweenness.items():
        Book.objects.filter(id=book_id).update(centrality_score=score)

    return betweenness


@log_time
def fetch_book_from_gutenberg(gutenberg_id):
    """Download a book from Project Gutenberg by ID."""
    # Create the storage directory if it doesn't exist
    os.makedirs(BOOKS_STORAGE_PATH, exist_ok=True)

    # Local file path
    local_path = os.path.join(BOOKS_STORAGE_PATH, f"book_{gutenberg_id}.txt")

    # If the file already exists, skip downloading
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read()
        return local_path, content

    # Try different URL formats from Project Gutenberg
    urls_to_try = [
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/pg{gutenberg_id}.txt",  # Standard format
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/pg{gutenberg_id}.txt.utf8",  # UTF-8 format
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/{gutenberg_id}.txt",  # Alternative format
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/{gutenberg_id}-0.txt",  # Another alternative
    ]

    content = None
    for url in urls_to_try:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                content = response.text
                break
        except requests.RequestException:
            continue

    if content:
        # Save content to file
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)
        return local_path, content

    return None, None
