"""Utility functions for the search engine."""

import os
import re
import json
import requests
import numpy as np
import nltk
import time
import itertools
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from django.db.models import Count, F
from django.db import transaction
from .models import Book, BookIndex, BookSimilarity
from .config import (
    BOOKS_STORAGE_PATH,
    GUTENBERG_MIRROR,
    MIN_WORDS_PER_BOOK,
    MAX_WORDS_TO_INDEX,
    JACCARD_SIMILARITY_THRESHOLD,
    MAX_GRAPH_WORKERS,
    GRAPH_CHUNK_SIZE,
)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Make sure punkt is downloaded
nltk.download("punkt", quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

# Initialize stopwords
STOP_WORDS = set(stopwords.words("english"))

# Minimum word length for indexing
MIN_WORD_LENGTH = 4


def clean_text(text):
    """Clean and normalize text for indexing and searching."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text, apply_stemming=True):
    """
    Tokenize text into words, removing stopwords and applying stemming.
    For book validation: Implements a strict tokenization that properly counts unique words.
    For search queries: Preserves multi-word phrases for better search results.

    Args:
        text: The text to tokenize
        apply_stemming: Whether to apply stemming (default: True)
    """
    try:
        # Add a safety check for None or empty text
        if not text:
            return []

        # Clean the text first
        cleaned_text = clean_text(text)

        # Use nltk's word_tokenize for proper tokenization
        tokens = word_tokenize(cleaned_text)

        # Special case for multi-word queries - preserve the original tokens too
        # This ensures we can find things like "open door" even if they get tokenized
        # But only for short text (likely search queries, not book content)
        original_phrase = None
        # if " " in text and len(text.split()) <= 5:
        #     # Add the original words (without cleaning) for better matching
        #     original_tokens = [w.lower() for w in text.split() if len(w) >= MIN_WORD_LENGTH]
        #     tokens.extend(original_tokens)

        #     # Also add the complete phrase as a token for exact matching
        #     if len(text) > 1:
        #         original_phrase = text.lower()
        #         tokens.append(original_phrase)

        # Remove stopwords and short tokens
        # For short text (search queries), keep all tokens of minimum length
        # For long text (book content), remove stopwords
        is_query = len(text) <= 100  # If the text is short, it's likely a query

        if is_query:
            tokens = [token for token in tokens if len(token) >= MIN_WORD_LENGTH]
        else:
            # For book content, be less aggressive with stopword removal
            # to ensure we get a more accurate word count for validation
            if len(text) > 8000:  # Likely a book, not a query
                # Keep words that appear in the text more than just as common stopwords
                # This helps get a more accurate word count for large texts
                # But still removes the most common words that would skew counts
                limited_stopwords = {
                    "the",
                    "and",
                    "a",
                    "to",
                    "of",
                    "in",
                    "i",
                    "that",
                    "is",
                    "was",
                    "it",
                    "for",
                    "as",
                    "with",
                    "be",
                    "on",
                    "at",
                    "by",
                    "this",
                    "have",
                }
                tokens = [
                    token
                    for token in tokens
                    if (token.lower() not in limited_stopwords)
                    and len(token) >= MIN_WORD_LENGTH
                ]
            else:
                # Standard processing for search queries and smaller texts
                tokens = [
                    token.lower()
                    for token in tokens
                    if token.lower() not in STOP_WORDS and len(token) >= MIN_WORD_LENGTH
                ]

        # Apply stemming to get the root form of words, except for the original phrase
        if apply_stemming:
            stemmed_tokens = []
            for token in tokens:
                # Don't stem the original multi-word phrase if it exists
                if token != original_phrase:
                    stemmed_tokens.append(stemmer.stem(token))
                else:
                    stemmed_tokens.append(token)
            tokens = stemmed_tokens

        # Remove duplicates but preserve order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token.lower() not in seen:
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens
    except Exception as e:
        print(f"Error in tokenize_text: {str(e)}")
        # Fallback to simple splitting if tokenization fails
        words = [w.lower() for w in text.split() if len(w) >= MIN_WORD_LENGTH]

        # Add the full text as a token too
        if len(text) > 1 and len(text.split()) <= 5:  # Only for short texts
            words.append(text.lower())

        # Apply stemming to fallback words
        if apply_stemming:
            words = [stemmer.stem(w) for w in words if len(w) >= MIN_WORD_LENGTH]

        return words


@transaction.atomic
def index_book(book):
    """Index a book for efficient searching."""
    # Delete any existing indices for this book
    BookIndex.objects.filter(book=book).delete()

    # Get content from file and tokenize it with stemming
    content = book.get_text_content()
    content_tokens = tokenize_text(content, apply_stemming=True)

    # Tokenize title and author for special weighting (with stemming)
    title_tokens = tokenize_text(book.title, apply_stemming=True)

    # Also store the unstemmed title tokens for exact matching
    unstemmed_title_tokens = tokenize_text(book.title, apply_stemming=False)

    author_tokens = tokenize_text(book.author, apply_stemming=True)

    # Count word occurrences in main text
    word_counts = Counter(content_tokens)

    # Apply major boosting to title words (20x weight) and author words (10x weight)
    # This ensures title/author matches are ranked much higher
    for word in title_tokens:
        if len(word) >= MIN_WORD_LENGTH:
            word_counts[word] += 20 * (
                word_counts.get(word, 0) + 1
            )  # Add significant weight to title words

    for word in author_tokens:
        if len(word) >= MIN_WORD_LENGTH:
            word_counts[word] += 10 * (
                word_counts.get(word, 0) + 1
            )  # Add weight to author words

    # Also add multi-word phrases from title to the index
    # Use the unstemmed title for exact phrase matching
    title_words = [
        word for word in book.title.lower().split() if len(word) >= MIN_WORD_LENGTH
    ]
    if len(title_words) > 1:
        # Add bigrams (pairs of consecutive words)
        for i in range(len(title_words) - 1):
            if (
                len(title_words[i]) >= MIN_WORD_LENGTH
                and len(title_words[i + 1]) >= MIN_WORD_LENGTH
            ):
                bigram = title_words[i] + " " + title_words[i + 1]
                word_counts[bigram] = 30  # Very high weight for exact title phrases

        # Add the entire title as an index entry if it's not too short
        full_title = book.title.lower()
        if len(full_title) > MIN_WORD_LENGTH:
            word_counts[full_title] = 50  # Extremely high weight for exact title match

    # Filter out words that are too short before creating indices
    filtered_word_counts = {
        word: count
        for word, count in word_counts.items()
        if len(word) >= MIN_WORD_LENGTH or " " in word
    }  # Keep multi-word phrases regardless of length

    # Convert filtered dictionary back to Counter for most_common method
    filtered_counter = Counter(filtered_word_counts)

    # Create index entries for the most common words
    indices = [
        BookIndex(book=book, word=word, occurrences=count)
        for word, count in filtered_counter.most_common(MAX_WORDS_TO_INDEX)
    ]

    # Bulk create all indices
    BookIndex.objects.bulk_create(indices, batch_size=1000)

    return len(indices)


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

    if content is None:
        raise Exception(
            f"Failed to download book {gutenberg_id}. Tried multiple URL formats."
        )

    # Save the book locally
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(content)

    return local_path, content


def extract_metadata_from_gutenberg(text):
    """Extract metadata (title, author) from Gutenberg text."""
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


def calculate_jaccard_similarity(book1, book2):
    """Calculate Jaccard similarity between two books based on indexed words."""
    # Get words from both books
    words1 = set(BookIndex.objects.filter(book=book1).values_list("word", flat=True))
    words2 = set(BookIndex.objects.filter(book=book2).values_list("word", flat=True))

    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    if union == 0:
        return 0

    return intersection / union


def _process_book_batch(book_pairs, word_cache=None):
    """Process a batch of book pairs to calculate similarity."""
    if word_cache is None:
        word_cache = {}

    results = []
    for book1_id, book1_data, book2_id, book2_data in book_pairs:
        # Get words from cache or database
        if book1_id not in word_cache:
            word_cache[book1_id] = set(
                BookIndex.objects.filter(book_id=book1_id).values_list(
                    "word", flat=True
                )
            )
        if book2_id not in word_cache:
            word_cache[book2_id] = set(
                BookIndex.objects.filter(book_id=book2_id).values_list(
                    "word", flat=True
                )
            )

        words1 = word_cache[book1_id]
        words2 = word_cache[book2_id]

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0

        if similarity >= JACCARD_SIMILARITY_THRESHOLD:
            results.append((book1_id, book2_id, similarity))

    return results


def build_jaccard_graph_chunk(chunk_id, total_chunks, use_parallel=True):
    """Build part of the Jaccard graph for a chunk of books."""
    G = nx.Graph()

    # Get all books
    all_books = list(Book.objects.all().values("id", "title", "author"))
    total_books = len(all_books)

    # Divide books into chunks
    chunk_size = (total_books + total_chunks - 1) // total_chunks  # Ceiling division
    start_idx = chunk_id * chunk_size
    end_idx = min((chunk_id + 1) * chunk_size, total_books)

    # This chunk's books
    chunk_books = all_books[start_idx:end_idx]
    print(f"Processing chunk {chunk_id+1}/{total_chunks} with {len(chunk_books)} books")

    # Add nodes for this chunk
    for book in chunk_books:
        G.add_node(book["id"], title=book["title"], author=book["author"])

    # Preload word cache for all books in chunk to avoid repeated DB queries
    word_cache = {}
    for book in chunk_books:
        word_cache[book["id"]] = set(
            BookIndex.objects.filter(book_id=book["id"]).values_list("word", flat=True)
        )

    # Compare books within this chunk
    book_pairs = []
    for i, book1 in enumerate(chunk_books):
        for book2 in chunk_books[i + 1 :]:
            book_pairs.append((book1["id"], book1, book2["id"], book2))

    # Compare with books from other chunks (only those after this chunk to avoid duplication)
    for i, book1 in enumerate(chunk_books):
        for book2 in all_books[end_idx:]:
            book_pairs.append((book1["id"], book1, book2["id"], book2))

    # Process book pairs - either serially or in parallel
    edges = []
    if (
        use_parallel and len(book_pairs) > 100
    ):  # Only use parallel for large enough batches
        # Split into smaller batches for parallel processing
        batch_size = min(1000, max(50, len(book_pairs) // MAX_GRAPH_WORKERS))
        batches = [
            book_pairs[i : i + batch_size]
            for i in range(0, len(book_pairs), batch_size)
        ]

        with ProcessPoolExecutor(max_workers=MAX_GRAPH_WORKERS) as executor:
            future_to_batch = {
                executor.submit(_process_book_batch, batch, word_cache): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch_edges = future.result()
                edges.extend(batch_edges)
    else:
        # Process serially
        edges = _process_book_batch(book_pairs, word_cache)

    # Add edges to graph
    for book1_id, book2_id, similarity in edges:
        G.add_edge(book1_id, book2_id, weight=similarity)

    print(f"Chunk {chunk_id+1} complete. Added {G.number_of_edges()} edges.")
    return G


def build_jaccard_graph():
    """Build a graph where books are nodes and edges represent Jaccard similarity."""
    start_time = time.time()
    print("Starting Jaccard graph construction...")

    # First, clear existing similarity relationships
    BookSimilarity.objects.all().delete()

    # Determine number of chunks based on book count
    total_books = Book.objects.count()
    # Use smaller chunks for larger book counts to limit memory usage
    num_chunks = max(1, min(10, total_books // GRAPH_CHUNK_SIZE))

    # Build graph in chunks
    graph_chunks = []
    for chunk_id in range(num_chunks):
        chunk_graph = build_jaccard_graph_chunk(chunk_id, num_chunks)
        graph_chunks.append(chunk_graph)

    # Combine all chunk graphs
    G = nx.Graph()
    for chunk_graph in graph_chunks:
        G.add_nodes_from(chunk_graph.nodes(data=True))
        G.add_edges_from(chunk_graph.edges(data=True))

    # Store the graph edges in the database
    similarity_objects = []
    for u, v, data in G.edges(data=True):
        # Create bidirectional edges since the graph is undirected
        similarity_objects.append(
            BookSimilarity(
                from_book_id=u, to_book_id=v, similarity_score=data["weight"]
            )
        )
        similarity_objects.append(
            BookSimilarity(
                from_book_id=v, to_book_id=u, similarity_score=data["weight"]
            )
        )

    # Bulk create the similarity relationships
    if similarity_objects:
        BookSimilarity.objects.bulk_create(
            similarity_objects, batch_size=1000, ignore_conflicts=True
        )

    print(f"Graph construction complete in {time.time() - start_time:.2f} seconds")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Stored {len(similarity_objects)} similarity relationships in database")

    return G


def calculate_pagerank(G):
    """Calculate PageRank for all books in the graph."""
    pagerank = nx.pagerank(G)
    59
    # Update centrality scores in the database
    with transaction.atomic():
        for book_id, score in pagerank.items():
            Book.objects.filter(id=book_id).update(centrality_score=score)

    return pagerank


def calculate_closeness_centrality(G):
    """Calculate closeness centrality for all books in the graph."""
    closeness = nx.closeness_centrality(G)

    # Update centrality scores in the database
    with transaction.atomic():
        for book_id, score in closeness.items():
            print(f"Book ID: {book_id}, Score: {score}")
            Book.objects.filter(id=book_id).update(centrality_score=score)

    return closeness


def calculate_betweenness_centrality(G):
    """Calculate betweenness centrality for all books in the graph."""
    betweenness = nx.betweenness_centrality(G)

    # Update centrality scores in the database
    with transaction.atomic():
        for book_id, score in betweenness.items():
            Book.objects.filter(id=book_id).update(centrality_score=score)

    return betweenness


def regex_search(pattern, book):
    """Search for a regex pattern in book content."""
    try:
        # Get content from file
        book_content = book.get_text_content()

        # Limit to first 100 matches for performance
        matches = re.finditer(pattern, book_content, re.IGNORECASE)
        return [match.group() for match in matches]
    except Exception as e:
        print(f"Error in regex_search: {str(e)}")
        return []
