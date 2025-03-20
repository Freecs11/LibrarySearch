"""Utility functions for the search engine."""
import os
import re
import json
import requests
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter

from django.db.models import Count, F
from django.db import transaction
from .models import Book, BookIndex
from .config import (
    BOOKS_STORAGE_PATH,
    GUTENBERG_MIRROR,
    MIN_WORDS_PER_BOOK,
    MAX_WORDS_TO_INDEX,
    JACCARD_SIMILARITY_THRESHOLD
)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Make sure punkt is downloaded
nltk.download('punkt', quiet=True)

# Initialize stopwords
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """Clean and normalize text for indexing and searching."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """Tokenize text into words, removing stopwords."""
    try:
        # Add a safety check for None or empty text
        if not text:
            return []
            
        # Clean the text first
        cleaned_text = clean_text(text)
        
        # Use nltk's word_tokenize for proper tokenization
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and single character tokens
        # Keep all tokens for search purposes, even if they're stopwords
        if len(text) <= 5:  # If the search query is short, don't filter
            tokens = [token for token in tokens if len(token) > 1]
        else:
            tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
            
        return tokens
    except Exception as e:
        print(f"Error in tokenize_text: {str(e)}")
        # Fallback to simple splitting if tokenization fails
        return [w for w in text.lower().split() if len(w) > 1]

@transaction.atomic
def index_book(book):
    """Index a book for efficient searching."""
    # Delete any existing indices for this book
    BookIndex.objects.filter(book=book).delete()
    
    # Tokenize the text
    tokens = tokenize_text(book.text_content)
    
    # Count word occurrences
    word_counts = Counter(tokens)
    
    # Create index entries for the most common words
    indices = [
        BookIndex(book=book, word=word, occurrences=count)
        for word, count in word_counts.most_common(MAX_WORDS_TO_INDEX)
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
        with open(local_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            content = f.read()
        return local_path, content
    
    # Try different URL formats from Project Gutenberg
    urls_to_try = [
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/pg{gutenberg_id}.txt",  # Standard format
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/pg{gutenberg_id}.txt.utf8",  # UTF-8 format
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/{gutenberg_id}.txt",  # Alternative format
        f"{GUTENBERG_MIRROR}/{gutenberg_id}/{gutenberg_id}-0.txt"  # Another alternative
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
        raise Exception(f"Failed to download book {gutenberg_id}. Tried multiple URL formats.")
    
    # Save the book locally
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return local_path, content

def extract_metadata_from_gutenberg(text):
    """Extract metadata (title, author) from Gutenberg text."""
    title = "Unknown Title"
    author = "Unknown Author"
    year = None
    
    # Try to extract title
    title_match = re.search(r'Title:\s*([^\r\n]+)', text)
    if title_match:
        title = title_match.group(1).strip()
    
    # Try to extract author
    author_match = re.search(r'Author:\s*([^\r\n]+)', text)
    if author_match:
        author = author_match.group(1).strip()
    
    # Try to extract year
    year_match = re.search(r'Release date:\s*(?:.*?)(\d{4})', text)
    if year_match:
        try:
            year = int(year_match.group(1))
        except ValueError:
            pass
    
    return title, author, year

def calculate_jaccard_similarity(book1, book2):
    """Calculate Jaccard similarity between two books based on indexed words."""
    # Get words from both books
    words1 = set(BookIndex.objects.filter(book=book1).values_list('word', flat=True))
    words2 = set(BookIndex.objects.filter(book=book2).values_list('word', flat=True))
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
    
    return intersection / union

def build_jaccard_graph():
    """Build a graph where books are nodes and edges represent Jaccard similarity."""
    G = nx.Graph()
    
    # Get all books
    books = list(Book.objects.all())
    
    # Add nodes
    for book in books:
        print(f"Adding node for {book.title}")
        G.add_node(book.id, title=book.title, author=book.author)
    
    # Calculate similarity and add edges
    for i, book1 in enumerate(books):
        print(f"Calculating similarity for {book1.title} ({i+1}/{len(books)})")
        for book2 in books[i+1:]:
            print(f"  - with {book2.title}")
            similarity = calculate_jaccard_similarity(book1, book2)
            print(f" - similarity {similarity}")
            if similarity >= JACCARD_SIMILARITY_THRESHOLD:
                G.add_edge(book1.id, book2.id, weight=similarity)
    
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

def regex_search(pattern, book_content):
    """Search for a regex pattern in book content."""
    matches = re.finditer(pattern, book_content, re.IGNORECASE)
    return [match.group() for match in matches]