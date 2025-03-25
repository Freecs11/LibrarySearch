"""
Django management command to find and list books with sufficient words.

This optimized script helps identify Gutenberg books that meet minimum word requirements
with significant performance improvements:
1. Uses local database caching to avoid rechecking books
2. Pre-screens books by file size before downloading content
3. Improved parallelization with batch processing
4. Better targeting of promising book ranges
"""

import os
import re
import sys
import requests
import random
import concurrent.futures
import nltk
import sqlite3
from datetime import datetime
from django.core.management.base import BaseCommand
from tqdm import tqdm

from searchengine.utils import tokenize_text
from searchengine.config import MIN_WORDS_PER_BOOK, GUTENBERG_MIRROR

# Ensure NLTK data is downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Range of Gutenberg book IDs to check
MIN_BOOK_ID = 1
MAX_BOOK_ID = 70000

# Ranges with higher likelihood of having longer books
PROMISING_RANGES = [
    (1, 1000),  # Early classics (often novels by Austen, Dickens, etc.)
    (1000, 2000),  # More classics
    (2000, 3000),  # Many novels
    (10000, 15000),  # Various literary works
    (20000, 25000),  # Some longer texts
    (30000, 35000),  # Various works
    (40000, 45000),  # More modern works
]

# Minimum file size to consider (in bytes) - approximately 50KB
# This helps filter out very short texts without downloading the full content
MIN_FILE_SIZE = 50 * 1024

# Database configuration
DB_FILE = "gutenberg_word_counts.db"


class Command(BaseCommand):
    help = "Find books in Project Gutenberg that meet minimum word count requirements"

    def add_arguments(self, parser):
        parser.add_argument(
            "--count", type=int, default=100, help="Number of good books to find"
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=30,  # Increased to 30 for more parallelism
            help="Maximum number of parallel workers",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="good_books.txt",
            help="Output file to save list of good book IDs",
        )
        parser.add_argument(
            "--random-sample",
            action="store_true",
            help="Use random sampling across all IDs instead of sequential search",
        )
        parser.add_argument(
            "--min-words",
            type=int,
            default=MIN_WORDS_PER_BOOK,
            help=f"Minimum number of words required (default: {MIN_WORDS_PER_BOOK})",
        )
        parser.add_argument(
            "--skip-cache",
            action="store_true",
            help="Skip using the local cache database",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of books to check in each batch",
        )
        parser.add_argument(
            "--category",
            type=str,
            help="Filter by category (e.g., 'novel', 'fiction')",
        )

    def setup_database(self):
        """Set up the SQLite database for caching book information."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            tokens INTEGER,
            raw_words INTEGER,
            file_size INTEGER,
            category TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.commit()
        return conn

    def handle(self, *args, **options):
        count = options["count"]
        max_workers = options["max_workers"]
        output_file = options["output"]
        use_random = options["random_sample"]
        min_words = options["min_words"]
        skip_cache = options["skip_cache"]
        batch_size = options["batch_size"]
        category = options.get("category")

        self.stdout.write(
            f"Looking for {count} books with at least {min_words} words..."
        )

        # Setup database for caching
        if not skip_cache:
            self.conn = self.setup_database()
            self.cursor = self.conn.cursor()

            # First, check if we already have enough books in the database
            query = "SELECT id, title, author, tokens FROM books WHERE tokens >= ?"
            params = [min_words]

            if category:
                query += " AND category LIKE ?"
                params.append(f"%{category}%")

            query += " ORDER BY tokens DESC LIMIT ?"
            params.append(count)

            self.cursor.execute(query, params)

            cached_good_books = [
                {
                    "id": row[0],
                    "title": row[1],
                    "author": row[2],
                    "tokens": row[3],
                    "is_good": True,
                }
                for row in self.cursor.fetchall()
            ]

            if len(cached_good_books) >= count:
                self.stdout.write(
                    f"Found {len(cached_good_books)} books meeting the criteria in the cache!"
                )
                self._save_results(
                    cached_good_books, output_file, len(cached_good_books)
                )
                return
            else:
                self.stdout.write(
                    f"Found {len(cached_good_books)} books in cache, need {count - len(cached_good_books)} more."
                )

            # Get the IDs we've already checked to avoid rechecking
            self.cursor.execute("SELECT id FROM books")
            checked_ids = {row[0] for row in self.cursor.fetchall()}
        else:
            self.conn = None
            checked_ids = set()
            cached_good_books = []

        # Prepare the list of book IDs to check, excluding already checked books
        book_ids_to_check = self._get_book_ids_to_check(
            count * 10, use_random
        )  # Get more books than needed
        book_ids_to_check = [id for id in book_ids_to_check if id not in checked_ids]

        if not book_ids_to_check:
            self.stdout.write("No new books to check!")
            return

        # Process books in batches for better performance
        good_books = cached_good_books.copy()
        total_checked = len(checked_ids)

        for i in range(0, len(book_ids_to_check), batch_size):
            batch = book_ids_to_check[i : i + batch_size]

            new_good_books, batch_checked = self._process_batch(
                batch, min_words, max_workers, category
            )
            good_books.extend(new_good_books)
            total_checked += batch_checked

            # Sort by token count (descending)
            good_books.sort(key=lambda x: x["tokens"], reverse=True)

            # Save intermediate results
            self._save_results(good_books, output_file, total_checked)

            # Check if we have enough books
            if len(good_books) >= count:
                break

            self.stdout.write(
                f"Progress: Found {len(good_books)}/{count} books after checking {total_checked} books..."
            )

        # Close the database connection
        if self.conn:
            self.conn.close()

    def _process_batch(self, book_ids, min_words, max_workers, category=None):
        """Process a batch of book IDs in parallel."""
        good_books = []
        book_count = len(book_ids)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # First, do a preliminary file size check to filter out small files
            size_futures = {
                executor.submit(self._check_file_size, book_id): book_id
                for book_id in book_ids
            }

            # Collect books that pass the file size check
            books_to_analyze = []
            with tqdm(total=book_count, desc="Checking file sizes") as pbar:
                for future in concurrent.futures.as_completed(size_futures):
                    book_id = size_futures[future]
                    try:
                        result = future.result()
                        if result and result["file_size"] >= MIN_FILE_SIZE:
                            books_to_analyze.append(
                                (book_id, result["url"], result["file_size"])
                            )
                    except Exception:
                        pass  # Silently ignore errors in size checking
                    pbar.update(1)

            # Now do the full content analysis only on books that passed the size check
            if not books_to_analyze:
                return [], book_count

            content_futures = {
                executor.submit(
                    self._analyze_book_content,
                    book_id,
                    url,
                    file_size,
                    min_words,
                    category,
                ): book_id
                for book_id, url, file_size in books_to_analyze
            }

            # Process the analysis results
            with tqdm(total=len(books_to_analyze), desc="Analyzing content") as pbar:
                for future in concurrent.futures.as_completed(content_futures):
                    book_id = content_futures[future]

                    try:
                        result = future.result()
                        if result:
                            # Add to database cache if we're using it
                            if self.conn and result.get("tokens"):
                                self._add_to_cache(result)

                            # Add to our results if it's a good book
                            if result.get("is_good"):
                                good_books.append(result)
                                self.stdout.write(
                                    f"Found good book: {result['id']} - {result['title']} - {result['tokens']} tokens"
                                )
                    except Exception as e:
                        self.stdout.write(f"Error analyzing book {book_id}: {str(e)}")

                    pbar.update(1)

        return good_books, book_count

    def _check_file_size(self, book_id):
        """Check the file size of a book before downloading the full content."""
        urls_to_try = [
            f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt",
            f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt.utf8",
            f"{GUTENBERG_MIRROR}/{book_id}/{book_id}.txt",
            f"{GUTENBERG_MIRROR}/{book_id}/{book_id}-0.txt",
        ]

        for url in urls_to_try:
            try:
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        file_size = int(content_length)
                        return {"book_id": book_id, "url": url, "file_size": file_size}
            except Exception:
                continue

        return None  # Could not determine file size

    def _analyze_book_content(self, book_id, url, file_size, min_words, category=None):
        """Analyze the content of a book to check word count."""
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                content = response.text

                # Extract metadata
                title = "Unknown Title"
                author = "Unknown Author"
                book_category = ""

                title_match = re.search(r"Title:\s*([^\r\n]+)", content)
                if title_match:
                    title = title_match.group(1).strip()

                author_match = re.search(r"Author:\s*([^\r\n]+)", content)
                if author_match:
                    author = author_match.group(1).strip()

                # Try to extract category information
                subject_match = re.search(r"Subject:\s*([^\r\n]+)", content)
                if subject_match:
                    book_category = subject_match.group(1).strip()

                # Check word count
                tokens = tokenize_text(content)
                token_count = len(tokens)
                raw_count = len(content.split())

                # Filter by category if specified
                category_match = True
                if category and book_category:
                    category_match = category.lower() in book_category.lower()

                return {
                    "id": book_id,
                    "title": title,
                    "author": author,
                    "tokens": token_count,
                    "raw_words": raw_count,
                    "file_size": file_size,
                    "category": book_category,
                    "is_good": token_count >= min_words and category_match,
                }
        except Exception:
            return None  # Error occurred

        return None  # Should not reach here, but just in case

    def _add_to_cache(self, book_data):
        """Add a book to the cache database."""
        try:
            self.cursor.execute(
                """INSERT OR REPLACE INTO books 
                   (id, title, author, tokens, raw_words, file_size, category) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    book_data["id"],
                    book_data["title"],
                    book_data["author"],
                    book_data["tokens"],
                    book_data["raw_words"],
                    book_data["file_size"],
                    book_data.get("category", ""),
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.stdout.write(f"Error adding book {book_data['id']} to cache: {str(e)}")

    def _get_book_ids_to_check(self, count, use_random):
        """Generate a list of book IDs to check."""
        if use_random:
            # Random sampling across all book IDs
            random_ids = set()

            # Start with promising ranges (higher probability of success)
            promising_ids = []
            for start, end in PROMISING_RANGES:
                promising_ids.extend(range(start, end))

            # Shuffle the promising IDs and take some of them
            random.shuffle(promising_ids)
            random_ids.update(promising_ids[: count // 2])

            # Fill the rest with truly random IDs
            remaining_count = count - len(random_ids)
            if remaining_count > 0:
                random_ids.update(
                    random.sample(
                        range(MIN_BOOK_ID, MAX_BOOK_ID),
                        min(remaining_count, MAX_BOOK_ID - MIN_BOOK_ID),
                    )
                )

            return list(random_ids)
        else:
            # Prioritize promising ranges
            book_ids = []

            # Add all promising ranges
            for start, end in PROMISING_RANGES:
                book_ids.extend(range(start, end))

            # Ensure we have enough IDs
            if len(book_ids) < count:
                # Add some sequential IDs if needed
                for i in range(1, MAX_BOOK_ID + 1, 100):  # Sample every 100th book
                    if i not in book_ids:
                        book_ids.append(i)
                    if len(book_ids) >= count:
                        break

            # Shuffle the IDs to diversify the search
            random.shuffle(book_ids)

            return book_ids[:count]

    def _save_results(self, good_books, output_file, checked_count):
        """Save the results to a file."""
        good_books_count = len(good_books)

        # Sort by token count (descending)
        good_books.sort(key=lambda x: x["tokens"], reverse=True)

        # Save results to file
        with open(output_file, "w") as f:
            f.write(
                f"# Found {good_books_count} good books out of {checked_count} checked\n"
            )
            f.write("# book_id,title,author,tokens\n")
            for book in good_books:
                f.write(
                    f"{book['id']},{book['title'].replace(',', ';')},{book['author'].replace(',', ';')},{book['tokens']}\n"
                )

        # Print summary
        success_rate = (
            (good_books_count / checked_count) * 100 if checked_count > 0 else 0
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Found {good_books_count} books meeting the word requirement "
                f"out of {checked_count} checked ({success_rate:.1f}% success rate)"
            )
        )
        self.stdout.write(f"Results saved to {output_file}")

        # Print the top 10 books
        if good_books:
            self.stdout.write("\nTop 10 books by word count:")
            for i, book in enumerate(good_books[:10]):
                self.stdout.write(
                    f"{i+1}. ID: {book['id']} - '{book['title']}' by {book['author']} - {book['tokens']} tokens"
                )
