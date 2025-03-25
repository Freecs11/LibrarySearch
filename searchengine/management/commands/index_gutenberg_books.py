"""
Django management command to index downloaded Gutenberg books with sqlite concurrency fixes.

This script:
1. Scans the books directory for downloaded books
2. Extracts metadata from each book
3. Creates database entries and indexes content
4. Calculates book similarity and centrality metrics
5. Implements proper database locking protections
"""

import os
import time
import re
import concurrent.futures
import threading
import queue
import random
from django.core.management.base import BaseCommand
from django.db import transaction, connection, OperationalError
from tqdm import tqdm

from searchengine.models import Book
from searchengine.utils import (
    tokenize_text,
    extract_metadata_from_gutenberg,
    index_book,
    build_jaccard_graph,
    calculate_pagerank,
    calculate_closeness_centrality,
    calculate_betweenness_centrality,
)
from searchengine.config import (
    BOOKS_STORAGE_PATH,
    MIN_WORDS_PER_BOOK,
    DEFAULT_RANKING_METHOD,
    MAX_GRAPH_WORKERS,
)


class Command(BaseCommand):
    help = "Index downloaded Gutenberg books with database lock protection"

    def add_arguments(self, parser):
        parser.add_argument(
            "--books-dir",
            type=str,
            default=BOOKS_STORAGE_PATH,
            help="Directory containing downloaded books",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=20,
            help="Maximum number of parallel content processing workers",
        )
        parser.add_argument(
            "--db-workers",
            type=int,
            default=1,
            help="Maximum number of database workers (keep at 1 for SQLite)",
        )
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Skip books that are already indexed in the database",
        )
        parser.add_argument(
            "--skip-centrality",
            action="store_true",
            help="Skip centrality calculation after indexing",
        )

        parser.add_argument(
            "--ranking-method",
            type=str,
            default=DEFAULT_RANKING_METHOD,
            choices=["pagerank", "closeness", "betweenness"],
            help="Centrality ranking method to use",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50,
            help="Number of books to process in each batch to save memory",
        )
        parser.add_argument(
            "--specific-books",
            type=str,
            help="Comma-separated list of specific book IDs to index",
        )
        parser.add_argument(
            "--verify-length",
            action="store_true",
            help="Verify book length meets minimum requirements",
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=5,
            help="Maximum retries for database operations when locked",
        )
        parser.add_argument(
            "--limit_book_number",
            type=int,
            default=100,
            help="Limit the number of books to be indexed",
        )

    def handle(self, *args, **options):
        self.books_dir = options["books_dir"]
        self.max_workers = options["max_workers"]
        self.db_workers = options["db_workers"]
        self.skip_existing = options["skip_existing"]
        self.skip_centrality = options["skip_centrality"]
        self.ranking_method = options["ranking_method"]
        self.batch_size = options["batch_size"]
        self.specific_books = options.get("specific_books")
        self.verify_length = options["verify_length"]
        self.max_retries = options["max_retries"]
        self.limit_book_number = options["limit_book_number"]

        # Database operations lock
        self.db_lock = threading.Lock()

        # Database operations queue
        self.db_queue = queue.Queue()

        # Check if the books directory exists
        if not os.path.exists(self.books_dir):
            self.stdout.write(
                self.style.ERROR(f"Books directory {self.books_dir} does not exist!")
            )
            return

        # Get list of book files
        all_files = os.listdir(self.books_dir)
        book_files = [
            f for f in all_files if f.startswith("book_") and f.endswith(".txt")
        ]

        # Limit the number of books to be indexed
        if self.limit_book_number:
            # get the last indexed book id from the database ( )
            last_indexed_book_id = Book.objects.count() + 1
            book_files = book_files[
                last_indexed_book_id : last_indexed_book_id + self.limit_book_number
            ]

        if not book_files:
            self.stdout.write(
                self.style.WARNING(f"No book files found in {self.books_dir}!")
            )
            return

        self.stdout.write(f"Found {len(book_files)} book files in {self.books_dir}")

        # Filter to specific books if requested
        if self.specific_books:
            specific_ids = [int(id.strip()) for id in self.specific_books.split(",")]
            book_files = [
                f for f in book_files if self._extract_book_id(f) in specific_ids
            ]
            self.stdout.write(f"Filtered to {len(book_files)} specific books")

        # Get existing book IDs from database if skipping existing
        existing_book_ids = set()
        if self.skip_existing:
            try:
                with self.db_lock:
                    existing_book_ids = set(
                        Book.objects.values_list("gutenberg_id", flat=True)
                    )
                self.stdout.write(
                    f"Found {len(existing_book_ids)} existing books in database"
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error accessing database: {str(e)}")
                )
                return

        # Filter out already indexed books if requested
        if self.skip_existing:
            book_files = [
                f
                for f in book_files
                if self._extract_book_id(f) not in existing_book_ids
            ]
            self.stdout.write(f"{len(book_files)} books need indexing")

        if not book_files:
            self.stdout.write(self.style.SUCCESS("No new books to index!"))
            return

        # Start database worker threads
        db_threads = []
        for i in range(self.db_workers):
            t = threading.Thread(target=self._database_worker)
            t.daemon = True
            t.start()
            db_threads.append(t)

        # Process books in batches to save memory
        books_to_process = book_files.copy()
        random.shuffle(
            books_to_process
        )  # Shuffle to reduce chance of concurrent access to same DB rows

        indexed_count = 0
        total_books = len(books_to_process)

        for batch_start in range(0, total_books, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_books)
            batch = books_to_process[batch_start:batch_end]

            self.stdout.write(
                f"Processing batch {batch_start//self.batch_size + 1} ({len(batch)} books)..."
            )
            newly_indexed = self._process_batch(batch)
            indexed_count += newly_indexed

            # Print progress
            progress = (batch_end / total_books) * 100
            self.stdout.write(
                f"Progress: {batch_end}/{total_books} ({progress:.1f}%) - {indexed_count} books indexed"
            )

        # Wait for all database operations to complete
        self.db_queue.join()

        # Signal database workers to exit
        for _ in range(self.db_workers):
            self.db_queue.put(None)

        # Wait for all database threads to exit
        for t in db_threads:
            t.join()

        self.stdout.write(
            self.style.SUCCESS(f"Successfully indexed {indexed_count} books")
        )

        # Calculate centrality if not skipped and we have at least 2 books
        if not self.skip_centrality:
            try:
                book_count = Book.objects.count()
                if book_count >= 2:
                    self._calculate_centrality(self.ranking_method)
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            "Not enough books to calculate centrality measures (need at least 2)"
                        )
                    )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error calculating centrality: {str(e)}")
                )

    def _database_worker(self):
        """Worker thread that processes database operations from the queue."""
        while True:
            item = self.db_queue.get()
            if item is None:  # Sentinel to exit
                self.db_queue.task_done()
                break

            operation, args = item

            # Reset Django's database connection to avoid sharing connections between threads
            connection.close()

            retries = 0
            while retries < self.max_retries:
                try:
                    if operation == "add_book":
                        self._add_book_to_database(*args)
                    # Add other database operations here if needed
                    break  # Success, exit retry loop
                except OperationalError as e:
                    if "database is locked" in str(e).lower():
                        retries += 1
                        # Exponential backoff with jitter
                        sleep_time = (2**retries) * 0.1 + (random.random() * 0.5)
                        time.sleep(sleep_time)
                    else:
                        # Some other database error
                        self.stdout.write(self.style.ERROR(f"Database error: {str(e)}"))
                        break
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"Error in database operation: {str(e)}")
                    )
                    break

            if retries == self.max_retries:
                self.stdout.write(
                    self.style.ERROR(f"Max retries reached for database operation")
                )

            self.db_queue.task_done()

    def _add_book_to_database(self, book_data):
        """Add a book to the database and index it."""
        try:
            with transaction.atomic():
                # Check if book already exists
                if Book.objects.filter(gutenberg_id=book_data["book_id"]).exists():
                    return {"status": "exists", "message": "Book already in database"}

                # Create the book entry
                book = Book.objects.create(
                    title=book_data["title"],
                    author=book_data["author"],
                    publication_year=book_data["year"],
                    file_path=book_data["filepath"],
                    gutenberg_id=book_data["book_id"],
                )

                # Index the book
                indexed_words = index_book(book)

                return {
                    "status": "success",
                    "book_id": book.id,
                    "indexed_words": indexed_words,
                }
        except Exception as e:
            raise e  # Re-raise to be caught by the worker's exception handler

    def _extract_book_id(self, filename):
        """Extract book ID from filename."""
        match = re.match(r"book_(\d+)\.txt", filename)
        if match:
            return int(match.group(1))
        return None

    def _process_batch(self, batch_files):
        """Process a batch of books in parallel."""
        indexed_count = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Create content processing tasks (not database operations)
            future_to_file = {
                executor.submit(self._process_book_content, filename): filename
                for filename in batch_files
            }

            # Process results with progress bar
            with tqdm(total=len(future_to_file), desc="Processing books") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    filename = future_to_file[future]

                    try:
                        result = future.result()

                        if result["status"] == "ready_for_db":
                            # Queue the database operation instead of doing it directly
                            self.db_queue.put(("add_book", (result,)))
                            indexed_count += 1
                            self.stdout.write(
                                f"Queued {result['book_id']}: '{result['title']}' ({result['words']} words)"
                            )
                        elif result["status"] == "too_short":
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Book {result['book_id']} too short ({result['words']} words < {MIN_WORDS_PER_BOOK})"
                                )
                            )
                        elif result["status"] == "error":
                            self.stdout.write(
                                self.style.ERROR(
                                    f"Error with book {result['book_id']}: {result['error']}"
                                )
                            )
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"Error processing {filename}: {str(e)}")
                        )

                    pbar.update(1)

        return indexed_count

    def _process_book_content(self, filename):
        """Process a single book file, extracting content and metadata but not touching database."""
        filepath = os.path.join(self.books_dir, filename)
        book_id = self._extract_book_id(filename)

        if book_id is None:
            return {
                "status": "error",
                "error": f"Could not extract book ID from filename: {filename}",
                "book_id": "unknown",
            }

        try:
            # Read book content
            with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()

            # Check book length if verification is enabled
            if self.verify_length:
                tokens = tokenize_text(content)
                if len(tokens) < MIN_WORDS_PER_BOOK:
                    return {
                        "status": "too_short",
                        "book_id": book_id,
                        "words": len(tokens),
                    }

            # Extract metadata
            title, author, year = extract_metadata_from_gutenberg(content)

            # Only analyze the book, don't do database operations here
            tokens = tokenize_text(content)

            return {
                "status": "ready_for_db",
                "book_id": book_id,
                "title": title,
                "author": author,
                "year": year,
                "words": len(tokens),
                "filepath": filepath,
            }

        except Exception as e:
            return {"status": "error", "book_id": book_id, "error": str(e)}

    def _calculate_centrality(self, ranking_method):
        """Calculate centrality measures for all books."""
        self.stdout.write("Building Jaccard similarity graph...")
        start_time = time.time()

        # Build the graph - this is read-only so shouldn't cause locking
        G = build_jaccard_graph()
        graph_time = time.time() - start_time

        self.stdout.write(f"Graph building completed in {graph_time:.2f} seconds")
        self.stdout.write(f"Calculating {ranking_method} centrality...")

        # Calculate centrality
        centrality_start = time.time()

        # These operations update the database, so protect with lock and retry logic
        retries = 0
        success = False

        while retries < self.max_retries and not success:
            try:
                if ranking_method == "pagerank":
                    calculate_pagerank(G)
                elif ranking_method == "closeness":
                    calculate_closeness_centrality(G)
                elif ranking_method == "betweenness":
                    calculate_betweenness_centrality(G)
                success = True
            except OperationalError as e:
                if "database is locked" in str(e).lower():
                    retries += 1
                    sleep_time = (2**retries) * 0.1 + (random.random() * 0.5)
                    self.stdout.write(
                        f"Database locked, retrying in {sleep_time:.2f} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    raise

        if not success:
            self.stdout.write(
                self.style.ERROR("Failed to calculate centrality after max retries")
            )
            return

        centrality_time = time.time() - centrality_start
        total_time = time.time() - start_time

        self.stdout.write(
            self.style.SUCCESS(
                f"Calculated {ranking_method} centrality in {centrality_time:.2f} seconds"
            )
        )
        self.stdout.write(
            self.style.SUCCESS(f"Total processing time: {total_time:.2f} seconds")
        )
