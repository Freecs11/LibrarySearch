"""
Django management command to reindex books from the local books_data folder.
- Optimized for SQLite performance
- Uses batch processing for faster database operations
- Implements proper concurrency controls
"""

import os
import re
import time
import nltk
import threading
import queue
import random
import concurrent.futures
import sqlite3
from django.core.management.base import BaseCommand
from django.db import transaction, connection, OperationalError
from tqdm import tqdm

from searchengine.models import Book
from searchengine.utils import (
    extract_metadata_from_gutenberg,
    index_book,
    tokenize_text,
    build_jaccard_graph,
    calculate_pagerank,
    calculate_closeness_centrality,
    calculate_betweenness_centrality,
    MIN_WORD_LENGTH,
)
from searchengine.config import (
    BOOKS_STORAGE_PATH,
    MIN_WORDS_PER_BOOK,
    DEFAULT_RANKING_METHOD,
)

# Ensure NLTK data is downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class Command(BaseCommand):
    help = "Reindex books with optimized database performance"

    def add_arguments(self, parser):
        parser.add_argument(
            "--ranking-method",
            type=str,
            default=DEFAULT_RANKING_METHOD,
            choices=["pagerank", "closeness", "betweenness", "occurrence"],
            help="Centrality ranking method to use",
        )
        parser.add_argument(
            "--skip-centrality",
            action="store_true",
            help="Skip centrality calculation (useful for large libraries)",
        )
        parser.add_argument(
            "--only-calculate-centrality",
            action="store_true",
            help="Only calculate centrality measures for existing books",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=20,
            help="Maximum number of parallel content processing workers",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of books to process in each batch",
        )
        parser.add_argument(
            "--db-batch-size",
            type=int,
            default=5,
            help="Number of books to add to database in a single transaction",
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=5,
            help="Maximum number of retries for database operations",
        )
        parser.add_argument(
            "--verify-length",
            action="store_true",
            help="Verify book length meets minimum requirements",
        )
        parser.add_argument(
            "--specific-books",
            type=str,
            help="Comma-separated list of specific book IDs to index",
        )
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Skip books that are already in the database",
        )
        parser.add_argument(
            "--optimize-db",
            action="store_true",
            help="Optimize SQLite database before starting",
        )
        parser.add_argument(
            "--min-word-length",
            type=int,
            default=MIN_WORD_LENGTH,
            help=f"Minimum word length to index (default: {MIN_WORD_LENGTH})",
        )
        parser.add_argument(
            "--reindex-all",
            action="store_true",
            help="Reindex all books, including those already indexed",
        )
        parser.add_argument(
            "--number_books_to_index",
            type=int,
            default=20,
            help="Number of books to index (for testing purposes)",
        )

    def handle(self, *args, **options):
        self.ranking_method = options["ranking_method"]
        self.skip_centrality = options["skip_centrality"]
        self.only_calculate_centrality = options["only_calculate_centrality"]
        self.max_workers = options["max_workers"]
        self.batch_size = options["batch_size"]
        self.db_batch_size = options["db_batch_size"]
        self.max_retries = options["max_retries"]
        self.verify_length = options["verify_length"]
        self.specific_books = options.get("specific_books")
        self.skip_existing = options["skip_existing"]
        self.optimize_db = options["optimize_db"]
        self.min_word_length = options["min_word_length"]
        self.reindex_all = options["reindex_all"]
        self.number_books_to_index = options["number_books_to_index"]

        # If custom min word length is used, display it
        if self.min_word_length != MIN_WORD_LENGTH:
            self.stdout.write(
                self.style.WARNING(
                    f"Using custom minimum word length: {self.min_word_length}"
                )
            )

        # Set up database queue with batch accumulation
        self.db_queue = queue.Queue()
        self.db_lock = threading.Lock()
        self.results_ready = threading.Event()

        # Optimize SQLite if requested
        if self.optimize_db:
            self._optimize_database()

        # Handle centrality-only mode
        if self.only_calculate_centrality:
            self.calculate_centrality(self.ranking_method)
            return

        self.stdout.write(self.style.SUCCESS("Starting to reindex local books..."))

        # Check if books_data directory exists
        if not os.path.exists(BOOKS_STORAGE_PATH):
            self.stdout.write(
                self.style.ERROR(
                    f"Books directory {BOOKS_STORAGE_PATH} does not exist!"
                )
            )
            return

        # Get a list of all book files
        book_files = [f for f in os.listdir(BOOKS_STORAGE_PATH) if f.endswith(".txt")]

        if not book_files:
            self.stdout.write(
                self.style.ERROR(f"No book files found in {BOOKS_STORAGE_PATH}!")
            )
            return

        # Limit the number of books to index if requested
        if self.number_books_to_index:
            book_files = book_files[: self.number_books_to_index]
            self.stdout.write(f"Indexing {self.number_books_to_index} books")

        # Filter to specific books if requested
        if self.specific_books:
            specific_ids = [int(id.strip()) for id in self.specific_books.split(",")]
            book_files = [
                f for f in book_files if self._extract_book_id(f) in specific_ids
            ]
            self.stdout.write(f"Filtered to {len(book_files)} specific books")

        # Get existing book IDs if we need to skip them
        existing_ids = set()
        if self.skip_existing and not self.reindex_all:
            try:
                with self.db_lock:
                    existing_ids = set(
                        Book.objects.values_list("gutenberg_id", flat=True)
                    )
                self.stdout.write(
                    f"Found {len(existing_ids)} existing books in database"
                )

                # Filter out books that already exist
                book_files = [
                    f
                    for f in book_files
                    if self._extract_book_id(f) not in existing_ids
                ]
                self.stdout.write(f"{len(book_files)} books need indexing")
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error accessing database: {str(e)}")
                )
        elif self.reindex_all:
            # If reindex_all is set, delete all existing book indices
            self.stdout.write(
                self.style.WARNING("Reindexing ALL books with improved stemming...")
            )
            try:
                with self.db_lock:
                    from searchengine.models import BookIndex

                    count = BookIndex.objects.count()
                    BookIndex.objects.all().delete()
                    self.stdout.write(
                        f"Deleted {count} existing book indices for reindexing"
                    )
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error clearing indices: {str(e)}"))

        if not book_files:
            self.stdout.write(self.style.SUCCESS("No new books to index!"))
            return

        self.stdout.write(f"Found {len(book_files)} books to index.")

        # Start database worker thread
        db_thread = threading.Thread(target=self._database_batch_worker)
        db_thread.daemon = True
        db_thread.start()

        # Process books in batches
        # Shuffle to reduce database contention
        random.shuffle(book_files)

        total_indexed = 0
        for i in range(0, len(book_files), self.batch_size):
            batch = book_files[i : i + self.batch_size]
            self.stdout.write(
                f"Processing batch {i//self.batch_size + 1} ({len(batch)} books)..."
            )

            indexed = self._process_batch(batch)
            total_indexed += indexed

            progress = min(100, (i + len(batch)) / len(book_files) * 100)
            self.stdout.write(
                f"Progress: {i + len(batch)}/{len(book_files)} ({progress:.1f}%) - {total_indexed} books indexed"
            )

        # Wait for all database operations to complete
        self.results_ready.set()  # Signal that no more items are coming
        self.db_queue.join()

        # Signal database worker to exit
        self.db_queue.put(None)
        db_thread.join()

        self.stdout.write(
            self.style.SUCCESS(f"Successfully indexed {total_indexed} books")
        )

        # Calculate centrality measures if not skipped
        if not self.skip_centrality:
            try:
                book_count = Book.objects.count()
                if book_count >= 2:
                    self.calculate_centrality(self.ranking_method)
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
        else:
            self.stdout.write("Skipping centrality calculation as requested.")

    def _optimize_database(self):
        """Apply SQLite optimizations."""
        self.stdout.write("Optimizing SQLite database...")

        # Get the database file path
        db_file = connection.settings_dict["NAME"]

        try:
            # Connect directly to SQLite to apply pragmas
            conn = sqlite3.connect(db_file)
            c = conn.cursor()

            # Enable WAL mode for better concurrency
            c.execute("PRAGMA journal_mode=WAL")

            # Set pragmas for better performance
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA cache_size=10000")
            c.execute("PRAGMA temp_store=MEMORY")
            c.execute("PRAGMA mmap_size=30000000000")

            # Run VACUUM to defragment
            c.execute("VACUUM")

            # Analyze to update statistics
            c.execute("ANALYZE")

            conn.close()

            self.stdout.write(self.style.SUCCESS("Database optimization completed"))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Database optimization failed: {str(e)}")
            )

    def _extract_book_id(self, filename):
        """Extract book ID from filename."""
        match = re.match(r"book_(\d+)\.txt", filename)
        if match:
            return int(match.group(1))
        return None

    def _database_batch_worker(self):
        """Worker thread that processes database operations in batches."""
        batch_buffer = []
        last_flush_time = time.time()

        while True:
            try:
                # Wait for item with timeout to allow periodic flushing
                try:
                    item = self.db_queue.get(timeout=5)
                except queue.Empty:
                    # No new items, check if we should flush the buffer
                    current_time = time.time()
                    if len(batch_buffer) > 0 and (
                        current_time - last_flush_time > 10
                        or (self.results_ready.is_set() and len(batch_buffer) > 0)
                    ):
                        self._process_book_batch(batch_buffer)
                        batch_buffer = []
                        last_flush_time = current_time
                    continue

                if item is None:  # Sentinel to exit
                    # Flush any remaining items
                    if batch_buffer:
                        self._process_book_batch(batch_buffer)
                    self.db_queue.task_done()
                    break

                # Add to batch buffer
                batch_buffer.append(item)

                # Flush when batch size is reached
                if len(batch_buffer) >= self.db_batch_size:
                    self._process_book_batch(batch_buffer)
                    batch_buffer = []
                    last_flush_time = time.time()

                self.db_queue.task_done()

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error in database batch worker: {str(e)}")
                )
                if batch_buffer:
                    # Mark items as done to avoid blocking
                    for _ in range(len(batch_buffer)):
                        self.db_queue.task_done()
                    batch_buffer = []

    def _process_book_batch(self, batch):
        """Process a batch of books in a single transaction."""
        if not batch:
            return

        # Reset Django's database connection
        connection.close()

        retries = 0
        while retries < self.max_retries:
            try:
                with transaction.atomic():
                    # Process each book in the batch within a single transaction
                    for book_data in batch:
                        # Check if book already exists
                        existing_book = Book.objects.filter(
                            gutenberg_id=book_data["gutenberg_id"]
                        ).first()

                        if existing_book and not self.reindex_all:
                            continue
                        elif existing_book and self.reindex_all:
                            # If reindexing all, use existing book but reindex
                            book = existing_book
                            # Delete any existing indices for this book
                            from searchengine.models import BookIndex

                            BookIndex.objects.filter(book=book).delete()
                        else:
                            # Create new book entry
                            book = Book.objects.create(
                                title=book_data["title"],
                                author=book_data["author"],
                                publication_year=book_data["year"],
                                file_path=book_data["file_path"],
                                gutenberg_id=book_data["gutenberg_id"],
                            )

                        # Index the book using the existing utility function with stemming
                        index_book(book)

                return True
            except OperationalError as e:
                if "database is locked" in str(e).lower():
                    retries += 1
                    sleep_time = (2**retries) * 0.1 + (random.random() * 0.5)
                    time.sleep(sleep_time)
                else:
                    self.stdout.write(self.style.ERROR(f"Database error: {str(e)}"))
                    return False
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error processing book batch: {str(e)}")
                )
                return False

        if retries == self.max_retries:
            self.stdout.write(self.style.ERROR(f"Max retries reached for book batch"))
            return False

    def _process_batch(self, batch_files):
        """Process a batch of book files in parallel."""
        indexed_count = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit tasks to analyze book content
            future_to_file = {
                executor.submit(self._analyze_book_content, file_name): file_name
                for file_name in batch_files
            }

            # Process the results as they complete
            with tqdm(total=len(future_to_file), desc="Analyzing books") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    file_name = future_to_file[future]

                    try:
                        result = future.result()

                        if result["status"] == "ready_for_db":
                            # Add to database queue for batch processing
                            self.db_queue.put(result)
                            indexed_count += 1
                        elif result["status"] == "too_short":
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Book {result['gutenberg_id']} too short ({result['word_count']} words)"
                                )
                            )
                        elif result["status"] == "error":
                            self.stdout.write(
                                self.style.ERROR(
                                    f"Error with book {file_name}: {result['error']}"
                                )
                            )
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"Error processing {file_name}: {str(e)}")
                        )

                    pbar.update(1)

        return indexed_count

    def _analyze_book_content(self, file_name):
        """Analyze book content without touching the database."""
        try:
            # Extract Gutenberg ID from filename
            book_id_match = re.match(r"book_(\d+)\.txt", file_name)
            if not book_id_match:
                return {
                    "status": "error",
                    "error": f"Could not extract ID from filename {file_name}",
                }

            gutenberg_id = int(book_id_match.group(1))
            file_path = os.path.join(BOOKS_STORAGE_PATH, file_name)

            # Read the book content
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()

            # Get tokens and check length, applying stemming
            tokens = tokenize_text(content, apply_stemming=True)

            # Filter out tokens that are too short based on min_word_length
            filtered_tokens = [
                token for token in tokens if len(token) >= self.min_word_length
            ]

            if self.verify_length and len(filtered_tokens) < MIN_WORDS_PER_BOOK:
                return {
                    "status": "too_short",
                    "gutenberg_id": gutenberg_id,
                    "word_count": len(filtered_tokens),
                }

            # Extract metadata
            title, author, year = extract_metadata_from_gutenberg(content)

            return {
                "status": "ready_for_db",
                "gutenberg_id": gutenberg_id,
                "title": title,
                "author": author,
                "year": year,
                "file_path": file_path,
                "word_count": len(filtered_tokens),
                "min_word_length": self.min_word_length,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def calculate_centrality(self, ranking_method):
        """Calculate centrality measures for all books."""
        start_time = time.time()
        self.stdout.write("Building Jaccard similarity graph...")

        # Clear existing similarity relationships
        try:
            from searchengine.models import BookSimilarity

            self.stdout.write("Clearing existing Jaccard graph data...")
            BookSimilarity.objects.all().delete()
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error clearing existing graph data: {str(e)}")
            )

        # Build the graph - the updated function will now store edges in the database
        G = build_jaccard_graph()
        graph_time = time.time() - start_time
        self.stdout.write(f"Graph building completed in {graph_time:.2f} seconds.")

        # Don't calculate if using occurrence ranking
        if ranking_method == "occurrence":
            self.stdout.write(
                "Using occurrence-based ranking, no centrality calculation needed."
            )
            return

        # Add timing information
        self.stdout.write(f"Calculating {ranking_method} centrality...")
        centrality_start = time.time()

        # Apply centrality calculation with retry logic
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

        # Count the number of similarity edges stored in the database
        try:
            from searchengine.models import BookSimilarity

            edge_count = BookSimilarity.objects.count()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Stored {edge_count} Jaccard graph edges in the database"
                )
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error counting graph edges: {str(e)}"))

        centrality_time = time.time() - centrality_start
        total_time = time.time() - start_time

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully calculated {ranking_method} centrality in {centrality_time:.2f} seconds."
            )
        )
        self.stdout.write(
            self.style.SUCCESS(f"Total processing time: {total_time:.2f} seconds.")
        )
