"""
Django management command to reindex books from the local books_data folder.
"""

import os
import re
import time
import nltk
from django.core.management.base import BaseCommand
from django.db import transaction

from searchengine.models import Book
from searchengine.utils import (
    extract_metadata_from_gutenberg,
    index_book,
    tokenize_text,
    build_jaccard_graph,
    calculate_pagerank,
    calculate_closeness_centrality,
    calculate_betweenness_centrality,
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
    help = "Reindex books from the local books_data folder"

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

    def handle(self, *args, **options):
        ranking_method = options["ranking_method"]
        skip_centrality = options["skip_centrality"]
        only_calculate_centrality = options["only_calculate_centrality"]

        if only_calculate_centrality:
            self.calculate_centrality(ranking_method)
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

        self.stdout.write(f"Found {len(book_files)} books to index.")

        successful_imports = 0
        for file_name in book_files:
            try:
                # Extract Gutenberg ID from filename
                book_id_match = re.match(r"book_(\d+)\.txt", file_name)
                if not book_id_match:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Could not extract ID from filename {file_name}, skipping."
                        )
                    )
                    continue

                gutenberg_id = int(book_id_match.group(1))
                file_path = os.path.join(BOOKS_STORAGE_PATH, file_name)

                self.import_book(file_path, gutenberg_id)
                successful_imports += 1

                self.stdout.write(
                    self.style.SUCCESS(
                        f"Indexed {successful_imports}/{len(book_files)} books"
                    )
                )

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error processing book {file_name}: {str(e)}")
                )

        self.stdout.write(
            self.style.SUCCESS(f"Successfully indexed {successful_imports} books")
        )

        # Calculate centrality measures if not skipped
        if not skip_centrality:
            self.calculate_centrality(ranking_method)
        else:
            self.stdout.write("Skipping centrality calculation as requested.")

    @transaction.atomic
    def import_book(self, file_path, gutenberg_id):
        """Import a book from a local file."""
        try:
            # Read the book content
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()

            # Check if the book is long enough
            tokens = tokenize_text(content)
            if len(tokens) < MIN_WORDS_PER_BOOK:
                raise ValueError(
                    f"Book {gutenberg_id} is too short ({len(tokens)} words)"
                )

            # Extract metadata
            title, author, year = extract_metadata_from_gutenberg(content)

            # Create the book entry
            book = Book.objects.create(
                title=title,
                author=author,
                publication_year=year,
                text_content=content,
                file_path=file_path,
                gutenberg_id=gutenberg_id,
            )

            # Index the book
            indexed_words = index_book(book)

            self.stdout.write(
                f'Indexed {indexed_words} words for "{title}" by {author}'
            )

            return book

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error processing book {gutenberg_id}: {str(e)}")
            )
            raise

    def calculate_centrality(self, ranking_method):
        """Calculate centrality measures for all books."""
        start_time = time.time()
        self.stdout.write("Building Jaccard similarity graph...")
        G = build_jaccard_graph()
        graph_time = time.time() - start_time
        self.stdout.write(f"Graph building completed in {graph_time:.2f} seconds.")

        # Add timing information
        self.stdout.write(f"Calculating {ranking_method} centrality...")
        centrality_start = time.time()

        if ranking_method == "pagerank":
            calculate_pagerank(G)
        elif ranking_method == "closeness":
            calculate_closeness_centrality(G)
        elif ranking_method == "betweenness":
            calculate_betweenness_centrality(G)
        elif ranking_method == "occurrence":
            self.stdout.write(
                "Using occurrence-based ranking, no centrality calculation needed."
            )

        centrality_time = time.time() - centrality_start
        total_time = time.time() - start_time
        
        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully calculated {ranking_method} centrality in {centrality_time:.2f} seconds."
            )
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Total processing time: {total_time:.2f} seconds."
            )
        )
