"""
Django management command to fetch books from Project Gutenberg.
"""

import time
import random
import nltk
import sys
from django.core.management.base import BaseCommand
from django.db import transaction

from searchengine.models import Book
from searchengine.utils import (
    fetch_book_from_gutenberg,
    extract_metadata_from_gutenberg,
    index_book,
    tokenize_text,
    build_jaccard_graph,
    calculate_pagerank,
    calculate_closeness_centrality,
    calculate_betweenness_centrality,
)
from searchengine.config import (
    MIN_BOOKS_REQUIRED,
    MIN_WORDS_PER_BOOK,
    DEFAULT_RANKING_METHOD,
)

# Ensure NLTK data is downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# List of Gutenberg book IDs to fetch
# These are some popular books from Project Gutenberg
GUTENBERG_POPULAR_BOOKS = [
    1342,  # Pride and Prejudice by Jane Austen
    11,  # Alice's Adventures in Wonderland by Lewis Carroll
    1661,  # The Adventures of Sherlock Holmes by Arthur Conan Doyle
    2701,  # Moby Dick by Herman Melville
    84,  # Frankenstein by Mary Shelley
    98,  # A Tale of Two Cities by Charles Dickens
    1400,  # Great Expectations by Charles Dickens
    1952,  # The Yellow Wallpaper by Charlotte Perkins Gilman
    345,  # Dracula by Bram Stoker
    1080,  # A Modest Proposal by Jonathan Swift
    74,  # The Adventures of Tom Sawyer by Mark Twain
    76,  # Adventures of Huckleberry Finn by Mark Twain
    5200,  # Metamorphosis by Franz Kafka
    2600,  # War and Peace by Leo Tolstoy
    1260,  # Jane Eyre by Charlotte Brontë
    1399,  # Anna Karenina by Leo Tolstoy
    174,  # The Picture of Dorian Gray by Oscar Wilde
    2814,  # Dubliners by James Joyce
    4300,  # Ulysses by James Joyce
    2500,  # Siddhartha by Hermann Hesse
    844,  # The Importance of Being Earnest by Oscar Wilde
    30254,  # The Scarlet Letter by Nathaniel Hawthorne
    158,  # Emma by Jane Austen
    161,  # Sense and Sensibility by Jane Austen
    236,  # The Jungle Book by Rudyard Kipling
    1232,  # The Prince by Niccolò Machiavelli
]

# Generate more book IDs to reach the minimum requirement
ADDITIONAL_BOOK_IDS = list(range(100, 2000)) + list(range(10000, 11000))


class Command(BaseCommand):
    help = "Fetch and index books from Project Gutenberg"

    def add_arguments(self, parser):
        parser.add_argument(
            "--count",
            type=int,
            default=MIN_BOOKS_REQUIRED,
            help=f"Number of books to fetch (minimum {MIN_BOOKS_REQUIRED})",
        )
        parser.add_argument(
            "--recalculate-centrality",
            action="store_true",
            help="Recalculate centrality measures for all books",
        )
        parser.add_argument(
            "--ranking-method",
            type=str,
            default=DEFAULT_RANKING_METHOD,
            choices=["pagerank", "closeness", "betweenness"],
            help="Centrality ranking method to use",
        )

    def handle(self, *args, **options):
        count = max(options["count"], MIN_BOOKS_REQUIRED)
        recalculate = options["recalculate_centrality"]
        ranking_method = options["ranking_method"]

        if recalculate:
            self.recalculate_centrality(ranking_method)
            return

        self.stdout.write(self.style.SUCCESS(f"Starting to fetch {count} books..."))

        # Clean all the books that have less than MIN_WORDS_PER_BOOK
        # self.clean_short_books()

        # Clean the database from books that are not in the books_data folder
        self.clean_db()

        # Combine lists of book IDs
        all_book_ids = GUTENBERG_POPULAR_BOOKS

        # Add additional book IDs if needed
        if len(all_book_ids) < count:
            # Shuffle the additional IDs to get a random selection
            random.shuffle(ADDITIONAL_BOOK_IDS)
            # Add only as many as needed to reach the count
            all_book_ids.extend(ADDITIONAL_BOOK_IDS[: count - len(all_book_ids)])

        # Get existing books to avoid duplicates
        existing_ids = set(Book.objects.values_list("gutenberg_id", flat=True))

        # number of books in the books_data folder
        successful_downloads = Book.objects.count()
        for book_id in all_book_ids:
            if successful_downloads >= count:
                break

            if book_id in existing_ids:
                self.stdout.write(f"Book {book_id} already exists, skipping.")
                continue

            try:
                self.fetch_and_index_book(book_id)
                successful_downloads += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Downloaded {successful_downloads}/{count} books"
                    )
                )

                # Sleep to avoid overwhelming the server
                time.sleep(1)

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error fetching book {book_id}: {str(e)}")
                )

        # Calculate centrality measures
        self.recalculate_centrality(ranking_method)

    @transaction.atomic
    def fetch_and_index_book(self, book_id):
        """Fetch a book from Gutenberg and add it to the database."""
        try:
            # Download the book
            file_path, content = fetch_book_from_gutenberg(book_id)

            # Check if the book is long enough
            tokens = tokenize_text(content)
            if len(tokens) < MIN_WORDS_PER_BOOK:
                raise ValueError(f"Book {book_id} is too short ({len(tokens)} words)")

            # Extract metadata
            title, author, year = extract_metadata_from_gutenberg(content)

            # Create the book entry
            book = Book.objects.create(
                title=title,
                author=author,
                publication_year=year,
                file_path=file_path,
                gutenberg_id=book_id,
            )

            # Index the book
            indexed_words = index_book(book)

            self.stdout.write(
                f'Indexed {indexed_words} words for "{title}" by {author}'
            )

            return book

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error processing book {book_id}: {str(e)}")
            )
            raise

    def clean_short_books(self):
        """Delete books that have fewer words than MIN_WORDS_PER_BOOK."""
        from searchengine.utils import tokenize_text
        import os

        self.stdout.write(
            f"Checking for books with fewer than {MIN_WORDS_PER_BOOK} words..."
        )
        books_to_check = Book.objects.all()
        removed_count = 0

        for book in books_to_check:
            tokens = tokenize_text(book.text_content)

            if len(tokens) < MIN_WORDS_PER_BOOK:
                self.stdout.write(
                    f'Removing book {book.gutenberg_id}: "{book.title}" ({len(tokens)} words)'
                )

                # Delete the file if it exists
                if book.file_path and os.path.exists(book.file_path):
                    try:
                        os.remove(book.file_path)
                    except OSError as e:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Error deleting file {book.file_path}: {str(e)}"
                            )
                        )

                # Delete from database
                book.delete()
                removed_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Removed {removed_count} books below the minimum word threshold"
            )
        )

    # clean db from books that are not in the books_data folder
    def clean_db(self):
        """Delete books that are not in the books_data folder."""
        import os

        self.stdout.write("Checking for books not in the books_data folder...")
        books_to_check = Book.objects.all()
        removed_count = 0

        for book in books_to_check:
            if not book.file_path or not os.path.exists(book.file_path):
                self.stdout.write(
                    f'Removing book {book.gutenberg_id}: "{book.title}" (not found)'
                )
                book.delete()
                removed_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Removed {removed_count} books not found in the books_data folder"
            )
        )

    def recalculate_centrality(self, ranking_method):
        """Recalculate centrality measures for all books."""
        self.stdout.write("Building Jaccard similarity graph...")
        G = build_jaccard_graph()

        self.stdout.write(f"Calculating {ranking_method} centrality...")

        if ranking_method == "pagerank":
            calculate_pagerank(G)
        elif ranking_method == "closeness":
            calculate_closeness_centrality(G)
        elif ranking_method == "betweenness":
            calculate_betweenness_centrality(G)

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully calculated {ranking_method} centrality for all books"
            )
        )
