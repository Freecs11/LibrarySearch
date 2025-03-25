"""
Django management command to download books from Project Gutenberg efficiently.
- Reads book IDs from good_books.txt file created by the find_good_books command
- Uses parallel processing for faster downloads
- Simple and focused on downloading only
"""

import os
import csv
import concurrent.futures
import requests
from tqdm import tqdm
from django.core.management.base import BaseCommand

from searchengine.config import BOOKS_STORAGE_PATH, GUTENBERG_MIRROR


class Command(BaseCommand):
    help = "Download books from Project Gutenberg using IDs from good_books.txt"

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            default="good_books.txt",
            help="Input file containing book IDs to download",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=BOOKS_STORAGE_PATH,
            help="Directory to save downloaded books",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=60,
            help="Maximum number of parallel download workers",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of books to download (default: all in input file)",
        )
        parser.add_argument(
            "--verify",
            action="store_true",
            help="Verify if files already exist before downloading",
        )
        parser.add_argument(
            "--retry",
            type=int,
            default=3,
            help="Number of retry attempts for failed downloads",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=60,
            help="Download timeout in seconds",
        )
        parser.add_argument(
            "--format",
            type=str,
            default="txt",
            choices=["txt", "epub", "all"],
            help="Format of books to download",
        )

    def handle(self, *args, **options):
        input_file = options["input"]
        output_dir = options["output_dir"]
        max_workers = options["max_workers"]
        limit = options["limit"]
        verify = options["verify"]
        retry_count = options["retry"]
        timeout = options["timeout"]
        book_format = options["format"]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Read book IDs from input file
        book_ids = self._read_book_ids(input_file, limit)
        total_books = len(book_ids)

        if total_books == 0:
            self.stdout.write(self.style.WARNING(f"No book IDs found in {input_file}"))
            return

        self.stdout.write(f"Found {total_books} books to download")

        # Check which books already exist if verify is enabled
        if verify:
            existing_files = os.listdir(output_dir)
            existing_ids = set()
            for filename in existing_files:
                if filename.startswith("book_") and filename.endswith(".txt"):
                    try:
                        book_id = int(filename.replace("book_", "").replace(".txt", ""))
                        existing_ids.add(book_id)
                    except ValueError:
                        continue

            # Filter out books that already exist
            filtered_book_ids = [id for id in book_ids if id not in existing_ids]

            skipped = total_books - len(filtered_book_ids)
            self.stdout.write(f"Skipping {skipped} books that already exist")
            book_ids = filtered_book_ids

            if not book_ids:
                self.stdout.write(self.style.SUCCESS("All books already downloaded!"))
                return

        # Start parallel downloading
        successful = 0
        failed = 0
        skipped = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create download tasks
            future_to_book_id = {
                executor.submit(
                    self._download_book,
                    book_id,
                    output_dir,
                    retry_count,
                    timeout,
                    book_format,
                ): book_id
                for book_id in book_ids
            }

            # Process results with progress bar
            with tqdm(total=len(future_to_book_id), desc="Downloading books") as pbar:
                for future in concurrent.futures.as_completed(future_to_book_id):
                    book_id = future_to_book_id[future]

                    try:
                        result = future.result()

                        if result["status"] == "success":
                            successful += 1
                            self.stdout.write(
                                f"Downloaded book {book_id} to {result['filepath']}"
                            )
                        elif result["status"] == "exists":
                            skipped += 1
                            self.stdout.write(
                                f"Book {book_id} already exists at {result['filepath']}"
                            )
                        else:
                            failed += 1
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Failed to download book {book_id}: {result['error']}"
                                )
                            )
                    except Exception as e:
                        failed += 1
                        self.stdout.write(
                            self.style.ERROR(
                                f"Error processing book {book_id}: {str(e)}"
                            )
                        )

                    pbar.update(1)

        # Print summary
        self.stdout.write(
            self.style.SUCCESS(
                f"Download summary: {successful} successful, {skipped} skipped, {failed} failed"
            )
        )

    def _read_book_ids(self, input_file, limit=None):
        """Read book IDs from the input file."""
        book_ids = []

        try:
            with open(input_file, "r") as f:
                # Skip first lines that start with #
                for line in f:
                    if not line.startswith("#"):
                        parts = line.strip().split(",")
                        if parts:
                            try:
                                book_id = int(parts[0])
                                book_ids.append(book_id)
                            except ValueError:
                                continue
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"Input file {input_file} not found."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error reading input file: {str(e)}"))

        # Apply limit if specified
        if limit is not None and limit > 0:
            book_ids = book_ids[:limit]

        return book_ids

    def _download_book(
        self, book_id, output_dir, retry_count=3, timeout=60, book_format="txt"
    ):
        """Download a book from Project Gutenberg."""
        # Define output file path
        filepath = os.path.join(output_dir, f"book_{book_id}.txt")

        # Check if the file already exists
        if os.path.exists(filepath):
            return {"status": "exists", "filepath": filepath}

        # Try different URL formats from Project Gutenberg
        urls_to_try = []

        if book_format in ["txt", "all"]:
            urls_to_try.extend(
                [
                    f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt",  # Standard format
                    f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt.utf8",  # UTF-8 format
                    f"{GUTENBERG_MIRROR}/{book_id}/{book_id}.txt",  # Alternative format
                    f"{GUTENBERG_MIRROR}/{book_id}/{book_id}-0.txt",  # Another alternative
                ]
            )

        if book_format in ["epub", "all"]:
            urls_to_try.extend(
                [
                    f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.epub",  # EPUB format
                ]
            )

        # Try each URL, with retries
        for url in urls_to_try:
            attempts = 0
            while attempts < retry_count:
                try:
                    response = requests.get(url, timeout=timeout)
                    if response.status_code == 200:
                        content = response.text

                        # Save the book locally
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)

                        return {
                            "status": "success",
                            "filepath": filepath,
                            "book_id": book_id,
                            "url": url,
                        }
                except requests.Timeout:
                    attempts += 1
                    continue  # Try again
                except requests.RequestException as e:
                    break  # Try next URL

        return {
            "status": "failed",
            "error": "All download attempts failed",
            "book_id": book_id,
        }
