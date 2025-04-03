import sqlite3
import os
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from searchengine.models import Book

class Command(BaseCommand):
    help = 'Verifies the optimal_index table and repairs any issues'

    def handle(self, *args, **options):
        # Connect to the database
        db_path = os.path.join(settings.BASE_DIR, 'db.sqlite3')
        self.stdout.write(f"Connecting to database at {db_path}")
        
        try:
            conn = sqlite3.connect(db_path, timeout=60)
            cursor = conn.cursor()
            
            # Check if the optimal_index table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='optimal_index'")
            if not cursor.fetchone():
                self.stdout.write(self.style.ERROR("The optimal_index table does not exist!"))
                self.create_optimal_index_table(cursor)
                conn.commit()
            
            # Check the structure of the table
            cursor.execute("PRAGMA table_info(optimal_index)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}
            
            expected_columns = {
                'word': 'TEXT',
                'book_ids': 'TEXT',
                'book_counts': 'TEXT',
                'total_occurrences': 'INTEGER'
            }
            
            # Verify columns
            missing_columns = [col for col in expected_columns if col not in columns]
            if missing_columns:
                self.stdout.write(self.style.WARNING(f"Missing columns: {missing_columns}"))
                self.repair_table_structure(cursor, missing_columns)
                conn.commit()
            
            # Check for valid JSON in book_ids and book_counts
            self.stdout.write("Verifying JSON data integrity...")
            cursor.execute("SELECT word, book_ids, book_counts FROM optimal_index LIMIT 100")
            invalid_records = []
            for word, book_ids, book_counts in cursor.fetchall():
                try:
                    json.loads(book_ids)
                    json.loads(book_counts)
                except json.JSONDecodeError:
                    invalid_records.append(word)
            
            if invalid_records:
                self.stdout.write(self.style.WARNING(f"Found {len(invalid_records)} records with invalid JSON"))
                self.repair_json_data(cursor, invalid_records)
                conn.commit()
            
            # Verify book references
            self.stdout.write("Verifying book references...")
            all_books = set(Book.objects.values_list('id', flat=True))
            
            cursor.execute("SELECT word, book_ids FROM optimal_index LIMIT 100")
            invalid_book_refs = []
            for word, book_ids in cursor.fetchall():
                try:
                    book_id_list = json.loads(book_ids)
                    for book_id in book_id_list:
                        if int(book_id) not in all_books:
                            invalid_book_refs.append((word, book_id))
                except (json.JSONDecodeError, ValueError):
                    continue
            
            if invalid_book_refs:
                self.stdout.write(self.style.WARNING(f"Found {len(invalid_book_refs)} invalid book references"))
            
            # Check for NULL values
            cursor.execute("SELECT COUNT(*) FROM optimal_index WHERE book_ids IS NULL OR book_counts IS NULL")
            null_count = cursor.fetchone()[0]
            if null_count:
                self.stdout.write(self.style.WARNING(f"Found {null_count} records with NULL values"))
                self.repair_null_values(cursor)
                conn.commit()
            
            conn.close()
            
            self.stdout.write(self.style.SUCCESS("Verification complete!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error verifying database: {str(e)}"))
    
    def create_optimal_index_table(self, cursor):
        self.stdout.write("Creating optimal_index table...")
        cursor.execute("""
            CREATE TABLE optimal_index (
                word TEXT PRIMARY KEY,
                book_ids TEXT NOT NULL,
                book_counts TEXT NOT NULL,
                total_occurrences INTEGER DEFAULT 0
            )
        """)
    
    def repair_table_structure(self, cursor, missing_columns):
        for col in missing_columns:
            if col == 'word':
                self.stdout.write("Cannot repair missing primary key column 'word'")
                continue
            
            if col == 'book_ids':
                self.stdout.write("Adding missing column 'book_ids'")
                cursor.execute("ALTER TABLE optimal_index ADD COLUMN book_ids TEXT DEFAULT '[]'")
            
            if col == 'book_counts':
                self.stdout.write("Adding missing column 'book_counts'")
                cursor.execute("ALTER TABLE optimal_index ADD COLUMN book_counts TEXT DEFAULT '{}'")
            
            if col == 'total_occurrences':
                self.stdout.write("Adding missing column 'total_occurrences'")
                cursor.execute("ALTER TABLE optimal_index ADD COLUMN total_occurrences INTEGER DEFAULT 0")
    
    def repair_json_data(self, cursor, invalid_records):
        for word in invalid_records:
            self.stdout.write(f"Repairing invalid JSON for word: {word}")
            cursor.execute(
                """
                UPDATE optimal_index 
                SET book_ids = '[]', book_counts = '{}' 
                WHERE word = ?
                """,
                (word,)
            )
    
    def repair_null_values(self, cursor):
        self.stdout.write("Repairing NULL values...")
        cursor.execute(
            """
            UPDATE optimal_index 
            SET book_ids = '[]' 
            WHERE book_ids IS NULL
            """
        )
        cursor.execute(
            """
            UPDATE optimal_index 
            SET book_counts = '{}' 
            WHERE book_counts IS NULL
            """
        )
        cursor.execute(
            """
            UPDATE optimal_index 
            SET total_occurrences = 0 
            WHERE total_occurrences IS NULL
            """
        )