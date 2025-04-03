import sqlite3
import os
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Optimizes the database for the optimal_index table'

    def handle(self, *args, **options):
        # Connect to the database
        db_path = os.path.join(settings.BASE_DIR, 'db.sqlite3')
        self.stdout.write(f"Connecting to database at {db_path}")
        
        try:
            conn = sqlite3.connect(db_path, timeout=60)
            cursor = conn.cursor()
            
            # Create indices if they don't exist
            self.stdout.write("Creating database indices for optimal_index...")
            
            # Index for word lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimal_index_word 
                ON optimal_index(word)
            """)
            
            # Index for word lookups using LIKE
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimal_index_book_ids 
                ON optimal_index(book_ids)
            """)
            
            # Update SQLite settings for better performance
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA cache_size = 10000")
            cursor.execute("PRAGMA temp_store = MEMORY")
            
            # Analyze the table for better query planning
            cursor.execute("ANALYZE optimal_index")
            
            # Vacuum the database to optimize storage
            self.stdout.write("Vacuuming database...")
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            self.stdout.write(self.style.SUCCESS("Database optimization complete!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error optimizing database: {str(e)}"))