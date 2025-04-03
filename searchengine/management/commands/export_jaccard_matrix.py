import csv
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.management.base import BaseCommand
from searchengine.models import Book, BookSimilarity


class Command(BaseCommand):
    help = 'Export Jaccard similarity matrix to various formats'

    def add_arguments(self, parser):
        parser.add_argument(
            '--format',
            type=str,
            default='csv',
            help='Output format (csv, json, png)',
        )
        parser.add_argument(
            '--sample',
            type=int,
            default=100,
            help='Sample size for visualization (default: 100 books)',
        )
        parser.add_argument(
            '--output',
            type=str,
            default='jaccard_matrix',
            help='Output file name without extension',
        )

    def handle(self, *args, **options):
        format_type = options['format']
        sample_size = options['sample']
        output_file = options['output']
        
        # Get all books or a sample
        books = list(Book.objects.all().order_by('id'))
        if len(books) > sample_size:
            self.stdout.write(f"Sampling {sample_size} books from {len(books)} total")
            books = books[:sample_size]
        
        # Create book ID to index mapping
        book_map = {book.id: i for i, book in enumerate(books)}
        
        # Initialize matrix with zeros
        n = len(books)
        matrix = np.zeros((n, n))
        
        # Get all similarities for books in our sample
        book_ids = [book.id for book in books]
        similarities = BookSimilarity.objects.filter(
            from_book_id__in=book_ids,
            to_book_id__in=book_ids
        )
        
        # Fill the matrix
        self.stdout.write(f"Building {n}x{n} similarity matrix...")
        for similarity in similarities:
            from_idx = book_map.get(similarity.from_book_id)
            to_idx = book_map.get(similarity.to_book_id)
            if from_idx is not None and to_idx is not None:
                matrix[from_idx][to_idx] = similarity.similarity_score
        
        # Export based on format
        if format_type == 'csv':
            self.export_csv(matrix, books, output_file)
        elif format_type == 'json':
            self.export_json(matrix, books, output_file)
        elif format_type == 'png':
            self.export_visualization(matrix, books, output_file)
        else:
            self.stdout.write(self.style.ERROR(f"Unknown format: {format_type}"))
            
        self.stdout.write(self.style.SUCCESS(f"Matrix export completed!"))

    def export_csv(self, matrix, books, output_file):
        output_path = f"{output_file}.csv"
        self.stdout.write(f"Exporting matrix to CSV: {output_path}")
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header row with book titles
            header = [''] + [f"{book.title[:30]}" for book in books]
            writer.writerow(header)
            
            # Write data rows
            for i, book in enumerate(books):
                row = [book.title[:30]] + [f"{x:.4f}" for x in matrix[i]]
                writer.writerow(row)
        
        self.stdout.write(self.style.SUCCESS(f"CSV export saved to {output_path}"))

    def export_json(self, matrix, books, output_file):
        output_path = f"{output_file}.json"
        self.stdout.write(f"Exporting matrix to JSON: {output_path}")
        
        # Prepare data structure
        data = {
            "books": [{"id": book.id, "title": book.title, "author": book.author} for book in books],
            "matrix": matrix.tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(f"JSON export saved to {output_path}"))

    def export_visualization(self, matrix, books, output_file):
        output_path = f"{output_file}.png"
        self.stdout.write(f"Creating visualization: {output_path}")
        
        # Create a DataFrame for easier plotting
        book_labels = [f"{book.title[:20]}" for book in books]
        df = pd.DataFrame(matrix, index=book_labels, columns=book_labels)
        
        # Create directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Plot settings
        plt.figure(figsize=(20, 16))
        
        # Draw heatmap
        sns.heatmap(df, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Jaccard Similarity'})
        
        plt.title("Jaccard Similarity Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.stdout.write(self.style.SUCCESS(f"Visualization saved to {output_path}"))