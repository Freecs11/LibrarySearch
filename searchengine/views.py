"""Views for the search engine."""

import re
import time
import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from io import BytesIO
from django.http import HttpResponse, FileResponse
from django.db.models import Q, Count, F, Sum, IntegerField, Avg
from django.db.models.functions import Coalesce
from django.shortcuts import get_object_or_404, render
from django.views.generic import TemplateView
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Book, OptimalIndex, SearchLog, BookClick, BookSimilarity
from .serializers import BookSerializer, BookDetailSerializer
from .utils import tokenize_text, search_books, get_recommendations
from .config import (
    RECOMMENDATION_LIMIT,
    MAX_SEARCH_RESULTS,
    JACCARD_SIMILARITY_THRESHOLD,
)


class SearchView(APIView):
    """API view for searching books by keyword."""

    def get(self, request):
        """Handle GET requests."""
        # Track timing for performance analysis
        start_time = time.time()

        # Get query parameters
        query = request.query_params.get("q", "")
        is_regex = request.query_params.get("regex", "false").lower() == "true"

        if not query:
            return Response(
                {"error": "No search query provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Log the search
        user = request.user if request.user.is_authenticated else None
        search_log = SearchLog.objects.create(
            user=user, search_term=query, is_regex=is_regex
        )

        # Perform the search using the optimal index approach
        search_start = time.time()
        if is_regex:
            print(f"Starting regex search for pattern: {query}")
        results = search_books(query, is_regex, MAX_SEARCH_RESULTS)
        search_time = time.time() - search_start
        if is_regex:
            print(f"Completed regex search in {search_time:.2f} seconds")

        # Get recommended books
        rec_start = time.time()
        # Ensure results is not None and not empty
        if results and len(results) > 0:
            recommendations = get_recommendations(results[:5], RECOMMENDATION_LIMIT)
        else:
            recommendations = []
        rec_time = time.time() - rec_start

        # Serialize results in chunks for better performance
        serialize_start = time.time()
        # Check for None before serializing
        serialized_results = BookSerializer(results or [], many=True).data
        serialized_recommendations = BookSerializer(
            recommendations or [], many=True
        ).data
        serialize_time = time.time() - serialize_start

        total_time = time.time() - start_time

        # Log performance metrics for debugging
        print(
            f"Search time: {search_time:.4f}s, Recommendations: {rec_time:.4f}s, Serialize: {serialize_time:.4f}s, Total: {total_time:.4f}s"
        )

        # Make sure results is not None before getting length
        results_count = len(results) if results is not None else 0

        return Response(
            {
                "query": query,
                "is_regex": is_regex,
                "results_count": results_count,
                "results": serialized_results or [],
                "recommendations": serialized_recommendations or [],
                "search_id": search_log.id,
            }
        )


class BookViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for browsing books."""

    queryset = (
        Book.objects.all()
        .order_by("-centrality_score")
        .only("id", "title", "author", "centrality_score", "total_clicks")
    )

    def get_serializer_class(self):
        if self.action == "retrieve":
            return BookDetailSerializer
        return BookSerializer

    @action(detail=True, methods=["post"])
    def click(self, request, pk=None):
        """Record a click on a book from a search."""
        book = self.get_object()
        search_id = request.data.get("search_id")

        if not search_id:
            return Response(
                {"error": "No search_id provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get the search log
        try:
            search_log = SearchLog.objects.get(id=search_id)
        except SearchLog.DoesNotExist:
            return Response(
                {"error": "Invalid search_id"}, status=status.HTTP_404_NOT_FOUND
            )

        # Use a direct SQL update to avoid locking issues
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                # First, insert the book click
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO searchengine_bookclick 
                    (search_log_id, book_id, timestamp) 
                    VALUES (%s, %s, datetime('now'))
                    """, 
                    [search_log.id, book.id]
                )
                
                # Then update the book's click count
                cursor.execute(
                    """
                    UPDATE searchengine_book 
                    SET total_clicks = total_clicks + 1 
                    WHERE id = %s
                    """, 
                    [book.id]
                )
            
            return Response({"status": "success"})
        except Exception as e:
            print(f"Error recording click: {str(e)}")
            # Fall back to Django ORM if there's an error
            try:
                # Record the click
                BookClick.objects.get_or_create(search_log=search_log, book=book)
                
                # Update the click count in a separate transaction
                from django.db import transaction
                with transaction.atomic():
                    Book.objects.filter(id=book.id).update(total_clicks=F("total_clicks") + 1)
                
                return Response({"status": "success"})
            except Exception as e2:
                print(f"Fallback error: {str(e2)}")
                return Response({"error": str(e2)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StatisticsView(APIView):
    """API view for getting search engine statistics."""

    def get(self, request):
        """Handle GET requests."""
        total_books = Book.objects.count()
        
        # Use the optimal_index table to get word count
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db.sqlite3'), timeout=30)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM optimal_index")
        total_indexed_words = cursor.fetchone()[0] or 0
        conn.close()
        
        top_books = Book.objects.order_by("-total_clicks")[:10]
        recent_searches = SearchLog.objects.order_by("-timestamp")[:10]

        return Response(
            {
                "total_books": total_books,
                "total_indexed_words": total_indexed_words,
                "top_books": BookSerializer(top_books, many=True).data,
                "recent_searches": [
                    {
                        "search_term": log.search_term,
                        "is_regex": log.is_regex,
                        "timestamp": log.timestamp,
                    }
                    for log in recent_searches
                ],
            }
        )


# Web UI Views


class SearchPageView(TemplateView):
    """View for the search page."""

    template_name = "searchengine/search.html"


class BooksPageView(TemplateView):
    """View for the books page."""

    template_name = "searchengine/books.html"


class BookDetailView(TemplateView):
    """View for the book detail page."""

    template_name = "searchengine/book_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        book_id = self.kwargs.get("pk")
        book = get_object_or_404(Book, id=book_id)

        # Get basic book information
        context["book"] = book

        # Get text content from file
        content = book.get_text_content()

        # Get word count
        context["word_count"] = len(tokenize_text(content))

        # Connect to the database to get information from optimal_index
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db.sqlite3'), timeout=30)
        cursor = conn.cursor()
        
        # Get indexed words count - using a more reliable query
        cursor.execute(
            """
            SELECT COUNT(*) FROM optimal_index 
            WHERE json_array_length(json_extract(book_ids, '$')) > 0
            AND book_ids LIKE ?
            """, 
            (f'%{book_id}%',)
        )
        result = cursor.fetchone()
        context["indexed_words_count"] = result[0] if result else 0
        
        # If count is 0, try a different approach
        if context["indexed_words_count"] == 0:
            cursor.execute(
                """
                SELECT COUNT(*) FROM optimal_index 
                WHERE book_counts LIKE ?
                """, 
                (f'%{book_id}%',)
            )
            result = cursor.fetchone()
            context["indexed_words_count"] = result[0] if result else 0
        
        # Get most common words - using a more reliable approach
        try:
            # First approach with json_extract
            cursor.execute(
                """
                SELECT word, json_extract(book_counts, '$.' || ?) as count
                FROM optimal_index 
                WHERE book_ids LIKE ?
                AND json_extract(book_counts, '$.' || ?) IS NOT NULL
                ORDER BY CAST(count AS INTEGER) DESC
                LIMIT 40
                """, 
                (str(book_id), f'%{book_id}%', str(book_id))
            )
            results = cursor.fetchall()
            
            # If no results, try a different approach
            if not results:
                # Alternative approach - parse the JSON in Python
                cursor.execute(
                    """
                    SELECT word, book_counts
                    FROM optimal_index 
                    WHERE book_counts LIKE ?
                    LIMIT 100
                    """, 
                    (f'%{book_id}%',)
                )
                import json
                word_counts = []
                for word, book_counts_json in cursor.fetchall():
                    try:
                        book_counts = json.loads(book_counts_json)
                        count = book_counts.get(str(book_id), 0)
                        if count > 0:
                            word_counts.append((word, count))
                    except:
                        pass
                
                # Sort by count and take top 40
                word_counts.sort(key=lambda x: x[1], reverse=True)
                results = word_counts[:40]
            
            common_words = [{'word': row[0], 'occurrences': row[1]} for row in results]
            context["common_words"] = common_words
        except Exception as e:
            print(f"Error getting common words: {str(e)}")
            context["common_words"] = []

        # Get book preview (first 1000 characters)
        preview_length = 1000
        context["book_preview"] = (
            content[:preview_length] + "..."
            if len(content) > preview_length
            else content
        )

        # Get similar books based on Jaccard graph neighborhood
        similar_books_ids = (
            BookSimilarity.objects.filter(
                from_book=book, similarity_score__gte=JACCARD_SIMILARITY_THRESHOLD
            )
            .order_by("-similarity_score")
            .values_list("to_book", flat=True)[:5]
        )

        # If we don't have similarity data, fall back to the word-based approach
        if not similar_books_ids:
            # Get the top 100 words for this book
            cursor.execute(
                """
                SELECT word FROM optimal_index 
                WHERE book_ids LIKE ?
                ORDER BY json_extract(book_counts, ?) DESC
                LIMIT 100
                """, 
                (f'%"{book_id}"%', f'$."{book_id}"')
            )
            top_words = [row[0] for row in cursor.fetchall()]
            
            # Find books that share these words
            book_counts = {}
            for word in top_words:
                cursor.execute(
                    """
                    SELECT book_ids FROM optimal_index 
                    WHERE word = ?
                    """, 
                    (word,)
                )
                result = cursor.fetchone()
                if result:
                    book_ids_json = result[0]
                    book_ids = json.loads(book_ids_json)
                    for bid in book_ids:
                        if int(bid) != book_id:  # Exclude current book
                            book_counts[int(bid)] = book_counts.get(int(bid), 0) + 1
            
            # Get the top 5 books with most shared words
            similar_books_ids = [bid for bid, count in sorted(book_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        conn.close()
        context["similar_books"] = Book.objects.filter(id__in=similar_books_ids)

        return context


class StatsPageView(TemplateView):
    """View for the statistics page."""

    template_name = "searchengine/stats.html"


class MatrixView(TemplateView):
    """View for visualizing the Jaccard similarity matrix."""

    template_name = "searchengine/matrix_view.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get parameters from request
        sample_size = min(200, int(self.request.GET.get("sample_size", 50)))
        threshold = float(
            self.request.GET.get("threshold", JACCARD_SIMILARITY_THRESHOLD)
        )

        # Store parameters in context
        context["sample_size"] = sample_size
        context["threshold"] = threshold

        # Create the static directory if it doesn't exist
        static_dir = os.path.join(settings.BASE_DIR, "static")
        os.makedirs(static_dir, exist_ok=True)

        # Generate the matrix visualization
        matrix_filename = f"jaccard_matrix_{sample_size}_{threshold}.png"
        matrix_path = os.path.join(static_dir, matrix_filename)
        context["matrix_url"] = f"{settings.STATIC_URL}{matrix_filename}"

        # Make sure the image exists or create it
        if not os.path.exists(matrix_path) or self.request.GET.get("refresh", False):
            # Always generate a new matrix image to ensure it exists
            self.generate_matrix_image(matrix_path, sample_size, threshold)

        return context

    def generate_matrix_image(self, output_path, sample_size, threshold):
        """Generate and save the matrix visualization."""
        # Get books
        books = list(Book.objects.all().order_by("id")[:sample_size])

        # Create book ID to index mapping
        book_map = {book.id: i for i, book in enumerate(books)}

        # Initialize matrix with zeros
        n = len(books)
        matrix = np.zeros((n, n))

        # Get similarities for books in our sample
        book_ids = [book.id for book in books]
        similarities = BookSimilarity.objects.filter(
            from_book_id__in=book_ids,
            to_book_id__in=book_ids,
            similarity_score__gte=threshold,
        )

        # Fill the matrix
        for similarity in similarities:
            from_idx = book_map.get(similarity.from_book_id)
            to_idx = book_map.get(similarity.to_book_id)
            if from_idx is not None and to_idx is not None:
                matrix[from_idx][to_idx] = similarity.similarity_score

        # Create a DataFrame for easier plotting
        book_labels = [f"{book.title[:20]}" for book in books]
        df = pd.DataFrame(matrix, index=book_labels, columns=book_labels)

        # Plot settings
        plt.figure(figsize=(20, 16))

        # Draw heatmap
        sns.heatmap(
            df, annot=False, cmap="YlGnBu", cbar_kws={"label": "Jaccard Similarity"}
        )

        plt.title("Jaccard Similarity Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def download_matrix(request):
    """View for downloading the Jaccard similarity matrix in various formats."""
    format_type = request.GET.get("format", "csv")
    sample_size = min(1000, int(request.GET.get("sample_size", 100)))

    # Get books
    books = list(Book.objects.all().order_by("id")[:sample_size])

    # Create book ID to index mapping
    book_map = {book.id: i for i, book in enumerate(books)}

    # Initialize matrix with zeros
    n = len(books)
    matrix = np.zeros((n, n))

    # Get similarities for books in our sample
    book_ids = [book.id for book in books]
    similarities = BookSimilarity.objects.filter(
        from_book_id__in=book_ids, to_book_id__in=book_ids
    )

    # Fill the matrix
    for similarity in similarities:
        from_idx = book_map.get(similarity.from_book_id)
        to_idx = book_map.get(similarity.to_book_id)
        if from_idx is not None and to_idx is not None:
            matrix[from_idx][to_idx] = similarity.similarity_score

    # Prepare response based on format
    if format_type == "csv":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="jaccard_matrix.csv"'

        writer = csv.writer(response)
        # Write header row with book titles
        header = [""] + [f"{book.title[:30]}" for book in books]
        writer.writerow(header)

        # Write data rows
        for i, book in enumerate(books):
            row = [book.title[:30]] + [f"{x:.4f}" for x in matrix[i]]
            writer.writerow(row)

        return response

    elif format_type == "json":
        # Prepare data structure
        data = {
            "books": [
                {"id": book.id, "title": book.title, "author": book.author}
                for book in books
            ],
            "matrix": matrix.tolist(),
        }

        response = HttpResponse(
            json.dumps(data, indent=2), content_type="application/json"
        )
        response["Content-Disposition"] = 'attachment; filename="jaccard_matrix.json"'
        return response

    else:
        return HttpResponse("Unsupported format", status=400)
