"""Views for the search engine."""

import re
from django.db.models import Q, Count, F
from django.shortcuts import get_object_or_404, render
from django.views.generic import TemplateView
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Book, BookIndex, SearchLog, BookClick
from .serializers import BookSerializer, BookDetailSerializer
from .utils import regex_search, tokenize_text
from .config import (
    RECOMMENDATION_LIMIT,
    MAX_SEARCH_RESULTS,
    JACCARD_SIMILARITY_THRESHOLD,
)


class SearchView(APIView):
    """API view for searching books by keyword."""

    def get(self, request):
        """Handle GET requests."""
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
        # user = request.user
        search_log = SearchLog.objects.create(
            user=user, search_term=query, is_regex=is_regex
        )

        # Perform the search
        if is_regex:
            results = self.regex_search(query)
        else:
            results = self.keyword_search(query)

        # Get recommended books
        recommendations = self.get_recommendations(results[:5], query)

        return Response(
            {
                "query": query,
                "is_regex": is_regex,
                "results_count": len(results),
                "results": BookSerializer(results, many=True).data,
                "recommendations": BookSerializer(recommendations, many=True).data,
                "search_id": search_log.id,
            }
        )


    def keyword_search(self, query):
        """Search books by keyword."""
        # Tokenize the query
        tokens = tokenize_text(query)

        if not tokens:
            print("No tokens")
            return []

        # Use exact matches for better performance
        query_filter = Q()
        for token in tokens:
            query_filter |= Q(indices__word=token)
            
        print(f"Searching for tokens: {tokens}")

        # Get matching books with occurrence count for ranking
        books = (
            Book.objects
            .filter(query_filter)
            .annotate(
                relevance_score=Count("indices", filter=Q(indices__word__in=tokens))
            )
            .order_by("-relevance_score", "-centrality_score")[:MAX_SEARCH_RESULTS]
        )

        print(f"Found {books.count()} books")
        
        return books

    def regex_search(self, pattern):
        """Search books by regex pattern."""
        try:
            # Test if the regex is valid
            re.compile(pattern)
        except re.error:
            print(f"Invalid regex pattern: {pattern}")
            return []

        try:
            print(f"Searching for regex pattern: {pattern}")
            # First match against the indexed words for efficiency with a limit
            matching_indices = BookIndex.objects.filter(word__regex=pattern)[:1000]
            book_ids = set(matching_indices.values_list("book_id", flat=True))
            
            if not book_ids:
                print("No matching indices found")
                return []
                
            print(f"Found {len(book_ids)} books with matching words")

            # Get books and order by relevance (number of matching words)
            # Use simpler query to improve performance
            books = Book.objects.filter(id__in=book_ids).order_by("-centrality_score")[:MAX_SEARCH_RESULTS]
            
            return books
            
        except Exception as e:
            print(f"Error in regex search: {str(e)}")
            return []

    def get_recommendations(self, search_results, query):
        """Get book recommendations based on search results."""
        try:
            if not search_results:
                # Return some high-ranking books if no results
                print("No search results, returning high-ranking books")
                return Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]

            # Get recommendations based on Jaccard similarity
            result_ids = [book.id for book in search_results]
            print(f"Finding recommendations for {len(result_ids)} search results")

            # Limit the number of common words to analyze for performance
            common_words = (
                BookIndex.objects.filter(book_id__in=result_ids[:3])
                .values_list("word", flat=True)
                .distinct()[:100]  # Limit to 100 common words
            )
            
            common_words_list = list(common_words)
            if not common_words_list:
                print("No common words found")
                return Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
                
            print(f"Found {len(common_words_list)} common words")

            # Get books that contain these words but are not in the results
            recommendations = (
                Book.objects.filter(indices__word__in=common_words_list)
                .exclude(id__in=result_ids)
                .distinct()
                .order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
            )
            
            print(f"Found {recommendations.count()} recommendations")
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]


class BookViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for browsing books."""

    queryset = Book.objects.all().order_by("-centrality_score")

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

        # Record the click
        BookClick.objects.create(search_log=search_log, book=book)

        # Update the book's click count
        book.total_clicks = F("total_clicks") + 1
        book.save(update_fields=["total_clicks"])

        return Response({"status": "success"})


class StatisticsView(APIView):
    """API view for getting search engine statistics."""

    def get(self, request):
        """Handle GET requests."""
        total_books = Book.objects.count()
        total_indexed_words = BookIndex.objects.values("word").distinct().count()
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
        book_id = self.kwargs.get('pk')
        book = get_object_or_404(Book, id=book_id)
        
        # Get basic book information
        context['book'] = book
        
        # Get word count
        context['word_count'] = len(tokenize_text(book.text_content))
        
        # Get indexed words count
        context['indexed_words_count'] = book.indices.count()
        
        # Get most common words
        context['common_words'] = book.indices.order_by('-occurrences')[:40]
        
        # Get book preview (first 1000 characters)
        preview_length = 1000
        context['book_preview'] = book.text_content[:preview_length] + "..." if len(book.text_content) > preview_length else book.text_content
        
        # Get similar books based on Jaccard similarity
        book_ids = BookIndex.objects.filter(
            word__in=book.indices.values_list('word', flat=True)[:100]
        ).exclude(book=book).values_list('book', flat=True).distinct()[:5]
        
        context['similar_books'] = Book.objects.filter(id__in=book_ids)
        
        return context


class StatsPageView(TemplateView):
    """View for the statistics page."""

    template_name = "searchengine/stats.html"
