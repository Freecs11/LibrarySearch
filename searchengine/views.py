"""Views for the search engine."""

import re
import time
from django.db.models import Q, Count, F, Sum, IntegerField
from django.db.models.functions import Coalesce
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

        # Perform the search
        search_start = time.time()
        if is_regex:
            results = self.regex_search(query)
        else:
            results = self.keyword_search(query)
        search_time = time.time() - search_start

        # Get recommended books
        rec_start = time.time()
        # Ensure results is not None and not empty
        if results and len(results) > 0:
            recommendations = self.get_recommendations(results[:5], query)
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

    def keyword_search(self, query):
        """Search books by keyword.

        Per project requirements:
        1. Use the index table to find books with matching words
        2. Order by relevance (occurrence count)
        3. Then by centrality ranking
        """
        # Tokenize the query and apply stemming
        tokens = tokenize_text(query, apply_stemming=True)

        if not tokens:
            print("No tokens")
            return []

        print(f"Searching for stemmed tokens: {tokens}")

        # STRICTLY USE INDEX TABLE as specified in requirements
        if len(tokens) == 1:
            # Single token search - simpler case
            token = tokens[0]

            # Get books with this word in their index
            matching_indices = BookIndex.objects.filter(word=token)

            # Group by book and sum occurrences for relevance score
            book_scores = {}
            for index in matching_indices:
                if index.book_id not in book_scores:
                    book_scores[index.book_id] = 0
                book_scores[index.book_id] += index.occurrences

            # Get the books
            if not book_scores:
                return []

            books = list(Book.objects.filter(id__in=book_scores.keys()))

            # Add relevance score attribute for sorting
            for book in books:
                book.relevance_score = book_scores[book.id]

            # Sort by relevance (occurrences) and then by centrality
            books.sort(
                key=lambda book: (
                    book.centrality_score,  # Centrality as tiebreaker
                    getattr(book, "relevance_score", 0),  # Occurrences in index
                ),
                reverse=True,
            )

            print(f"Found {len(books)} books for token '{token}'")
            return books[:MAX_SEARCH_RESULTS]

        else:
            # Multi-token search
            # Get all books matching any token
            query_filter = Q()
            for token in tokens:
                query_filter |= Q(indices__word=token)

            # Also try to match the full original query
            if " " in query and len(query) > 2:
                query_filter |= Q(indices__word=query.lower())

            matching_books = Book.objects.filter(query_filter).distinct()

            # Calculate relevance scores based on token occurrences
            book_scores = {}
            for book in matching_books:
                # Get matching indices for this book
                indices = BookIndex.objects.filter(book=book, word__in=tokens)

                # Also check for full query match
                if " " in query:
                    full_query_indices = BookIndex.objects.filter(
                        book=book, word=query.lower()
                    )
                    indices = list(indices) + list(full_query_indices)

                # Count distinct tokens matched
                matched_tokens = len(set(idx.word for idx in indices))

                # Sum occurrences for total relevance
                total_occurrences = sum(idx.occurrences for idx in indices)

                # Store both metrics for sorting
                book_scores[book.id] = (matched_tokens, total_occurrences)

                # Add relevance attributes to book
                book.matched_tokens = matched_tokens
                book.total_occurrences = total_occurrences

            # Sort by matched token count (primary) and total occurrences (secondary)
            matching_books = list(matching_books)
            matching_books.sort(
                key=lambda book: (
                    book.centrality_score,  # Centrality as final tiebreaker
                    getattr(
                        book, "matched_tokens", 0
                    ),  # Number of query tokens matched
                    getattr(
                        book, "total_occurrences", 0
                    ),  # Total occurrences of all matched tokens
                ),
                reverse=True,
            )

            # For exact phrase matches, we would need to search the text content
            # But according to requirements, we should stick to the index table

            print(f"Found {len(matching_books)} books with {len(tokens)} tokens")
            return matching_books[:MAX_SEARCH_RESULTS]

    def regex_search(self, pattern):
        """Search books by regex pattern.

        Per project requirements:
        1. Search index table for words matching the regex
        2. Return books that have matching words in index
        3. Order by occurrences and centrality
        """
        try:
            # Test if the regex is valid
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(f"Invalid regex pattern: {pattern}")
            return []

        try:
            print(f"Searching for regex pattern: {pattern}")

            # STRICTLY USE INDEX TABLE as specified in requirements
            # For "open door" as regex, we need to also check for both "open" and "door"
            # First, if the pattern contains spaces, split it and create filters for each word
            word_filters = []

            # First, try the exact pattern in the index
            exact_pattern_filter = Q(word__regex=pattern)
            word_filters.append(exact_pattern_filter)

            # If the pattern has spaces, also look for the individual words
            if " " in pattern:
                words = pattern.split()
                for word in words:
                    if (
                        len(word) >= MIN_WORD_LENGTH
                    ):  # Only use words that meet minimum length
                        # Look for individual words that might appear in the index
                        word_filters.append(Q(word__regex=word))

                        # Also try the stemmed version of each word
                        stemmed_word = stemmer.stem(word)
                        if (
                            stemmed_word != word
                        ):  # Only add if stemming changed the word
                            word_filters.append(Q(word__regex=stemmed_word))

            # Combine all filters with OR
            if word_filters:
                combined_filter = word_filters[0]
                for filter_item in word_filters[1:]:
                    combined_filter |= filter_item

                # Get all matching indices
                matching_indices = BookIndex.objects.filter(combined_filter)

                if not matching_indices.exists():
                    print(f"No matching indices found for regex pattern '{pattern}'")
                    return []

                # Group by book and sum occurrences for relevance
                book_scores = {}
                for index in matching_indices:
                    if index.book_id not in book_scores:
                        book_scores[index.book_id] = 0

                    # Words matching the full pattern get extra weight
                    if re.search(pattern, index.word, re.IGNORECASE):
                        book_scores[index.book_id] += (
                            index.occurrences * 2
                        )  # Double weight for full pattern match
                    else:
                        book_scores[index.book_id] += index.occurrences

                # Get the books
                books = list(Book.objects.filter(id__in=book_scores.keys()))

                # Add relevance scores for sorting
                for book in books:
                    book.relevance_score = book_scores[book.id]

                # Sort by relevance score and centrality
                books.sort(
                    key=lambda book: (
                        getattr(
                            book, "relevance_score", 0
                        ),  # Primary: relevance score from index
                        book.centrality_score,  # Secondary: centrality tiebreaker
                    ),
                    reverse=True,
                )

                print(f"Found {len(books)} books matching regex pattern '{pattern}'")
                return books[:MAX_SEARCH_RESULTS]
            else:
                print(f"No valid word filters created for regex pattern '{pattern}'")
                return []

        except Exception as e:
            print(f"Error in regex search: {str(e)}")
            return []

    def get_recommendations(self, search_results, query):
        """Get book recommendations based on search results.

        Per project requirements:
        - Use Jaccard similarity to find similar content
        - Return books that are neighbors in the Jaccard graph
        """
        try:
            if not search_results:
                # Return some high-ranking books if no results
                print("No search results, returning high-ranking books")
                return list(
                    Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
                )

            # Get recommendations based on Jaccard similarity (as required)
            result_ids = [book.id for book in search_results]
            print(f"Finding recommendations for {len(result_ids)} search results")

            # Use a maximum of 5 top results to find common words
            result_ids = result_ids[:5]

            if len(result_ids) == 0:
                return list(
                    Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
                )

            # Convert to string for SQLite compatibility
            result_ids_str = ",".join(str(id) for id in result_ids)

            # Get words from the top books
            book_words = BookIndex.objects.filter(book_id__in=result_ids)

            # Count occurrences of each word across top books
            word_counts = {}
            for index in book_words:
                if index.word not in word_counts:
                    word_counts[index.word] = 0
                word_counts[index.word] += index.occurrences

            # Get the most common words (up to 20)
            common_words = sorted(
                word_counts.items(), key=lambda x: x[1], reverse=True
            )[:20]
            common_words_list = [word for word, _ in common_words]

            if not common_words_list:
                print("No common words found")
                return list(
                    Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
                )

            # Find books containing these common words (but not in search results)
            book_counts = {}
            for word in common_words_list:
                # Get books with this word that aren't in search results
                matching_indices = BookIndex.objects.filter(word=word).exclude(
                    book_id__in=result_ids
                )

                # Count occurrences for each book
                for index in matching_indices:
                    if index.book_id not in book_counts:
                        book_counts[index.book_id] = 0
                    book_counts[index.book_id] += 1

            # Get books sorted by how many common words they share
            sorted_books = sorted(book_counts.items(), key=lambda x: x[1], reverse=True)
            rec_book_ids = [
                book_id for book_id, _ in sorted_books[: RECOMMENDATION_LIMIT * 2]
            ]

            if not rec_book_ids:
                # Fallback to centrality ranking
                return list(
                    Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
                )

            # Get the books
            recommendations = list(Book.objects.filter(id__in=rec_book_ids))

            # Sort by how many common words they share
            id_to_count = {book_id: count for book_id, count in sorted_books}
            recommendations.sort(
                key=lambda book: (
                    id_to_count.get(book.id, 0),  # Common word count
                    book.centrality_score,  # Centrality as tiebreaker
                ),
                reverse=True,
            )

            print(f"Found {len(recommendations)} Jaccard-based recommendations")
            return recommendations[:RECOMMENDATION_LIMIT]

        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return list(
                Book.objects.order_by("-centrality_score")[:RECOMMENDATION_LIMIT]
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
        book_id = self.kwargs.get("pk")
        book = get_object_or_404(Book, id=book_id)

        # Get basic book information
        context["book"] = book

        # Get text content from file
        content = book.get_text_content()

        # Get word count
        context["word_count"] = len(tokenize_text(content))

        # Get indexed words count
        context["indexed_words_count"] = book.indices.count()

        # Get most common words
        context["common_words"] = book.indices.order_by("-occurrences")[:40]

        # Get book preview (first 1000 characters)
        preview_length = 1000
        context["book_preview"] = (
            content[:preview_length] + "..."
            if len(content) > preview_length
            else content
        )

        # Get similar books based on Jaccard similarity
        book_ids = (
            BookIndex.objects.filter(
                word__in=book.indices.values_list("word", flat=True)[:100]
            )
            .exclude(book=book)
            .values_list("book", flat=True)
            .distinct()[:5]
        )

        context["similar_books"] = Book.objects.filter(id__in=book_ids)

        return context


class StatsPageView(TemplateView):
    """View for the statistics page."""

    template_name = "searchengine/stats.html"
