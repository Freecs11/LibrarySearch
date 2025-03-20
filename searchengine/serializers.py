"""Serializers for the search engine."""
from rest_framework import serializers
from .models import Book, SearchLog, BookClick

class BookSerializer(serializers.ModelSerializer):
    """Serializer for the Book model."""
    
    class Meta:
        model = Book
        fields = ['id', 'title', 'author', 'publication_year', 'centrality_score', 'total_clicks']

class BookDetailSerializer(serializers.ModelSerializer):
    """Serializer for detailed Book information."""
    word_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Book
        fields = [
            'id', 'title', 'author', 'publication_year',
            'centrality_score', 'total_clicks', 'word_count',
            'gutenberg_id'
        ]
    
    def get_word_count(self, obj):
        return obj.indices.count()

class SearchLogSerializer(serializers.ModelSerializer):
    """Serializer for the SearchLog model."""
    
    class Meta:
        model = SearchLog
        fields = ['id', 'search_term', 'is_regex', 'timestamp']

class BookClickSerializer(serializers.ModelSerializer):
    """Serializer for the BookClick model."""
    book = BookSerializer()
    
    class Meta:
        model = BookClick
        fields = ['id', 'book', 'timestamp']