from django.contrib import admin
from .models import Book, OptimalIndex, SearchLog, BookClick, BookSimilarity

# Register models for admin interface
@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'publication_year', 'total_clicks', 'centrality_score')
    search_fields = ('title', 'author')
    list_filter = ('publication_year',)

@admin.register(OptimalIndex)
class OptimalIndexAdmin(admin.ModelAdmin):
    list_display = ('word', 'total_occurrences')
    search_fields = ('word',)
    ordering = ('-total_occurrences',)

@admin.register(SearchLog)
class SearchLogAdmin(admin.ModelAdmin):
    list_display = ('search_term', 'is_regex', 'timestamp', 'user')
    list_filter = ('is_regex', 'timestamp')
    search_fields = ('search_term',)

@admin.register(BookClick)
class BookClickAdmin(admin.ModelAdmin):
    list_display = ('book', 'search_log', 'timestamp')
    list_filter = ('timestamp',)

@admin.register(BookSimilarity)
class BookSimilarityAdmin(admin.ModelAdmin):
    list_display = ('from_book', 'to_book', 'similarity_score')
    list_filter = ('similarity_score',)
    search_fields = ('from_book__title', 'to_book__title')
