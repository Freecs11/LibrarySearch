from django.db import models
from django.contrib.auth.models import User
import json

class Book(models.Model):
    """Model representing a book in the library."""
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    publication_year = models.IntegerField(null=True, blank=True)
    file_path = models.CharField(max_length=512, blank=True, default="books_data/unknown.txt")
    gutenberg_id = models.IntegerField(null=True, blank=True, unique=True)
    
    # Fields for ranking
    centrality_score = models.FloatField(default=0.0)
    total_clicks = models.IntegerField(default=0)
    
    # Related books through Jaccard similarity graph
    similar_books = models.ManyToManyField(
        'self', 
        through='BookSimilarity',
        symmetrical=False,
        related_name='related_books'
    )
    
    def get_text_content(self):
        """Read book content from file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {self.file_path}: {str(e)}")
            return ""
    
    def __str__(self):
        return f"{self.title} by {self.author}"

class SearchLog(models.Model):
    """Model to log user searches for recommendation features."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    search_term = models.CharField(max_length=255)
    is_regex = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    clicked_books = models.ManyToManyField(Book, through='BookClick')
    
    def __str__(self):
        return f"Search for '{self.search_term}' at {self.timestamp}"

class BookClick(models.Model):
    """Model to track which books users click on after searches."""
    search_log = models.ForeignKey(SearchLog, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('search_log', 'book')
    
    def __str__(self):
        return f"Click on {self.book.title} from search '{self.search_log.search_term}'"

class BookSimilarity(models.Model):
    """Model representing an edge in the Jaccard similarity graph."""
    from_book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='outgoing_similarities')
    to_book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='incoming_similarities')
    similarity_score = models.FloatField(default=0.0)
    
    class Meta:
        unique_together = ('from_book', 'to_book')
        indexes = [
            models.Index(fields=['from_book']),
            models.Index(fields=['to_book']),
            models.Index(fields=['similarity_score']),
        ]
    
    def __str__(self):
        return f"Similarity between {self.from_book.title} and {self.to_book.title}: {self.similarity_score:.4f}"

class OptimalIndex(models.Model):
    """
    Optimal index model where each word stores a list of book IDs.
    This model represents the structure of the optimal_index table.
    """
    word = models.CharField(max_length=100, primary_key=True)
    book_ids = models.TextField()  # JSON array of book IDs
    book_counts = models.TextField()  # JSON object mapping book IDs to occurrence counts
    total_occurrences = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'optimal_index'
        
    def get_book_ids(self):
        """Get the list of book IDs."""
        return json.loads(self.book_ids)
    
    def get_book_counts(self):
        """Get the dictionary mapping book IDs to counts."""
        return json.loads(self.book_counts)
    
    def get_books_containing_word(self):
        """Get the Book objects containing this word."""
        return Book.objects.filter(id__in=self.get_book_ids())
    
    def __str__(self):
        book_count = len(self.get_book_ids())
        return f"{self.word} (in {book_count} books, {self.total_occurrences} occurrences)"
