from django.db import models
from django.contrib.auth.models import User

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

class BookIndex(models.Model):
    """Model for indexing words in books for faster searching."""
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='indices')
    word = models.CharField(max_length=100, db_index=True)
    occurrences = models.IntegerField(default=1)
    
    class Meta:
        unique_together = ('book', 'word')
        indexes = [models.Index(fields=['word'])]
    
    def __str__(self):
        return f"{self.word} in {self.book.title}"

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
