from django.db import models
from django.contrib.auth.models import User

class Book(models.Model):
    """Model representing a book in the library."""
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    publication_year = models.IntegerField(null=True, blank=True)
    text_content = models.TextField()
    file_path = models.CharField(max_length=512, blank=True, null=True)
    gutenberg_id = models.IntegerField(null=True, blank=True, unique=True)
    
    # Fields for ranking
    centrality_score = models.FloatField(default=0.0)
    total_clicks = models.IntegerField(default=0)
    
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
