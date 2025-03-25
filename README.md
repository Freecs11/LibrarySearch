# Library Search Engine

A sophisticated search engine for a library of text documents, built with Django. This project allows users to search through a large collection of texts (from Project Gutenberg) using both keyword search and advanced regular expression search.

## Features

- **Keyword Search**: Find books containing specific words or phrases
- **Advanced RegEx Search**: Use regular expressions for more complex search patterns
- **Intelligent Ranking**: Results are ordered by relevance and centrality ranking
- **Content Recommendations**: Get suggestions for similar books based on your search
- **Interactive Web Interface**: User-friendly interface with responsive design
- **Detailed Statistics**: View insights about the library and search patterns

## Database Storage Optimization

The system is optimized to reduce database size by:

1. Storing book text files on disk rather than in the database
2. Using file references in the DB model
3. Loading content on-demand via the `get_text_content()` method

### Migration from Content Storage to File References

If upgrading from a previous version that stored full text content in the database:

1. Run migrations to update the DB schema:
   ```
   python manage.py migrate
   ```

2. Run the SQL cleanup script to reclaim space:
   ```
   sqlite3 db.sqlite3 < migration_cleanup.sql
   ```

## Project Structure

- **Models**:
  - `Book`: Stores book metadata and file path references
  - `BookIndex`: Indexes words in books for efficient searching
  - `SearchLog`: Logs user searches for recommendations
  - `BookClick`: Tracks which books users click on
  
- **Main Components**:
  - Search engine with keyword and RegEx capabilities
  - Centrality ranking algorithms (PageRank, Closeness, Betweenness)
  - Book recommendation system based on Jaccard similarity
  - Responsive web UI with Bootstrap
  - RESTful API for all functionality

## Technical Implementation

- Built with Django and Django REST Framework
- Uses algorithms from network theory for ranking
- Implements Jaccard similarity for content recommendations
- Employs efficient database indexing techniques
- Provides both API endpoints and rendered templates
- File-based content storage for reduced database size

## Requirements

- Python 3.8+
- Django 4.2+
- Django REST Framework
- NumPy
- SciPy
- NetworkX
- NLTK
- Other dependencies in requirements.txt

## Setup and Running

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: 
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Download required NLTK data: `python manage.py download_nltk_data`
6. Apply migrations: `python manage.py migrate`
7. Fetch books from Project Gutenberg: `python manage.py fetch_gutenberg_books`
8. Run the server: `python manage.py runserver`
9. Visit `http://localhost:8000/` in your browser


  pip install tqdm

  # Run the improved fetcher
  python manage.py fetch_gutenberg_books_improved --count 1664 --max-workers 20

  # If you want to clean up orphaned files and short books
  python manage.py fetch_gutenberg_books_improved --cleanup

## Troubleshooting

If you encounter NLTK-related errors:
- Make sure to run `python manage.py download_nltk_data`
- Or manually download required data:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

## Final Project - PRIMARY CHOICE

This project fulfills the requirements of the DAAR Final Assignment - PRIMARY CHOICE.

© 2024