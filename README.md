# Library Search Engine

A search engine for a digital library of books, featuring efficient indexing, keyword and regex search, and book recommendations.

## Features

- **Efficient Indexing**: Each word is stored once with a list of book IDs for space efficiency
- **Keyword Search**: Fast search for books containing specific words
- **Regex Search**: Advanced search using regular expressions
- **Book Recommendations**: Get recommendations based on book similarity
- **Jaccard Similarity**: Books are related by their word similarity (Jaccard index)
- **Centrality Ranking**: Books ranked by their importance in the similarity graph

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the database:
   ```
   ./update_schema.sh
   ```
4. Download NLTK data (required for text processing):
   ```
   python manage.py download_nltk_data
   ```
5. Index some books:
   ```
   python manage.py reindex_local_books
   ```

## Running the Application

Use the optimized runner to start the server:

```
./optimize.sh
```

This script:
1. Verifies the optimal_index table structure
2. Optimizes the database for better performance
3. Starts the Django development server

The server will be available at http://127.0.0.1:8000/

## Management Commands

- `python manage.py reindex_local_books`: Re-index books from the local storage
- `python manage.py download_gutenberg_books`: Download books from Project Gutenberg
- `python manage.py verify_optimal_index`: Verify and repair the optimal_index table
- `python manage.py optimize_database`: Optimize the database for better performance
- `python manage.py export_jaccard_matrix`: Export the Jaccard similarity matrix

## Maintenance

### Cleaning Up Unnecessary Files

Run the cleanup script to remove unnecessary files from previous development iterations:

```
./cleanup.sh
```

### Updating the Database Schema

If you make changes to the models, update the database schema:

```
./update_schema.sh
```

## Architecture

This project uses:

- **Django**: Web framework
- **Django REST Framework**: API endpoints
- **SQLite**: Database
- **NLTK**: Natural language processing
- **NetworkX**: Graph analysis for book similarity

### Database Models

- **Book**: Represents a book in the library
- **OptimalIndex**: Efficient word indexing (one word → many books)
- **BookSimilarity**: Graph edges representing similarity between books
- **SearchLog**: Logs of user searches
- **BookClick**: Tracks which books users click on after searching

### Optimal Indexing

The core of the system is the `optimal_index` table, which stores:
- **word**: The indexed word (primary key)
- **book_ids**: JSON array of book IDs containing this word
- **book_counts**: JSON object mapping book IDs to occurrence counts
- **total_occurrences**: Total occurrences across all books

This structure provides significant space efficiency compared to traditional indexing.

## API Endpoints

- `/api/search/?q=<query>&regex=<bool>`: Search for books
- `/api/books/`: List all books
- `/api/books/<id>/`: Get book details
- `/api/books/<id>/click/`: Record a click on a book
- `/api/stats/`: Get search engine statistics

## Web Interface

- `/`: Home page with search interface
- `/books/`: List of all books
- `/book/<id>/`: Book detail page
- `/stats/`: Statistics page
- `/matrix/`: Jaccard similarity matrix visualization