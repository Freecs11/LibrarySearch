"""Configuration for the search engine."""

# Folder path to store downloaded books
BOOKS_STORAGE_PATH = "books_data"

# Gutenberg Project API settings
GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub"

# Minimum required books in the library
MIN_BOOKS_REQUIRED = 1664

# Minimum words per book
MIN_WORDS_PER_BOOK = 10000

# Ranking settings
# Options: 'occurrence', 'closeness', 'betweenness', 'pagerank'
DEFAULT_RANKING_METHOD = "pagerank"

# Number of similar books to suggest
RECOMMENDATION_LIMIT = 5

# Jaccard similarity threshold (0-1)
# Higher value (0.1-0.2) means fewer connections, better performance
JACCARD_SIMILARITY_THRESHOLD = 0.15

# Maximum results per search query
MAX_SEARCH_RESULTS = 500

# Maximum words to index per book
# Setting too high may impact performance
MAX_WORDS_TO_INDEX = 10000

# Number of parallel workers for graph building
MAX_GRAPH_WORKERS = 4

# Chunk size for batch processing in graph building
GRAPH_CHUNK_SIZE = 25
