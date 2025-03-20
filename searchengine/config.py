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
DEFAULT_RANKING_METHOD = "occurrence"

# Number of similar books to suggest
RECOMMENDATION_LIMIT = 5

# Jaccard similarity threshold (0-1)
JACCARD_SIMILARITY_THRESHOLD = 0.05

# Maximum results per search query
MAX_SEARCH_RESULTS = 50

# Maximum words to index per book
# Setting too high may impact performance
MAX_WORDS_TO_INDEX = 20000
