"""
Performance tests for the search engine.
"""
import time
import os
import json
import random
import statistics
from django.test import TestCase, Client
from django.conf import settings
from django.urls import reverse

from searchengine.models import Book
from searchengine.utils import (
    search_books,
    calculate_pagerank,
    calculate_closeness_centrality,
    calculate_betweenness_centrality,
    build_jaccard_graph,
)

# Test configuration
TEST_WORDS = ['love', 'war', 'peace', 'death', 'life', 'king', 'queen', 'sword', 'knight', 'monster']
TEST_REGEX = ['lo[vw]e', 'w[aoi]r', 'pe[ae]ce', '[dk]ing', '[qk]ueen']
TEST_COMPLEX_REGEX = ['[A-Z][a-z]{3,6}', 'th[aeiou]', '[a-z]{2}ing', '[0-9]{1,2}th']

# Results storage
RESULTS_DIR = os.path.join(settings.BASE_DIR, 'test_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class SearchPerformanceTest(TestCase):
    """Performance tests for the search engine."""

    def setUp(self):
        """Set up the test environment."""
        self.client = Client()
        print(f"Database contains {Book.objects.count()} books")

    def test_keyword_search_performance(self):
        """Test keyword search performance."""
        print("\nRunning keyword search performance test...")
        results = []

        for word in TEST_WORDS:
            print(f"Testing keyword search for '{word}'...")
            
            # Run the search and measure time
            start_time = time.time()
            books = search_books(word)
            end_time = time.time()
            
            search_time = end_time - start_time
            book_count = len(books)
            
            print(f"Found {book_count} books in {search_time:.4f} seconds")
            
            # Store the results
            results.append({
                'word': word,
                'time': search_time,
                'book_count': book_count
            })
            
        # Save results to JSON file
        with open(os.path.join(RESULTS_DIR, 'keyword_search_performance.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'keyword_search_performance.json')}")

    def test_regex_search_performance(self):
        """Test regex search performance."""
        print("\nRunning regex search performance test...")
        results = []

        for regex in TEST_REGEX:
            print(f"Testing regex search for '{regex}'...")
            
            # Run the search and measure time
            start_time = time.time()
            books = search_books(regex, is_regex=True)
            end_time = time.time()
            
            search_time = end_time - start_time
            book_count = len(books)
            
            print(f"Found {book_count} books in {search_time:.4f} seconds")
            
            # Store the results
            results.append({
                'regex': regex,
                'time': search_time,
                'book_count': book_count
            })
            
        # Save results to JSON file
        with open(os.path.join(RESULTS_DIR, 'regex_search_performance.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'regex_search_performance.json')}")
    
    def test_api_performance(self):
        """Test API endpoint performance."""
        print("\nRunning API performance test...")
        results = []

        for word in TEST_WORDS[:5]:  # Use fewer words to speed up the test
            print(f"Testing API search for '{word}'...")
            
            # Run the API search and measure time
            start_time = time.time()
            response = self.client.get(reverse('search-books') + f'?query={word}')
            end_time = time.time()
            
            api_time = end_time - start_time
            
            # Check if the response is valid
            self.assertEqual(response.status_code, 200)
            data = response.json()
            book_count = len(data.get('books', []))
            
            print(f"API returned {book_count} books in {api_time:.4f} seconds")
            
            # Store the results
            results.append({
                'word': word,
                'time': api_time,
                'book_count': book_count
            })
            
        # Save results to JSON file
        with open(os.path.join(RESULTS_DIR, 'api_performance.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'api_performance.json')}")


class CentralityPerformanceTest(TestCase):
    """Test case for centrality calculation performance."""
    
    def test_centrality_performance(self):
        """Test the performance of centrality calculations."""
        print("\nRunning centrality calculation performance test...")
        results = {
            'pagerank': {},
            'closeness': {},
            'betweenness': {}
        }
        
        # Build the graph
        print("Building Jaccard similarity graph...")
        graph_start_time = time.time()
        G = build_jaccard_graph()
        graph_time = time.time() - graph_start_time
        
        print(f"Graph built in {graph_time:.4f} seconds")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Test PageRank
        print("Testing PageRank calculation...")
        pagerank_start_time = time.time()
        pagerank = calculate_pagerank(G)
        pagerank_time = time.time() - pagerank_start_time
        
        print(f"PageRank calculated in {pagerank_time:.4f} seconds")
        results['pagerank'] = {
            'time': pagerank_time,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges()
        }
        
        # Test Closeness Centrality
        print("Testing Closeness Centrality calculation...")
        closeness_start_time = time.time()
        closeness = calculate_closeness_centrality(G)
        closeness_time = time.time() - closeness_start_time
        
        print(f"Closeness Centrality calculated in {closeness_time:.4f} seconds")
        results['closeness'] = {
            'time': closeness_time,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges()
        }
        
        # Test Betweenness Centrality
        print("Testing Betweenness Centrality calculation...")
        betweenness_start_time = time.time()
        betweenness = calculate_betweenness_centrality(G)
        betweenness_time = time.time() - betweenness_start_time
        
        print(f"Betweenness Centrality calculated in {betweenness_time:.4f} seconds")
        results['betweenness'] = {
            'time': betweenness_time,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges()
        }
        
        # Save results to JSON file
        with open(os.path.join(RESULTS_DIR, 'centrality_performance.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'centrality_performance.json')}")


class LoadTestCase(TestCase):
    """Test case for load testing."""
    
    def setUp(self):
        """Set up the test environment."""
        self.client = Client()
        
    def test_load_simulation(self):
        """Simulate load on the search API."""
        print("\nRunning load simulation test...")
        results = []
        
        # Simulate different numbers of concurrent users
        for num_users in [1, 5, 10, 20]:
            print(f"Testing with {num_users} concurrent users...")
            response_times = []
            
            # Run queries for each simulated user
            for _ in range(num_users):
                # Choose a random query
                query = random.choice(TEST_WORDS + TEST_REGEX)
                
                # Select the appropriate endpoint
                is_regex = query in TEST_REGEX
                endpoint = reverse('search-books-regex') if is_regex else reverse('search-books')
                
                # Measure response time
                start_time = time.time()
                response = self.client.get(f"{endpoint}?query={query}")
                end_time = time.time()
                
                # Check if the response is valid
                self.assertEqual(response.status_code, 200)
                
                # Calculate response time
                response_time = end_time - start_time
                response_times.append(response_time)
            
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95 = sorted_times[int(len(sorted_times) * 0.95) - 1] if len(sorted_times) >= 20 else max_time
            
            print(f"Results: Avg={avg_time:.4f}s, Min={min_time:.4f}s, Max={max_time:.4f}s, P95={p95:.4f}s")
            
            # Store results
            results.append({
                'num_users': num_users,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'p95': p95
            })
            
        # Save results to JSON file
        with open(os.path.join(RESULTS_DIR, 'load_test.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'load_test.json')}")


# Create a simple script to visualize the results
def create_visualization_script():
    """Create a Python script to visualize the test results."""
    script_path = os.path.join(RESULTS_DIR, 'visualize_results.py')
    
    with open(script_path, 'w') as f:
        f.write('''
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Directory where test results are stored
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filename):
    """Load JSON data from file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def create_keyword_search_chart():
    """Create a chart showing keyword search performance."""
    data = load_json('keyword_search_performance.json')
    if not data:
        print("No keyword search performance data found")
        return
    
    # Extract data
    words = [item['word'] for item in data]
    times = [item['time'] for item in data]
    book_counts = [item['book_count'] for item in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot search times
    ax1.bar(words, times)
    ax1.set_title('Keyword Search Performance')
    ax1.set_xlabel('Search Term')
    ax1.set_ylabel('Search Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot book counts
    ax2.bar(words, book_counts)
    ax2.set_title('Number of Books Found')
    ax2.set_xlabel('Search Term')
    ax2.set_ylabel('Number of Books')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'keyword_search_chart.png'))
    plt.close()
    print("Created keyword search chart")

def create_regex_search_chart():
    """Create a chart showing regex search performance."""
    data = load_json('regex_search_performance.json')
    if not data:
        print("No regex search performance data found")
        return
    
    # Extract data
    patterns = [item['regex'] for item in data]
    times = [item['time'] for item in data]
    book_counts = [item['book_count'] for item in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot search times
    ax1.bar(patterns, times)
    ax1.set_title('Regex Search Performance')
    ax1.set_xlabel('Regex Pattern')
    ax1.set_ylabel('Search Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot book counts
    ax2.bar(patterns, book_counts)
    ax2.set_title('Number of Books Found')
    ax2.set_xlabel('Regex Pattern')
    ax2.set_ylabel('Number of Books')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'regex_search_chart.png'))
    plt.close()
    print("Created regex search chart")

def create_api_performance_chart():
    """Create a chart showing API performance."""
    data = load_json('api_performance.json')
    if not data:
        print("No API performance data found")
        return
    
    # Extract data
    words = [item['word'] for item in data]
    times = [item['time'] for item in data]
    book_counts = [item['book_count'] for item in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot search times
    ax1.bar(words, times)
    ax1.set_title('API Performance')
    ax1.set_xlabel('Search Term')
    ax1.set_ylabel('Response Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot book counts
    ax2.bar(words, book_counts)
    ax2.set_title('Number of Books Found')
    ax2.set_xlabel('Search Term')
    ax2.set_ylabel('Number of Books')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'api_performance_chart.png'))
    plt.close()
    print("Created API performance chart")

def create_centrality_chart():
    """Create a chart comparing centrality calculation performance."""
    data = load_json('centrality_performance.json')
    if not data:
        print("No centrality performance data found")
        return
    
    # Extract data
    methods = list(data.keys())
    times = [data[method]['time'] for method in methods]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(methods, times)
    plt.title('Centrality Calculation Performance')
    plt.xlabel('Method')
    plt.ylabel('Calculation Time (seconds)')
    plt.xticks(rotation=0)
    
    # Add node and edge counts as text
    if 'pagerank' in data:
        node_count = data['pagerank']['node_count']
        edge_count = data['pagerank']['edge_count']
        plt.figtext(0.5, 0.01, f'Graph: {node_count} nodes, {edge_count} edges', 
                   ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'centrality_chart.png'))
    plt.close()
    print("Created centrality performance chart")

def create_load_test_chart():
    """Create a chart showing load test results."""
    data = load_json('load_test.json')
    if not data:
        print("No load test data found")
        return
    
    # Extract data
    users = [item['num_users'] for item in data]
    avg_times = [item['avg_time'] for item in data]
    min_times = [item['min_time'] for item in data]
    max_times = [item['max_time'] for item in data]
    p95_times = [item['p95'] for item in data]
    
    # Create line chart
    plt.figure(figsize=(10, 6))
    plt.plot(users, avg_times, 'o-', label='Average Time')
    plt.plot(users, min_times, 's-', label='Minimum Time')
    plt.plot(users, max_times, '^-', label='Maximum Time')
    plt.plot(users, p95_times, 'D-', label='95th Percentile')
    
    plt.title('Load Test Performance')
    plt.xlabel('Number of Concurrent Users')
    plt.ylabel('Response Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'load_test_chart.png'))
    plt.close()
    print("Created load test chart")

def create_comparison_chart():
    """Create a chart comparing different search methods."""
    keyword_data = load_json('keyword_search_performance.json')
    regex_data = load_json('regex_search_performance.json')
    api_data = load_json('api_performance.json')
    
    if not keyword_data or not regex_data:
        print("Missing data for comparison chart")
        return
    
    # Calculate averages
    avg_keyword_time = sum(item['time'] for item in keyword_data) / len(keyword_data)
    avg_regex_time = sum(item['time'] for item in regex_data) / len(regex_data)
    avg_api_time = sum(item['time'] for item in api_data) / len(api_data) if api_data else 0
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    methods = ['Keyword Search', 'Regex Search', 'API Response']
    avg_times = [avg_keyword_time, avg_regex_time, avg_api_time]
    
    plt.bar(methods, avg_times)
    plt.title('Average Response Time Comparison')
    plt.xlabel('Search Method')
    plt.ylabel('Average Time (seconds)')
    
    # Add values as text labels
    for i, v in enumerate(avg_times):
        plt.text(i, v + 0.01, f'{v:.4f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_chart.png'))
    plt.close()
    print("Created comparison chart")

if __name__ == '__main__':
    print("Generating charts from test results...")
    
    # Create charts
    create_keyword_search_chart()
    create_regex_search_chart()
    create_api_performance_chart()
    create_centrality_chart()
    create_load_test_chart()
    create_comparison_chart()
    
    print("All charts generated successfully!")
''')
    
    print(f"Created visualization script at {script_path}")
    return script_path

# Create the visualization script when tests are loaded
create_visualization_script()