
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
