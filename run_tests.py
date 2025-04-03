#!/usr/bin/env python
"""
Simple performance tests for the library search engine.
"""
import os
import sys
import time
import json
import matplotlib.pyplot as plt

# Set up Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "librarysearch.settings")
import django

django.setup()

from searchengine.models import Book
from searchengine.utils import search_books

# Create results directory
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "performance_results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test keywords
KEYWORDS = ["love", "tartarin", "death", "life", "king", "peace"]
REGEX_PATTERNS = ["lo[vw]e", "tart.*rin", "[dk]ing", "h[aeiou]t"]


def run_keyword_tests():
    """Run keyword search performance tests."""
    print(f"\n=== Testing keyword search performance ===")
    results = []

    for keyword in KEYWORDS:
        print(f"Searching for '{keyword}'...")

        start_time = time.time()
        books = search_books(keyword)
        end_time = time.time()

        search_time = end_time - start_time
        book_count = len(books)

        print(f"Found {book_count} books in {search_time:.4f} seconds")

        results.append(
            {"keyword": keyword, "time": search_time, "book_count": book_count}
        )

    # Save results
    with open(os.path.join(RESULTS_DIR, "keyword_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Create chart
    create_keyword_chart(results)

    return results


def run_regex_tests():
    """Run regex search performance tests."""
    print(f"\n=== Testing regex search performance ===")
    results = []

    for pattern in REGEX_PATTERNS:
        print(f"Searching for pattern '{pattern}'...")

        start_time = time.time()
        books = search_books(pattern, is_regex=True)
        end_time = time.time()

        search_time = end_time - start_time
        book_count = len(books)

        print(f"Found {book_count} books in {search_time:.4f} seconds")

        results.append(
            {"pattern": pattern, "time": search_time, "book_count": book_count}
        )

    # Save results
    with open(os.path.join(RESULTS_DIR, "regex_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Create chart
    create_regex_chart(results)

    return results


def create_keyword_chart(results):
    """Create a chart for keyword search results."""
    keywords = [r["keyword"] for r in results]
    times = [r["time"] for r in results]
    counts = [r["book_count"] for r in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot search times
    bars = ax1.bar(keywords, times)
    ax1.set_title("Keyword Search Performance")
    ax1.set_xlabel("Keyword")
    ax1.set_ylabel("Search Time (seconds)")

    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{time_val:.3f}s",
            ha="center",
            va="bottom",
        )

    # Plot book counts
    ax2.bar(keywords, counts)
    ax2.set_title("Books Found per Keyword")
    ax2.set_xlabel("Keyword")
    ax2.set_ylabel("Number of Books")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "keyword_search_chart.png"))
    print(f"Chart saved to {os.path.join(RESULTS_DIR, 'keyword_search_chart.png')}")


def create_regex_chart(results):
    """Create a chart for regex search results."""
    patterns = [r["pattern"] for r in results]
    times = [r["time"] for r in results]
    counts = [r["book_count"] for r in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot search times
    bars = ax1.bar(patterns, times)
    ax1.set_title("Regex Search Performance")
    ax1.set_xlabel("Pattern")
    ax1.set_ylabel("Search Time (seconds)")

    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{time_val:.3f}s",
            ha="center",
            va="bottom",
        )

    # Plot book counts
    ax2.bar(patterns, counts)
    ax2.set_title("Books Found per RegEx Pattern")
    ax2.set_xlabel("Pattern")
    ax2.set_ylabel("Number of Books")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "regex_search_chart.png"))
    print(f"Chart saved to {os.path.join(RESULTS_DIR, 'regex_search_chart.png')}")


def create_comparison_chart(keyword_results, regex_results):
    """Create a comparison chart between keyword and regex search."""
    # Calculate averages
    avg_keyword_time = sum(r["time"] for r in keyword_results) / len(keyword_results)
    avg_regex_time = sum(r["time"] for r in regex_results) / len(regex_results)

    # Create bar chart
    methods = ["Keyword Search", "Regex Search"]
    times = [avg_keyword_time, avg_regex_time]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, times)

    # Add labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{time_val:.3f}s",
            ha="center",
            va="bottom",
        )

    plt.title("Average Search Time Comparison")
    plt.ylabel("Average Time (seconds)")
    plt.savefig(os.path.join(RESULTS_DIR, "search_comparison_chart.png"))
    print(
        f"Comparison chart saved to {os.path.join(RESULTS_DIR, 'search_comparison_chart.png')}"
    )


if __name__ == "__main__":
    print(f"Running performance tests...")
    print(f"Library contains {Book.objects.count()} books")

    # Run tests
    keyword_results = run_keyword_tests()
    regex_results = run_regex_tests()

    # Create comparison chart
    create_comparison_chart(keyword_results, regex_results)

    print(f"\nAll tests completed! Results saved to {RESULTS_DIR}")
