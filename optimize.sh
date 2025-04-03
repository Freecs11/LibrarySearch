#!/bin/bash

echo "Verifying optimal_index table structure..."
# python manage.py verify_optimal_index

echo "Optimizing database for better performance..."
# python manage.py optimize_database

echo "Starting server..."
python manage.py runserver