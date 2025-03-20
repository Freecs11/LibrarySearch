"""
Django management command to download required NLTK data.
"""
import nltk
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Download required NLTK data for the search engine'

    def handle(self, *args, **options):
        self.stdout.write('Downloading NLTK data...')
        
        # Download required NLTK datasets
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.stdout.write(self.style.SUCCESS('Successfully downloaded NLTK data'))