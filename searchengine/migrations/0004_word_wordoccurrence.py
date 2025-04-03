# Generated manually
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('searchengine', '0003_booksimilarity_book_similar_books_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Word',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.CharField(db_index=True, max_length=100, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='WordOccurrence',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('occurrences', models.IntegerField(default=1)),
                ('book', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='word_occurrences', to='searchengine.book')),
                ('word', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='book_occurrences', to='searchengine.word')),
            ],
            options={
                'unique_together': {('book', 'word')},
            },
        ),
        migrations.AddIndex(
            model_name='wordoccurrence',
            index=models.Index(fields=['book'], name='searchengin_book_id_7c6b52_idx'),
        ),
        migrations.AddIndex(
            model_name='wordoccurrence',
            index=models.Index(fields=['word'], name='searchengin_word_id_8c6ea7_idx'),
        ),
    ]