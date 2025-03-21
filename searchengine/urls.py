"""URL patterns for the search engine."""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router for viewsets
router = DefaultRouter()
router.register(r'books', views.BookViewSet)

urlpatterns = [
    # API endpoints for search functionality
    path('api/search/', views.SearchView.as_view(), name='search'),
    path('api/statistics/', views.StatisticsView.as_view(), name='statistics'),
    path('api/', include(router.urls)),
    
    # Web UI
    path('', views.SearchPageView.as_view(), name='search_page'),
    path('books/', views.BooksPageView.as_view(), name='books_page'),
    path('book/<int:pk>/', views.BookDetailView.as_view(), name='book_detail'),
    path('stats/', views.StatsPageView.as_view(), name='stats_page'),
]