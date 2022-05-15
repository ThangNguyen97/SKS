from django.urls import path
from .views import KeywordView

# app_name = 'main'
urlpatterns = [
    path('', KeywordView.as_view(), name='keyword'),
]