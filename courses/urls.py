from django.urls import path
from . import views

app_name = 'courses'

urlpatterns = [
    path('', views.CourseListView.as_view(), name='list'),
    path('<slug:slug>/', views.CourseDetailView.as_view(), name='detail'),
]
