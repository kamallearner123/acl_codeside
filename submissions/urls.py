from django.urls import path
from . import views

app_name = 'submissions'

urlpatterns = [
    path('', views.MySubmissionsView.as_view(), name='my_submissions'),
    path('<int:pk>/', views.SubmissionDetailView.as_view(), name='detail'),
]
