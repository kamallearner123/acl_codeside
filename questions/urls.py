from django.urls import path
from . import views

app_name = 'questions'

urlpatterns = [
    path('', views.QuestionListView.as_view(), name='list'),
    path('<slug:slug>/', views.QuestionDetailView.as_view(), name='detail'),
    path('<slug:slug>/submit/', views.submit_solution, name='submit'),
    path('<slug:slug>/comment/', views.post_comment, name='post_comment'),
    path('stats/', views.solved_stats, name='stats'),
    path('<slug:slug>/solvers/', views.question_solvers, name='solvers'),
]
