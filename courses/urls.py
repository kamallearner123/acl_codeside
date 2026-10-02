from django.urls import path
from . import views

app_name = 'courses'

urlpatterns = [
    path('', views.CourseListView.as_view(), name='list'),
    path('agentic-ai/', views.removed_course, name='removed_agentic_ai'),
    path('exploring-zephyr-using-stm32/book/', views.zephyr_book_index, name='zephyr_book_index'),
    path('exploring-zephyr-using-stm32/book/<path:subpath>', views.zephyr_book_serve, name='zephyr_book_serve'),
    path('<slug:slug>/', views.CourseDetailView.as_view(), name='detail'),
    path('<slug:slug>/enroll/', views.enroll_course, name='enroll'),
]
