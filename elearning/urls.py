from django.urls import path
from . import views

app_name = 'elearning'

urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', views.student_login, name='login'),
    path('logout/', views.student_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('enroll/<int:course_id>/', views.enroll_course, name='enroll'),
    path('course/<int:course_id>/', views.course_detail, name='course_detail'),
    path('course/<int:course_id>/update-topic/', views.update_topic_progress, name='update_topic_progress'),
    path('profile/', views.student_profile, name='profile'),
]
