from django.urls import path
from . import views

app_name = 'tutor'

urlpatterns = [
    path('', views.index, name='index'),
    path('run-code/', views.run_code, name='run_code'),
]
