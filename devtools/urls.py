from django.urls import path
from . import views

app_name = 'devtools'

urlpatterns = [
    path('', views.DevToolListView.as_view(), name='list'),
]
