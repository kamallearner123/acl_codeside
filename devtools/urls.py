from django.urls import path
from . import views

app_name = 'devtools'

urlpatterns = [
    path('', views.DevToolListView.as_view(), name='list'),
    path('roughnote/', views.roughnote_detail, name='roughnote_detail'),
    path('<slug:slug>/', views.DevToolDetailView.as_view(), name='detail'),
]
