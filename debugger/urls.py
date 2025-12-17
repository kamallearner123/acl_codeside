from django.urls import path
from . import views

app_name = 'debugger'

urlpatterns = [
    path('', views.index, name='home'),
    path('rust/', views.rust_editor, name='rust_editor'),
    path('python/', views.python_editor, name='python_editor'),
    # DSA routes removed per request
    path('execute/', views.execute_code, name='execute_code'),
    path('python/execute/', views.execute_python_code, name='execute_python_code'),
    path('cpp/execute/', views.execute_cpp, name='execute_cpp'),
    path('execution/<int:execution_id>/', views.get_execution, name='get_execution'),
]
