from django.urls import path
from . import views

app_name = "materials"

urlpatterns = [
    path("", views.MaterialsDashboardView.as_view(), name="dashboard"),
    path("ml-using-python/", views.MLUsingPythonView.as_view(), name="ml_python"),
    path("ml-handson/", views.MLHandsonView.as_view(), name="ml_handson"),
    path("python-programming/", views.PythonMaterialsView.as_view(), name="python"),
    path("rust-programming/", views.RustMaterialsView.as_view(), name="rust"),
    path("c-programming/", views.CProgrammingMaterialsView.as_view(), name="c_programming"),
    path("dsa-concepts/", views.DSAMaterialsView.as_view(), name="dsa_concepts"),
]
