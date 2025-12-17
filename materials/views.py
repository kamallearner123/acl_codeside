from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView


class MaterialsDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "materials/dashboard.html"


class MLUsingPythonView(LoginRequiredMixin, TemplateView):
    template_name = "materials/ml_using_python.html"


class MLHandsonView(LoginRequiredMixin, TemplateView):
    template_name = "materials/ml_handson.html"


class PythonMaterialsView(LoginRequiredMixin, TemplateView):
    template_name = "materials/python_programming.html"


class RustMaterialsView(LoginRequiredMixin, TemplateView):
    template_name = "materials/rust_programming.html"


class CProgrammingMaterialsView(LoginRequiredMixin, TemplateView):
    template_name = "materials/c_programming.html"


class DSAMaterialsView(LoginRequiredMixin, TemplateView):
    template_name = "materials/dsa_concepts.html"
