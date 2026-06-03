from django.views.generic import ListView, DetailView
from .models import Service

class ServiceListView(ListView):
    model = Service
    template_name = 'services/list.html'
    context_object_name = 'services'

class ServiceDetailView(DetailView):
    model = Service
    template_name = 'services/detail.html'
    context_object_name = 'service'
