from django.views.generic import ListView, DetailView
from .models import Event
from django.utils import timezone

class EventListView(ListView):
    model = Event
    template_name = 'events/list.html'
    context_object_name = 'events'
    
    def get_queryset(self):
        return Event.objects.all()

class EventDetailView(DetailView):
    model = Event
    template_name = 'events/detail.html'
    context_object_name = 'event'
