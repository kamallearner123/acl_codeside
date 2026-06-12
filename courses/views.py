from django.views.generic import ListView, DetailView
from .models import Course

class CourseListView(ListView):
    model = Course
    template_name = 'courses/list.html'
    context_object_name = 'courses'
    
    def get_queryset(self):
        """Return courses deduplicated by title (case-insensitive), keeping the first occurrence."""
        qs = list(super().get_queryset().order_by('title'))
        seen = set()
        unique = []
        for c in qs:
            key = (c.title or '').strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

class CourseDetailView(DetailView):
    model = Course
    template_name = 'courses/detail.html'
    context_object_name = 'course'