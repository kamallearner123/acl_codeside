from django.views.generic import ListView
from .models import DevTool


class DevToolListView(ListView):
    model = DevTool
    template_name = 'devtools/list.html'
    context_object_name = 'tools'
