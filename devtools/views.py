from django.views.generic import ListView, DetailView
from .models import DevTool
from django.shortcuts import render



class DevToolListView(ListView):
    model = DevTool
    template_name = 'devtools/list.html'
    context_object_name = 'tools'

class DevToolDetailView(DetailView):
    model = DevTool
    template_name = 'devtools/detail.html'
    context_object_name = 'tool'
def roughnote_detail(request):
    return render(request, 'devtools/roughnote_detail.html')