from django.views.generic import ListView, DetailView
from .models import DevTool, Rating
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json


class DevToolListView(ListView):
    model = DevTool
    template_name = 'devtools/list.html'
    context_object_name = 'tools'

class DevToolDetailView(DetailView):
    model = DevTool
    template_name = 'devtools/detail.html'
    context_object_name = 'tool'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        ratings = self.object.ratings.all()
        if ratings:
            avg_rating = sum(r.stars for r in ratings) / len(ratings)
            context['average_rating'] = round(avg_rating, 1)
            context['rating_count'] = len(ratings)
        else:
            context['average_rating'] = 0
            context['rating_count'] = 0
        return context

def roughnote_detail(request):
    return render(request, 'devtools/roughnote_detail.html')

@require_POST
def rate_tool(request, slug):
    tool = get_object_or_404(DevTool, slug=slug)
    try:
        data = json.loads(request.body)
        stars = int(data.get('stars', 0))
        if 1 <= stars <= 5:
            Rating.objects.create(tool=tool, stars=stars)
            
            # Calculate new average
            ratings = tool.ratings.all()
            avg_rating = sum(r.stars for r in ratings) / len(ratings)
            
            return JsonResponse({'status': 'success', 'average_rating': round(avg_rating, 1), 'rating_count': len(ratings)})
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid star rating'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)