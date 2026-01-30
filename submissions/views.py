from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView
from .models import Submission


class MySubmissionsView(LoginRequiredMixin, ListView):
    model = Submission
    template_name = 'submissions/my_submissions.html'
    context_object_name = 'submissions'
    paginate_by = 20
    
    def get_queryset(self):
        return Submission.objects.filter(user=self.request.user).order_by('-submitted_at')


from django.db.models import Avg, StdDev, Count

class SubmissionDetailView(LoginRequiredMixin, DetailView):
    model = Submission
    template_name = 'submissions/detail.html'
    context_object_name = 'submission'
    
    def get_queryset(self):
        return Submission.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get all successful submissions for this question to calculate distribution
        all_successful = Submission.objects.filter(
            question=self.object.question,
            status='accepted'
        ).values_list('runtime', flat=True)
        
        context['all_runtimes'] = list(all_successful)
        
        # Simple percentile calculation
        if self.object.status == 'accepted' and self.object.runtime is not None:
            faster_count = sum(1 for r in all_successful if r < self.object.runtime)
            total_count = len(all_successful)
            if total_count > 0:
                context['percentile'] = round((1 - faster_count / total_count) * 100, 1)
        
        return context
