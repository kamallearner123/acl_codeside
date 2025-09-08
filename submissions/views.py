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


class SubmissionDetailView(LoginRequiredMixin, DetailView):
    model = Submission
    template_name = 'submissions/detail.html'
    context_object_name = 'submission'
    
    def get_queryset(self):
        return Submission.objects.filter(user=self.request.user)
