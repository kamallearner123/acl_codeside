from django.shortcuts import render, redirect
from django.contrib.auth import login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import UserRegistrationForm, UserProfileForm
from .models import UserProfile
from django.db.models import Count
from questions.models import Question
from submissions.models import Submission
import sys

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            login(request, user)
            return redirect('questions:list')
    else:
        form = UserRegistrationForm()
    return render(request, 'accounts/register.html', {'form': form})


@login_required
def logout(request):
    auth_logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('accounts:login')


@login_required
def profile(request):
    user = request.user
    
    # Aggregate solved problems by difficulty
    solved_stats = Question.objects.filter(
        submissions__user=user,
        submissions__status='accepted'
    ).distinct().values('difficulty').annotate(count=Count('id'))
    
    difficulty_counts = {
        'easy': 0,
        'medium': 0,
        'hard': 0
    }
    for stat in solved_stats:
        difficulty_counts[stat['difficulty']] = stat['count']
        
    # Total active questions by difficulty
    total_stats = Question.objects.filter(is_active=True).values('difficulty').annotate(count=Count('id'))
    total_counts = {
        'easy': 0,
        'medium': 0,
        'hard': 0
    }
    for stat in total_stats:
        total_counts[stat['difficulty']] = stat['count']
        
    context = {
        'user': user,
        'difficulty_counts': difficulty_counts,
        'total_counts': total_counts,
    }
    
    return render(request, 'accounts/profile.html', context)


@login_required
def edit_profile(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('accounts:profile')
    else:
        form = UserProfileForm(instance=request.user.profile)
    return render(request, 'accounts/edit_profile.html', {'form': form})


@login_required
def leaderboard(request):
    users = UserProfile.objects.select_related('user').order_by('-points', '-problems_solved')[:50]
    return render(request, 'accounts/leaderboard.html', {'users': users})