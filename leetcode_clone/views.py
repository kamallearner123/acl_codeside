from django.shortcuts import render
from questions.models import Question, DailyChallenge
from django.utils import timezone
import random

def home(request):
    today = timezone.now().date()
    daily_challenge = DailyChallenge.objects.filter(date=today).first()
    
    if not daily_challenge:
        # Pick a random active question to be today's challenge
        questions = Question.objects.filter(is_active=True)
        if questions.exists():
            random_question = random.choice(questions)
            daily_challenge = DailyChallenge.objects.create(
                question=random_question,
                date=today
            )
            
    return render(request, 'home.html', {'daily_challenge': daily_challenge})

def events(request):
    return render(request, 'events.html')
