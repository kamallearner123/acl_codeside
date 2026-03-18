from django.shortcuts import render
from questions.models import Question, DailyChallenge
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import random

@login_required
def youtube_progress(request):
    topics = [
        {"domain": "Core Rust", "topic": "Ownership Deep Dive", "date": "2026-04-01"},
        {"domain": "Core Rust", "topic": "Borrowing & References", "date": "2026-04-04"},
        {"domain": "Core Rust", "topic": "Lifetimes Explained", "date": "2026-04-07"},
        {"domain": "Core Rust", "topic": "Structs & Enums in Depth", "date": "2026-04-10"},
        {"domain": "Core Rust", "topic": "Traits & Generics", "date": "2026-04-13"},
        {"domain": "Core Rust", "topic": "Error Handling (Result & Option)", "date": "2026-04-16"},
        {"domain": "System Programming", "topic": "Memory Layout in Rust", "date": "2026-04-19"},
        {"domain": "System Programming", "topic": "Unsafe Rust Explained", "date": "2026-04-22"},
        {"domain": "System Programming", "topic": "FFI with C/C++", "date": "2026-04-25"},
        {"domain": "System Programming", "topic": "Multithreading & Send/Sync", "date": "2026-04-28"},
        {"domain": "System Programming", "topic": "Async vs Threads", "date": "2026-05-01"},
        {"domain": "System Programming", "topic": "File Handling & mmap", "date": "2026-05-04"},
        {"domain": "Automotive", "topic": "Why Rust for Automotive?", "date": "2026-05-07"},
        {"domain": "Automotive", "topic": "Embedded Rust Overview", "date": "2026-05-10"},
        {"domain": "Automotive", "topic": "Working with CAN Protocol", "date": "2026-05-13"},
        {"domain": "Automotive", "topic": "RTOS Concepts in Rust", "date": "2026-05-16"},
        {"domain": "Automotive", "topic": "Memory Safety in ECUs", "date": "2026-05-19"},
        {"domain": "Automotive", "topic": "Rust vs C++ in Safety Systems", "date": "2026-05-22"},
        {"domain": "Web Development", "topic": "Intro to Actix Web", "date": "2026-05-25"},
        {"domain": "Web Development", "topic": "Building REST API in Rust", "date": "2026-05-28"},
        {"domain": "Web Development", "topic": "Database Integration (Diesel/SQLx)", "date": "2026-05-31"},
        {"domain": "Web Development", "topic": "Authentication & JWT", "date": "2026-06-03"},
        {"domain": "Web Development", "topic": "Deploying Rust Web App", "date": "2026-06-06"},
        {"domain": "Web Development", "topic": "WebAssembly with Rust", "date": "2026-06-09"},
        {"domain": "Machine Learning", "topic": "Rust for ML Overview", "date": "2026-06-12"},
        {"domain": "Machine Learning", "topic": "Using ndarray in Rust", "date": "2026-06-15"},
        {"domain": "Machine Learning", "topic": "Rust + Python Interop (PyO3)", "date": "2026-06-18"},
        {"domain": "Machine Learning", "topic": "Running ONNX Models in Rust", "date": "2026-06-21"},
        {"domain": "Machine Learning", "topic": "High Performance Inference", "date": "2026-06-24"},
        {"domain": "Machine Learning", "topic": "Parallel Data Processing", "date": "2026-06-27"},
        {"domain": "Networking", "topic": "TCP/UDP in Rust", "date": "2026-06-30"},
        {"domain": "Networking", "topic": "Building a TCP Server", "date": "2026-07-03"},
        {"domain": "Networking", "topic": "Async Networking with Tokio", "date": "2026-07-06"},
        {"domain": "Networking", "topic": "Protocol Design Basics", "date": "2026-07-09"},
        {"domain": "Networking", "topic": "Packet Parsing in Rust", "date": "2026-07-12"},
        {"domain": "Networking", "topic": "gRPC in Rust", "date": "2026-07-15"},
        {"domain": "Cyber Security", "topic": "Secure Coding in Rust", "date": "2026-07-18"},
        {"domain": "Cyber Security", "topic": "Memory Safety & Exploit Prevention", "date": "2026-07-21"},
        {"domain": "Cyber Security", "topic": "Cryptography Libraries in Rust", "date": "2026-07-24"},
        {"domain": "Cyber Security", "topic": "Building a Simple Auth System", "date": "2026-07-27"},
        {"domain": "Cyber Security", "topic": "Threat Modeling for Rust Apps", "date": "2026-07-30"},
        {"domain": "Cyber Security", "topic": "Rust for Secure Systems", "date": "2026-08-02"}
    ]
    return render(request, 'youtube_progress.html', {'topics': topics})

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
