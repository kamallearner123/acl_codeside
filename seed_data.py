import os
import django
import sys

sys.path.append('/home/kamal/Documents/1.Github/acl_codeside')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from services.models import Service
from blogs.models import Post
from events.models import Event
from django.utils import timezone
from datetime import timedelta

print("Seeding Services...")
services_data = [
    {"title": "Industry Training & MaaS", "slug": "industry-training", "icon_class": "fas fa-briefcase", "short_description": "Comprehensive corporate training and Model-as-a-Service solutions.", "description": "Full description of our industry training program.", "is_featured": True, "order": 1},
    {"title": "Rust Automotive Program", "slug": "rust-automotive", "icon_class": "fab fa-rust", "short_description": "Pioneering Rust in automotive software systems.", "description": "Full description of Rust Automotive.", "is_featured": True, "order": 2},
    {"title": "Automotive IDS/IDPS", "slug": "automotive-ids", "icon_class": "fas fa-shield-alt", "short_description": "Advanced intrusion detection and prevention systems.", "description": "Details about Automotive IDS.", "is_featured": False, "order": 3},
    {"title": "Building AI Agents", "slug": "building-ai-agents", "icon_class": "fas fa-robot", "short_description": "Design and implement autonomous AI agents for real-world tasks.", "description": "Agent architectures, reinforcement learning basics, and safe deployment practices.", "is_featured": False, "order": 4},
]
for data in services_data:
    Service.objects.get_or_create(slug=data['slug'], defaults=data)

print("Seeding Blogs...")
blogs_data = [
    {"title": "Why Rust is the Future of Automotive", "slug": "rust-future-automotive", "author": "Kamal", "excerpt": "Exploring memory safety in critical automotive systems.", "content": "Full article content goes here...", "image_url": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&q=80&w=800"},
    {"title": "Building V2X Architecture", "slug": "building-v2x", "author": "Dhanush", "excerpt": "A deep dive into Vehicle-to-Everything communication protocols.", "content": "Full article content goes here...", "image_url": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800"},
]
for data in blogs_data:
    Post.objects.get_or_create(slug=data['slug'], defaults=data)

print("Seeding Events...")
now = timezone.now()
events_data = [
    {"title": "Rust for Embedded Systems Workshop", "slug": "rust-embedded-workshop", "date": now + timedelta(days=10), "location": "Bangalore Campus / Online", "description": "A 2-day hands-on workshop.", "registration_link": "https://example.com/register"},
    {"title": "Automotive Cybersecurity Summit", "slug": "auto-cyber-summit", "date": now + timedelta(days=30), "location": "Online", "description": "Discussing the latest trends in IDS/IDPS.", "registration_link": "https://example.com/register"},
]
for data in events_data:
    Event.objects.get_or_create(slug=data['slug'], defaults=data)

print("Seeding Complete!")

# --- Courses ---
try:
    from courses.models import Course
    print("Seeding Courses...")
    courses_data = [
        {
            'title': 'Agentic AI',
            'slug': 'agentic-ai',
            'short_description': 'Design and deploy autonomous agent systems.',
            'description': 'Comprehensive course on agent architectures, RL, safety and deployment.',
            'duration_weeks': 6,
            'skill_level': 'Advanced',
            'technologies': 'AI,Agents,Reinforcement Learning'
        }
    ]
    for data in courses_data:
        Course.objects.get_or_create(slug=data['slug'], defaults=data)
except Exception:
    # If courses app isn't available in this environment, skip silently
    pass
