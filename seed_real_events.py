import os
import django
import sys
from datetime import datetime
from django.utils import timezone

sys.path.append('/home/kamal/Documents/1.Github/acl_codeside')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from events.models import Event

events_data = [
    {
        "title": "Engaging with Young Engineering Minds",
        "slug": "engaging-engineering-minds",
        "date": timezone.make_aware(datetime(2025, 8, 15, 14, 0)),
        "location": "Bangalore Campus",
        "image_url": "/static/img/embedd_1.jpeg",
        "description": """
<p><strong>It was an amazing time!</strong> 🎓</p>
<p>Today, I met young and enthusiastic minds: <strong>Poornateja S A, Ketan Kumar S, A Mohammad Nasir, Mithun Chakravarthy S, Harshith, and Rahul</strong>. Their energy and passion for technology were truly inspiring!</p>

<h3>A Journey Back in Time</h3>
<p>They took me back to my B.Tech days when I used to struggle with small ICs, blasting transistors, and countless experiments. So many stories, so many fun incidents came flooding back! Time flew by, and before I knew it, 3.5 hours had passed in what felt like minutes.</p>

<h3>Two-Way Knowledge Exchange:</h3>
<p>This wasn't just a mentoring session—it was a genuine two-way knowledge sharing experience. I heard amazing ideas from these brilliant students, and their perspectives on modern embedded systems, IoT, and emerging technologies were refreshing and innovative.</p>

<h3>A Message to Working Professionals:</h3>
<p>I would strongly suggest to working professionals, irrespective of how much experience you have: <strong>talk to current engineering students</strong>. There is so much to learn from them! The energy you get from their enthusiasm, their fresh perspectives on technology, and their fearless approach to problem-solving is invaluable.</p>

<blockquote>"The best way to stay young in technology is to keep engaging with young minds. They remind us that innovation starts with curiosity and ends with persistence."</blockquote>
        """
    },
    {
        "title": "Tech Summit Day 2 at NMIT Bangalore",
        "slug": "tech-summit-nmit-bangalore",
        "date": timezone.make_aware(datetime(2025, 9, 13, 9, 0)),
        "location": "NMIT Bangalore",
        "image_url": "/static/img/Kamal_at_NITTE.jpeg",
        "description": """
<p><strong>It was a wonderful experience to be part of Tech Summit Day 2 at NMIT Bangalore on 13th September!</strong> 🌟</p>
<p>I had the privilege to participate in multiple capacities, contributing to an impactful day of learning, innovation, and entrepreneurship.</p>

<h3>My Roles at the Summit</h3>
<ul>
    <li><strong>Keynote Speaker:</strong> Explained the importance of <strong>Secure Coding</strong> and why it's crucial to build this skill from the start of one's career. Highlighted the significance of <strong>Rust Programming</strong> with the motto: <em>"Drive Safe with Rust 🚗💨!"</em></li>
    <li><strong>Panel Discussion Member:</strong> Shared insights on <strong>"Lab to Market"</strong> exposure and bridging the gap between academic research and industry implementation.</li>
    <li><strong>Jury Member:</strong> Evaluated innovative ideas in the <strong>"Startup Pitch-In"</strong>, providing constructive feedback to aspiring entrepreneurs on their pitches' feasibility, innovation, and potential impact.</li>
</ul>

<h3>Reflections & Gratitude:</h3>
<p>It was a great opportunity to meet exceptional people from industries, academia, and most importantly, the enthusiastic students. The talent and curiosity demonstrated by the students were remarkable.</p>
<p>My heartfelt thanks to the <strong>Principal, faculty, and management of NMIT Bangalore</strong> for organizing such an impactful event and inviting me to contribute. I also extend my gratitude to <strong>Magesh</strong> for introducing me to the organizing committee.</p>
        """
    },
    {
        "title": "Rust Programming Workshop at Vishnu Institute of Technology",
        "slug": "rust-workshop-vit",
        "date": timezone.make_aware(datetime(2025, 9, 19, 9, 0)),
        "location": "Bhimavaram, Andhra Pradesh",
        "image_url": "/static/img/rust-vit-19-20-sept.png",
        "description": """
<p><strong>Excited to share the success of our recent two-day workshop on Rust Programming!</strong> 🌟</p>
<p>It was an incredible experience introducing engineering students and faculty to the power of <strong>Rust</strong>—a language known for its memory safety, performance, and cargo features. The workshop aimed to equip participants with foundational skills in Rust programming, exploring its applications in systems development.</p>

<h3>Workshop Coverage</h3>
<ul>
    <li>Need of Secure programming language in present world</li>
    <li>Core Rust concepts: <strong>ownership, borrowing, lifetimes</strong></li>
    <li>Advanced topics: <strong>traits, crates, structs, collections</strong></li>
    <li>Hands-on coding sessions and practical implementations</li>
    <li>Real-world project development and debugging</li>
</ul>

<h3>Acknowledgments:</h3>
<p>A huge thank you to the enthusiastic students and faculty at <strong>Vishnu Institute of Technology</strong> for their active participation, insightful questions, and collaborative learning environment.</p>
<p>Special appreciation to <strong>Prof. Ratna Babu M</strong> for referring me to this event as trainer and facilitating this valuable educational partnership.</p>
        """
    }
]

print("Seeding past events...")
# Optional: Clear old dummy events if desired
Event.objects.filter(slug__in=['rust-embedded-workshop', 'auto-cyber-summit']).delete()

for data in events_data:
    Event.objects.update_or_create(slug=data['slug'], defaults=data)
print("Events seeded successfully!")
