import os
import django
import sys
from datetime import datetime
from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from events.models import Event

events_data = [
    {
        "title": "Inauguration of AIROGEN – Student Forum at JNNCE, Shivamogga",
        "slug": "airogen-inauguration-jnnce-shivamogga",
        "date": timezone.make_aware(datetime(2026, 11, 9, 10, 0)),
        "location": "JNNCE, Shivamogga, Karnataka",
        "registration_link": "https://www.linkedin.com/feed/update/urn:li:activity:7525574545911599105/",
        "image_url": "/static/img/jnnce_airogen_2026.jpg",
        "description": """
<p><strong>Honoured to be the Chief Guest at the inauguration of AIROGEN – Student Forum 2026–27</strong>, Department of Robotics and Artificial Intelligence, Jawaharlal Nehru National College of Engineering (JNNCE), Shivamogga.</p>

<p>It was a wonderful experience interacting with students and sharing my thoughts on <em>“Beyond Automation: How AI and Robotics Will Shape the Future of Work, Careers and Entrepreneurship.”</em></p>

<h3>Key Discussion Highlights</h3>
<ul>
    <li><strong>Evolution of Automation:</strong> Transitioning from repetitive task execution to intelligent, adaptive, and cognitive robotic systems.</li>
    <li><strong>AI &amp; Robotics Convergence:</strong> The critical role of edge compute, real-time sensing, and embedded software in modern autonomous platforms.</li>
    <li><strong>Career &amp; Entrepreneurship Pathways:</strong> The exponential demand for systems engineers capable of bridging silicon, firmware, and deep learning algorithms.</li>
    <li><strong>Student Mentorship:</strong> Engaging directly with future engineers, exploring their innovative concepts, and fostering an entrepreneurial mindset.</li>
</ul>

<h3>A Special Honour &amp; Gratitude</h3>
<p>A special honour to receive a recognition memento featuring <strong>Shri Ratan Tata</strong>—a symbol of visionary leadership and nation-building.</p>
<p>My sincere thanks to the leadership and faculty of JNNCE for their warm welcome and flawless organisation:</p>
<ul>
    <li><strong>Dr. Y. Vijaya Kumar</strong>, Principal, JNNCE</li>
    <li><strong>Dr. Basappaji K M</strong>, HOD, Department of Robotics and AI</li>
    <li><strong>Dr. Kiran M C</strong> and <strong>Deepankar V K</strong>, for coordinating this impactful engagement.</li>
</ul>

<blockquote>"Innovation in robotics and AI is not just about automating existing tasks; it is about augmenting human potential and creating entirely new industries."</blockquote>
        """
    },
    {
        "title": "Short-Term Industry Training on Embedded Systems & STM32 at AMBIT Belagavi",
        "slug": "embedded-systems-training-ambit-belagavi",
        "date": timezone.make_aware(datetime(2026, 11, 8, 9, 30)),
        "location": "AMBIT, Belagavi, Karnataka",
        "registration_link": "https://www.linkedin.com/feed/update/urn:li:ugcPost:7525516619133988864/",
        "image_url": "/static/img/ambit_stm32_training_2026.jpg",
        "description": """
<p><strong>Yes, I do agree… students are struggling even to blink an LED using STM32. But I would say they are struggling at the right time!</strong> 💡</p>

<p>Recently, I had the opportunity to conduct a comprehensive <strong>Short-Term Industry Training Course</strong> organised by the <strong>AMBIT NextGen Mobility Centre (NextGeM)</strong> at <strong>Anuvartik Mirji Bharatesh Institute of Technology (AMBIT)</strong>, Belagavi, Karnataka, with the generous support of the <strong>AMBIT-IRISE Foundation</strong>.</p>

<h3>Hands-On Mastery of Embedded Systems</h3>
<p>It gives me immense satisfaction to teach core concepts that give wings to students' thoughts and ideas. The students showed remarkable energy and enthusiasm, diving deep into hands-on firmware development:</p>
<ul>
    <li><strong>From Bare-Metal to HAL:</strong> Demystifying microcontroller architectures, clock trees, register manipulation, and vector interrupt controllers.</li>
    <li><strong>STM32 and Rust Programming:</strong> Exploring why mastering core concepts in <strong>STM32 and Rust</strong> holds immense promise for next-generation systems.</li>
    <li><strong>Cross-Industry Applications:</strong> Practical firmware design patterns across <strong>IoT, Edge Devices, Automotive, Medical Devices, and Industrial Automation</strong>.</li>
    <li><strong>Laboratory Bring-Up:</strong> Real-time hardware debugging with ST-LINK and oscilloscopes, building robust foundations for engineering careers.</li>
</ul>

<h3>Institutional Leadership &amp; Acknowledgments</h3>
<p>I would like to take this opportunity to convey my sincere thanks to the leaders behind this initiative:</p>
<ul>
    <li><strong>Dr. M. S. Patil</strong>, Principal, AMBIT</li>
    <li><strong>Dr. Veena Karchi</strong>, Director, AMBIT-IRISE Foundation</li>
    <li><strong>Dr. Gomatesh M. Ravanavar</strong>, Administrator, AMBIT</li>
    <li><strong>Dr. Shrikanth Tagade</strong>, for envisioning this initiative and taking concrete steps to execute it.</li>
</ul>
<p>A big thank you to all the enthusiastic students who participated—including <em>Tejashwi Sawant, Babajan Panari, Deepak S Mane, Chetan Hallad, Yallapp</em>, and the entire cohort. Your trust, curiosity, and sincere effort to learn new technologies made this a truly memorable experience!</p>

<blockquote>"Mastering core concepts in Embedded Systems and Rust requires struggling through physical hardware realities—and that struggle is the foundation of true engineering excellence."</blockquote>
        """
    },
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

<h3>Reflections &amp; Gratitude:</h3>
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

print("Seeding past & new events...")
Event.objects.filter(slug__in=['rust-embedded-workshop', 'auto-cyber-summit']).delete()

for data in events_data:
    obj, created = Event.objects.update_or_create(slug=data['slug'], defaults=data)
    print(("Created" if created else "Updated"), "event:", obj.title, f"({obj.date.date()})")
print("Events seeded successfully!")
