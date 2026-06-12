import os
import sys
import django
from collections import defaultdict

# Ensure project root is on sys.path so Django settings can be imported
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leetcode_clone.settings')
django.setup()

from courses.models import Course

def fix_agentic():
    c = Course.objects.filter(slug='agentic-ai').first()
    if c:
        c.title = 'Agentic AI'
        c.duration_weeks = 6
        c.short_description = 'Build autonomous agent systems with RL, planning and safe deployment.'
        c.skill_level = 'Advanced'
        c.technologies = 'AI,Agents,Reinforcement Learning'
        c.save()
        print('Updated agentic-ai:', c.id)
    else:
        print('No agentic-ai record found')

def dedupe_titles():
    d = defaultdict(list)
    for c in Course.objects.all().order_by('id'):
        key = (c.title or '').strip().lower()
        d[key].append(c)

    for key, items in d.items():
        if len(items) > 1:
            keep = items[0]
            for rem in items[1:]:
                # Avoid deleting agentic-ai accidentally; if rem is agentic-ai and keep isn't, swap
                if rem.slug == 'agentic-ai' and keep.slug != 'agentic-ai':
                    keep, rem = rem, keep
                print(f"Deleting duplicate: {rem.id} {rem.slug} ({rem.title})")
                rem.delete()

if __name__ == '__main__':
    fix_agentic()
    dedupe_titles()
    print('Fix script completed')
