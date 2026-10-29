"""
Seed script for the "DIY with STM32 Arm Cortex-M Boards: The Complete 50-Project Embedded Engineering Masterclass" blog post.

Author: Kamal Kumar Mukiri
Company: Apt Computing Labs
This script is idempotent - it can be re-run safely (get_or_create + update).

Usage:
    python seed_stm32_diy_blog.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")

import django
django.setup()

from django.utils import timezone
from blogs.models import Post

from scripts.blog_sections.prereqs import get_prerequisites_section
from scripts.blog_sections.hardware_and_rules import get_hardware_and_rules_section
from scripts.blog_sections.level1 import get_level1_section
from scripts.blog_sections.level2 import get_level2_section
from scripts.blog_sections.level3 import get_level3_section
from scripts.blog_sections.level4 import get_level4_section
from scripts.blog_sections.level5 import get_level5_section
from scripts.blog_sections.conclusion import get_conclusion_section
from scripts.blog_sections.svg_loader import inline_all_stm32_svgs

SLUG = "diy-with-stm32-arm-cortex-m-boards"
TITLE = "DIY with STM32 Arm Cortex-M Boards: The Complete 50-Project Embedded Engineering Masterclass"
AUTHOR = "Kamal Kumar Mukiri"
EXCERPT = (
    "A comprehensive 50-project practical curriculum for mastering STM32 ARM Cortex-M microcontrollers—from "
    "bare-metal GPIOs and hardware timers to sensor acquisition, communication protocols (I2C, SPI, UART, CAN), "
    "and mission-critical integrated systems."
)
COVER_IMAGE = "/static/blogs/stm32-diy/stm32-diy-masterclass-cover.jpg"

def build_content():
    parts = [
        get_prerequisites_section(),
        get_hardware_and_rules_section(),
        get_level1_section(),
        get_level2_section(),
        get_level3_section(),
        get_level4_section(),
        get_level5_section(),
        get_conclusion_section()
    ]
    raw_html = "\n\n".join(parts)
    return inline_all_stm32_svgs(raw_html)

def seed_blog():
    html_content = build_content()

    post, created = Post.objects.get_or_create(
        slug=SLUG,
        defaults={
            "title": TITLE,
            "author": AUTHOR,
            "excerpt": EXCERPT,
            "content": html_content,
            "image_url": COVER_IMAGE,
            "published_date": timezone.now(),
        }
    )

    if not created:
        post.title = TITLE
        post.author = AUTHOR
        post.excerpt = EXCERPT
        post.content = html_content
        post.image_url = COVER_IMAGE
        post.published_date = timezone.now()
        post.save()
        print(f"Updated existing post '{SLUG}' (ID: {post.id})")
    else:
        print(f"Created new post '{SLUG}' (ID: {post.id})")

if __name__ == "__main__":
    seed_blog()

