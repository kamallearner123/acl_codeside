#!/usr/bin/env python3
"""
generate_part7_8_full.py - Generates masterclass chapters for:
Part 7:
  Module 33: Hardware Interfacing & MMIO
  Module 34: Linker Scripts & Memory Maps
  Module 35: Startup Code & Reset Sequence
  (Module 36 already complete at 886 lines)
  Module 37: RTOS Fundamentals & FreeRTOS
Part 8:
  Module 38: MISRA C:2012 / MISRA C:2023 Compliance
  Module 39: CERT C & Defensive Coding Practices
"""

import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_course_materials import generate_sidebar_html

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "..", "private_books", "c_programming_professional", "chapters")

def replace_article_in_file(filename, new_article_html):
    filepath = os.path.join(CHAPTERS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = r'<article class="prose">.*?</article>'
    replacement = f'<article class="prose">\n{new_article_html}\n      </article>'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        raise ValueError(f"Could not find <article class='prose'> in {filename}")
    
    sidebar_html = generate_sidebar_html(active_file=filename, is_index=False)
    sidebar_pattern = r'<nav class="sidebar-nav">.*?</nav>'
    sidebar_replacement = f'<nav class="sidebar-nav">\n{sidebar_html}\n      </nav>'
    new_content, s_count = re.subn(sidebar_pattern, sidebar_replacement, new_content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    line_count = len(new_content.splitlines())
    sz_kb = len(new_content.encode("utf-8")) // 1024
    print(f"Updated {filename}: {line_count} lines, {sz_kb} KB")

# -------------------------------------------------------------
# Import the modular articles from generate_part7_8_helpers.py
# -------------------------------------------------------------
from generate_part7_8_helpers import (
    MOD33_ARTICLE, MOD34_ARTICLE, MOD35_ARTICLE,
    MOD37_ARTICLE, MOD38_ARTICLE, MOD39_ARTICLE
)

if __name__ == "__main__":
    print("--- Upgrading Part 7 & 8 Modules ---")
    replace_article_in_file("mod33-hardware-interfacing-and-mmio.html", MOD33_ARTICLE)
    replace_article_in_file("mod34-linker-scripts-and-memory-maps.html", MOD34_ARTICLE)
    replace_article_in_file("mod35-startup-code-and-reset-sequence.html", MOD35_ARTICLE)
    replace_article_in_file("mod37-rtos-fundamentals-and-freertos.html", MOD37_ARTICLE)
    replace_article_in_file("mod38-misra-c-compliance.html", MOD38_ARTICLE)
    replace_article_in_file("mod39-cert-c-and-defensive-coding.html", MOD39_ARTICLE)
    print("--- Part 7 & 8 Completed Successfully! ---")
