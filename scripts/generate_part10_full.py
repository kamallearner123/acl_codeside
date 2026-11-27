#!/usr/bin/env python3
"""
generate_part10_full.py - Generates all 8 Capstone engineering masterclass chapters:
Capstone 01: Bare-Metal Bootloader with CRC32 Verification & Dual-Bank Fallback
Capstone 02: Deterministic Memory Pool Allocator for Hard Real-Time Systems
Capstone 03: DMA-Driven Circular Ring Buffer Serial Engine
Capstone 04: Minimal Preemptive RTOS Kernel for ARM Cortex-M
Capstone 05: Production Embedded CLI Shell with Command Auto-Completion
Capstone 06: MISRA C Compliant Multi-Sensor Fusion Pipeline
Capstone 07: Post-Mortem Crash Dump Diagnostic Engine
Capstone 08: High-Performance Multi-Client POSIX epoll Daemon
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

from generate_part10_helpers1 import (
    CAPSTONE01_ARTICLE, CAPSTONE02_ARTICLE,
    CAPSTONE03_ARTICLE, CAPSTONE04_ARTICLE
)
from generate_part10_helpers2 import (
    CAPSTONE05_ARTICLE, CAPSTONE06_ARTICLE,
    CAPSTONE07_ARTICLE, CAPSTONE08_ARTICLE
)

if __name__ == "__main__":
    print("--- Upgrading Part 10 Capstones ---")
    replace_article_in_file("capstone01-bare-metal-bootloader.html", CAPSTONE01_ARTICLE)
    replace_article_in_file("capstone02-memory-pool-allocator.html", CAPSTONE02_ARTICLE)
    replace_article_in_file("capstone03-dma-serial-engine.html", CAPSTONE03_ARTICLE)
    replace_article_in_file("capstone04-preemptive-rtos-kernel.html", CAPSTONE04_ARTICLE)
    replace_article_in_file("capstone05-embedded-cli-shell.html", CAPSTONE05_ARTICLE)
    replace_article_in_file("capstone06-misra-sensor-fusion.html", CAPSTONE06_ARTICLE)
    replace_article_in_file("capstone07-crash-dump-analyzer.html", CAPSTONE07_ARTICLE)
    replace_article_in_file("capstone08-posix-epoll-daemon.html", CAPSTONE08_ARTICLE)
    print("--- Part 10 Completed Successfully! ---")
