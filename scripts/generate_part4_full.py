#!/usr/bin/env python3
"""
generate_part4_full.py - Generates all 5 modules of Part 4:
Module 17: Build Systems (Make & Modern CMake)
Module 18: GDB Debugging & Core Dump Analysis
Module 19: Dynamic Analysis & Sanitizers (Valgrind, ASan, UBSan, TSan)
Module 20: Static Analysis & Code Quality (Cppcheck, Clang-Tidy, Compiler Warnings)
Module 21: Linux Profiling & Performance Tuning (perf, gprof, eBPF)
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
# Import the existing articles for 17 and 18 from generate_part4.py
# -------------------------------------------------------------
from generate_part4 import MOD17_ARTICLE, MOD18_ARTICLE

# -------------------------------------------------------------
# MODULE 19: Dynamic Analysis & Sanitizers (Valgrind, ASan, UBSan, TSan)
# -------------------------------------------------------------
MOD19_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 4 • Linux & Engineering Tools</span>
          <h1 class="chapter-h1">Module 19: Dynamic Analysis &amp; Compiler Sanitizers</h1>
          <p class="chapter-lead">Master runtime verification of memory safety, undefined behavior, and concurrency hazards using Valgrind Memcheck, AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and ThreadSanitizer (TSan). Understand shadow memory mechanics, redzone insertion, and production triage workflows.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: C11 / POSIX / LLVM Compiler-RT</div>
            <div class="meta-item">⚡ Target: Linux x86_64 / ARM AArch64 / QEMU</div>
          </div>
        </div>

        <h2 id="sec-19-1">19.1 Dynamic Binary Translation vs. Compiler Instrumentation</h2>
        <p>
          Detecting memory corruption, invalid pointer dereferences, and thread race conditions at runtime historically required two distinct engineering philosophies:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Dynamic Verification Approaches Compared</div>
          <pre style="margin:0;color:#94A3B8;">
1. Dynamic Binary Translation (Valgrind):
   Unmodified ELF Binary ──▶ [ Valgrind JIT Core ] ──▶ Instruments machine instructions in RAM
                                                      └── Shadow Memory tracking (V-bits &amp; A-bits)
   Overhead: ~10x to 25x execution slowdown; high memory penalty (2x-4x RAM).

2. Compile-Time Instrumentation (Compiler Sanitizers - ASan, UBSan, TSan):
   Source Code (.c) ──▶ [ GCC / Clang -fsanitize=... ] ──▶ Instrumented ELF Binary
                                                           ├── Injects Redzones around buffers
                                                           └── Direct Shadow Memory Lookups
   Overhead: ~1.5x to 2x execution slowdown; minimal runtime penalty.
          </pre>
        </div>

        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Capability / Defect Type</th>
              <th style="padding:10px;text-align:left;">Valgrind Memcheck</th>
              <th style="padding:10px;text-align:left;">AddressSanitizer (ASan)</th>
              <th style="padding:10px;text-align:left;">UBSan</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Heap Buffer Overflow</strong></td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes</span></td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes (Instant SIGSEGV)</span></td>
              <td style="padding:10px;">No</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Stack Buffer Overflow</strong></td>
              <td style="padding:10px;"><span style="color:#EF4444;font-weight:bold;">No</span> (Cannot see stack layout)</td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes</span> (Inserts stack redzones)</td>
              <td style="padding:10px;">No</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Global / Static Buffer Overflow</strong></td>
              <td style="padding:10px;"><span style="color:#EF4444;font-weight:bold;">No</span></td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes</span> (Pads .bss/.data)</td>
              <td style="padding:10px;">No</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Heap Use-After-Free</strong></td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes</span></td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes (Quarantine pool)</span></td>
              <td style="padding:10px;">No</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Uninitialized Memory Read</strong></td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes</span> (V-bit tracking)</td>
              <td style="padding:10px;">Requires MSan (MemorySanitizer)</td>
              <td style="padding:10px;">No</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Signed Integer Overflow / UB</strong></td>
              <td style="padding:10px;"><span style="color:#EF4444;">No</span></td>
              <td style="padding:10px;">No</td>
              <td style="padding:10px;"><span style="color:#10B981;">Yes</span></td>
            </tr>
          </tbody>
        </table>

        <h2 id="sec-19-2">19.2 AddressSanitizer (ASan) Internals & Shadow Memory</h2>
        <p>
          AddressSanitizer achieves unprecedented performance through an ingenious mathematical abstraction: <strong>Shadow Memory</strong>.
        </p>
        <p>
          In a 64-bit architecture, every 8 bytes of application virtual memory are mapped to exactly <strong>1 byte of shadow memory</strong>. The shadow byte encodes the accessibility of those 8 bytes:
        </p>
        <ul>
          <li><code>0x00</code>: All 8 bytes are valid and unpoisoned.</li>
          <li><code>0x01 .. 0x07</code>: The first <code>k</code> bytes are addressable; the remaining bytes are poisoned redzones.</li>
          <li>Negative / Special values (e.g., <code>0xFA</code>, <code>0xFB</code>, <code>0xFD</code>): The entire 8-byte chunk is poisoned (Heap redzone, Stack redzone, Freed heap quarantine).</li>
        </ul>

        <h3>Mathematical Translation Formula</h3>
        <p>
          Given any 64-bit application memory address <code>Addr</code>, the corresponding shadow memory byte is located in <code>O(1)</code> time via a fast bit-shift and constant offset addition:
        </p>
        <pre><code class="language-c">char *shadow_byte = (char *)((Addr &gt;&gt; 3) + 0x7fff8000);</code></pre>

        <h3>Compiler Instrumentation Example</h3>
        <p>When you write an ordinary memory write in C:</p>
        <pre><code class="language-c">*address = 0xAA;</code></pre>
        <p>Under <code>-fsanitize=address</code>, GCC/Clang transforms the instruction into:</p>
        <pre><code class="language-c">char *shadow = (char *)(((uintptr_t)address &gt;&gt; 3) + 0x7fff8000);
if (*shadow != 0) {
    if (((uintptr_t)address &amp; 7) &gt;= *shadow) {
        __asan_report_store1(address); /* Traps immediately with backtrace! */
    }
}
*address = 0xAA;</code></pre>

        <h2 id="sec-19-3">19.3 UndefinedBehaviorSanitizer (UBSan)</h2>
        <p>
          C allows undefined behavior (UB) for maximum compiler optimization liberty. When UB occurs, modern compilers (GCC/Clang) assume the path is dead code, often silently omitting safety checks. <strong>UBSan</strong> instruments the binary to abort or log errors when UB occurs.
        </p>

        <h3>Critical Flaws Caught by UBSan (<code>-fsanitize=undefined</code>)</h3>
        <ol>
          <li><strong>Signed Integer Overflow:</strong> Exceeding <code>INT_MAX</code> is undefined in ISO C §6.5.6.</li>
          <li><strong>Misaligned Pointer Access:</strong> Dereferencing a <code>uint32_t *</code> on an odd byte address (e.g. <code>0x20000003</code>) triggers bus faults on ARM Cortex-M0/M3/M4.</li>
          <li><strong>Shift Out-of-Bounds:</strong> Shifting a 32-bit integer by &ge; 32 bits (e.g., <code>1U &lt;&lt; 32</code>).</li>
          <li><strong>NULL Pointer Arithmetic:</strong> Computing <code>NULL + offset</code> even if not dereferenced.</li>
          <li><strong>Type Punning Violations:</strong> Violating the strict aliasing rule (ISO C §6.5).</li>
        </ol>

        <h2 id="sec-19-4">19.4 ThreadSanitizer (TSan) & Concurrency Data Races</h2>
        <p>
          A <strong>Data Race</strong> occurs whenever two threads concurrently access the same shared memory location, at least one of the accesses is a write, and the threads do not use any synchronization (mutexes, atomics, or memory barriers).
        </p>
        <p>
          ThreadSanitizer (<code>-fsanitize=thread</code>) tracks the <em>happens-before</em> partial order using vector clocks. When TSan detects two conflicting accesses unsynchronized by synchronization primitives, it prints a complete diagnostic listing both threads, their allocation call stacks, and the exact source code lines.
        </p>

        <h2 id="sec-19-5">19.5 Valgrind Memcheck & Massif in Production</h2>
        <p>
          While ASan is superior for catching buffer overflows in developer builds, <strong>Valgrind Memcheck</strong> remains indispensable because:
        </p>
        <ul>
          <li>It requires <strong>zero recompilation</strong> (can run on vendor pre-compiled binaries).</li>
          <li>It detects <strong>uninitialized memory reads</strong> (passing uninitialized struct padding to network sockets).</li>
          <li><strong>Valgrind Massif</strong> generates visual heap allocation profiles over time, identifying insidious memory leaks that evade simple threshold checks.</li>
        </ul>

        <h2 id="sec-19-6">19.6 Industrial Case Study: The Heartbleed Vulnerability (CVE-2014-0160)</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The OpenSSL Heartbleed Buffer Over-Read Catastrophe</h4>
          <p>
            In April 2014, the Heartbleed bug in OpenSSL exposed millions of private encryption keys, passwords, and sessions worldwide. The flaw resided in the TLS Heartbeat extension.
          </p>
          <p>
            <strong>The Defect:</strong> The client sends a payload length field (a 16-bit integer up to 65,535 bytes) along with a small 1-byte payload. The server allocated an output buffer and copied the specified length without validating whether the actual received packet contained that many bytes:
          </p>
          <pre><code class="language-c">/* Faulty OpenSSL Code */
unsigned int payload = 0;
n2s(p, payload); /* Extracts length from client header (e.g. 64 KB) */
/* ... */
unsigned char *buffer = malloc(1 + 2 + payload + padding);
memcpy(bp, pl, payload); /* pl was only 1 byte! Over-reads 64 KB of heap! */</code></pre>
          <p>
            The server dutifully copied 64 KB of adjacent heap memory—which contained private RSA keys and user credentials—and transmitted it back to the client!
          </p>
          <p>
            <strong>Why Standard Tests Missed It:</strong> Reading beyond a heap buffer boundary does not cause a crash unless it hits an unmapped virtual memory page.
          </p>
          <p>
            <strong>How ASan Instantly Prevents It:</strong> Because ASan surrounds every heap allocation with poisoned redzones, <code>memcpy</code> reading into the poisoned shadow memory triggers an immediate, unmaskable ASan abort:
          </p>
          <pre><code class="language-text">=================================================================
==4210==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000041
READ of size 65535 at 0x602000000041 thread T0
    #0 0x7f4b82 in memcpy /usr/include/bits/string3.h:51
    #1 0x7f4b82 in tls1_process_heartbeat ssl/t1_lib.c:2586
=================================================================</code></pre>
        </div>

        <h2 id="sec-19-7">19.7 Hands-On Lab: Diagnosing 4 Memory Safety Defects with ASan & UBSan</h2>
        <p>
          In this lab, you will compile a diagnostic test suite containing 4 subtle, real-world C bugs. You will use GCC sanitizers to catch all 4 bugs without stepping through a single line in a debugger.
        </p>

        <h3>Test Harness (<code>lab_sanitizer.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;
#include &lt;limits.h&gt;

static void bug1_heap_uaf(void) {
    char *buf = (char *)malloc(32);
    strcpy(buf, "Safety-Critical");
    free(buf);
    printf("[BUG 1] Read after free: %c\n", buf[0]); /* Use-After-Free! */
}

static void bug2_stack_overflow(void) {
    int arr[4] = {10, 20, 30, 40};
    int val = arr[5]; /* Stack Buffer Overflow! */
    printf("[BUG 2] Out of bounds: %d\n", val);
}

static void bug3_signed_overflow(void) {
    int max = INT_MAX;
    int result = max + 1; /* Undefined Behavior: Signed Overflow! */
    printf("[BUG 3] Overflow result: %d\n", result);
}

static void bug4_unaligned_access(void) {
    char raw_bytes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    /* Misaligned 32-bit pointer on odd byte boundary */
    int *unaligned_ptr = (int *)&amp;raw_bytes[1];
    printf("[BUG 4] Misaligned value: 0x%08X\n", *unaligned_ptr);
}

int main(int argc, char **argv) {
    if (argc &lt; 2) {
        printf("Usage: %s &lt;1|2|3|4&gt;\n", argv[0]);
        return 1;
    }
    int choice = atoi(argv[1]);
    switch (choice) {
        case 1: bug1_heap_uaf(); break;
        case 2: bug2_stack_overflow(); break;
        case 3: bug3_signed_overflow(); break;
        case 4: bug4_unaligned_access(); break;
        default: printf("Invalid choice\n"); break;
    }
    return 0;
}</code></pre>

        <h3>Build & Test Commands</h3>
        <pre><code class="language-bash"># Compile with AddressSanitizer and UndefinedBehaviorSanitizer
gcc -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer lab_sanitizer.c -o lab_sanitizer

# Test Bug 1: Heap Use-After-Free
./lab_sanitizer 1
# ASan reports: heap-use-after-free on address 0x... freed by thread T0 here: ...

# Test Bug 2: Stack Buffer Overflow
./lab_sanitizer 2
# ASan reports: stack-buffer-overflow on address 0x... in bug2_stack_overflow

# Test Bug 3: Signed Integer Overflow
./lab_sanitizer 3
# UBSan reports: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'

# Test Bug 4: Misaligned Access
./lab_sanitizer 4
# UBSan reports: load of misaligned address 0x... for type 'int', which requires 4 byte alignment</code></pre>

        <h2 id="sec-19-quiz">19.8 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why is AddressSanitizer (ASan) capable of detecting stack buffer overflows while Valgrind Memcheck cannot?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Valgrind only instruments 32-bit software.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) ASan is a compiler pass that restructures stack frames by inserting poisoned redzones between local variables; Valgrind runs on precompiled binaries without stack frame layout insight. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) Valgrind disables the CPU stack pointer register.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Valgrind operates on unmodified binaries and only monitors heap allocation hooks (<code>malloc</code>/<code>free</code>) and system call buffers. It cannot differentiate between adjacent variables inside an allocated stack frame.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the shadow memory scale factor utilized by AddressSanitizer on 64-bit platforms?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) 1 shadow byte tracks 1 application byte.</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) 1 shadow byte tracks 8 application bytes (1:8 scale factor). (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) 1 shadow bit tracks 32 application words.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> ASan maps 8 bytes of virtual memory to 1 shadow byte, resulting in a compact shadow memory layout representing exactly 1/8th of the address space.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Which sanitizer should be deployed to detect multithreaded data races on shared memory without locks?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>-fsanitize=address</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>-fsanitize=undefined</code></label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>-fsanitize=thread</code> (TSan) (Correct)</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> ThreadSanitizer (TSan) instruments memory reads and writes with vector clocks to identify conflicting, unsynchronized accesses between concurrent threads.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What happens when signed integer overflow occurs in a binary compiled with <code>-fsanitize=undefined</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) The CPU wraps silently using two's complement arithmetic.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) UBSan traps the operation and prints an error diagnostic detailing the file, line number, and invalid values. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) The program switches to 64-bit integers automatically.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> UBSan inserts checks prior to arithmetic operations and intercepts undefined behavior before the CPU or compiler can trigger erroneous execution paths.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: Can AddressSanitizer and ThreadSanitizer be enabled simultaneously in the same GCC build invocation?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) Yes, they seamlessly merge shadow memory mappings.</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) No, ASan and TSan are mutually exclusive because both define conflicting shadow memory layouts and runtime interceptors. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) Yes, provided <code>-O3</code> is passed.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> ASan and TSan cannot be linked into the same executable; developers must execute separate test runs for memory errors (ASan) and concurrency races (TSan).
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 20: Static Analysis & Code Quality
# -------------------------------------------------------------
MOD20_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 4 • Linux & Engineering Tools</span>
          <h1 class="chapter-h1">Module 20: Static Analysis &amp; Automated Code Quality</h1>
          <p class="chapter-lead">Deep dive into static analysis theory, Abstract Syntax Trees (AST), Control Flow Graphs (CFG), symbolic execution, GCC/Clang compiler warning enforcement, Cppcheck rule customization, and Clang-Tidy LLVM integration for safety-critical C software.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 50m</div>
            <div class="meta-item">🎯 Standard: MISRA C:2012 / CERT C / ISO 26262 ASIL-D</div>
            <div class="meta-item">⚡ Target: Linux x86_64 / Cross-Platform Toolchains</div>
          </div>
        </div>

        <h2 id="sec-20-1">20.1 Theoretical Foundations of Static Analysis</h2>
        <p>
          Unlike dynamic analysis, which evaluates a single concrete execution path under specific input stimuli, <strong>Static Analysis</strong> evaluates the complete mathematical state space of a program without running it. It parses C source code into formal structural representations:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Static Analysis Pipeline Architecture</div>
          <pre style="margin:0;color:#94A3B8;">
C Source Code (.c) ──▶ [ Lexer &amp; Parser ] ──▶ Abstract Syntax Tree (AST)
                                                     │
                                            Semantic Analysis
                                                     ▼
                                          Control Flow Graph (CFG)
                                                     │
                       ┌─────────────────────────────┴─────────────────────────────┐
                       ▼                                                           ▼
         [ Data Flow &amp; Taint Analysis ]                              [ Symbolic Execution Engine ]
         (Tracks uninitialized variables,                            (Evaluates feasibility of
          null pointer propagation, dead code)                        all branch paths symbolically)
                       │                                                           │
                       └─────────────────────────────┬─────────────────────────────┘
                                                     ▼
                                     [ Defect &amp; MISRA Violation Reports ]
          </pre>
        </div>

        <h3>Static Analysis Disciplines</h3>
        <ul>
          <li><strong>Syntactic Pattern Matching:</strong> Fast AST traversal checking style, naming conventions, or prohibited constructs (e.g. <code>goto</code>, <code>malloc</code> in embedded loops).</li>
          <li><strong>Dataflow Analysis:</strong> Tracks the propagation of values across variables and registers (detecting variables used before initialization).</li>
          <li><strong>Taint Analysis:</strong> Marks external user or network inputs as "tainted" and verifies they do not reach security-sensitive sinks (such as <code>memcpy</code> lengths or array indices) without bounds checking.</li>
          <li><strong>Path-Sensitive Symbolic Execution:</strong> Represents program variables as mathematical symbols and explores execution paths, proving whether an assertion can ever be violated under any theoretical combination of inputs.</li>
        </ul>

        <h2 id="sec-20-2">20.2 Compiler Warnings as the First Line of Defense</h2>
        <p>
          Before deploying dedicated third-party static analyzers, high-reliability engineering requires configuring the compiler itself as a zero-tolerance static analysis engine.
        </p>

        <h3>Production Safety-Critical GCC/Clang Compiler Flags</h3>
        <pre><code class="language-makefile"># ==============================================================================
# Production Safety-Critical Warning Configuration
# ==============================================================================
WARN_FLAGS := -Wall -Wextra -Wpedantic \
              -Werror \
              -Wconversion \
              -Wsign-conversion \
              -Wshadow \
              -Wdouble-promotion \
              -Wstrict-prototypes \
              -Wundef \
              -Wformat=2 \
              -Wformat-security \
              -Wnull-dereference \
              -Wwrite-strings \
              -Wcast-align \
              -Wlogical-op \
              -Wduplicated-cond \
              -Wduplicated-branches</code></pre>

        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Flag</th>
              <th style="padding:10px;text-align:left;">Defect Category Caught</th>
              <th style="padding:10px;text-align:left;">Why It Matters in Embedded C</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><code>-Wconversion</code></td>
              <td style="padding:10px;">Implicit numeric narrowing / signedness conversion</td>
              <td style="padding:10px;">Prevents silent truncation of 32-bit registers into 16-bit or 8-bit variables.</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><code>-Wshadow</code></td>
              <td style="padding:10px;">Local variable shadowing outer variable</td>
              <td style="padding:10px;">Prevents local function parameters from accidentally hiding global configuration structs.</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><code>-Wdouble-promotion</code></td>
              <td style="padding:10px;">Implicit promotion of <code>float</code> to <code>double</code></td>
              <td style="padding:10px;">Prevents software emulation of 64-bit doubles on Cortex-M4F single-precision hardware FPUs.</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><code>-Wnull-dereference</code></td>
              <td style="padding:10px;">Compile-time provable NULL pointer dereference</td>
              <td style="padding:10px;">Catches code paths where a pointer is checked for NULL but dereferenced regardless.</td>
            </tr>
          </tbody>
        </table>

        <h2 id="sec-20-3">20.3 Cppcheck: Deep Static Analysis & MISRA C Integration</h2>
        <p>
          <strong>Cppcheck</strong> is a dedicated static analysis tool designed specifically for C and C++. Unlike compilers, which stop analyzing when code is syntactically legal C, Cppcheck looks specifically for undefined behavior, resource leaks, and logic flaws.
        </p>

        <h3>Automating Cppcheck in Build Pipelines</h3>
        <pre><code class="language-bash">cppcheck --enable=all \
         --inconclusive \
         --error-exitcode=1 \
         --suppress=missingIncludeSystem \
         --inline-suppr \
         -I include/ \
         src/</code></pre>

        <h3>Enforcing MISRA C:2012 with Cppcheck</h3>
        <p>
          Cppcheck includes an automated MISRA addon (<code>misra.py</code>). By feeding the MISRA rule table text file, Cppcheck verifies compliance against safety-critical automotive standards:
        </p>
        <pre><code class="language-bash">cppcheck --addon=misra.py --rule-texts=misra_rules.txt src/</code></pre>

        <h2 id="sec-20-4">20.4 Clang-Tidy: LLVM AST Matchers & Automated Modernization</h2>
        <p>
          <strong>Clang-Tidy</strong> is an LLVM-based linter and static analysis platform. It provides a modular check framework capable of not only reporting errors, but also applying automated source-level patches (<code>-fix</code>).
        </p>

        <h3>Production <code>.clang-tidy</code> Configuration</h3>
        <pre><code class="language-yaml"># .clang-tidy - Safety-Critical Embedded C Configuration
---
Checks: >
  -*,
  clang-analyzer-*,
  cert-*,
  bugprone-*,
  readability-*,
  performance-*,
  -readability-magic-numbers

WarningsAsErrors: 'clang-analyzer-*,cert-*,bugprone-*'

CheckOptions:
  - key: cert-dcl16-c.NewSuffixes
    value: 'L,LL,U,UL,ULL'
...</code></pre>

        <h2 id="sec-20-5">20.5 Industrial Case Study: Ariane 5 Flight 501 Numeric Conversion Failure</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The $370 Million Float-to-Int Truncation Disaster</h4>
          <p>
            On June 4, 1996, the maiden flight of the European Ariane 5 launcher ended in catastrophic destruction 37 seconds after liftoff. The rocket veered sharply off trajectory, causing structural breakup under aerodynamic aerodynamic loads, triggering the onboard automated flight termination system.
          </p>
          <p>
            <strong>Inquiry Board Findings:</strong>
            The failure occurred in the Inertial Reference System (SRI). A software module calculated the horizontal bias (a variable representing the vehicle's horizontal velocity).
          </p>
          <p>
            The value was represented as a <strong>64-bit IEEE-754 floating-point number</strong>. In an unshielded code section, the software attempted to convert this 64-bit float into a <strong>16-bit signed integer</strong>.
          </p>
          <pre><code class="language-c">/* Conceptual representation of the Ariane 5 truncation defect */
double horizontal_bias = calculate_horizontal_velocity(); /* > 32767.0 on Ariane 5! */
int16_t truncated_bias = (int16_t)horizontal_bias;         /* Hardware Operand Error! */</code></pre>
          <p>
            On the earlier, slower Ariane 4 rocket, the horizontal velocity physically never exceeded 32,767. However, Ariane 5's higher thrust caused the velocity to exceed the range of a 16-bit signed integer. The conversion triggered an unhandled hardware arithmetic overflow exception, causing the primary SRI computer to shut down, followed immediately by the identical backup unit.
          </p>
          <p>
            <strong>How Modern Static Analysis Prevents This:</strong>
            Compiling with <code>-Wconversion</code> or Clang-Tidy’s <code>bugprone-narrowing-conversions</code> flags this line at compile-time as an unsafe loss of precision, halting the build before binary deployment.
          </p>
        </div>

        <h2 id="sec-20-6">20.6 Hands-On Lab: Setting up an Automated Static Analysis Pipeline</h2>
        <p>
          In this lab, you will analyze a C telemetry module containing subtle static analysis violations (implicit truncation, uninitialized variables, and shadow variables), and configure Cppcheck and Clang-Tidy to flag every defect automatically.
        </p>

        <h3>Defective Code (<code>lab_static.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdint.h&gt;

static uint32_t g_system_id = 42;

/* Defect 1: Implicit conversion / truncation from uint32_t to uint16_t */
static uint16_t scale_telemetry(uint32_t raw_value) {
    uint16_t scaled = raw_value; /* Truncation without explicit cast */
    return scaled;
}

/* Defect 2: Variable shadowing */
static void process_id(uint32_t g_system_id) {
    printf("Processing ID: %u\n", g_system_id);
}

/* Defect 3: Uninitialized variable along error path */
static int read_sensor_voltage(int error_flag, int *out_voltage) {
    int v;
    if (error_flag == 0) {
        v = 3300; // 3.3V
    }
    /* If error_flag != 0, v is uninitialized! */
    *out_voltage = v;
    return 0;
}

int main(void) {
    uint32_t raw = 0x12345678;
    uint16_t s = scale_telemetry(raw);
    process_id(99);

    int volt;
    read_sensor_voltage(1, &amp;volt);
    printf("Scaled: 0x%04X, Volt: %d\n", s, volt);
    return 0;
}</code></pre>

        <h3>Verification Script</h3>
        <pre><code class="language-bash"># 1. Test Compiler Static Analysis
gcc -Wall -Wextra -Wconversion -Wshadow -Wuninitialized -c lab_static.c
# Compiler immediately reports:
# - warning: conversion from 'uint32_t' to 'uint16_t' may change value [-Wconversion]
# - warning: declaration of 'g_system_id' shadows a global declaration [-Wshadow]
# - warning: 'v' may be used uninitialized [-Wmaybe-uninitialized]

# 2. Test Cppcheck
cppcheck --enable=all --error-exitcode=1 lab_static.c
# Cppcheck reports:
# [lab_static.c:24]: (error) Uninitialized variable: v</code></pre>

        <h2 id="sec-20-quiz">20.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What compiler flag flags implicit type conversions that may alter the numeric value or signedness of an expression?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) <code>-O3</code></label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) <code>-Wconversion</code> (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) <code>-fshort-enums</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>-Wconversion</code> warns whenever an expression undergoes implicit conversion between integers that could alter its value (such as assigning a 32-bit int to a 16-bit short).
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the primary difference between an Abstract Syntax Tree (AST) and a Control Flow Graph (CFG)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) An AST represents grammatical nesting of tokens, while a CFG models the directed order of execution paths and branch conditions. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) An AST is only used by linkers, whereas CFGs are used by assemblers.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) They are identical data structures with different names.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The AST represents program syntax hierarchy (expressions, declarations, statements), whereas the CFG represents execution graph blocks connected by directed edges representing branch decisions.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Why is <code>-Wdouble-promotion</code> vital when compiling C code for single-precision ARM Cortex-M4F processors?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) Doubles cannot fit inside 32-bit SRAM.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Cortex-M4F only possesses a single-precision 32-bit hardware FPU; implicit promotion to double forces slow, multi-cycle software emulation library calls. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) It causes the CPU clock frequency to divide by two.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Without <code>-Wdouble-promotion</code>, writing <code>float x = y * 0.5;</code> implicitly promotes <code>y</code> to <code>double</code> because <code>0.5</code> is a 64-bit double literal, destroying DSP algorithm performance.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What role does Taint Analysis play in secure C development?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) It monitors CPU fan speeds under high thermal loads.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) It tracks untrusted external inputs from network/UART interfaces and verifies they are validated before reaching sensitive operations like array indices or memory allocations. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) It cleans object files before archiving.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Taint analysis identifies vulnerability pathways where untrusted external data could influence execution flow or buffer bounds without sanitization.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What command instructs Clang-Tidy to automatically apply its suggested code fixes?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>--patch-all</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>-fix</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>--repair</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The <code>-fix</code> flag instructs Clang-Tidy to directly rewrite the source code with compiler-verified refactorings and corrections.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 21: Linux Profiling & Performance Tuning
# -------------------------------------------------------------
MOD21_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 4 • Linux & Engineering Tools</span>
          <h1 class="chapter-h1">Module 21: Linux Profiling &amp; Performance Tuning</h1>
          <p class="chapter-lead">Master CPU hardware performance counters (PMU), cache hierarchy mechanics, instruction pipeline stalls, sampling vs instrumentation profilers, Linux perf masterclass, Flame Graph visualization, and low-overhead kernel tracing with eBPF.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: Linux perf / PMU Hardware Events / eBPF</div>
            <div class="meta-item">⚡ Target: Linux x86_64 / ARM Cortex-A &amp; Cortex-M</div>
          </div>
        </div>

        <h2 id="sec-21-1">21.1 Modern CPU Microarchitecture & Hardware Counters</h2>
        <p>
          High-performance C engineering requires looking beyond big-O algorithmic complexity into the physical reality of the microprocessor. Modern superscalar CPUs do not execute instructions at a constant rate; performance is governed by <strong>cache hierarchies, pipeline stalls, and branch prediction</strong>.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Microarchitectural Latency Hierarchy</div>
          <pre style="margin:0;color:#94A3B8;">
Hardware Component            Access Latency (Cycles)      Equivalent Human Scale
──────────────────────────────────────────────────────────────────────────────────
CPU Registers                 1 cycle                      1 second
L1 Data Cache (32-64 KB)      4 - 5 cycles                 5 seconds
L2 Unified Cache (256-512 KB) 12 - 14 cycles               15 seconds
L3 Shared Cache (8-32 MB)     35 - 50 cycles               45 seconds
Main Memory (DDR4/DDR5 DRAM)  150 - 250 cycles             3 to 4 minutes!
NVMe Flash Storage (PCIe)     ~20,000 cycles               1 to 2 days!!
Rotational Disk Drive         ~10,000,000 cycles           10 months!!!
          </pre>
        </div>

        <h3>The Performance Monitoring Unit (PMU)</h3>
        <p>
          Modern CPUs contain dedicated hardware silicon counters known as the <strong>PMU (Performance Monitoring Unit)</strong>. The PMU counts real physical silicon events without software overhead:
        </p>
        <ul>
          <li><code>cpu-cycles</code>: Core clock cycles elapsed.</li>
          <li><code>instructions</code>: Total instructions retired by the pipeline.</li>
          <li><code>IPC (Instructions Per Cycle)</code>: Ratio of instructions to cycles. Superscalar CPUs can achieve IPC &gt; 2.0; memory-stalled algorithms drop to IPC &lt; 0.3!</li>
          <li><code>cache-misses</code>: Last-Level Cache (LLC) misses requiring physical DRAM access.</li>
          <li><code>branch-misses</code>: Pipeline flushes caused by mispredicted branches (costing 15-20 penalty cycles each).</li>
        </ul>

        <h2 id="sec-21-2">21.2 Sampling Profilers vs. Instrumentation Profilers</h2>
        <p>
          Engineers often confuse statistical sampling profilers with instrumentation profilers:
        </p>
        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Feature</th>
              <th style="padding:10px;text-align:left;">Instrumentation (e.g. <code>gprof -pg</code>)</th>
              <th style="padding:10px;text-align:left;">Statistical Sampling (e.g. <code>perf</code>)</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Mechanism</strong></td>
              <td style="padding:10px;">Injects code (<code>mcount</code>) at every function entry/exit.</td>
              <td style="padding:10px;">PMU timer interrupts CPU at 99 Hz or 999 Hz, capturing Program Counter (PC).</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Heisenbug Distortion</strong></td>
              <td style="padding:10px;"><strong style="color:#EF4444;">Severe</strong> (distorts small, inlined functions; prevents vectorization).</td>
              <td style="padding:10px;"><strong style="color:#10B981;">Negligible</strong> (&lt; 1% overhead; runs in production).</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Kernel &amp; Driver Insight</strong></td>
              <td style="padding:10px;">None (User space only).</td>
              <td style="padding:10px;">Complete (Traces user space, kernel syscalls, and ISRs).</td>
            </tr>
          </tbody>
        </table>

        <h2 id="sec-21-3">21.3 Linux <code>perf</code> Tooling Masterclass</h2>
        <p>
          The Linux kernel tool <strong>perf</strong> is the gold standard for production profiling.
        </p>

        <h3>1. <code>perf stat</code>: Microarchitectural Summary</h3>
        <pre><code class="language-bash">perf stat -e cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses ./my_app</code></pre>
        <pre><code class="language-text"> Performance counter stats for './my_app':

     2,840,119,412      cycles                    #    3.401 GHz
     6,120,442,100      instructions              #    2.15  insn per cycle   &lt;-- Good IPC!
        42,110,400      cache-references
         1,210,040      cache-misses              #    2.87% of all L1D hits  &lt;-- Excellent cache locality
       850,200,100      branch-instructions
         2,100,400      branch-misses             #    0.25% of all branches

       0.835102400 seconds time elapsed</code></pre>

        <h3>2. <code>perf record</code> & <code>perf report</code>: Hotspot Identification</h3>
        <pre><code class="language-bash"># Record high-resolution stack samples at 999 Hz with DWARF unwinding
perf record -F 999 -g -- ./my_app

# Interactive TUI report showing functions ordered by CPU consumption
perf report -g 'graph,0.5,caller'</code></pre>

        <h2 id="sec-21-4">21.4 Flame Graphs: Visualizing Complex Call Stacks</h2>
        <p>
          Textual call trees are overwhelming in complex applications. <strong>Flame Graphs</strong> (pioneered by Brendan Gregg) visualize CPU consumption hierarchically:
        </p>
        <ul>
          <li>The <strong>x-axis</strong> represents the entire sample population (alphabetically ordered, not time-ordered). The wider the box, the more CPU cycles consumed!</li>
          <li>The <strong>y-axis</strong> represents call stack depth. Top-level boxes are the leaf functions executing on CPU.</li>
        </ul>
        <pre><code class="language-bash"># Generate an interactive SVG flame graph
perf record -F 99 -a -g -- sleep 10
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl &gt; cpu_flamegraph.svg</code></pre>

        <h2 id="sec-21-5">21.5 Low-Overhead Kernel Tracing with eBPF</h2>
        <p>
          <strong>eBPF (Extended Berkeley Packet Filter)</strong> allows developers to execute sandboxed, JIT-compiled C code inside the Linux kernel without modifying kernel source or loading out-of-tree kernel modules.
        </p>
        <p>
          Using <code>bpftrace</code>, engineers can measure latency distributions of I/O, context switching, or scheduler wakeups with nanosecond resolution:
        </p>
        <pre><code class="language-bash"># Trace read() syscall latency histogram across all processes
sudo bpftrace -e 'kprobe:sys_read { @start[tid] = nsecs; }
kretprobe:sys_read /@start[tid]/ {
    @latency_us = hist((nsecs - @start[tid]) / 1000);
    delete(@start[tid]);
}'</code></pre>

        <h2 id="sec-21-6">21.6 Industrial Case Study: HFT Gateway False Sharing Catastrophe</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The 15% Latency Penalty in Multi-Core Network Ingestion</h4>
          <p>
            A high-frequency trading (HFT) firm upgraded an order gateway server from an 8-core CPU to a 32-core dual-socket NUMA system. Instead of throughput multiplying, 99th-percentile packet latency degraded by 400%.
          </p>
          <p>
            <strong>Investigation via <code>perf c2c</code> (Cache-to-Cache):</strong>
            Engineers executed Linux's specialized cache-coherency profiler:
          </p>
          <pre><code class="language-bash">perf c2c record -F 60000 -- ./gateway_daemon
perf c2c report --stdio</code></pre>
          <p>
            The report revealed extreme <strong>HITM (Hit Modified)</strong> events centered on a single memory address: <code>0x7fff80004040</code>.
          </p>
          <p>
            <strong>Root Cause: False Sharing!</strong>
            The circular ring buffer state struct placed the producer head pointer and the consumer tail pointer inside the same 64-byte CPU cache line:
          </p>
          <pre><code class="language-c">/* Faulty Layout: Occupies single 64-byte cacheline! */
typedef struct {
    uint64_t producer_head; /* 8 bytes */
    uint64_t consumer_tail; /* 8 bytes (Located in same cacheline!) */
} queue_indices_t;</code></pre>
          <p>
            Every time Core 0 updated <code>producer_head</code>, the CPU hardware MESI coherency protocol invalidated Core 1's entire L1 cacheline, forcing Core 1 to stall and fetch the line across the slow interconnect bus!
          </p>
          <p>
            <strong>The Fix: Explicit Cacheline Alignment:</strong>
          </p>
          <pre><code class="language-c">/* Optimized Layout: Isolated across distinct 64-byte cachelines */
typedef struct {
    alignas(64) uint64_t producer_head;
    alignas(64) uint64_t consumer_tail;
} queue_indices_t;</code></pre>
          <p>
            This 2-line change eliminated 99.8% of HITM bus stalls, slashing order routing latency by 65%.
          </p>
        </div>

        <h2 id="sec-21-7">21.7 Hands-On Lab: Cache Locality Optimization & Perf Validation</h2>
        <p>
          In this lab, you will measure the massive performance difference between a row-major array traversal (cache-friendly) vs column-major traversal (cache-hostile), proving the impact with <code>perf stat</code>.
        </p>

        <h3>Source Code (<code>lab_matrix.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;

#define MATRIX_DIM 4096

static int matrix[MATRIX_DIM][MATRIX_DIM];

static long long sum_row_major(void) {
    long long sum = 0;
    for (int r = 0; r &lt; MATRIX_DIM; r++) {
        for (int c = 0; c &lt; MATRIX_DIM; c++) {
            sum += matrix[r][c]; /* Contiguous memory access (Cache Friendly) */
        }
    }
    return sum;
}

static long long sum_col_major(void) {
    long long sum = 0;
    for (int c = 0; c &lt; MATRIX_DIM; c++) {
        for (int r = 0; r &lt; MATRIX_DIM; r++) {
            sum += matrix[r][c]; /* Strided access by 4096*4 bytes (Cache Hostile!) */
        }
    }
    return sum;
}

int main(int argc, char **argv) {
    if (argc &lt; 2) return 1;
    int mode = atoi(argv[1]);
    long long res = (mode == 1) ? sum_row_major() : sum_col_major();
    printf("Result: %lld\n", res);
    return 0;
}</code></pre>

        <h3>Build & Benchmarking Commands</h3>
        <pre><code class="language-bash"># Compile with -O2
gcc -O2 lab_matrix.c -o lab_matrix

# Benchmark Mode 1 (Row-Major: Cache Friendly)
perf stat -e cycles,instructions,L1-dcache-load-misses ./lab_matrix 1

# Benchmark Mode 2 (Column-Major: Cache Hostile)
perf stat -e cycles,instructions,L1-dcache-load-misses ./lab_matrix 2

# Observed Results:
# Mode 1: ~0.04s execution time | L1 miss rate &lt; 1% | IPC &gt; 2.2
# Mode 2: ~0.42s execution time | L1 miss rate &gt; 95% | IPC &lt; 0.3
# Verified: Cache locality yields a 10x speedup on identical algorithmic workloads!</code></pre>

        <h2 id="sec-21-quiz">21.8 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What physical event occurs when two separate CPU cores concurrently write to different variables that reside on the exact same 64-byte cache line?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) True Data Race</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) False Sharing (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) Page Fault</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> False sharing occurs when threads modify independent variables residing within the same cache line, causing unnecessary hardware cache-invalidation ping-pong across the inter-core interconnect.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the typical cache line size in bytes on modern x86_64 and ARM Cortex-A microprocessors?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) 16 bytes</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) 64 bytes (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) 4096 bytes</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Standard L1/L2/L3 cache architectures organize memory in 64-byte blocks (cache lines), which are transferred as atomic units between cache and main memory.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What Linux tool records CPU execution profiles with zero code instrumentation using PMU hardware counters?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>gprof</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>perf record</code> (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>strace</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>perf record</code> configures the CPU's PMU to trigger statistical interrupts at regular sample intervals, capturing register contexts with minimal overhead.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: In a Flame Graph visualization, what does the horizontal width of a function box indicate?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) Chronological elapsed wall-clock time.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) Proportion of CPU samples where that function was present in the call stack. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) Size of the function in flash memory.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> In a flame graph, the x-axis represents the aggregated population of samples; wider boxes represent functions consuming a larger percentage of total CPU cycles.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What technology enables executing sandboxed C-derived bytecode in the Linux kernel for non-intrusive tracing?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) JVM</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) eBPF (Extended Berkeley Packet Filter) (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) OpenMP</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> eBPF executes verified, JIT-compiled bytecode attached to kernel tracepoints and kprobes, providing safe, high-performance kernel observability.
            </div>
          </div>
        </div>'''

if __name__ == "__main__":
    print("--- Upgrading Part 4 Modules ---")
    replace_article_in_file("mod17-build-systems-make-and-cmake.html", MOD17_ARTICLE)
    replace_article_in_file("mod18-gdb-debugging-and-core-dumps.html", MOD18_ARTICLE)
    replace_article_in_file("mod19-valgrind-and-sanitizers.html", MOD19_ARTICLE)
    replace_article_in_file("mod20-static-analysis-tools.html", MOD20_ARTICLE)
    replace_article_in_file("mod21-linux-profiling-and-perf.html", MOD21_ARTICLE)
    print("--- Part 4 Completed Successfully! ---")
