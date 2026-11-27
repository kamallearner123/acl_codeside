#!/usr/bin/env python3
"""
generate_part6_full.py - Generates all 7 modules of Part 6:
Module 26: File Descriptors & POSIX I/O
Module 27: Scalable I/O Multiplexing (epoll)
Module 28: Process Lifecycle & IPC
Module 29: POSIX Threads & Multithreading
Module 30: Thread Synchronization & Locks
Module 31: POSIX Signals & Real-Time Timers
Module 32: Network Sockets Programming
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
# MODULE 26: File Descriptors & POSIX I/O
# -------------------------------------------------------------
MOD26_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 26: File Descriptors &amp; POSIX Low-Level I/O</h1>
          <p class="chapter-lead">Deep dive into Linux Virtual Filesystem (VFS) internals, file descriptor tables, system calls vs buffered stdio, non-blocking I/O (O_NONBLOCK), close-on-exec (O_CLOEXEC), atomic persistence with fsync/fdatasync, and scatter/gather vectorized I/O (readv/writev).</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: POSIX.1-2008 / Linux VFS / C11</div>
            <div class="meta-item">⚡ Target: Linux x86_64 / Embedded Linux (ARM/MIPS)</div>
          </div>
        </div>

        <h2 id="sec-26-1">26.1 Linux VFS Architecture & Kernel Descriptor Tables</h2>
        <p>
          In Unix-like systems, the foundational design maxim is <em>"everything is a file"</em>. A <strong>File Descriptor (FD)</strong> is not a file itself; it is a small, non-negative integer serving as an index into a per-process kernel table.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Linux Kernel VFS Object Relationships</div>
          <pre style="margin:0;color:#94A3B8;">
User Process (task_struct)
  └── files_struct (Descriptor Table)
       ├── FD 0 (stdin)  ──▶ [ struct file (Offset, Flags) ] ──▶ [ struct inode ] ──▶ Hardware / TTY
       ├── FD 1 (stdout) ──▶ [ struct file (Offset, Flags) ] ──▶ [ struct inode ] ──▶ Hardware / TTY
       ├── FD 2 (stderr) ──▶ [ struct file (Offset, Flags) ] ──▶ [ struct inode ] ──▶ Hardware / TTY
       └── FD 3 (telemetry.dat) ──▶ [ struct file ] ───────────▶ [ struct inode ] ──▶ Ext4 / NVMe
                                     (f_pos = 1024)                 (i_size = 4096)
          </pre>
        </div>

        <h3>Three Kernel Layers of File Abstraction</h3>
        <ol>
          <li><strong>File Descriptor Table (Per-Process):</strong> Maps integer integers (0, 1, 2, 3...) to open file descriptions. Preserved across <code>fork()</code>.</li>
          <li><strong>Open File Table (System-Wide):</strong> Stores current byte offset (<code>f_pos</code>), access status flags (<code>O_RDONLY</code>, <code>O_NONBLOCK</code>, <code>O_APPEND</code>), and reference count.</li>
          <li><strong>Inode Table (Filesystem-Wide):</strong> Represents the physical storage object on disk, containing file size, disk block extents, permissions, and file type (regular file, socket, pipe, character device).</li>
        </ol>

        <h2 id="sec-26-2">26.2 System Calls vs. Standard C Buffered I/O</h2>
        <p>
          Software engineers often confuse low-level POSIX system calls (<code>open</code>, <code>read</code>, <code>write</code>, <code>close</code>) with ISO C standard library streams (<code>fopen</code>, <code>fread</code>, <code>fwrite</code>, <code>fclose</code>).
        </p>

        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Dimension</th>
              <th style="padding:10px;text-align:left;">POSIX System Calls (<code>unistd.h</code>)</th>
              <th style="padding:10px;text-align:left;">Standard C Library (<code>stdio.h</code>)</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Handle Type</strong></td>
              <td style="padding:10px;">Integer file descriptor (<code>int fd</code>)</td>
              <td style="padding:10px;">Pointer to struct (<code>FILE *fp</code>)</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Buffering</strong></td>
              <td style="padding:10px;">Unbuffered (Direct kernel syscall)</td>
              <td style="padding:10px;">User-space buffered (Typically 4 KB or 8 KB buffer)</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Overhead for small writes</strong></td>
              <td style="padding:10px;">High (Every write triggers a CPU context switch)</td>
              <td style="padding:10px;">Minimal (Aggregates small writes in user-space buffer)</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Direct Hardware / Non-blocking</strong></td>
              <td style="padding:10px;"><span style="color:#10B981;font-weight:bold;">Yes</span> (Ideal for epoll, sockets, devices)</td>
              <td style="padding:10px;"><span style="color:#EF4444;font-weight:bold;">No</span> (Blocks unexpectedly when buffer drains)</td>
            </tr>
          </tbody>
        </table>

        <h2 id="sec-26-3">26.3 Atomic Persistence: fsync vs fdatasync</h2>
        <p>
          When you invoke <code>write(fd, buf, len)</code>, the kernel copies the data from user space into the <strong>Linux Page Cache</strong> in RAM. The physical write to flash or disk occurs asynchronously seconds later!
        </p>
        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">Data Loss Hazard:</strong> If power is severed immediately after <code>write()</code> or <code>fflush()</code>, unwritten page cache data in RAM evaporates, leaving files corrupted or empty on disk!
        </div>
        <p>
          To guarantee non-volatile durability, production embedded systems must explicitly flush the page cache:
        </p>
        <ul>
          <li><code>fsync(fd)</code>: Flushes all modified dirty file data AND metadata (inode modification time, file size) to physical disk. Forces head/flash commit.</li>
          <li><code>fdatasync(fd)</code>: Flushes dirty file data, but omits metadata updates unless required to retrieve data (e.g., file size expansion). Significantly faster than <code>fsync()</code> on mechanical storage.</li>
        </ul>

        <h2 id="sec-26-4">26.4 Scatter/Gather Vectorized I/O (writev / readv)</h2>
        <p>
          When transmitting structured packets (e.g. Header + Payload + Checksum), naive programs either concatenate the buffers into a temporary array (costing CPU copies and RAM allocations) or issue multiple <code>write()</code> syscalls (costing multiple kernel transitions).
        </p>
        <p>
          <strong>Vectorized I/O (<code>writev</code>)</strong> eliminates both penalties by passing an array of pointers (<code>struct iovec</code>) to the kernel in a single atomic syscall:
        </p>
        <pre><code class="language-c">#include &lt;sys/uio.h&gt;

struct iovec iov[3];
iov[0].iov_base = &amp;header;      iov[0].iov_len = sizeof(header);
iov[1].iov_base = payload_data; iov[1].iov_len = payload_size;
iov[2].iov_base = &amp;crc32;       iov[2].iov_len = sizeof(crc32);

/* Single atomic syscall! Zero intermediate buffer copies! */
ssize_t written = writev(fd, iov, 3);</code></pre>

        <h2 id="sec-26-5">26.5 Industrial Case Study: Database Journal Data Loss Disaster</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Industrial SCADA Power Outage File Truncation Incident</h4>
          <p>
            An industrial wastewater treatment facility experienced a sudden power grid failure. The backup generator kicked in within 3 seconds, but the SCADA telemetry logging system failed to restart, reporting corrupted transaction journals.
          </p>
          <p>
            <strong>Root Cause Analysis:</strong>
            The logging daemon wrote critical alarm logs using standard C library calls:
          </p>
          <pre><code class="language-c">fprintf(log_fp, "[ALARM] Motor Overheat State at %lu\n", timestamp);
fflush(log_fp); /* Flushes stdio buffer into Linux Page Cache ONLY! */</code></pre>
          <p>
            The developers believed <code>fflush()</code> ensured disk persistence. In reality, <code>fflush()</code> simply issues a <code>write()</code> syscall to the Linux kernel page cache. Because the system lost power 500 milliseconds later, the kernel never flushed the dirty pages from volatile DDR RAM to physical eMMC flash memory!
          </p>
          <p>
            <strong>The Fix:</strong> All critical configuration and journal commits were updated to invoke <code>fdatasync(fileno(log_fp))</code>, ensuring physical non-volatile storage commit before returning control to the caller.
          </p>
        </div>

        <h2 id="sec-26-6">26.6 Hands-On Lab: Building an Atomic Configuration Writer in C</h2>
        <p>
          In this lab, you will write a robust, crash-resilient file persistence engine in C that prevents partial file corruption using the <em>Atomic Write-and-Rename</em> idiom combined with <code>fdatasync()</code>.
        </p>

        <h3>Production Atomic Writer (<code>atomic_write.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;
#include &lt;fcntl.h&gt;
#include &lt;unistd.h&gt;
#include &lt;errno.h&gt;

int atomic_write_file(const char *target_path, const void *data, size_t size) {
    char temp_path[512];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp.%d", target_path, (int)getpid());

    /* 1. Open temporary file with exclusive creation flags */
    int fd = open(temp_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd &lt; 0) return -1;

    /* 2. Write all data, handling partial writes */
    const char *ptr = (const char *)data;
    size_t remaining = size;
    while (remaining &gt; 0) {
        ssize_t written = write(fd, ptr, remaining);
        if (written &lt;= 0) {
            if (written &lt; 0 &amp;&amp; errno == EINTR) continue;
            close(fd);
            unlink(temp_path);
            return -2;
        }
        ptr += written;
        remaining -= (size_t)written;
    }

    /* 3. Flush page cache to physical non-volatile medium */
    if (fdatasync(fd) != 0) {
        close(fd);
        unlink(temp_path);
        return -3;
    }

    /* 4. Close file descriptor */
    if (close(fd) != 0) {
        unlink(temp_path);
        return -4;
    }

    /* 5. Atomic POSIX directory rename: swaps directory entry atomically */
    if (rename(temp_path, target_path) != 0) {
        unlink(temp_path);
        return -5;
    }

    return 0; // Success: File guaranteed durable on disk!
}

int main(void) {
    const char *payload = "DEVICE_ID=42\nCALIBRATION_OFFSET=1024\nBAUD_RATE=115200\n";
    const char *config_file = "device_config.cfg";

    printf("Executing atomic configuration write...\n");
    if (atomic_write_file(config_file, payload, strlen(payload)) == 0) {
        printf("Configuration committed durably and atomically to %s\n", config_file);
    } else {
        perror("Failed to persist configuration");
        return 1;
    }
    return 0;
}</code></pre>

        <h2 id="sec-26-quiz">26.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What does the <code>O_CLOEXEC</code> flag do when opening a file descriptor with <code>open()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Closes all open network sockets in the program.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) Automatically closes the file descriptor if the process executes an <code>execve()</code> system call, preventing descriptor leakage to child programs. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) Deletes the file immediately upon closing.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>O_CLOEXEC</code> sets the close-on-exec flag atomically, preventing security vulnerabilities where sensitive file descriptors or sockets are accidentally inherited by child processes.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the crucial difference between <code>fflush()</code> and <code>fsync()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>fflush()</code> transfers data from user space stdio buffers into the OS page cache; <code>fsync()</code> forces the OS page cache to physically write to non-volatile storage. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>fflush()</code> only works on network sockets.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) They are identical POSIX functions.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>fflush()</code> only leaves the user-space C library buffer; data is still vulnerable in volatile kernel RAM until <code>fsync()</code> or <code>fdatasync()</code> forces commit to disk.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What system call allows transmitting multiple non-contiguous memory buffers in a single atomic I/O operation?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>sendfile()</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>writev()</code> (Vectorized I/O) (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>mmap()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>writev()</code> consumes an array of <code>struct iovec</code>, gathering disparate memory buffers into a single kernel I/O request without intermediate copies.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: Why does the Atomic Write-and-Rename pattern use <code>rename()</code> instead of overwriting the original file directly?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>rename()</code> compresses the file.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) In POSIX filesystems, <code>rename()</code> is an atomic directory metadata swap; if a power cut occurs midway, the system retains the intact original file rather than a half-written corrupt file. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) It encrypts the disk sector.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Direct in-place writes leave the original file in a partially overwritten, corrupted state if interrupted; atomic rename guarantees that readers always observe either the old intact file or the new intact file.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What return value from <code>read()</code> indicates that an end-of-file (EOF) condition has been reached?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>-1</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>0</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>EOF (-1)</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> In POSIX I/O, <code>read()</code> returns 0 on EOF, -1 on error (with <code>errno</code> set), and positive values indicating the number of bytes successfully read.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# Import remaining articles from separate modular definitions
# -------------------------------------------------------------
from generate_part6_helpers import (
    MOD27_ARTICLE, MOD28_ARTICLE, MOD29_ARTICLE,
    MOD30_ARTICLE, MOD31_ARTICLE, MOD32_ARTICLE
)

if __name__ == "__main__":
    print("--- Upgrading Part 6 Modules ---")
    replace_article_in_file("mod26-file-descriptors-and-posix-io.html", MOD26_ARTICLE)
    replace_article_in_file("mod27-scalable-io-multiplexing-epoll.html", MOD27_ARTICLE)
    replace_article_in_file("mod28-process-lifecycle-and-ipc.html", MOD28_ARTICLE)
    replace_article_in_file("mod29-posix-threads-and-concurrency.html", MOD29_ARTICLE)
    replace_article_in_file("mod30-thread-synchronization-and-locks.html", MOD30_ARTICLE)
    replace_article_in_file("mod31-posix-signals-and-timers.html", MOD31_ARTICLE)
    replace_article_in_file("mod32-network-sockets-programming.html", MOD32_ARTICLE)
    print("--- Part 6 Completed Successfully! ---")
