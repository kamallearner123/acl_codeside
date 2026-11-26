#!/usr/bin/env python3
import os
import sys

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "private_books", "c_programming_professional", "chapters")
INDEX_FILE = os.path.join(os.path.dirname(__file__), "private_books", "c_programming_professional", "index.html")

# Define all curriculum parts and modules
CURRICULUM = [
    {
        "part_id": "P1",
        "part_title": "C Programming Foundations",
        "modules": [
            {"id": "mod01-intro-to-c", "title": "Mod 1: Intro to C & Software Dev", "time": "45m", "file": "mod01-intro-to-c.html"},
            {"id": "mod02-compilation-and-build-process", "title": "Mod 2: Compilation & Build Process", "time": "60m", "file": "mod02-compilation-and-build-process.html"},
            {"id": "mod03-data-types-and-variables", "title": "Mod 3: C Data Types & Variables", "time": "50m", "file": "mod03-data-types-and-variables.html"},
            {"id": "mod04-operators-and-expressions", "title": "Mod 4: Operators & Expressions", "time": "45m", "file": "mod04-operators-and-expressions.html"},
            {"id": "mod05-control-flow", "title": "Mod 5: Control Flow & State Machines", "time": "55m", "file": "mod05-control-flow.html"},
            {"id": "mod06-functions-and-modularity", "title": "Mod 6: Functions & Modular Design", "time": "50m", "file": "mod06-functions-and-modularity.html"},
        ]
    },
    {
        "part_id": "P2",
        "part_title": "Memory & Data Structures",
        "modules": [
            {"id": "mod07-pointers-and-memory-fundamentals", "title": "Mod 7: Pointers & Memory Architecture", "time": "60m", "file": "mod07-pointers-and-memory-fundamentals.html"},
            {"id": "mod08-pointer-arithmetic-and-arrays", "title": "Mod 8: Pointer Arithmetic & Arrays", "time": "55m", "file": "mod08-pointer-arithmetic-and-arrays.html"},
            {"id": "mod09-function-pointers-and-callbacks", "title": "Mod 9: Function Pointers & Callbacks", "time": "50m", "file": "mod09-function-pointers-and-callbacks.html"},
            {"id": "mod10-structures-unions-and-bitfields", "title": "Mod 10: Structures, Alignment & Bitfields", "time": "60m", "file": "mod10-structures-unions-and-bitfields.html"},
            {"id": "mod11-dynamic-memory-and-custom-allocators", "title": "Mod 11: Dynamic Memory & Allocators", "time": "65m", "file": "mod11-dynamic-memory-and-custom-allocators.html"},
        ]
    },
    {
        "part_id": "P3",
        "part_title": "Advanced C Programming",
        "modules": [
            {"id": "mod12-preprocessor-metaprogramming", "title": "Mod 12: Preprocessor Metaprogramming & X-Macros", "time": "55m", "file": "mod12-preprocessor-metaprogramming.html"},
            {"id": "mod13-type-qualifiers-and-storage", "title": "Mod 13: Type Qualifiers & Storage Classes", "time": "50m", "file": "mod13-type-qualifiers-and-storage.html"},
            {"id": "mod14-inline-assembly-and-hardware", "title": "Mod 14: Inline Assembly & Hardware Control", "time": "60m", "file": "mod14-inline-assembly-and-hardware.html"},
            {"id": "mod15-gnu-c-extensions-and-attributes", "title": "Mod 15: GNU C Extensions & Attributes", "time": "50m", "file": "mod15-gnu-c-extensions-and-attributes.html"},
            {"id": "mod16-compiler-optimizations-and-lto", "title": "Mod 16: Compiler Optimizations & LTO", "time": "55m", "file": "mod16-compiler-optimizations-and-lto.html"},
        ]
    },
    {
        "part_id": "P4",
        "part_title": "Linux & Engineering Tools",
        "modules": [
            {"id": "mod17-build-systems-make-and-cmake", "title": "Mod 17: Build Systems (Make & CMake)", "time": "60m", "file": "mod17-build-systems-make-and-cmake.html"},
            {"id": "mod18-gdb-debugging-and-core-dumps", "title": "Mod 18: GDB Debugging & Core Dumps", "time": "65m", "file": "mod18-gdb-debugging-and-core-dumps.html"},
            {"id": "mod19-valgrind-and-sanitizers", "title": "Mod 19: Valgrind & Sanitizers (ASan/UBSan)", "time": "50m", "file": "mod19-valgrind-and-sanitizers.html"},
            {"id": "mod20-static-analysis-tools", "title": "Mod 20: Static Analysis (Cppcheck & Clang-Tidy)", "time": "45m", "file": "mod20-static-analysis-tools.html"},
            {"id": "mod21-linux-profiling-and-perf", "title": "Mod 21: Linux Profiling & perf", "time": "55m", "file": "mod21-linux-profiling-and-perf.html"},
        ]
    },
    {
        "part_id": "P5",
        "part_title": "Testing & QA Engineering",
        "modules": [
            {"id": "mod22-unit-testing-unity-cmock", "title": "Mod 22: Unit Testing with Unity & CMock", "time": "55m", "file": "mod22-unit-testing-unity-cmock.html"},
            {"id": "mod23-mocking-hardware-and-hal", "title": "Mod 23: Mocking Hardware & HAL", "time": "50m", "file": "mod23-mocking-hardware-and-hal.html"},
            {"id": "mod24-code-coverage-gcov-lcov", "title": "Mod 24: Code Coverage (gcov & lcov)", "time": "45m", "file": "mod24-code-coverage-gcov-lcov.html"},
            {"id": "mod25-cicd-for-embedded-c", "title": "Mod 25: CI/CD Pipelines for Embedded C", "time": "50m", "file": "mod25-cicd-for-embedded-c.html"},
        ]
    },
    {
        "part_id": "P6",
        "part_title": "Linux System Programming",
        "modules": [
            {"id": "mod26-file-descriptors-and-posix-io", "title": "Mod 26: File Descriptors & POSIX I/O", "time": "55m", "file": "mod26-file-descriptors-and-posix-io.html"},
            {"id": "mod27-scalable-io-multiplexing-epoll", "title": "Mod 27: Scalable I/O with epoll", "time": "60m", "file": "mod27-scalable-io-multiplexing-epoll.html"},
            {"id": "mod28-process-lifecycle-and-ipc", "title": "Mod 28: Process Lifecycle & IPC (Pipes, Shared Memory)", "time": "60m", "file": "mod28-process-lifecycle-and-ipc.html"},
            {"id": "mod29-posix-threads-and-concurrency", "title": "Mod 29: POSIX Threads & Multithreading", "time": "55m", "file": "mod29-posix-threads-and-concurrency.html"},
            {"id": "mod30-thread-synchronization-and-locks", "title": "Mod 30: Thread Synchronization & Mutexes", "time": "60m", "file": "mod30-thread-synchronization-and-locks.html"},
            {"id": "mod31-posix-signals-and-timers", "title": "Mod 31: POSIX Signals & Real-Time Timers", "time": "50m", "file": "mod31-posix-signals-and-timers.html"},
            {"id": "mod32-network-sockets-programming", "title": "Mod 32: Sockets & Binary Protocols", "time": "60m", "file": "mod32-network-sockets-programming.html"},
        ]
    },
    {
        "part_id": "P7",
        "part_title": "Embedded C & Firmware",
        "modules": [
            {"id": "mod33-hardware-interfacing-and-mmio", "title": "Mod 33: Memory-Mapped I/O & Registers", "time": "65m", "file": "mod33-hardware-interfacing-and-mmio.html"},
            {"id": "mod34-linker-scripts-and-memory-maps", "title": "Mod 34: Linker Scripts & Memory Maps", "time": "60m", "file": "mod34-linker-scripts-and-memory-maps.html"},
            {"id": "mod35-startup-code-and-reset-sequence", "title": "Mod 35: Startup Code & Reset Sequence", "time": "55m", "file": "mod35-startup-code-and-reset-sequence.html"},
            {"id": "mod36-interrupts-nvic-and-concurrency", "title": "Mod 36: Interrupts, NVIC & Concurrency", "time": "65m", "file": "mod36-interrupts-nvic-and-concurrency.html"},
            {"id": "mod37-rtos-fundamentals-and-freertos", "title": "Mod 37: RTOS Fundamentals & FreeRTOS", "time": "70m", "file": "mod37-rtos-fundamentals-and-freertos.html"},
        ]
    },
    {
        "part_id": "P8",
        "part_title": "Secure C Programming",
        "modules": [
            {"id": "mod38-misra-c-compliance", "title": "Mod 38: MISRA C:2012 Compliance", "time": "60m", "file": "mod38-misra-c-compliance.html"},
            {"id": "mod39-cert-c-and-defensive-coding", "title": "Mod 39: CERT C & Defensive Coding", "time": "55m", "file": "mod39-cert-c-and-defensive-coding.html"},
        ]
    },
    {
        "part_id": "P9",
        "part_title": "Software Architecture",
        "modules": [
            {"id": "mod40-layered-hal-and-driver-design", "title": "Mod 40: Layered HAL & Device Driver Design", "time": "60m", "file": "mod40-layered-hal-and-driver-design.html"},
            {"id": "mod41-oop-patterns-and-interfaces", "title": "Mod 41: OOP Patterns & Interfaces in C", "time": "55m", "file": "mod41-oop-patterns-and-interfaces.html"},
            {"id": "mod42-publish-subscribe-event-runtimes", "title": "Mod 42: Event-Driven & Pub/Sub Runtimes", "time": "55m", "file": "mod42-publish-subscribe-event-runtimes.html"},
        ]
    },
    {
        "part_id": "P10",
        "part_title": "Capstone Projects",
        "modules": [
            {"id": "capstone01-bare-metal-bootloader", "title": "Project 1: Bare-Metal Bootloader (CRC32)", "time": "90m", "file": "capstone01-bare-metal-bootloader.html"},
            {"id": "capstone02-memory-pool-allocator", "title": "Project 2: Deterministic Memory Pool Allocator", "time": "90m", "file": "capstone02-memory-pool-allocator.html"},
            {"id": "capstone03-dma-serial-engine", "title": "Project 3: DMA Serial Protocol Engine", "time": "90m", "file": "capstone03-dma-serial-engine.html"},
            {"id": "capstone04-preemptive-rtos-kernel", "title": "Project 4: Preemptive Cooperative RTOS Kernel", "time": "120m", "file": "capstone04-preemptive-rtos-kernel.html"},
            {"id": "capstone05-embedded-cli-shell", "title": "Project 5: Embedded CLI Shell with Tab Completion", "time": "80m", "file": "capstone05-embedded-cli-shell.html"},
            {"id": "capstone06-misra-sensor-fusion", "title": "Project 6: MISRA-Compliant Sensor Fusion Pipeline", "time": "90m", "file": "capstone06-misra-sensor-fusion.html"},
            {"id": "capstone07-crash-dump-analyzer", "title": "Project 7: Post-Mortem Crash Dump Diagnostic Engine", "time": "90m", "file": "capstone07-crash-dump-analyzer.html"},
            {"id": "capstone08-posix-epoll-daemon", "title": "Project 8: POSIX Non-Blocking epoll Network Daemon", "time": "100m", "file": "capstone08-posix-epoll-daemon.html"},
        ]
    }
]

def generate_sidebar_html(active_file=None, is_index=False):
    prefix = "" if is_index else ""
    chapter_prefix = "chapters/" if is_index else ""

    html = []
    # Global navigation items
    html.append('''        <!-- Global Portal Navigation -->
        <div class="nav-module-group" style="margin-bottom:0.75rem;padding-bottom:0.75rem;border-bottom:1px solid var(--border-subtle);">
          <a href="/" class="chapter-nav-item" style="color:var(--text-secondary);">
            <span class="chapter-check-icon">🏠</span>
            <span class="chapter-title-text">Home (ACL)</span>
          </a>
          <a href="/courses/" class="chapter-nav-item" style="color:var(--text-secondary);">
            <span class="chapter-check-icon">📚</span>
            <span class="chapter-title-text">All Courses</span>
          </a>
          <a href="/courses/professional-c-in-embedded-systems/" class="chapter-nav-item" style="color:var(--text-secondary);">
            <span class="chapter-check-icon">📋</span>
            <span class="chapter-title-text">Course Dashboard</span>
          </a>
        </div>

        <!-- Overview -->
        <div class="nav-module-group">
          <a href="''' + ("index.html" if is_index else "../index.html") + '''" class="chapter-nav-item''' + (' active' if is_index else '') + '''">
            <span class="chapter-check-icon">📖</span>
            <span class="chapter-title-text">Handbook Overview</span>
          </a>
        </div>''')

    for part in CURRICULUM:
        html.append(f'''
        <!-- {part["part_id"]}: {part["part_title"]} -->
        <div class="nav-module-group">
          <button class="module-title-btn">
            <div class="module-title-left">
              <span class="module-badge">{part["part_id"]}</span>
              <span>{part["part_title"]}</span>
            </div>
            <span class="module-chevron">▼</span>
          </button>
          <ul class="chapter-sublist">''')
        
        for mod in part["modules"]:
            file_link = f"{chapter_prefix}{mod['file']}"
            is_active = (active_file == mod['file'])
            active_class = " active" if is_active else ""
            html.append(f'''            <li><a href="{file_link}" class="chapter-nav-item{active_class}"><span
                  class="chapter-check-icon">●</span><span class="chapter-title-text">{mod["title"]}</span><span
                  class="chapter-read-time">{mod["time"]}</span></a></li>''')
            
        html.append('''          </ul>
        </div>''')

    return "\n".join(html)

print("Sidebar generator initialized.")
