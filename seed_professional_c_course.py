import os
import django
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from courses.models import Course

html_content = """
<div class="space-y-8">
  <div class="bg-brand-navy text-white p-6 rounded-xl shadow-lg border-l-4 border-brand-coral">
    <h3 class="text-xl font-bold mb-3 flex items-center">
      <i class="fas fa-microchip mr-2 text-brand-coral"></i> Professional C in Embedded Systems: Architecture, Firmware & Real-Time Engineering
    </h3>
    <ul class="space-y-2 text-gray-200 text-sm md:text-base">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Comprehensive Curriculum</strong>: 10 structured parts, 42 in-depth modules, and 8 hands-on capstone engineering projects.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Dual Target Architectures</strong>: Linux POSIX systems (x86_64) + Bare-Metal & RTOS on ARM Cortex-M microcontrollers.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Modern Industry Standards</strong>: C11 / C17 / C23 standards, MISRA C:2012 compliance, and CERT C secure coding rules.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Low-Level Mastery</strong>: Linker scripts, ELF binary analysis, memory alignment, hardware register manipulation, cache coherence, and MMU/MPU setup.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Professional Toolchain</strong>: GCC, Clang, Make/CMake, GDB, Valgrind, AddressSanitizer, perf, logic analyzers, and OpenOCD.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Digital Reference Handbook</strong>: Direct access to comprehensive interactive chapters, reference architectures, and downloadable code templates.</li>
    </ul>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">Course Curriculum & Modules</h3>
    
    <div class="space-y-6">
      <div class="p-5 bg-white rounded-xl shadow-sm border border-gray-100">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <span class="bg-brand-coral text-white text-xs px-2.5 py-1 rounded-md font-mono mr-2">Part 1</span>
          C Programming Foundations & Build Pipeline (Mod 1–6)
        </h4>
        <p class="text-brand-textSecondary text-sm mb-3">
          Deep dive into the compilation pipeline (preprocessing, parsing, code generation, assembly, linking). Memory layout of C programs (.text, .rodata, .data, .bss, stack, heap), standard data types, integer promotions, bitwise operators, modular multi-file architecture, and state machine design.
        </p>
        <div class="text-xs text-gray-500 font-mono">Topics: mod01 Intro to C • mod02 Build Process & Linkers • mod03 Types & IEEE-754 • mod04 Operators • mod05 Control Flow & FSM • mod06 Modular Design</div>
      </div>

      <div class="p-5 bg-white rounded-xl shadow-sm border border-gray-100">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <span class="bg-brand-navy text-white text-xs px-2.5 py-1 rounded-md font-mono mr-2">Part 2</span>
          Memory Management & Data Structures in Systems (Mod 7–11)
        </h4>
        <p class="text-brand-textSecondary text-sm mb-3">
          Pointer arithmetic, multi-level pointers, function pointers, callbacks, structure padding/alignment, bitfields, union memory punning, dynamic memory allocators (malloc/free internals), ring buffers, intrusive linked lists, and memory pool allocators for deterministic embedded runtimes.
        </p>
      </div>

      <div class="p-5 bg-white rounded-xl shadow-sm border border-gray-100">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <span class="bg-brand-navy text-white text-xs px-2.5 py-1 rounded-md font-mono mr-2">Part 3</span>
          Advanced C, Metaprogramming & Optimization (Mod 12–16)
        </h4>
        <p class="text-brand-textSecondary text-sm mb-3">
          Advanced preprocessor macros (X-Macros, variadic macros, token pasting), const/volatile/restrict semantics, inline assembly, GNU C attributes (__attribute__((packed, aligned, section))), compiler optimization flags (-O2, -O3, -Os, LTO), and vectorization.
        </p>
      </div>

      <div class="p-5 bg-white rounded-xl shadow-sm border border-gray-100">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <span class="bg-brand-navy text-white text-xs px-2.5 py-1 rounded-md font-mono mr-2">Part 4</span>
          Linux Engineering Tools, GDB & Performance Profiling (Mod 17–21)
        </h4>
        <p class="text-brand-textSecondary text-sm mb-3">
          Mastery of Makefile and CMake build systems, command-line debugging with GDB (breakpoints, watchpoints, stack frame inspection, core dump analysis), memory leak diagnostics via Valgrind and AddressSanitizer (ASan/UBSan), static analysis with Cppcheck and Clang-Tidy, and Linux perf profiling.
        </p>
      </div>

      <div class="p-5 bg-white rounded-xl shadow-sm border border-gray-100">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <span class="bg-brand-navy text-white text-xs px-2.5 py-1 rounded-md font-mono mr-2">Part 5</span>
          Embedded C, Hardware Interfacing & Firmware Architecture (Mod 33–37)
        </h4>
        <p class="text-brand-textSecondary text-sm mb-3">
          Memory-mapped I/O (MMIO), register-level programming without vendor HALs, custom linker scripts (.ld) and startup files, interrupt service routine (ISR) design, atomic operations, DMA zero-copy transfers, communication protocols (UART, SPI, I2C, CAN), and FreeRTOS task scheduling.
        </p>
      </div>

      <div class="p-5 bg-white rounded-xl shadow-sm border border-gray-100">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <span class="bg-brand-navy text-white text-xs px-2.5 py-1 rounded-md font-mono mr-2">Part 6</span>
          Safety-Critical Standards & Secure C (Mod 38–42)
        </h4>
        <p class="text-brand-textSecondary text-sm mb-3">
          MISRA C:2012 guidelines, SEI CERT C coding standards, buffer overflow prevention, integer wrap handling, defensive error propagation, watchdog timer integration, and hardware fault exception handlers (HardFault, BusFault, MemManage).
        </p>
      </div>

      <div class="p-5 bg-blue-50/80 rounded-xl border border-blue-200">
        <h4 class="text-lg font-bold text-brand-navy mb-2 flex items-center">
          <i class="fas fa-project-diagram mr-2 text-brand-coral"></i>
          Industry Capstone Engineering Projects (1–8)
        </h4>
        <ul class="grid md:grid-cols-2 gap-2 text-sm text-gray-700">
          <li><strong>Project 1:</strong> Custom Bare-Metal Bootloader with CRC32 Verification</li>
          <li><strong>Project 2:</strong> Deterministic Fixed-Size Block Memory Allocator</li>
          <li><strong>Project 3:</strong> High-Throughput DMA-Driven Serial Protocol Engine</li>
          <li><strong>Project 4:</strong> Priority-Based Preemptive Cooperative Kernel</li>
          <li><strong>Project 5:</strong> Command Line Shell (CLI) with Tab Completion for RTOS</li>
          <li><strong>Project 6:</strong> MISRA-Compliant Sensor Fusion Pipeline (I2C/SPI)</li>
          <li><strong>Project 7:</strong> Fault Logger & Post-Mortem Crash Dump Analyzer</li>
          <li><strong>Project 8:</strong> POSIX Non-Blocking Network Daemon (epoll/sockets)</li>
        </ul>
      </div>
    </div>
  </div>
</div>
"""

course, created = Course.objects.update_or_create(
    slug="professional-c-in-embedded-systems",
    defaults={
        "title": "Professional C in Embedded Systems",
        "short_description": "Master professional C programming for embedded systems, Linux system programming, firmware, and safety-critical architectures (MISRA/CERT C).",
        "description": html_content,
        "duration_weeks": 8,
        "duration_hours": 32,
        "skill_level": "Intermediate to Advanced",
        "technologies": "C11/C17, Embedded C, ARM Cortex-M, Linux POSIX, GDB, CMake, FreeRTOS, MISRA C",
        "fee": "₹8000/- + GST",
        "timing": "Mon to Fri • 6:00 PM – 7:30 PM IST",
        "start_date": "Starts 2nd Dec 2026 to 22nd Dec 2026 (Live Interactive Classes)",
        "trainers": "Kamal Kumar Mukiri & Industry Architects",
        "recordings_info": "Full HD Session Recordings & Lifetime Access",
        "materials_info": "Digital Handbook, Linker Scripts, Drivers & Capstone Repositories",
        "prerequisites": "Basic C programming familiarity and understanding of computer architecture.",
        "training_mode": "Live Online & Interactive Hands-on Labs",
    }
)

print(f"Course '{course.title}' (slug: {course.slug}) {'created' if created else 'updated'} successfully!")
