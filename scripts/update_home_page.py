import re

with open("private_books/c_programming_professional/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# We want to replace the hero and add the new comprehensive sections right above "<h2>Curriculum & Learning Modules</h2>"
target = """        <h2>Curriculum &amp; Learning Modules</h2>
        <p>
          The course emphasizes practical systems engineering skills, industry-standard tools, software architecture, debugging,
          testing, security, and performance—not just basic C syntax. Explore each part below:
        </p>"""

new_sections = """        <!-- SECTION 1: WHY LEARN PROFESSIONAL C -->
        <section id="why-learn-c" style="margin-top:2.5rem;margin-bottom:3rem;">
          <h2 style="display:flex;align-items:center;gap:10px;border-bottom:2px solid var(--border-subtle);padding-bottom:0.75rem;">
            <span>⚡ Why Professional C in Embedded Systems?</span>
          </h2>
          <p class="chapter-lead" style="font-size:1.05rem;line-height:1.7;color:var(--text-secondary);margin-bottom:1.5rem;">
            C remains the undisputed foundational language of the physical computing world. From medical ventilators, flight control systems, and automotive electronic control units (ECUs) to Linux kernel subsystems, satellite transponders, and micro-robotics, <strong>C executes closest to the silicon</strong> with unmatched determinism, predictable memory footprint, and bare-metal latency.
          </p>

          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:1.5rem;margin:1.5rem 0;">
            <div style="background:var(--bg-surface, #1e2433);border:1px solid var(--border-subtle, rgba(255,255,255,0.1));border-radius:12px;padding:1.5rem;box-shadow:0 4px 14px rgba(0,0,0,0.1);">
              <div style="font-size:1.75rem;margin-bottom:0.75rem;">🎯</div>
              <h3 style="font-size:1.15rem;margin:0 0 0.5rem 0;color:var(--text-primary);">Deterministic Real-Time Execution</h3>
              <p style="font-size:0.9rem;line-height:1.6;color:var(--text-secondary);margin:0;">
                No hidden runtime garbage collection pauses, no unpredictable JIT warm-ups, and no dynamic heap allocations required. What you write in C directly translates into predictable, single-cycle ARM Thumb-2 or x86 assembly instructions.
              </p>
            </div>

            <div style="background:var(--bg-surface, #1e2433);border:1px solid var(--border-subtle, rgba(255,255,255,0.1));border-radius:12px;padding:1.5rem;box-shadow:0 4px 14px rgba(0,0,0,0.1);">
              <div style="font-size:1.75rem;margin-bottom:0.75rem;">🔌</div>
              <h3 style="font-size:1.15rem;margin:0 0 0.5rem 0;color:var(--text-primary);">Direct Hardware &amp; Silicon Access</h3>
              <p style="font-size:0.9rem;line-height:1.6;color:var(--text-secondary);margin:0;">
                Interact directly with memory-mapped I/O (MMIO), volatile hardware control registers, NVIC interrupt controllers, DMA engines, and linker-placed memory segments without intervening abstraction penalties.
              </p>
            </div>

            <div style="background:var(--bg-surface, #1e2433);border:1px solid var(--border-subtle, rgba(255,255,255,0.1));border-radius:12px;padding:1.5rem;box-shadow:0 4px 14px rgba(0,0,0,0.1);">
              <div style="font-size:1.75rem;margin-bottom:0.75rem;">🛡️</div>
              <h3 style="font-size:1.15rem;margin:0 0 0.5rem 0;color:var(--text-primary);">Industry Standard in Safety-Critical Systems</h3>
              <p style="font-size:0.9rem;line-height:1.6;color:var(--text-secondary);margin:0;">
                Automotive (ISO 26262), aerospace (DO-178C), industrial automation (IEC 61508), and medical devices (IEC 62304) mandate safety-standard compliance like MISRA C:2012 and SEI CERT C. C is the global universal currency of mission-critical engineering.
              </p>
            </div>
          </div>

          <div class="callout callout-tip" style="margin-top:1.5rem;">
            <div class="callout-header">💡 The Engineering Reality: Syntax vs Systems Mastery</div>
            <div class="callout-body">
              <p>
                Knowing loop syntax or basic function definitions in C is simple; writing rock-solid production firmware that executes 24/7 for 15 years inside an automotive brake-by-wire controller without a stack overflow, race condition, or memory leak requires an entirely different discipline. This course bridges the chasm between "academic textbook C" and high-reliability industry systems engineering.
              </p>
            </div>
          </div>
        </section>

        <!-- SECTION 2: IMPORTANCE & INDUSTRY RELEVANCE -->
        <section id="why-it-matters" style="margin-bottom:3rem;">
          <h2 style="display:flex;align-items:center;gap:10px;border-bottom:2px solid var(--border-subtle);padding-bottom:0.75rem;">
            <span>💎 The Core Pillars: What Makes This Handbook Unique</span>
          </h2>
          <p style="color:var(--text-secondary);font-size:1rem;line-height:1.7;">
            Typical tutorials teach C in a vacuum on desktop OS command prompts with artificial toys like calculators and text manipulation. In modern industry, embedded software engineering is deeply coupled with computer architecture, memory constraints, debugging protocols, operating system internals, and stringent automated testing:
          </p>

          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:1.5rem;margin:1.5rem 0;">
            <div style="background:rgba(255,91,91,0.04);border-left:4px solid var(--brand-coral, #ff5b5b);padding:1.25rem;border-radius:0 8px 8px 0;background-color:var(--bg-surface);">
              <h4 style="margin:0 0 0.5rem 0;color:var(--text-primary);font-size:1.05rem;">1. Full Hardware &amp; System Symbiosis</h4>
              <p style="margin:0;font-size:0.875rem;color:var(--text-secondary);line-height:1.6;">
                We connect high-level C abstractions to physical realities: CPU registers, cache line alignment, bus fault exceptions, ARM Cortex-M NVIC prioritization, DMA ring-buffers, and POSIX epoll event loops.
              </p>
            </div>

            <div style="background:rgba(0,184,212,0.04);border-left:4px solid #00b8d4;padding:1.25rem;border-radius:0 8px 8px 0;background-color:var(--bg-surface);">
              <h4 style="margin:0 0 0.5rem 0;color:var(--text-primary);font-size:1.05rem;">2. Rigorous Safety &amp; Defensive Standards</h4>
              <p style="margin:0;font-size:0.875rem;color:var(--text-secondary);line-height:1.6;">
                Every line of code is held to MISRA C:2012 guidelines and SEI CERT C defensive patterns. Learn how undefined behavior (UB), signed integer overflow, pointer aliasing, and reentrancy bugs destroy systems.
              </p>
            </div>

            <div style="background:rgba(139,92,246,0.04);border-left:4px solid #8b5cf6;padding:1.25rem;border-radius:0 8px 8px 0;background-color:var(--bg-surface);">
              <h4 style="margin:0 0 0.5rem 0;color:var(--text-primary);font-size:1.05rem;">3. Professional Tooling &amp; Automation</h4>
              <p style="margin:0;font-size:0.875rem;color:var(--text-secondary);line-height:1.6;">
                Move beyond clicking "build" buttons. Master GNU Make, CMake, GCC compiler flags, GDB remote debugging, hardware register inspection, Valgrind, AddressSanitizer (ASan), Cppcheck, and automated Unity/CMock test harnesses in CI/CD.
              </p>
            </div>

            <div style="background:rgba(16,185,129,0.04);border-left:4px solid #10b981;padding:1.25rem;border-radius:0 8px 8px 0;background-color:var(--bg-surface);">
              <h4 style="margin:0 0 0.5rem 0;color:var(--text-primary);font-size:1.05rem;">4. Scalable Software Architecture</h4>
              <p style="margin:0;font-size:0.875rem;color:var(--text-secondary);line-height:1.6;">
                Structure complex embedded codebases using clean Hardware Abstraction Layers (HAL), object-oriented encapsulation and polymorphic vtables in pure C, and decoupled zero-copy publish-subscribe event systems.
              </p>
            </div>
          </div>
        </section>

        <!-- SECTION 3: QUICK GUIDE - HOW TO USE THIS MATERIAL -->
        <section id="how-to-use" style="margin-bottom:3.5rem;">
          <h2 style="display:flex;align-items:center;gap:10px;border-bottom:2px solid var(--border-subtle);padding-bottom:0.75rem;">
            <span>📖 Quick Guide: How to Study &amp; Master This Material</span>
          </h2>
          <p style="color:var(--text-secondary);font-size:1rem;line-height:1.7;">
            This handbook is organized into <strong>10 progressive parts, 42 in-depth masterclass modules, and 8 production capstones</strong>. Whether you are an embedded engineer, Linux systems developer, or firmware architect, follow this battle-tested 4-step workflow for every chapter:
          </p>

          <div style="margin:2rem 0;display:flex;flex-direction:column;gap:1.25rem;">
            <div style="display:flex;gap:1.25rem;align-items:flex-start;background:var(--bg-surface);padding:1.25rem;border-radius:10px;border:1px solid var(--border-subtle);">
              <div style="background:var(--brand-coral);color:#fff;font-weight:800;font-size:1.1rem;min-width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(255,91,91,0.4);">
                1
              </div>
              <div>
                <h4 style="margin:0 0 0.35rem 0;color:var(--text-primary);font-size:1.1rem;">Study the Theory &amp; Architecture Diagrams</h4>
                <p style="margin:0;font-size:0.9rem;color:var(--text-secondary);line-height:1.6;">
                  Begin with the conceptual breakdown, memory layouts, register diagrams, and standards analysis (ISO C, POSIX.1-2017, MISRA). Pay close attention to the execution models and hardware bus diagrams before touching the keyboard.
                </p>
              </div>
            </div>

            <div style="display:flex;gap:1.25rem;align-items:flex-start;background:var(--bg-surface);padding:1.25rem;border-radius:10px;border:1px solid var(--border-subtle);">
              <div style="background:#00b8d4;color:#fff;font-weight:800;font-size:1.1rem;min-width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,184,212,0.4);">
                2
              </div>
              <div>
                <h4 style="margin:0 0 0.35rem 0;color:var(--text-primary);font-size:1.1rem;">Hand-Type and Compile the Production Code</h4>
                <p style="margin:0;font-size:0.9rem;color:var(--text-secondary);line-height:1.6;">
                  Do not copy-paste code snippets. Hand-type the implementation into your editor. Compile with explicit strict flags (<code>-Wall -Wextra -Werror -pedantic -std=c11</code>) and observe compiler diagnostics. Study the defensive bounds checking and error handling strategies.
                </p>
              </div>
            </div>

            <div style="display:flex;gap:1.25rem;align-items:flex-start;background:var(--bg-surface);padding:1.25rem;border-radius:10px;border:1px solid var(--border-subtle);">
              <div style="background:#8b5cf6;color:#fff;font-weight:800;font-size:1.1rem;min-width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(139,92,246,0.4);">
                3
              </div>
              <div>
                <h4 style="margin:0 0 0.35rem 0;color:var(--text-primary);font-size:1.1rem;">Execute the Hands-On Lab &amp; Diagnostic Exercises</h4>
                <p style="margin:0;font-size:0.9rem;color:var(--text-secondary);line-height:1.6;">
                  Every single module features a comprehensive, multi-step <strong>Hands-On Lab</strong>. Run the code under debuggers (GDB/LLDB), inspect memory blocks, step through disassembly, run sanitizers (ASan/UBSan), or verify with a logic analyzer / QEMU emulator. Complete the lab checklists to prove operational readiness.
                </p>
              </div>
            </div>

            <div style="display:flex;gap:1.25rem;align-items:flex-start;background:var(--bg-surface);padding:1.25rem;border-radius:10px;border:1px solid var(--border-subtle);">
              <div style="background:#10b981;color:#fff;font-weight:800;font-size:1.1rem;min-width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(16,185,129,0.4);">
                4
              </div>
              <div>
                <h4 style="margin:0 0 0.35rem 0;color:var(--text-primary);font-size:1.1rem;">Validate with the Knowledge Check Quiz</h4>
                <p style="margin:0;font-size:0.9rem;color:var(--text-secondary);line-height:1.6;">
                  Finish each chapter by taking the interactive 5-question conceptual quiz. Read the technical rationales for both correct and incorrect options to solidify your mental model of undefined behavior, memory hazards, and hardware register semantics.
                </p>
              </div>
            </div>
          </div>

          <!-- TARGET LEARNING PATHS -->
          <div style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:12px;padding:1.75rem;margin-top:2rem;">
            <h3 style="margin:0 0 1rem 0;font-size:1.2rem;color:var(--text-primary);display:flex;align-items:center;gap:8px;">
              <span>🎯 Recommended Learning Tracks</span>
            </h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:1.25rem;">
              <div style="padding:1rem;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
                <span style="font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;font-weight:700;color:var(--brand-coral);">Track A</span>
                <h4 style="margin:0.25rem 0 0.5rem 0;color:var(--text-primary);font-size:1rem;">Bare-Metal &amp; RTOS Firmware</h4>
                <p style="font-size:0.85rem;color:var(--text-secondary);line-height:1.5;margin-bottom:0.75rem;">For microcontrollers (ARM Cortex-M, STM32, RISC-V, ESP32):</p>
                <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.6;">
                  <code>P1 (Foundations)</code> ➔ <code>P2 (Memory/Pointers)</code> ➔ <code>P3 (Advanced C)</code> ➔ <code>P7 (Bare-Metal/RTOS)</code> ➔ <code>P8 (Safety)</code> ➔ <code>P9 (HAL/PubSub)</code> ➔ <code>P10 (Capstones 1-6)</code>
                </div>
              </div>

              <div style="padding:1rem;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
                <span style="font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;font-weight:700;color:#00b8d4;">Track B</span>
                <h4 style="margin:0.25rem 0 0.5rem 0;color:var(--text-primary);font-size:1rem;">Embedded Linux &amp; Systems Engineering</h4>
                <p style="font-size:0.85rem;color:var(--text-secondary);line-height:1.5;margin-bottom:0.75rem;">For gateway software, daemons, edge compute, and IPC:</p>
                <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.6;">
                  <code>P1 (Foundations)</code> ➔ <code>P2 (Memory)</code> ➔ <code>P4 (Tools/GDB/Perf)</code> ➔ <code>P5 (QA/Testing)</code> ➔ <code>P6 (POSIX &amp; epoll)</code> ➔ <code>P9 (Architecture)</code> ➔ <code>P10 (Capstones 7-8)</code>
                </div>
              </div>

              <div style="padding:1rem;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
                <span style="font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;font-weight:700;color:#8b5cf6;">Track C</span>
                <h4 style="margin:0.25rem 0 0.5rem 0;color:var(--text-primary);font-size:1rem;">Automotive &amp; Medical Safety (MISRA / CERT)</h4>
                <p style="font-size:0.85rem;color:var(--text-secondary);line-height:1.5;margin-bottom:0.75rem;">For engineers building ISO 26262 / DO-178C certified devices:</p>
                <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.6;">
                  <code>P3 (Advanced C)</code> ➔ <code>P5 (Unit Testing &amp; Coverage)</code> ➔ <code>P8 (MISRA C &amp; CERT C)</code> ➔ <code>P9 (Layered HAL)</code> ➔ <code>Capstone 06 (Sensor Fusion)</code>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- SECTION 4: HARDWARE & TOOLCHAIN REQUIREMENTS -->
        <section id="hardware-toolchain" style="margin-bottom:3.5rem;">
          <h2 style="display:flex;align-items:center;gap:10px;border-bottom:2px solid var(--border-subtle);padding-bottom:0.75rem;">
            <span>🛠️ Target Hardware &amp; Toolchain Setup</span>
          </h2>
          <p style="color:var(--text-secondary);font-size:1rem;line-height:1.7;">
            To gain the most from this handbook, having hands-on access to a development workstation and physical or simulated target hardware is recommended:
          </p>

          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:1.5rem;margin:1.5rem 0;">
            <div class="callout callout-tip" style="margin:0;">
              <div class="callout-header">💻 Host Software &amp; Tooling (Linux / macOS / WSL2)</div>
              <div class="callout-body">
                <ul style="margin:0.5rem 0 0 1.25rem;padding:0;font-size:0.875rem;line-height:1.6;">
                  <li><strong>Compiler:</strong> GCC 11+ or Clang 14+ (supporting C11/C17/C23)</li>
                  <li><strong>Cross Toolchain:</strong> <code>arm-none-eabi-gcc</code>, <code>binutils</code>, <code>newlib</code></li>
                  <li><strong>Build Tools:</strong> GNU Make 4.0+, CMake 3.20+, Ninja</li>
                  <li><strong>Diagnostics:</strong> GDB, Valgrind 3.18+, AddressSanitizer, UBSan, Cppcheck, Clang-Tidy</li>
                  <li><strong>Emulation:</strong> QEMU (<code>qemu-system-arm</code>, <code>qemu-arm</code>)</li>
                </ul>
              </div>
            </div>

            <div class="callout callout-hardware" style="margin:0;">
              <div class="callout-header">⚡ Reference Target Hardware</div>
              <div class="callout-body">
                <ul style="margin:0.5rem 0 0 1.25rem;padding:0;font-size:0.875rem;line-height:1.6;">
                  <li><strong>Primary Target:</strong> STM32 NUCLEO-F446RE (ARM Cortex-M4 @ 180 MHz, 512KB Flash, 128KB SRAM) or NUCLEO-F401RE / F411RE</li>
                  <li><strong>Alternative Target:</strong> Raspberry Pi Pico (RP2040) / ESP32 or Linux x86_64 host system</li>
                  <li><strong>Debug Interface:</strong> On-board ST-LINK/V2-1 or V3, OpenOCD, J-Link</li>
                  <li><strong>Lab Instrumentation:</strong> 8-channel 24 MHz USB Logic Analyzer (with PulseView/sigrok)</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <h2>Curriculum &amp; Learning Modules</h2>
        <p>
          The course emphasizes practical systems engineering skills, industry-standard tools, software architecture, debugging,
          testing, security, and performance—not just basic C syntax. Explore each part below:
        </p>"""

if target in content:
    content = content.replace(target, new_sections)
    with open("private_books/c_programming_professional/index.html", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: index.html updated successfully with Home Page deep dives.")
else:
    print("ERROR: Target text not found in index.html")
