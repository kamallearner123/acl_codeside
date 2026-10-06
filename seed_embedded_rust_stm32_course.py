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
    <h3 class="text-xl font-bold mb-3 flex items-center"><i class="fas fa-microchip mr-2 text-brand-coral"></i> Embedded Rust with STM32: From Bare-Metal Firmware to Safe Embedded Products</h3>
    <ul class="space-y-2 text-gray-200">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>26 chapters with labs and quizzes</strong> on the NUCLEO-F446RE board.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Not Book 1 in Rust</strong>: learn to think differently - ownership, borrowing, lifetimes, Option and Result, no_std.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Peripherals the Rust way</strong>: GPIO, interrupts, timers, PWM, UART, I&sup2;C, SPI, ADC, CAN and DMA.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Professional workflow</strong>: probe-rs, defmt, GDB, host tests, mocks and CI.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Product capstone</strong>: a fail-safe fan and environment controller with watchdog and CAN telemetry.</li>
    </ul>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">Syllabus</h3>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 1:</span> Why Embedded Rust?</h4>
      <p class="text-brand-textSecondary">C-to-Rust transition: pointer lifetime, buffer bounds, shared mutable state and unchecked errors.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 2:</span> Rust for C Developers</h4>
      <p class="text-brand-textSecondary">Variables, types, functions, integer behaviour and arrays.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 3:</span> Ownership</h4>
      <p class="text-brand-textSecondary">Moves, copies, Drop and peripherals as owned values.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 4:</span> Borrowing</h4>
      <p class="text-brand-textSecondary">References, the aliasing rule and slices.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 5:</span> Lifetimes</h4>
      <p class="text-brand-textSecondary">Lifetime annotations and embedded implications such as DMA buffers.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 6:</span> Structs &amp; Enums</h4>
      <p class="text-brand-textSecondary">Hardware abstraction, newtypes and state machines.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 7:</span> Traits &amp; Generics</h4>
      <p class="text-brand-textSecondary">Driver abstraction with embedded-hal.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 8:</span> Option &amp; Result</h4>
      <p class="text-brand-textSecondary">Error handling without NULL or error codes.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 9:</span> no_std</h4>
      <p class="text-brand-textSecondary">The embedded environment, core and heapless.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 10:</span> Cross Compilation</h4>
      <p class="text-brand-textSecondary">ARM targets and cargo configuration.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 11:</span> Linker &amp; Memory Layout</h4>
      <p class="text-brand-textSecondary">Flash, SRAM, memory.x and sections.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 12:</span> Embedded Rust Toolchain</h4>
      <p class="text-brand-textSecondary">Cargo, rustup, probe-rs, flip-link and defmt.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 13:</span> GPIO</h4>
      <p class="text-brand-textSecondary">First Rust firmware with pin type-states.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 14:</span> Interrupts</h4>
      <p class="text-brand-textSecondary">ISR architecture, atomics, critical sections and RTIC.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 15:</span> Timers &amp; PWM</h4>
      <p class="text-brand-textSecondary">Hardware control with typed duty cycles.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 16:</span> UART</h4>
      <p class="text-brand-textSecondary">Serial communication and safe command parsing.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 17:</span> I&sup2;C</h4>
      <p class="text-brand-textSecondary">Writing a generic sensor driver.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 18:</span> SPI</h4>
      <p class="text-brand-textSecondary">Displays, flash and storage.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 19:</span> ADC</h4>
      <p class="text-brand-textSecondary">Sensor acquisition with typed units.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 20:</span> CAN</h4>
      <p class="text-brand-textSecondary">Automotive communication, bit timing and HAL support.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 21:</span> DMA</h4>
      <p class="text-brand-textSecondary">High-performance peripherals with ownership-safe buffers.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 22:</span> Testing</h4>
      <p class="text-brand-textSecondary">Unit, mocked and on-target tests, plus CI.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 23:</span> Debugging</h4>
      <p class="text-brand-textSecondary">ST-LINK, probe-rs, GDB, RTT and hard faults.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 24:</span> Rust Driver Design</h4>
      <p class="text-brand-textSecondary">Reusable, portable driver crates and type-state.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 25:</span> Embedded Architecture</h4>
      <p class="text-brand-textSecondary">Modules, interfaces, RTIC vs Embassy and reliability.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 26:</span> Capstone</h4>
      <p class="text-brand-textSecondary">A safe embedded product: fan and environment controller node.</p>
    </div>
  </div>

  <div class="mt-10 mb-4 text-center print:hidden">
    <a href="/courses/embedded-rust-with-stm32/book/" target="_blank" rel="noopener noreferrer" class="bg-brand-coral hover:bg-opacity-90 text-white px-8 py-3.5 rounded-full font-bold shadow-lg inline-flex items-center justify-center text-lg">
      Go to Course <i class="fas fa-arrow-right ml-2"></i>
    </a>
  </div>
</div>
"""

course, created = Course.objects.update_or_create(
    slug="embedded-rust-with-stm32",
    defaults={
        "title": "Embedded Rust with STM32",
        "short_description": "For STM32/C developers: learn to think in Rust - ownership, borrowing, lifetimes, Option/Result, no_std - then build safe firmware with GPIO, interrupts, timers, PWM, UART, I2C, SPI, ADC, CAN and DMA, with testing, debugging and a product capstone.",
        "description": html_content,
        "duration_weeks": 10,
        "duration_hours": 60,
        "skill_level": "Intermediate",
        "technologies": "Rust, STM32, no_std, stm32f4xx-hal, embedded-hal, probe-rs, defmt, RTIC, Embassy",
        "fee": "9000/- + GST",
        "timing": "Weekend course, 4:00pm to 5:30pm",
        "start_date": "To be announced",
        "trainers": "Kamal Kumar Mukiri",
        "recordings_info": "Sessions are recorded and shared and maintained for 3 months.",
        "materials_info": "Full digital book with 26 chapters, code samples, lab checklists, quizzes and a capstone.",
        "prerequisites": "STM32 Firmware Development with C (or equivalent C and STM32 experience).",
        "training_mode": "Online",
        "image": "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?auto=format&fit=crop&q=80&w=800",
    },
)

print(("Created" if created else "Updated"), "course:", course.title, "(ID: %s)" % course.id)
