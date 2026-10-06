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
    <h3 class="text-xl font-bold mb-3 flex items-center"><i class="fas fa-microchip mr-2 text-brand-coral"></i> ARM Cortex-M Architecture: From CPU Core to STM32 Firmware</h3>
    <ul class="space-y-2 text-gray-200">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>36 chapters with labs and quizzes</strong>, relatively vendor-independent, with STM32 as the worked example.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Core internals</strong>: programmer's model, registers, Thumb-2, memory map, NVIC, exceptions and fault handling.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>System view</strong>: startup, vector table, linker script, clocks, MPU, FPU, DMA and caches.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Tools</strong>: SWD/JTAG, GDB, CMSIS, and tracing C code down to machine code.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>The engineering reference</strong> behind the STM32 with C and Embedded Rust courses.</li>
    </ul>
  </div>
  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">Syllabus</h3>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 1:</span> Embedded Processor Fundamentals</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 2:</span> ARM Architecture Overview</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 3:</span> The Cortex-M Family</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 4:</span> Cortex-M0, M0+, M3, M4, M7 and M33</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 5:</span> The Programmer's Model</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 6:</span> Registers</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 7:</span> R0-R12, SP, LR, PC and xPSR</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 8:</span> Instruction Set Basics</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 9:</span> Thumb and Thumb-2</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 10:</span> The Memory Map</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 11:</span> Flash Memory</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 12:</span> SRAM</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 13:</span> The Stack</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 14:</span> The Heap</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 15:</span> Memory-Mapped I/O</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 16:</span> NVIC</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 17:</span> Interrupts</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 18:</span> Exceptions</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 19:</span> SysTick</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 20:</span> SVC and PendSV</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 21:</span> Fault Handling</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 22:</span> Reset and Startup</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 23:</span> The Vector Table</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 24:</span> The Linker Script</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 25:</span> The Clock System</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 26:</span> The MPU</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 27:</span> The FPU</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 28:</span> DMA</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 29:</span> Cache and Memory Systems</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 30:</span> Debug Architecture</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 31:</span> SWD and JTAG</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 32:</span> GDB</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 33:</span> CMSIS</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 34:</span> STM32 Peripheral Architecture</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 35:</span> From Reset to main()</h4>
    </div>
    <div class="mb-3 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 36:</span> From C Code to Machine Code</h4>
    </div>
  </div>
</div>
"""

course, created = Course.objects.update_or_create(
    slug="arm-cortex-m-architecture",
    defaults={
        "title": "ARM Cortex-M Architecture for Embedded Engineers",
        "short_description": "From CPU core to STM32 firmware: programmer's model, memory map, NVIC, exceptions, faults, startup, linker script, clocks, MPU, FPU, DMA, SWD/GDB, CMSIS and C to machine code.",
        "description": html_content,
        "duration_weeks": 10,
        "duration_hours": 60,
        "skill_level": "Intermediate",
        "technologies": "ARM Cortex-M, Thumb-2, NVIC, CMSIS, arm-none-eabi-gcc, GDB, OpenOCD, STM32",
        "fee": "9000/- + GST",
        "timing": "Weekend course, 4:00pm to 5:30pm",
        "start_date": "To be announced",
        "trainers": "Kamal Kumar Mukiri",
        "recordings_info": "Sessions are recorded and shared and maintained for 3 months.",
        "materials_info": "Full digital book with 36 chapters, code samples, lab checklists and quizzes.",
        "prerequisites": "Basic C programming and digital electronics.",
        "training_mode": "Online",
        "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800",
    },
)

print(("Created" if created else "Updated"), "course:", course.title, "(ID: %s)" % course.id)
