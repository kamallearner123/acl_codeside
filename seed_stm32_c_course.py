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
    <h3 class="text-xl font-bold mb-3 flex items-center"><i class="fas fa-microchip mr-2 text-brand-coral"></i> Programming STM32 with C: From CubeMX to Hardware Debugging</h3>
    <ul class="space-y-2 text-gray-200">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>9 chapters, 9 labs, 5 mini projects and a capstone</strong> on the NUCLEO-F446RE board.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Foundation to peripherals</strong>: Embedded C, GPIO, interrupts, timers, PWM, ADC, UART, DMA, I&sup2;C, SPI and CAN.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Real hardware</strong>: sensors, OLED displays, servos and DC/stepper motors.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Debug like a professional</strong>: ST-LINK, SWD, GDB, logic analyzer and HardFault diagnosis.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Full digital book</strong> with code samples, quizzes and lab checklists.</li>
    </ul>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">Syllabus</h3>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 1:</span> Embedded C &amp; STM32 Architecture</h4>
      <p class="text-brand-textSecondary">Embedded C, volatile, bit manipulation, memory map, buses, clocks, boot sequence.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 2:</span> STM32CubeMX &amp; STM32CubeIDE</h4>
      <p class="text-brand-textSecondary">Pinout and clock tree configuration, HAL code generation, safe user code sections.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 3:</span> GPIO &amp; Interrupts</h4>
      <p class="text-brand-textSecondary">Input/output modes, EXTI, NVIC priorities, ISR patterns, debouncing.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 4:</span> Timers, PWM &amp; ADC</h4>
      <p class="text-brand-textSecondary">PSC/ARR maths, PWM duty control, ADC sampling and unit conversion.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 5:</span> USART/UART &amp; DMA</h4>
      <p class="text-brand-textSecondary">Polling, interrupt and DMA serial, idle-line detection, printf retargeting, command parser.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 6:</span> I&sup2;C, SPI &amp; CAN</h4>
      <p class="text-brand-textSecondary">Addressing, clock modes, filters and bit timing with working HAL code.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 7:</span> Sensors, Displays &amp; Motors</h4>
      <p class="text-brand-textSecondary">BME280/MPU6050, SSD1306 OLED, servos, DC and stepper motors.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 8:</span> ST-LINK, SWD, GDB &amp; Fault Diagnosis</h4>
      <p class="text-brand-textSecondary">Debug probe, OpenOCD/GDB, logic analyzer, HardFault decoding.</p>
    </div>
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Chapter 9:</span> Mini Projects &amp; Capstone</h4>
      <p class="text-brand-textSecondary">Five mini projects and a CAN-connected environmental data logger.</p>
    </div>
  </div>

  <div class="mt-10 mb-4 text-center print:hidden">
    <a href="/courses/stm32-firmware-development-with-c/book/" target="_blank" rel="noopener noreferrer" class="bg-brand-coral hover:bg-opacity-90 text-white px-8 py-3.5 rounded-full font-bold shadow-lg inline-flex items-center justify-center text-lg">
      Go to Course <i class="fas fa-arrow-right ml-2"></i>
    </a>
  </div>
</div>
"""

course, created = Course.objects.update_or_create(
    slug="stm32-firmware-development-with-c",
    defaults={
        "title": "STM32 Firmware Development with C",
        "short_description": "Foundation course: Embedded C, STM32 architecture, CubeMX/CubeIDE, GPIO, interrupts, timers, PWM, ADC, UART, I2C, SPI, CAN, DMA, sensors, displays, motors, and hardware debugging with ST-LINK, SWD, GDB and a logic analyzer.",
        "description": html_content,
        "duration_weeks": 8,
        "duration_hours": 45,
        "skill_level": "Beginner to Intermediate",
        "technologies": "C, STM32, CubeMX, CubeIDE, HAL, GPIO, UART, I2C, SPI, CAN, DMA, GDB, ST-LINK",
        "fee": "7000/- + GST",
        "timing": "Weekend course, 4:00pm to 5:30pm",
        "start_date": "To be announced",
        "trainers": "Kamal Kumar Mukiri",
        "recordings_info": "Sessions are recorded and shared and maintained for 3 months.",
        "materials_info": "Full digital book with 9 chapters, code samples, lab checklists, quizzes and a capstone.",
        "prerequisites": "Basic C programming and elementary electronics.",
        "training_mode": "Online",
        "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800",
    },
)

print(("Created" if created else "Updated"), "course:", course.title, "(ID: %s)" % course.id)
