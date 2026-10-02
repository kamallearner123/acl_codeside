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
    <h3 class="text-xl font-bold mb-3 flex items-center"><i class="fas fa-microchip mr-2 text-brand-coral"></i> What to Expect from this Course</h3>
    <ul class="space-y-2 text-gray-200">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>6 comprehensive modules (36 hands-on sessions)</strong> taking you from bare-metal mindset to Zephyr RTOS mastery.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Hardware-focused curriculum</strong> specifically targeting STMicroelectronics Nucleo boards (Nucleo-F401RE, Nucleo-G071RB, and Nucleo-L476RG).</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Deep dive into DeviceTree &amp; Kconfig</strong>: Master the industry-standard hardware description model used in Linux and Zephyr.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Real-time kernel &amp; peripheral drivers</strong>: Preemptive multithreading, message queues, async DMA UART, I2C sensor APIs, and hardware PWM.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Production engineering &amp; Capstone</strong>: Build an Industrial Multi-Sensor Gateway with MCUBoot dual-slot DFU, Twister automated tests, and Watchdog supervisor.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Full digital book access</strong>: Interactive web guide with DeviceTree visualizer, board switcher, and runnable sample projects.</li>
    </ul>
  </div>

  <div class="bg-gray-50 p-6 rounded-xl border border-gray-100 shadow-sm">
    <h3 class="text-xl font-bold text-brand-navy mb-4">Course Philosophy</h3>
    <p class="mb-2 text-brand-textSecondary text-lg"><strong>Traditional:</strong> Memorize registers &rarr; Write monolithic bare-metal code &rarr; Struggle with portability.</p>
    <p class="mb-4 text-brand-textSecondary text-lg"><strong>Problem-driven Modern RTOS:</strong> Real-time hardware requirement &rarr; DeviceTree abstraction &rarr; Zephyr Driver API &rarr; Production-grade portable firmware.</p>
    <p class="text-brand-textSecondary text-lg">The course treats Zephyr RTOS as an enterprise-grade embedded software platform. Rather than writing fragile register manipulation code, learners master DeviceTree overlays, asynchronous kernel services, and production toolchains like <code>west</code> and <code>twister</code>.</p>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">The Learning Journey (6 Modules &amp; Capstone)</h3>

    <!-- Module 1 -->
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Module 1:</span> Foundations &amp; Toolchain Setup</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Embedded projects outgrow bare-metal code and FreeRTOS lack standardized driver models, leading to vendor lock-in. How does Zephyr solve this?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Zephyr Architecture vs FreeRTOS &bull; The <code>west</code> meta-tool &bull; Zephyr SDK &amp; ARM GCC toolchain &bull; STM32 ST-Link &amp; Virtual COM Port (VCP) setup &bull; CMake &amp; Ninja build flow.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Set up a complete Zephyr workspace and build/flash your first heartbeat Blinky with VCP serial printk telemetry on STM32 Nucleo.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> A clean, reproducible Zephyr development environment ready for production embedded workflows.</p>
    </div>

    <!-- Module 2 -->
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Module 2:</span> DeviceTree &amp; Kconfig Deep Dive</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Hardcoding peripheral memory addresses and pin numbers creates non-portable, buggy code that breaks when switching microcontroller revisions.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Open Firmware standard &bull; DTS node hierarchy &bull; Bindings (YAML) &bull; Phandles &amp; label references &bull; Board overlays (<code>.overlay</code>) &bull; Kconfig symbols &bull; <code>prj.conf</code> customization.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Write a custom board overlay mapping external LEDs and pushbuttons; inspect preprocessed DTS output and compile conditional Kconfig drivers.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Fluent understanding of Zephyr's hardware abstraction layer and zero-cost compile-time configuration.</p>
    </div>

    <!-- Module 3 -->
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Module 3:</span> Zephyr Kernel Fundamentals</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Complex embedded devices need to run multiple tasks simultaneously without race conditions, stack overflows, or priority inversion.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Preemptive vs cooperative thread scheduling &bull; Dynamic vs static thread creation &bull; Stack size tuning &bull; Mutexes, Semaphores &amp; condition variables &bull; Thread-safe Message Queues (<code>k_msgq</code>) &bull; Workqueues &amp; Timers.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build a multi-threaded telemetry pipeline where producer threads collect sensor readings and a consumer thread batches data safely.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Mastery of deterministic real-time multitasking on ARM Cortex-M.</p>
    </div>

    <!-- Module 4 -->
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Module 4:</span> STM32 Peripherals &amp; Drivers</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Interfacing with real-world sensors and actuators requires clean, non-blocking peripheral drivers that do not stall the CPU.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Zephyr Unified Device Driver Model &bull; Pin interrupts &amp; debouncing &bull; Asynchronous DMA-driven UART &bull; I2C Sensor API (HTS221 / BME280) &bull; Hardware PWM for motor/LED dimming &bull; ADC sampling &bull; Power management (Low Power Suspend/Sleep).</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Implement an environmental weather station streaming temperature, humidity, and battery voltage over async UART with dynamic PWM indicator.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Production mastery over STM32 hardware peripherals using Zephyr's standard driver interfaces.</p>
    </div>

    <!-- Module 5 -->
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Module 5:</span> Subsystems, Shell &amp; Industrial Protocols</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Production firmware requires interactive on-device diagnostics, configurable logging levels, and robust industrial connectivity without reinventing protocols.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Zephyr Interactive Shell (command trees, autocompletion, parameters) &bull; High-performance deferred logging subsystem &bull; CAN bus communication (bxCAN / FDCAN) &bull; BLE peripheral telemetry &bull; Storage subsystems (NVS / Flash circular log).</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Construct a diagnostic shell with custom CLI commands to query live peripheral state, trigger calibration cycles, and monitor CAN messages.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build field-maintainable firmware with professional telemetry and industrial connectivity.</p>
    </div>

    <!-- Module 6 -->
    <div class="mb-8 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Module 6:</span> Production Engineering &amp; Capstone</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Shipping commercial IoT and automotive devices requires fail-safe bootloaders, automated CI/CD testing, and hardware watchdog protection against hangs.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> MCUBoot dual-slot flash partitioning &bull; Secure Device Firmware Upgrade (DFU) &bull; Automated unit &amp; integration testing with Twister &amp; Ztest &bull; Hardware Watchdog (WDT) supervisors &bull; OpenOCD &amp; GDB remote debugging.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Integrate MCUBoot into an STM32 project, verify safe rollback on firmware faults, and configure an automated Twister test matrix.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete certification-ready embedded firmware engineering practices.</p>
    </div>
  </div>

  <div class="bg-brand-navy text-white p-8 rounded-2xl shadow-xl mt-12 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative overflow-hidden">
    <div class="absolute inset-0 bg-brand-navy/90"></div>
    <div class="relative z-10">
      <h3 class="text-3xl font-bold mb-4 text-white flex items-center"><i class="fas fa-rocket mr-3 text-brand-coral"></i> Capstone Project: Industrial Multi-Sensor Gateway</h3>
      <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
        <p class="text-lg"><strong class="text-brand-coral uppercase tracking-wider text-sm block mb-1">Objective</strong> Architect and deploy a robust, production-grade Industrial Telemetry &amp; Safety Gateway on STM32 Nucleo with real-time sensor aggregation, interactive diagnostic shell, MCUBoot dual-slot DFU, and watchdog fail-safe recovery.</p>
      </div>
      <ul class="list-none mb-8 space-y-3">
        <li class="flex items-center text-gray-200"><i class="fas fa-check-circle text-brand-coral mr-3"></i> Multi-threaded kernel pipeline with bounded message queues and sensor priority tiers.</li>
        <li class="flex items-center text-gray-200"><i class="fas fa-check-circle text-brand-coral mr-3"></i> DeviceTree overlays supporting multiple Nucleo boards (Nucleo-F401RE, Nucleo-G071RB, Nucleo-L476RG).</li>
        <li class="flex items-center text-gray-200"><i class="fas fa-check-circle text-brand-coral mr-3"></i> Dual-slot MCUBoot partition table for fail-safe field firmware upgrades.</li>
        <li class="flex items-center text-gray-200"><i class="fas fa-check-circle text-brand-coral mr-3"></i> Automated Twister test suite running over Ztest mock harnesses.</li>
      </ul>
      <h4 class="text-xl font-bold mb-4 text-white border-b border-gray-600 pb-2">Core Deliverables</h4>
      <ul class="list-none space-y-3">
        <li class="flex items-center text-gray-200"><i class="fas fa-code-branch text-brand-coral mr-3"></i> Clean, Git-tracked West workspace repository with sample drivers and board overlays.</li>
        <li class="flex items-center text-gray-200"><i class="fas fa-book text-brand-coral mr-3"></i> Complete interactive digital book with instant search, DeviceTree inspector, and knowledge quizzes.</li>
      </ul>
    </div>
  </div>

  <div class="mt-10 mb-4 text-center flex flex-col sm:flex-row justify-center gap-4 print:hidden">
    <a href="/courses/exploring-zephyr-using-stm32/book/" target="_blank" rel="noopener noreferrer" class="bg-brand-coral hover:bg-opacity-90 text-white px-8 py-3.5 rounded-full font-bold shadow-lg transition-transform hover:-translate-y-0.5 inline-flex items-center justify-center text-lg group">
      Go to Course <i class="fas fa-arrow-right ml-2 transform group-hover:translate-x-1 transition-transform"></i>
    </a>
    <button type="button" onclick="window.print()" class="bg-gray-800 hover:bg-gray-700 text-white px-8 py-3.5 rounded-full font-bold shadow-lg transition-transform hover:-translate-y-0.5 inline-flex items-center justify-center">
      <i class="fas fa-file-pdf mr-2"></i> Download as PDF
    </button>
  </div>

  <div class="mt-8 text-center sm:flex-row justify-center gap-4">
    <a href="/courses/" class="inline-block px-8 py-3 bg-gray-100 text-brand-navy font-bold rounded-full hover:bg-gray-200 shadow">Browse more courses</a>
  </div>
</div>
"""

course, created = Course.objects.update_or_create(
    slug="exploring-zephyr-using-stm32",
    defaults={
        "title": "Exploring Zephyr using STM32",
        "short_description": "Comprehensive hands-on digital guide and engineering program covering Zephyr RTOS, DeviceTree, Kconfig, multithreading, STM32 peripherals, shell subsystems, and production MCUBoot.",
        "description": html_content,
        "duration_weeks": 6,
        "duration_hours": 40,
        "skill_level": "Intermediate to Advanced",
        "technologies": "Zephyr RTOS, STM32, DeviceTree, Kconfig, ARM Cortex-M, C, Embedded Systems, West, MCUBoot",
        "fee": "7000/- + GST",
        "timing": "Weekend course, 4:00pm to 5:30pm",
        "start_date": "10th Oct 2026 - 15th Nov 2026",
        "trainers": "Kamal Kumar Mukiri",
        "recordings_info": "Sessions are recorded and shared and maintained for 3 months.",
        "materials_info": "Full interactive digital book with 6 modules, source code samples, Devicetree overlays, quizzes, and project blueprints.",
        "prerequisites": "C programming fundamentals and basic microcontroller concepts.",
        "training_mode": "Online",
        "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800"
    }
)

if created:
    print(f"Created new course: {course.title} (ID: {course.id})")
else:
    print(f"Updated course: {course.title} (ID: {course.id})")
