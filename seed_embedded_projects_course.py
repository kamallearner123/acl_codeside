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
  <!-- Top Banner -->
  <div class="bg-brand-navy text-white p-6 rounded-xl shadow-lg border-l-4 border-brand-coral">
    <h3 class="text-xl font-bold mb-3 flex items-center"><i class="fas fa-microchip mr-2 text-brand-coral"></i> Designing & Developing Embedded Systems Projects (Using STM32 / TI / ESP32)</h3>
    <ul class="space-y-2 text-gray-200">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Rigorous Embedded SDLC &amp; V-Model methodology</strong>: from requirements engineering, hardware/software partitioning, and detailed design to HIL verification, production flashing, and field diagnostics.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Multi-vendor mastery</strong>: compare, evaluate, and develop on <strong>STM32</strong> (ARM Cortex-M4/M7), <strong>Texas Instruments</strong> (MSP430 / C2000 / SimpleLink), and <strong>ESP32</strong> (Xtensa &amp; RISC-V).</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Hardware &amp; firmware co-design</strong>: schematic review, power budgeting, clock distribution, decoupling, pin muxing, and board bring-up.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Bare-metal to RTOS</strong>: low-level register and HAL/LL driver development through FreeRTOS preemptive multitasking, queues, semaphores, and memory management.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Industrial &amp; wireless communications</strong>: CAN 2.0B / CAN-FD, RS-485 Modbus, high-speed SPI/I2C with DMA, BLE 5.0, Wi-Fi, and MQTT over TLS.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Production readiness</strong>: ultra-low-power sleep modes, watchdogs, Brown-out Reset (BOR), custom bootloaders, flash partitioning, and fail-safe OTA updates.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Full digital reference manual</strong> with code samples, schematic diagrams, checklists, and 4 production capstone walk-throughs.</li>
    </ul>
  </div>

  <!-- SDLC & V-Model Framework Highlight Card -->
  <div class="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-10 h-10 rounded-xl bg-brand-coral/10 text-brand-coral flex items-center justify-center font-bold text-xl">
        <i class="fas fa-sitemap"></i>
      </div>
      <div>
        <h3 class="text-xl font-bold text-brand-navy">The Embedded Systems V-Model &amp; SDLC Framework</h3>
        <p class="text-xs text-gray-500 font-semibold uppercase tracking-wider">End-to-End Traceability from Requirements to Field Diagnostics</p>
      </div>
    </div>
    <p class="text-sm text-gray-600 mb-4 leading-relaxed">
      Unlike pure web/cloud software where runtime environments are virtualized and bugs can be hotfixed in minutes, embedded systems are governed by physical silicon, power envelopes, and hard real-time deadlines. Every project in this course follows the <strong>Embedded V-Model lifecycle</strong>:
    </p>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <span class="font-bold text-blue-600 block mb-1">Phase 1: Requirements</span>
        Real-time latency deadlines (&lt;10µs), power budgets, memory limits, and ISO 26262/IEC 61508 safety standards.
      </div>
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <span class="font-bold text-purple-600 block mb-1">Phase 2: Architecture &amp; Partitioning</span>
        Hardware/software division, MCU selection matrix (STM32 vs TI vs ESP32), pin muxing, and clock tree design.
      </div>
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <span class="font-bold text-amber-600 block mb-1">Phase 3: Detailed Design</span>
        CMSIS/HAL driver layering, FreeRTOS task priorities, NVIC preemption levels, and linker script memory maps.
      </div>
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <span class="font-bold text-emerald-600 block mb-1">Phase 4: Implementation</span>
        Defensive Embedded C, MISRA-C:2012 guidelines, atomic register writes, circular DMA, and volatile safeguards.
      </div>
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <span class="font-bold text-indigo-600 block mb-1">Phase 5: Multi-Tier Verification</span>
        Host unit tests with Unity/CMock, logic analyzer bus tracing (CAN/SPI/I2C), GDB breakpoints, and HIL testing.
      </div>
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100">
        <span class="font-bold text-teal-600 block mb-1">Phase 6: Production Flashing</span>
        Bed-of-nails SWD programming, Readout Protection (RDP Level 1/2), dual-bank bootloaders, and fail-safe OTA.
      </div>
      <div class="bg-gray-50 p-3 rounded-lg border border-gray-100 col-span-1 md:col-span-2">
        <span class="font-bold text-red-600 block mb-1">Phase 7: Maintenance &amp; Field Diagnostics</span>
        Non-volatile crash dumps, Cortex-M HardFault exception frame decoding (R0-R3, R12, LR, PC, xPSR), and field patches.
      </div>
    </div>
  </div>

  <!-- Hardware Platforms Overview -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <div class="bg-white p-5 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
      <div class="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center mb-3">
        <i class="fas fa-microchip text-lg"></i>
      </div>
      <h4 class="text-lg font-bold text-brand-navy mb-2">STMicroelectronics (STM32)</h4>
      <p class="text-sm text-gray-600 mb-3">Focus on industrial control, motor drives, CAN bus, high-speed DMA, and deterministic real-time processing on Cortex-M4/M7.</p>
      <div class="text-xs font-semibold text-brand-navy bg-gray-50 p-2 rounded">
        <strong>Hardware:</strong> NUCLEO-F446RE / G474RE<br>
        <strong>Tools:</strong> STM32CubeIDE, ST-LINK, GDB
      </div>
    </div>
    <div class="bg-white p-5 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
      <div class="w-10 h-10 rounded-lg bg-red-50 text-red-600 flex items-center justify-center mb-3">
        <i class="fas fa-battery-three-quarters text-lg"></i>
      </div>
      <h4 class="text-lg font-bold text-brand-navy mb-2">Texas Instruments (TI)</h4>
      <p class="text-sm text-gray-600 mb-3">Focus on ultra-low-power energy harvesting, sub-microamp standby, precision sensing, and robust industrial automation.</p>
      <div class="text-xs font-semibold text-brand-navy bg-gray-50 p-2 rounded">
        <strong>Hardware:</strong> MSP430FR5994 / C2000 LaunchPad<br>
        <strong>Tools:</strong> Code Composer Studio (CCS), XDS110
      </div>
    </div>
    <div class="bg-white p-5 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
      <div class="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center mb-3">
        <i class="fas fa-wifi text-lg"></i>
      </div>
      <h4 class="text-lg font-bold text-brand-navy mb-2">Espressif (ESP32)</h4>
      <p class="text-sm text-gray-600 mb-3">Focus on dual-core multitasking, Wi-Fi 802.11 b/g/n, BLE 5.0, MQTT IoT telemetry, cryptographic accelerators, and secure OTA.</p>
      <div class="text-xs font-semibold text-brand-navy bg-gray-50 p-2 rounded">
        <strong>Hardware:</strong> ESP32-S3 / ESP32-C3 DevKit<br>
        <strong>Tools:</strong> ESP-IDF, ESP-Prog, FreeRTOS
      </div>
    </div>
  </div>

  <!-- Syllabus -->
  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">Course Syllabus &amp; Modules</h3>
    
    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 1 (SDLC Phase 1):</span> Requirements Engineering &amp; Embedded Feasibility</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Translating user requirements into formal PRDs and SRS. Quantifying hard real-time latency deadlines, power budgets, Flash/SRAM capacity, environmental specs, and functional safety standards (ISO 26262, IEC 61508).</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 2 (SDLC Phase 2):</span> Architecture Co-Design &amp; MCU Selection Matrix</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Hardware vs. Software partitioning (silicon accelerators vs. CPU algorithms). Trade-off analysis between STM32 (deterministic real-time), TI MSP430 (ultra-low power FRAM), and ESP32 (dual-core wireless IoT). Pin allocation and bus topology planning.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 3 (SDLC Phase 2):</span> Hardware Schematics, Power Delivery &amp; Component Selection</h4>
      <p class="text-brand-textSecondary text-sm mb-1">LDO vs. Buck switching regulators, decoupling capacitor networks and ESR, crystal oscillator layout, ESD protection on external lines, reset circuitry, and JTAG/SWD headers.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 4 (SDLC Phase 3):</span> Detailed Firmware Architecture &amp; Driver Layering</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Layered embedded architecture: CMSIS, HAL/LL, Board Support Package (BSP), and Application layers. State machine modeling, memory map layout, vector table relocation (VTOR), and linker script configuration.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 5 (SDLC Phase 3):</span> Toolchains, Build Systems &amp; Hardware Debuggers</h4>
      <p class="text-brand-textSecondary text-sm mb-1">GCC arm-none-eabi, Clang, TI CCS toolchain, and ESP-IDF CMake build systems. Working with hardware debuggers: ST-LINK V2/V3, TI XDS110, ESP-Prog, and Segger J-Link over SWD/JTAG.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 6 (SDLC Phase 4):</span> Bare-Metal Driver Engineering: Clocks, GPIO &amp; NVIC</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Defensive Embedded C: MISRA-C:2012 guidelines, fixed-width types, and `volatile` semantics. Clock tree PLLs, atomic BSRR GPIO toggles, nested interrupt priorities, race condition avoidance, and switch debouncing.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 7 (SDLC Phase 4):</span> High-Throughput Serial Communication &amp; Circular DMA</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Zero-copy UART reception using circular DMA and idle-line interrupt detection. High-speed SPI master/slave transactions with DMA. I2C bus arbitration, clock stretching, and recovery routines.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 8 (SDLC Phase 4):</span> Industrial Fieldbus Networks: CAN 2.0B, CAN-FD &amp; RS-485 Modbus</h4>
      <p class="text-brand-textSecondary text-sm mb-1">CAN physical transceiver wiring, termination resistors, bit timing calculation, mailbox filters, frame IDs, and payload parsing. RS-485 differential signaling and Modbus RTU implementation.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 9 (SDLC Phase 4):</span> Analog Signal Acquisition, ADC Filtering &amp; Sensor Calibration</h4>
      <p class="text-brand-textSecondary text-sm mb-1">SAR ADC vs Sigma-Delta, sampling rate vs impedance matching, DMA scanning, moving-average and FIR digital filtering, op-amp signal conditioning, and sensor calibration curves.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 10 (SDLC Phase 4):</span> Real-Time Operating Systems (FreeRTOS) Architecture</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Preemptive vs cooperative scheduling, task priority assignment, task stacks, queues, binary/counting semaphores, mutexes with priority inheritance, event groups, and ISR-safe IPC.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 11 (SDLC Phase 4):</span> Ultra-Low-Power Firmware &amp; Energy Budgeting</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Sleep, Stop, Standby and Deep Sleep modes. RTC periodic wakeups, pin interrupt wakeup, peripheral power gating. Current profiling using Nordic PPK2 / multimeters. Energy harvesting with TI MSP430 FRAM.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 12 (SDLC Phase 4):</span> Wireless IoT Networking with ESP32 (Wi-Fi, BLE 5.0 &amp; MQTT TLS)</h4>
      <p class="text-brand-textSecondary text-sm mb-1">ESP32 dual-core Wi-Fi station and soft-AP configuration. BLE GATT server and client architecture, custom services, characteristics, notifications, and MQTT over TLS (mTLS) cloud integration.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 13 (SDLC Phase 5):</span> Multi-Tier Verification, Logic Analyzers &amp; HIL Testing</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Unit testing with Unity/CMock on host PCs. Hardware-in-the-Loop (HIL) testing with simulated sensor inputs. Protocol bus decoding with USB logic analyzers (PulseView/Saleae). Automated test runner integration.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 14 (SDLC Phase 6):</span> Bootloaders, Production Flashing &amp; Secure OTA</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Flash partitioning, vector table relocation, CRC32 verification, automated bed-of-nails flashing scripts, Readout Protection (RDP Level 1/2), and fail-safe dual-bank OTA updates with rollback protection.</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 15 (SDLC Phase 7):</span> Reliability, Watchdogs &amp; In-Field Crash Diagnostics</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Independent and window watchdogs (IWDG / WWDG), brown-out reset (BOR), stack overflow detection in RTOS. Non-volatile crash dump logging and Cortex-M HardFault exception frame decoding (R0-R3, R12, LR, PC, xPSR).</p>
    </div>

    <div class="mb-6 pl-4 border-l-2 border-brand-coral/40">
      <h4 class="text-lg font-bold text-brand-navy mb-1"><span class="text-brand-coral">Module 16 (SDLC Integration):</span> Heterogeneous Multi-MCU System Bring-Up</h4>
      <p class="text-brand-textSecondary text-sm mb-1">Architecting distributed embedded products: high-speed SPI/UART/CAN bridge between a deterministic STM32 real-time motor/sensor controller and an ESP32 wireless edge gateway, tested through the complete V-Model lifecycle.</p>
    </div>
  </div>

  <!-- Capstone Projects Showcase -->
  <div class="bg-gray-50 p-6 rounded-2xl border border-gray-200">
    <h3 class="text-2xl font-bold text-brand-navy mb-4"><i class="fas fa-trophy mr-2 text-brand-coral"></i> 4 Real-World Production Capstones</h3>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="bg-white p-4 rounded-xl border border-gray-100 shadow-sm">
        <span class="text-xs font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full uppercase">Project 1 • STM32</span>
        <h4 class="font-bold text-brand-navy mt-2 mb-1">Industrial CAN-FD Sensor Node &amp; PID Motor Actuator</h4>
        <p class="text-xs text-gray-600">STM32F4/G4 closed-loop speed control, quadrature encoder reading via hardware timers, high-frequency ADC current sensing, and fault-protected CAN-FD telemetry.</p>
      </div>
      <div class="bg-white p-4 rounded-xl border border-gray-100 shadow-sm">
        <span class="text-xs font-bold text-red-600 bg-red-50 px-2.5 py-1 rounded-full uppercase">Project 2 • TI MSP430</span>
        <h4 class="font-bold text-brand-navy mt-2 mb-1">Ultra-Low Power Solar-Harvested Environmental Beacon</h4>
        <p class="text-xs text-gray-600">TI MSP430FR with FRAM non-volatile memory, sub-microamp LPM3.5 sleep cycling, BME280 sensor data logging, and low-power wireless telemetry lasting years on a supercapacitor.</p>
      </div>
      <div class="bg-white p-4 rounded-xl border border-gray-100 shadow-sm">
        <span class="text-xs font-bold text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded-full uppercase">Project 3 • ESP32</span>
        <h4 class="font-bold text-brand-navy mt-2 mb-1">Secure Edge IoT Gateway with BLE Mesh &amp; Cloud TLS</h4>
        <p class="text-xs text-gray-600">Dual-core FreeRTOS architecture on ESP32-S3: Core 0 ingests BLE sensor beacons; Core 1 runs an embedded web dashboard and pushes telemetry via MQTT over TLS with OTA.</p>
      </div>
      <div class="bg-white p-4 rounded-xl border border-gray-100 shadow-sm">
        <span class="text-xs font-bold text-purple-600 bg-purple-50 px-2.5 py-1 rounded-full uppercase">Project 4 • Heterogeneous</span>
        <h4 class="font-bold text-brand-navy mt-2 mb-1">Integrated Dual-MCU Smart Industrial Controller</h4>
        <p class="text-xs text-gray-600">An STM32 real-time motor controller linked via SPI/DMA packet protocol to an ESP32 wireless communications module, providing remote diagnostics, emergency stops, and cloud analytics.</p>
      </div>
    </div>
  </div>

  <div class="mt-8 mb-4 text-center print:hidden">
    <a href="/courses/designing-embedded-systems-projects-stm32-ti-esp32/book/" target="_blank" rel="noopener noreferrer" class="bg-brand-coral hover:bg-opacity-90 text-white px-8 py-3.5 rounded-full font-bold shadow-lg inline-flex items-center justify-center text-lg transition-transform hover:-translate-y-0.5">
      <i class="fas fa-book-open mr-2"></i> Go to Digital Course Handbook <i class="fas fa-arrow-right ml-2 text-xs"></i>
    </a>
  </div>
</div>
"""

course, created = Course.objects.update_or_create(
    slug="designing-embedded-systems-projects-stm32-ti-esp32",
    defaults={
        "title": "Designing & Developing Embedded Systems Projects (Using STM32 / TI / ESP32)",
        "short_description": "End-to-end embedded project engineering following the rigorous V-Model SDLC: requirements analysis, hardware/software co-design, bare-metal HAL/LL drivers, FreeRTOS multitasking, CAN bus, wireless IoT, power optimization, HIL testing, and production deployment across STM32, Texas Instruments, and ESP32.",
        "description": html_content,
        "duration_weeks": 12,
        "duration_hours": 60,
        "skill_level": "Intermediate to Advanced",
        "technologies": "Embedded SDLC & V-Model, STM32 (Cortex-M4/M7), TI MSP430/C2000, ESP32-S3/C3, FreeRTOS, KiCad, CAN Bus, BLE 5.0, Wi-Fi, MQTT TLS, ST-LINK, XDS110, HIL Testing",
        "fee": "12000/- + GST",
        "timing": "Weekend course, 4:00pm to 6:30pm",
        "start_date": "To be announced",
        "trainers": "Kamal Kumar Mukiri",
        "recordings_info": "Sessions are recorded, shared, and accessible for 3 months.",
        "materials_info": "Full digital handbook with 16 modules, schematics, source code repositories, lab checklists, SDLC templates, and 4 production capstone projects.",
        "prerequisites": "Proficiency in C programming, basic microcontroller architecture and fundamental circuit electronics.",
        "training_mode": "Online (Live Hands-on Project Labs)",
        "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800",
    },
)

print(("Created" if created else "Updated"), "course:", course.title, "(ID: %s)" % course.id)
