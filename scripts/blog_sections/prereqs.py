"""
Prerequisites Section for STM32 Arm Cortex-M DIY Blog.
"""

def get_prerequisites_section():
    return r"""
<p class="lead" style="font-size: 1.25rem; font-weight: 500; color: #1E3557; line-height: 1.8;">
    Mastering embedded systems development requires transcending the realm of hobbyist breadboard sketches and stepping into deterministic, register-aware, bare-metal and HAL-based firmware engineering. The STMicroelectronics STM32 family—powered by the 32-bit ARM Cortex-M architecture (ranging from ultra-low-power Cortex-M0+ to high-performance Cortex-M4F and Cortex-M7 cores)—is the undisputed workhorse of modern automotive powertrains, industrial automation, medical robotics, flight telemetry, and edge IoT devices.
</p>

<p style="font-size: 1.1rem; line-height: 1.8; color: #334155;">
    This 50-assignment engineering roadmap is built for practicing developers, university students, and systems engineers who want to build verified, production-grade embedded products. Rather than spoon-feeding unoptimized copy-paste snippets, each assignment establishes clear <strong>functional expectations, observable electrical outputs, rigorous non-blocking architectural rules, deterministic test cases, and diagnostic suggestions</strong> to debug real hardware in the lab.
</p>

<div class="toc-box mt-8 mb-10 p-6 bg-brand-light border border-gray-100 rounded-2xl shadow-sm">
    <p class="font-bold text-brand-navy mb-3 text-lg"><i class="fas fa-list-ol mr-2 text-brand-coral"></i> Complete Roadmap Navigation</p>
    <ol class="space-y-1.5" style="margin:0; padding-left: 1.5rem; line-height: 1.9; font-size: 0.95rem; list-style-type: decimal;">
        <li><a href="#prerequisites" class="hover:text-brand-coral transition-colors font-medium">1. Foundational Prerequisites: C, Architecture, Circuits, &amp; Toolchain</a></li>
        <li><a href="#hardware-kit" class="hover:text-brand-coral transition-colors font-medium">2. Recommended Hardware Kits &amp; Bill of Materials</a></li>
        <li><a href="#mandatory-rules" class="hover:text-brand-coral transition-colors font-medium">3. Mandatory Engineering Rules &amp; Non-Blocking Architectural Contract</a></li>
        <li><a href="#level-1" class="hover:text-brand-coral transition-colors font-medium">4. Level 1 — GPIO, Timers &amp; Basic Control (Assignments 1–10 · Beginner)</a></li>
        <li><a href="#level-2" class="hover:text-brand-coral transition-colors font-medium">5. Level 2 — Sensors &amp; Analog Interfaces (Assignments 11–20 · Beginner to Intermediate)</a></li>
        <li><a href="#level-3" class="hover:text-brand-coral transition-colors font-medium">6. Level 3 — Displays, Distance &amp; Motion (Assignments 21–30 · Intermediate)</a></li>
        <li><a href="#level-4" class="hover:text-brand-coral transition-colors font-medium">7. Level 4 — Communication Protocols (Assignments 31–40 · Intermediate to Advanced)</a></li>
        <li><a href="#level-5" class="hover:text-brand-coral transition-colors font-medium">8. Level 5 — Integrated Embedded Systems (Assignments 41–50 · Advanced)</a></li>
        <li><a href="#verification-portfolio" class="hover:text-brand-coral transition-colors font-medium">9. Verification Methodology &amp; Building an Industry-Grade Portfolio</a></li>
    </ol>
</div>

---

<h2 id="prerequisites" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-graduation-cap text-brand-coral mr-3"></i> 1. Foundational Prerequisites: What You Must Master Initially
</h2>

<p>
    Before configuring peripherals in STM32CubeMX or pressing the flash button in STM32CubeIDE, an embedded engineer must master four interconnected foundational pillars: <strong>Embedded C Programming</strong>, <strong>ARM Cortex-M Architecture</strong>, <strong>Circuit Theory &amp; Interfacing</strong>, and <strong>The STM32 Toolchain</strong>. Neglecting any of these domains leads to mysterious HardFault crashes, blown microcontroller pins, or firmware that locks up under real-time interrupts.
</p>

<div class="space-y-8 my-8">
    <div class="p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
        <h3 class="text-xl font-bold text-brand-navy mb-3 flex items-center">
            <i class="fas fa-code text-blue-600 mr-2"></i> 1.1 Embedded C Programming
        </h3>
        <p class="text-sm leading-relaxed text-slate-700 mb-3">
            Embedded C differs fundamentally from desktop C. It operates directly against memory-mapped hardware registers, strict cache coherency rules, and volatile asynchronous states.
        </p>
        <ul class="text-sm space-y-2 list-disc pl-5 text-slate-700">
            <li><strong>Memory Segments:</strong> Flash (ROM) stores non-volatile code (<code>.text</code>) and constants (<code>.rodata</code>). SRAM stores initialized variables (<code>.data</code>, copied from Flash on reset), zero-initialized data (<code>.bss</code>), the dynamic heap, and the downward-growing execution stack.</li>
            <li><strong>Bitwise Manipulation:</strong> Hardware registers are bitfields. You must be fluent in setting bits (<code>REG |= (1U &lt;&lt; 5);</code>), clearing bits (<code>REG &amp;= ~(1U &lt;&lt; 5);</code>), toggling bits (<code>REG ^= (1U &lt;&lt; 5);</code>), and testing bits (<code>if (REG &amp; (1U &lt;&lt; 5))</code>).</li>
            <li><strong>The Crucial <code>volatile</code> Keyword:</strong> Instructs the compiler optimizer that a variable’s value can mutate asynchronously outside the current instruction stream (e.g., peripheral status registers, DMA buffers, or flags modified inside an Interrupt Service Routine). Omitting <code>volatile</code> causes the compiler to cache variable reads in CPU core registers, creating infinite polling loops.</li>
            <li><strong>CMSIS Register Pointers:</strong> Pointer arithmetic casting arbitrary physical base addresses into structured register maps:
                <pre class="bg-slate-900 text-slate-100 p-2 rounded text-xs font-mono my-1"><code>#define GPIOA_BASE (0x40020000UL)
#define GPIOA      ((GPIO_TypeDef *) GPIOA_BASE)
GPIOA-&gt;ODR |= (1U &lt;&lt; 5); /* Direct register manipulation */</code></pre>
            </li>
            <li><strong>Re-entrancy &amp; Critical Sections:</strong> Writing thread-safe functions that can be interrupted safely without corrupting shared global state.</li>
        </ul>
    </div>

    <div class="p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
        <h3 class="text-xl font-bold text-brand-navy mb-3 flex items-center">
            <i class="fas fa-microchip text-indigo-600 mr-2"></i> 1.2 ARM Cortex-M Architecture
        </h3>
        <p class="text-sm leading-relaxed text-slate-700 mb-3">
            The ARM Cortex-M processor core defines the execution pipeline, exception model, and bus matrix governing every STM32 microcontroller.
        </p>
        <ul class="text-sm space-y-2 list-disc pl-5 text-slate-700">
            <li><strong>Register Bank:</strong> Thirteen 32-bit general-purpose registers (R0–R12), SP (R13 - banked as Main Stack Pointer <code>MSP</code> and Process Stack Pointer <code>PSP</code>), Link Register (R14 - <code>LR</code> containing return addresses and <code>EXC_RETURN</code> exception payload), Program Counter (R15 - <code>PC</code>), and Status Registers (xPSR).</li>
            <li><strong>Modes &amp; Privilege:</strong> Two execution modes—<strong>Thread Mode</strong> (for normal foreground execution and RTOS tasks) and <strong>Handler Mode</strong> (entered automatically upon exception/interrupt servicing). Cortex-M supports Privileged and Unprivileged states for memory protection.</li>
            <li><strong>Bus Topology (AHB / APB):</strong> Microcontrollers route transactions across the Advanced High-performance Bus (AHB) for SRAM, Flash, and DMA, and Advanced Peripheral Buses (APB1 for low-speed peripherals up to 45 MHz, APB2 for high-speed peripherals up to 90/180 MHz). Understanding bus clock dividers is critical for calculating timer prescalers.</li>
            <li><strong>Nested Vectored Interrupt Controller (NVIC):</strong> Hardware-driven interrupt controller providing ultra-low-latency interrupt dispatching (12 cycles on Cortex-M4), programmable preemption priorities and sub-priorities, and automatic hardware stacking/unstacking of R0–R3, R12, LR, PC, and xPSR.</li>
            <li><strong>SysTick Timer:</strong> A 24-bit decrementing system timer integrated directly into the Cortex-M core, generating standard 1 millisecond interrupts to drive the STM32 HAL tick base (<code>HAL_GetTick()</code>) and RTOS schedulers.</li>
        </ul>
    </div>

    <div class="p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
        <h3 class="text-xl font-bold text-brand-navy mb-3 flex items-center">
            <i class="fas fa-bolt text-amber-500 mr-2"></i> 1.3 Circuit &amp; Hardware Interfacing
        </h3>
        <p class="text-sm leading-relaxed text-slate-700 mb-3">
            Embedded software is inseparable from physical electronics. A solid grasp of analog and digital circuit behavior prevents hardware damage and intermittent bugs.
        </p>
        <ul class="text-sm space-y-2 list-disc pl-5 text-slate-700">
            <li><strong>Ohm’s Law &amp; Current Budgets:</strong> STM32 GPIO pins can source or sink a maximum of 20–25 mA per pin, with total chip current capped at ~120–150 mA. When driving an external LED with forward voltage drop <code>V<sub>F</sub> = 2.0V</code> at <code>I<sub>F</sub> = 10 mA</code> from a 3.3V rail:
                <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                    R = (V<sub>CC</sub> - V<sub>F</sub>) / I<sub>F</sub> = (3.3V - 2.0V) / 0.010A = 130 &Omega;
                </div>
                Always use a standard 220 &Omega; or 330 &Omega; resistor to ensure safe pin dissipation.
            </li>
            <li><strong>Pin Output Topologies:</strong> <strong>Push-Pull</strong> actively drives pins HIGH (VDD) and LOW (GND). <strong>Open-Drain</strong> drives only LOW (GND) and requires an external pull-up resistor to pull HIGH—essential for multi-master buses like I2C and shared 1-Wire buses.</li>
            <li><strong>Switch Contact Bouncing:</strong> Mechanical switch plates physically vibrate against each other for 5 to 25 milliseconds upon closure. Without hardware low-pass filtering (10 k&Omega; + 100nF capacitor) or non-blocking software state debounce filters, a single press registers as dozens of spurious transitions.</li>
            <li><strong>Inductive Kickback Protection:</strong> Inductive devices (relays, DC motors, solenoids, electromagnetic buzzers) release stored magnetic energy when power is interrupted, generating catastrophic negative voltage transients:
                <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                    V = -L &times; (di / dt)
                </div>
                Never drive inductive loads directly from GPIO. Always employ an NPN BJT (e.g. 2N2222) or N-channel MOSFET (e.g. 2N7000, IRLZ44N) and place an antiparallel <strong>flyback diode</strong> (1N4148 or 1N4007) across the coil.
            </li>
            <li><strong>3.3V vs 5V Tolerance:</strong> STM32 runs on 3.3V logic. While many digital input pins are marked "FT" (Five-Volt Tolerant) in ST datasheets, <strong>analog ADC pins are strictly non-5V-tolerant (maximum 3.6V)</strong>. Applying 5V to an analog channel will destroy the internal ADC multiplexer.</li>
        </ul>
    </div>

    <div class="p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow">
        <h3 class="text-xl font-bold text-brand-navy mb-3 flex items-center">
            <i class="fas fa-desktop text-emerald-600 mr-2"></i> 1.4 STM32CubeMX &amp; STM32CubeIDE
        </h3>
        <p class="text-sm leading-relaxed text-slate-700 mb-3">
            STMicroelectronics provides an integrated graphical configurator and Eclipse-based IDE that standardizes embedded C development.
        </p>
        <ul class="text-sm space-y-2 list-disc pl-5 text-slate-700">
            <li><strong>Clock Tree Synthesis:</strong> Configuring the clock distribution network in CubeMX. Routing High-Speed External (HSE) crystal oscillators (e.g., 8 MHz) or High-Speed Internal (HSI) RC oscillators through PLL multipliers and dividers to achieve maximum core clock speed (e.g., 180 MHz on STM32F446RE, 72 MHz on STM32F103).</li>
            <li><strong>Alternate Function (AF) Multiplexing:</strong> Mapping on-chip peripherals (TIM, USART, SPI, I2C, CAN) to designated physical package pins without electrical routing conflicts.</li>
            <li><strong>User Code Sandboxing:</strong> CubeMX generates modular C source files (<code>main.c</code>, <code>stm32f4xx_it.c</code>). To prevent code from being wiped out upon regeneration, write firmware <strong>strictly between the designated comment pairs</strong>:
                <pre class="bg-slate-900 text-slate-100 p-2 rounded text-xs font-mono my-1"><code>/* USER CODE BEGIN 2 */
/* Custom initialization logic here */
/* USER CODE END 2 */</code></pre>
            </li>
            <li><strong>HAL vs LL Libraries:</strong> The Hardware Abstraction Layer (HAL) provides high-level portability across the STM32 portfolio, while Low-Layer (LL) APIs offer zero-overhead inline register access for time-critical loops.</li>
            <li><strong>In-Circuit Debugging:</strong> Leveraging the ST-LINK on-board debugger via Serial Wire Debug (SWD: SWDIO, SWCLK). Setting hardware execution breakpoints, configuring Live Watch variables without stopping the CPU, and utilizing the Fault Analyzer to diagnose <code>HardFault</code>, <code>BusFault</code>, and <code>UsageFault</code> stack frames.</li>
        </ul>
    </div>
</div>
"""
