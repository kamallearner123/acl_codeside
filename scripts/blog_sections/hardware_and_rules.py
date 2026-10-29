"""
Hardware Kits and Mandatory Rules Section for STM32 Arm Cortex-M DIY Blog.
"""

def get_hardware_and_rules_section():
    return r"""
<h2 id="hardware-kit" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-boxes text-brand-coral mr-3"></i> 2. The Recommended Hardware Lab Kit &amp; Bill of Materials
</h2>

<p>
    To execute all 50 assignments in this curriculum, you need a workbench equipped with the following five modular component kits. Every component has been selected for wide commercial availability, industrial relevance, and strict electrical compatibility with 3.3V STM32 microcontrollers.
</p>

<div class="space-y-8 my-8">
    <!-- Kit 1 -->
    <div class="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row gap-6 items-center">
            <div class="w-full md:w-1/3">
                <img src="/static/blogs/stm32-diy/nucleo-f446re-pinout.svg" alt="STM32 NUCLEO-F446RE Pinout and Architecture Diagram" class="rounded-lg shadow border border-gray-200 w-full object-cover">
            </div>
            <div class="w-full md:w-2/3">
                <span class="inline-block px-3 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full mb-2">Core Platform</span>
                <h3 class="text-2xl font-bold text-brand-navy mb-2">1. STM32 Development Board</h3>
                <p class="text-slate-700 text-sm mb-3">
                    The recommended target is the <strong>NUCLEO-F446RE</strong> (or <strong>NUCLEO-L476RG</strong> for ultra-low-power focus). Alternatively, the classic STM32F103C8T6 "Blue Pill" board paired with an external ST-LINK V2 programmer can be used for fundamental assignments.
                </p>
                <div class="bg-slate-50 p-4 rounded-lg border border-slate-200 text-xs text-slate-800 space-y-1">
                    <p><strong>NUCLEO-F446RE Specs:</strong> ARM Cortex-M4F core @ 180 MHz, 512 KB Flash, 128 KB SRAM, Hardware FPU.</p>
                    <p><strong>Peripherals:</strong> 3x ADC (12-bit, up to 2.4 MSPS), 2x DAC (12-bit), 2x CAN 2.0B controllers, 4x USART/UART, 4x SPI, 3x I2C, 2x Advanced Timers, 8x General Purpose Timers.</p>
                    <p><strong>On-board ST-LINK/V2-1:</strong> Integrated hardware programmer/debugger providing SWD programming, Virtual COM Port (VCP) over USB, and direct drag-and-drop flash programming.</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Kit 2 -->
    <div class="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row gap-6 items-center">
            <div class="w-full md:w-1/3">
                <img src="/static/blogs/stm32-diy/basic-kit-components.svg" alt="Basic Electronics Prototyping Kit Diagram" class="rounded-lg shadow border border-gray-200 w-full object-cover">
            </div>
            <div class="w-full md:w-2/3">
                <span class="inline-block px-3 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full mb-2">Prototyping Kit</span>
                <h3 class="text-2xl font-bold text-brand-navy mb-2">2. Basic Electronics Kit</h3>
                <p class="text-slate-700 text-sm mb-3">
                    The fundamental building blocks for digital I/O, tactile input, signal attenuation, and driver circuits.
                </p>
                <ul class="text-xs text-slate-700 space-y-1 list-disc pl-5">
                    <li>Full-size 830-point solderless breadboard and premium male-to-male / male-to-female DuPont jumper wires.</li>
                    <li>Tactile push buttons with breadboard-friendly pins (4-pin momentary switches).</li>
                    <li>3mm and 5mm high-efficiency LEDs (Red, Green, Yellow, Blue) and an RGB Common-Cathode LED module.</li>
                    <li>Metal film resistor assortment: 220 &Omega;, 330 &Omega;, 1 k&Omega;, 4.7 k&Omega; (I2C pull-ups), 10 k&Omega; (pull-downs/potentiometers), and 120 &Omega; (CAN termination).</li>
                    <li>10 k&Omega; linear rotary potentiometers for analog voltage divider inputs.</li>
                    <li>Active 5V/3.3V piezo buzzer and passive electromagnetic buzzer for tone/frequency synthesis.</li>
                    <li>General-purpose NPN transistors (2N2222 / BC547), N-channel logic MOSFETs (2N7000), and 1N4148 / 1N4007 flyback diodes.</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Kit 3 -->
    <div class="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row gap-6 items-center">
            <div class="w-full md:w-1/3">
                <img src="/static/blogs/stm32-diy/display-sensing-pinouts.svg" alt="Display and Sensing Module Pinout Diagram" class="rounded-lg shadow border border-gray-200 w-full object-cover">
            </div>
            <div class="w-full md:w-2/3">
                <span class="inline-block px-3 py-1 bg-emerald-100 text-emerald-800 text-xs font-bold rounded-full mb-2">Human Interface &amp; Sensors</span>
                <h3 class="text-2xl font-bold text-brand-navy mb-2">3. Display and Sensing Kit</h3>
                <p class="text-slate-700 text-sm mb-3">
                    Essential visual feedback units, precision clocks, and physical transducer sensors:
                </p>
                <ul class="text-xs text-slate-700 space-y-1 list-disc pl-5">
                    <li><strong>SSD1306 0.96-inch Monochrome OLED:</strong> 128x64 resolution with I2C interface (address 0x3C / 0x3D).</li>
                    <li><strong>DS3231 High-Precision Real-Time Clock (RTC):</strong> Temperature-compensated crystal oscillator (TCXO) with I2C bus and coin-cell battery backup.</li>
                    <li><strong>LM35 / TMP36 Analog Temperature Sensor:</strong> 10 mV/°C linear analog output voltage.</li>
                    <li><strong>DS18B20 1-Wire Digital Thermometer:</strong> 9-bit to 12-bit programmable resolution with digital CRC.</li>
                    <li><strong>Light Dependent Resistor (LDR / Photoresistor):</strong> CdS photocell for ambient light measurement.</li>
                    <li><strong>HC-SR501 PIR Motion Sensor:</strong> Pyroelectric infrared sensor with adjustable sensitivity and trigger delay.</li>
                    <li><strong>TTP223 Capacitive Touch Sensor:</strong> Momentary and latching digital capacitive touch detection.</li>
                    <li><strong>SW-420 Piezo Vibration Sensor:</strong> High-sensitivity vibration sensor with onboard comparator.</li>
                    <li><strong>MPU6050 6-Axis IMU:</strong> 3-axis accelerometer and 3-axis gyroscope with integrated 16-bit ADCs and I2C output.</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Kit 4 -->
    <div class="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row gap-6 items-center">
            <div class="w-full md:w-1/3">
                <img src="/static/blogs/stm32-diy/detection-modules-pinouts.svg" alt="Detection and Actuation Modules Pinout Diagram" class="rounded-lg shadow border border-gray-200 w-full object-cover">
            </div>
            <div class="w-full md:w-2/3">
                <span class="inline-block px-3 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full mb-2">Robotics &amp; Actuation</span>
                <h3 class="text-2xl font-bold text-brand-navy mb-2">4. Detection and Actuation Modules</h3>
                <p class="text-slate-700 text-sm mb-3">
                    Modules for environmental gas monitoring, rangefinding, motion actuation, and robotic navigation:
                </p>
                <ul class="text-xs text-slate-700 space-y-1 list-disc pl-5">
                    <li><strong>HC-SR04 Ultrasonic Distance Sensor:</strong> 2 cm to 400 cm range with 10 µs trigger pulse and duration echo output (requires 5V to 3.3V voltage divider on Echo pin).</li>
                    <li><strong>TCRT5000 Infrared Reflective Sensor Array:</strong> Dual or multi-channel IR line tracking modules.</li>
                    <li><strong>MQ-2 Smoke &amp; Combustible Gas Sensor:</strong> Analog tin dioxide (SnO2) sensing element with onboard heating coil.</li>
                    <li><strong>MQ-3 Alcohol Sensor Module:</strong> Ethanol-sensitive metal oxide semiconductor sensor.</li>
                    <li><strong>Low-Power Eye-Safe Laser Emitter &amp; Photodiode Receiver:</strong> 650 nm 5 mW red laser diode module for optical tripwires.</li>
                    <li><strong>SG90 Micro-Servo Motor:</strong> 180-degree PWM position control (50 Hz / 1.0 ms–2.0 ms pulse width).</li>
                    <li><strong>L298N or TB6612FNG Dual H-Bridge Motor Driver:</strong> Bidirectional DC motor speed and direction driver stage.</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Kit 5 -->
    <div class="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <div class="flex flex-col md:flex-row gap-6 items-center">
            <div class="w-full md:w-1/3">
                <img src="/static/blogs/stm32-diy/can-bus-transceiver-circuit.svg" alt="CAN Bus and Communication Transceiver Circuit Diagram" class="rounded-lg shadow border border-gray-200 w-full object-cover">
            </div>
            <div class="w-full md:w-2/3">
                <span class="inline-block px-3 py-1 bg-rose-100 text-rose-800 text-xs font-bold rounded-full mb-2">Industrial Fieldbuses</span>
                <h3 class="text-2xl font-bold text-brand-navy mb-2">5. Industrial Communication Kit</h3>
                <p class="text-slate-700 text-sm mb-3">
                    Transceivers and memory modules to master industrial multi-node communication:
                </p>
                <ul class="text-xs text-slate-700 space-y-1 list-disc pl-5">
                    <li><strong>3.3V CAN Transceivers:</strong> 2x SN65HVD230 (or VP230) 3.3V CAN transceivers designed to interface directly with STM32 CAN TX/RX pins without level translation.</li>
                    <li><strong>CAN Bus Transmission Line:</strong> Twisted-pair cabling terminated with precise 120 &Omega; resistors at both physical cable ends.</li>
                    <li><strong>Second CAN Node:</strong> A second STM32 development board, or a USB-to-CAN analyzer (e.g., Candlelight / CANable) running Wireshark/cangen for protocol verification.</li>
                    <li><strong>External SPI NOR Flash:</strong> W25Q64 (8 MB) or W25Q128 (16 MB) SPI flash memory module for non-volatile logging.</li>
                    <li><strong>USB-to-UART Bridge:</strong> CP2102 or FT232RL breakout adapter for external terminal debugging and telemetry capture.</li>
                </ul>
            </div>
        </div>
    </div>
</div>

---

<h2 id="mandatory-rules" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-clipboard-check text-brand-coral mr-3"></i> 3. Mandatory Engineering Rules &amp; Non-Blocking Architectural Contract
</h2>

<p>
    In professional embedded systems firmware, sloppy habits like calling blocking delays, ignoring electrical bounce, or driving coils directly from silicon pins are unacceptable. <strong>Every assignment in this curriculum must strictly adhere to the following seven architectural rules:</strong>
</p>

<div class="space-y-4 my-6">
    <div class="p-4 bg-rose-50 border-l-4 border-rose-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-rose-900 text-base mb-1">Rule 1: Non-Blocking Timing Architecture (Zero Long HAL_Delay Calls)</h4>
        <p class="text-xs leading-relaxed">
            <code>HAL_Delay()</code> is a blocking spin-lock that freezes CPU execution, completely blinding the microcontroller to user inputs, sensor deadlines, and communication buffers. <strong>Long blocking delays are strictly prohibited.</strong> All periodic timing, state transitions, and timeouts must be implemented using non-blocking delta-time calculations with <code>HAL_GetTick()</code>, hardware timer interrupts, or cooperative state machine dispatchers:
        </p>
        <pre class="bg-slate-900 text-slate-100 p-2 rounded text-xs font-mono mt-2"><code>/* Correct Non-Blocking Delta-Tick Pattern */
static uint32_t last_time = 0;
uint32_t now = HAL_GetTick();
if (now - last_time &gt;= INTERVAL_MS) {
    last_time = now;
    Execute_Periodic_Task();
}</code></pre>
    </div>

    <div class="p-4 bg-amber-50 border-l-4 border-amber-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-amber-900 text-base mb-1">Rule 2: Push-Button Debouncing &amp; Sensor Input Validation</h4>
        <p class="text-xs leading-relaxed">
            Never trust a raw GPIO read for a mechanical switch. Firmware must implement a robust software debouncing state machine (sampling the button state across a 20 ms to 50 ms stability window) to ensure that exactly one clean event is registered per physical actuation. Similarly, all analog and digital sensor readings must undergo sanity filtering (range checking, median filtering, or outlier rejection) before driving actuators.
        </p>
    </div>

    <div class="p-4 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-blue-900 text-base mb-1">Rule 3: Measurable Acceptance Criteria &amp; Timing Accuracy</h4>
        <p class="text-xs leading-relaxed">
            Every implementation must define unambiguous, quantifiable pass/fail criteria: exact millisecond intervals, frequency tolerances (&plusmn;2%), duty cycle resolutions, and explicit timeout bounds. "It seems to work" is not an engineering specification.
        </p>
    </div>

    <div class="p-4 bg-indigo-50 border-l-4 border-indigo-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-indigo-900 text-base mb-1">Rule 4: Explicit Finite State Machines (FSMs) for Multi-Mode Systems</h4>
        <p class="text-xs leading-relaxed">
            Any project possessing more than one functional mode must be architected as a formal Finite State Machine. Use C <code>typedef enum</code> for state declarations, maintain a dedicated state variable, isolate state transition logic from output generation, and handle unexpected states gracefully in a <code>default</code> clause.
        </p>
    </div>

    <div class="p-4 bg-emerald-50 border-l-4 border-emerald-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-emerald-900 text-base mb-1">Rule 5: Comprehensive Verification Submission Package</h4>
        <p class="text-xs leading-relaxed">
            For each completed assignment, maintain a complete verification portfolio containing: (1) Clean, commented C source code (CubeMX <code>.ioc</code> and <code>Core/Src/</code> files), (2) Electrical wiring diagram / breadboard schematic, (3) Documented test matrix with actual vs expected observations, and (4) Oscilloscope / logic analyzer waveforms validating timing.
        </p>
    </div>

    <div class="p-4 bg-purple-50 border-l-4 border-purple-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-purple-900 text-base mb-1">Rule 6: Fieldbus &amp; Communication Bus Hygiene</h4>
        <p class="text-xs leading-relaxed">
            For CAN bus implementations, verify that a terminating 120 &Omega; resistor is physically connected between CANH and CANL at both ends of the transmission line, and always test with a real second node. For I2C buses, ensure mandatory 4.7 k&Omega; pull-up resistors are connected to 3.3V. For SPI, verify that Clock Polarity (CPOL) and Clock Phase (CPHA) match the peripheral datasheet exactly.
        </p>
    </div>

    <div class="p-4 bg-red-50 border-l-4 border-red-500 rounded-r-lg text-slate-800">
        <h4 class="font-bold text-red-900 text-base mb-1">Rule 7: Strict Electrical Isolation &amp; Inductive Protection</h4>
        <p class="text-xs leading-relaxed">
            <strong>Never drive motors, relays, solenoids, or high-current electromagnetic buzzers directly from STM32 GPIO pins.</strong> Doing so will induce brownouts or permanently blow the internal FET output stages. Always use an external transistor/MOSFET driver stage powered by an appropriate external supply, ensure common ground (GND) bonding, and install a flyback diode across inductive coils.
        </p>
    </div>
</div>
"""
