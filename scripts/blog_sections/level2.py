"""
Level 2 Section: Sensors and Analog Interfaces (Assignments 11–20)
"""

def get_level2_section():
    return r"""
<h2 id="level-2" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-thermometer-half text-brand-coral mr-3"></i> 5. Level 2 — Sensors and Analog Interfaces (Assignments 11–20 · Beginner to Intermediate)
</h2>

<p class="text-slate-700 leading-relaxed mb-6">
    Level 2 transitions from synthetic digital inputs to physical analog transducers, environmental sensors, digital one-wire protocols, and inertial measurement units (IMUs). You will master ADC multi-channel sampling, calibration mathematics, hysteresis filtering, threshold triggering, and I2C graphical display integration.
</p>

<!-- Assignment 11 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 11: Analog Temperature Alarm (LM35 / TMP36)</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">LM35 / TMP36 Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">12-bit ADC (ADC1)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Alarm LED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Buzzer</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Interface an analog precision temperature sensor (LM35 producing 10.0 mV/°C linear output, or TMP36 with 500 mV offset). Sample the sensor voltage every 500 ms using a 12-bit ADC channel. Compute real-world temperature in Celsius with 0.1°C precision. When the measured temperature crosses a configurable threshold (e.g., T &gt; 35.0&deg;C), trip a persistent thermal alarm: illuminate a red warning LED and sound an acoustic buzzer.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Analog voltage reading on ADC pin (e.g. 250 mV at 25°C for LM35).</li>
                <li>Digital alarm pin asserts HIGH immediately when temperature exceeds 35.0°C.</li>
                <li>UART terminal stream logging: <code>"ADC: 310 | Vout: 0.250V | Temp: 25.0 C | Status: NORMAL"</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>ADC Voltage Conversion Formula:
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        V<sub>sense</sub> = (ADC_RAW &times; 3.3V) / 4095<br>
                        For LM35: T(&deg;C) = V<sub>sense</sub> / 0.010V/&deg;C = V<sub>sense</sub> &times; 100
                    </div>
                </li>
                <li>Implement a 16-sample software oversampling or exponential moving average (EMA) filter to eradicate ADC quantification noise:
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        y[n] = &alpha; &times; x[n] + (1 - &alpha;) &times; y[n-1]
                    </div>
                </li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC11.1</td>
                        <td class="p-2 border">Ambient room condition (T &asymp; 24&deg;C).</td>
                        <td class="p-2 border">System logs 24.0°C; Alarm LED remains completely OFF; Buzzer silent.</td>
                        <td class="p-2 border">Reported temp matches calibrated digital thermometer within &plusmn;1.0&deg;C.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC11.2</td>
                        <td class="p-2 border">Apply warm thermal source (finger or warm air) raising T &gt; 35&deg;C.</td>
                        <td class="p-2 border">Temperature rises; as soon as T &ge; 35.1&deg;C, Red Alarm LED turns ON and buzzer sounds.</td>
                        <td class="p-2 border">Trip latency &lt; 500 ms following threshold crossing.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC11.3</td>
                        <td class="p-2 border">Remove thermal source; allow temperature to drop below 35.0°C.</td>
                        <td class="p-2 border">Alarm automatically resets or unlatches (if configured for auto-clear).</td>
                        <td class="p-2 border">Buzzer ceases sounding; LED turns off.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Breadboard power rails often suffer from voltage sag. Check the exact voltage on the <code>VDDA / VREF+</code> pin with a precision DMM. If <code>VDDA</code> is 3.24V instead of 3.30V, calibrate your ADC calculation factor accordingly, or use the internal STM32 V<sub>REFINT</sub> channel (Channel 17) to calculate actual supply voltage dynamically.
        </div>
    </div>
</div>

<!-- Assignment 12 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 12: Digital Temperature Monitor (DS18B20 1-Wire)</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">DS18B20 1-Wire Digital Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">4.7kΩ Pull-Up Resistor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">GPIO in Open-Drain Mode</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Alarm LED</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement the Dallas/Maxim 1-Wire protocol bit-banging driver with microsecond precision. Communicate with a digital DS18B20 temperature sensor over a single bidirectionally driven open-drain bus wire. Send initialization reset pulses, detect sensor presence pulse (60–240 µs response), issue <code>SKIP ROM (0xCC)</code> and <code>CONVERT T (0x44)</code> commands, poll or wait 750 ms for 12-bit conversion, read back the 9-byte scratchpad, and compute the hardware Cyclic Redundancy Check (CRC-8). Display temperature and assert alarm if temperature exceeds a setpoint.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>1-Wire bus logic analyzer trace showing clean reset pulse (480 µs), presence pulse, and write/read time slots (15 µs write-0 / write-1 phases).</li>
                <li>UART terminal stream: <code>"DS18B20 Temp: +27.625 C | CRC: OK (0x5A)"</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Mandatory 4.7 k&Omega; pull-up resistor from 1-Wire data line to 3.3V.</li>
                <li>Configure the STM32 GPIO pin as <code>Output Open-Drain</code> with no internal pull.</li>
                <li>Use a hardware timer (e.g. TIM2 running at 1 MHz) for microsecond delay timing (<code>delay_us()</code>). Disable interrupts (<code>__disable_irq()</code>) only during critical 1-Wire bit read/write time-slots to prevent ISR jitter.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC12.1</td>
                        <td class="p-2 border">Bus Initialization / Reset.</td>
                        <td class="p-2 border">Master pulls low for 480 µs; sensor pulls low for 100 µs presence pulse.</td>
                        <td class="p-2 border">Presence pulse detected; driver returns <code>PRESENCE_OK</code>.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC12.2</td>
                        <td class="p-2 border">Read Scratchpad and compute CRC-8.</td>
                        <td class="p-2 border">Received 9 bytes decoded; calculated CRC matches Byte 8.</td>
                        <td class="p-2 border">CRC matches; zero data corruption.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC12.3</td>
                        <td class="p-2 border">Disconnect sensor physically during operation.</td>
                        <td class="p-2 border">Driver detects missing presence pulse; logs <code>"SENSOR_DISCONNECTED"</code> error.</td>
                        <td class="p-2 border">Firmware handles error gracefully without hanging in infinite while-loop.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Never write an unbounded <code>while(HAL_GPIO_ReadPin(...))</code> waiting for a response! Always include an iteration timeout counter to prevent an unplugged sensor from locking your entire microcontroller execution forever.
        </div>
    </div>
</div>

<!-- Assignment 13 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 13: OLED Temperature Dashboard</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">0.96" SSD1306 I2C OLED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Temperature Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">I2C1 Peripheral</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Reset Min/Max Button</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build an interactive digital temperature dashboard using an SSD1306 128x64 I2C OLED display. The firmware periodically samples the temperature and renders a clean graphical user interface showing:
                (1) Current Temperature in large bold font (e.g. <code>"24.5 °C"</code>),
                (2) Minimum temperature recorded since boot,
                (3) Maximum temperature recorded since boot, and
                (4) A dynamic horizontal progress bar showing temperature relative to a safe 10°C–50°C scale.
                Pressing a tactile button resets the stored Min and Max records to the current ambient reading.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Crisp 128x64 monochrome graphic rendering on OLED updated smoothly at 2 Hz.</li>
                <li>Min and Max values latch and update whenever current temperature exceeds prior limits.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>I2C bus speed configured to Fast Mode (400 kHz) to minimize frame rendering latency.</li>
                <li>Maintain a 1024-byte local RAM framebuffer (<code>uint8_t oled_buffer[128 * 64 / 8]</code>). Perform all text and graphics drawing operations in RAM, and transmit the buffer to the display in a single burst using <code>HAL_I2C_Mem_Write()</code> or DMA.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC13.1</td>
                        <td class="p-2 border">System startup.</td>
                        <td class="p-2 border">OLED initializes; splash screen displays for 1.5s; current, min, and max all equal initial reading.</td>
                        <td class="p-2 border">Display is clear, bright, with zero corrupt pixel artifacts.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC13.2</td>
                        <td class="p-2 border">Warm sensor to 38°C, then cool down to 20°C.</td>
                        <td class="p-2 border">Max updates to 38.0°C; Min updates to 20.0°C; Current reflects real-time value.</td>
                        <td class="p-2 border">Extreme values preserved accurately.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC13.3</td>
                        <td class="p-2 border">Press Min/Max Reset button.</td>
                        <td class="p-2 border">Min and Max instantly overwrite to match current ambient reading.</td>
                        <td class="p-2 border">Reset verified on OLED within 100 ms of button press.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Ensure the SSD1306 charge pump is properly enabled in your initialization command sequence (command bytes <code>0x8D</code> followed by <code>0x14</code>). Without the internal charge pump turned on, the OLED panel will stay completely dark even if I2C communications return HAL_OK!
        </div>
    </div>
</div>

<!-- Assignment 14 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 14: Automatic Night Lamp with Schmitt Trigger Hysteresis</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Light Dependent Resistor (LDR)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">10kΩ Fixed Resistor (Voltage Divider)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">12-bit ADC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Lamp LED / Relay Driver</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement an automated twilight street lamp with software Schmitt trigger hysteresis. An LDR forms a voltage divider with a 10 k&Omega; resistor connected to an ADC pin. As darkness falls, the LDR resistance climbs, causing the ADC voltage to rise. If a single threshold is used, slight light fluctuations or self-illumination cause rapid, destructive lamp flickering. You must implement a dual-threshold hysteresis band:
                Turn Lamp <strong>ON</strong> when dark: V<sub>ADC</sub> &ge; 2.50V (upper threshold).
                Turn Lamp <strong>OFF</strong> when bright: V<sub>ADC</sub> &le; 1.80V (lower threshold).
                In the intermediate region (1.80V to 2.50V), the lamp maintains its previous state.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Relay/LED output turns ON firmly when light drops below darkness threshold.</li>
                <li>Zero rapid fluttering/chattering when ambient light hovers around the threshold border.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>State logic:
                    <pre class="bg-slate-900 text-slate-100 p-2 rounded text-xs font-mono my-1"><code>if (adc_val &gt;= UPPER_THRESHOLD &amp;&amp; !lamp_state) {
    lamp_state = 1;
    HAL_GPIO_WritePin(LAMP_PORT, LAMP_PIN, GPIO_PIN_SET);
} else if (adc_val &lt;= LOWER_THRESHOLD &amp;&amp; lamp_state) {
    lamp_state = 0;
    HAL_GPIO_WritePin(LAMP_PORT, LAMP_PIN, GPIO_PIN_RESET);
}</code></pre>
                </li>
                <li>Sample ADC at 10 Hz with low-pass filtering.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC14.1</td>
                        <td class="p-2 border">Expose to direct flashlight (V<sub>ADC</sub> &asymp; 0.5V).</td>
                        <td class="p-2 border">Lamp output firmly OFF.</td>
                        <td class="p-2 border">GPIO pin reads 0.0V.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC14.2</td>
                        <td class="p-2 border">Gradually shade LDR until V<sub>ADC</sub> crosses 2.50V.</td>
                        <td class="p-2 border">Lamp turns ON immediately upon hitting 2.50V.</td>
                        <td class="p-2 border">GPIO pin switches to 3.3V.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC14.3</td>
                        <td class="p-2 border">Introduce slight ambient jitter around 2.20V (within hysteresis band).</td>
                        <td class="p-2 border">Lamp stays solid ON without flickering.</td>
                        <td class="p-2 border">Zero relay chattering or PWM flicker observed.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            If your LDR is placed physically close to the lamp LED, the light emitted by the LED can shine back into the LDR, artificially dropping the sensed resistance and triggering an endless ON-OFF oscillation. Always orient the sensor away from the lamp or add a minimum 3-second holdoff timer after state transitions.
        </div>
    </div>
</div>

<!-- Assignment 15 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 15: Smoke-Level Gas Warning Monitor (MQ-2)</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">MQ-2 Gas/Smoke Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">12-bit ADC Channel</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Buzzer (PWM Tone Modulated)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Status LEDs</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Interface an MQ-2 metal-oxide semiconductor combustible gas sensor. On initial power-up, enforce a mandatory 20-second preheating warm-up phase (during which an amber LED blinks slowly and gas alarms are inhibited). Once preheated, calculate baseline ambient resistance R<sub>0</sub>. When smoke or gas concentrations rise above baseline, progressively escalate acoustic warnings:
                <strong>Safe Level:</strong> Buzzer silent, Green LED ON.
                <strong>Moderate Level:</strong> Buzzer beeps intermittently (1 Hz repetition rate), Yellow LED ON.
                <strong>Hazardous Level:</strong> Buzzer sounds continuous high-pitch alarm (2.5 kHz), Red LED flashes rapidly.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Analog voltage output from MQ-2 load resistor rises in proportion to smoke density.</li>
                <li>Audible beep rate scales dynamically with measured gas concentration.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li><strong>Safe Testing Protocol:</strong> Test sensor responses safely using an unlit butane lighter release in a well-ventilated room, or by extinguishing a wooden match near the sensor. Never expose the module to open flames or hazardous combustible gas concentrations in unventilated spaces.</li>
                <li>MQ-2 heating element draws ~150 mA at 5V; power the sensor from the board's 5V pin or external supply, never from a 3.3V GPIO.</li>
                <li>Verify that the analog output voltage does not exceed 3.3V before feeding into the STM32 ADC channel (use a resistive divider if module outputs 0–5V).</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC15.1</td>
                        <td class="p-2 border">Boot device.</td>
                        <td class="p-2 border">System enters 20-second preheat mode; amber LED blinks; alarm disabled.</td>
                        <td class="p-2 border">Timer runs for exactly 20.0s before switching to active monitoring.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC15.2</td>
                        <td class="p-2 border">Introduce slight puff of extinguished match smoke.</td>
                        <td class="p-2 border">Gas reading elevates into Moderate band; Yellow LED lights; buzzer emits 1 Hz warning pulses.</td>
                        <td class="p-2 border">Beep rate matches 1.0 Hz &plusmn;5%.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC15.3</td>
                        <td class="p-2 border">Clear air with gentle fan.</td>
                        <td class="p-2 border">Gas concentration returns below safe threshold; buzzer returns to silent.</td>
                        <td class="p-2 border">Green LED illuminates; safe state restored.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            New MQ-2 sensors require up to 24–48 hours of initial continuous power burn-in for their internal chemical coatings to reach optimal equilibrium. For laboratory assignments, a 2-minute preheating period is sufficient for consistent comparative differential testing.
        </div>
    </div>
</div>

<!-- Assignment 16 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 16: Alcohol Vapor Monitor (MQ-3)</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">MQ-3 Alcohol Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">12-bit ADC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Piezo Buzzer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART Console / Bar Graph</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Interface an MQ-3 ethanol gas sensor to detect airborne alcohol vapors. Calibrate the baseline sensor resistance in clean ambient air. As alcohol vapor concentrations increase, the firmware maps the analog sensor voltage into an 8-segment ASCII bar graph printed over UART: <code>"[====    ] 42% Intensity"</code>. Modulate an acoustic buzzer frequency or beep rate proportionally to the reading. <em>Explicit Safety Notice:</em> This implementation is an educational demonstrator of chemical semiconductor sensing; it must never be interpreted as a certified breathalyzer or legal Blood Alcohol Content (BAC) measurement.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Real-time UART telemetry displaying raw ADC counts, normalized vapor index (0–100%), and visual ASCII bar graph.</li>
                <li>Acoustic beeper increases in repetition frequency from 0.5 Hz (low vapor) up to 10 Hz (heavy vapor saturation).</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Provide clean baseline auto-zeroing: upon user command, average 50 samples to establish the baseline zero-offset.</li>
                <li>Limit voltage fed into the STM32 ADC pin to 3.3V maximum.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC16.1</td>
                        <td class="p-2 border">Ambient baseline measurement.</td>
                        <td class="p-2 border">Index reads 0%–5%; bar graph empty; buzzer silent.</td>
                        <td class="p-2 border">Clean baseline stability.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC16.2</td>
                        <td class="p-2 border">Wave an isopropyl alcohol wipe 5 cm away from sensor.</td>
                        <td class="p-2 border">ADC voltage surges; index climbs to 60%–80%; buzzer beeps at high repetition rate.</td>
                        <td class="p-2 border">Beep rate scales linearly with ADC voltage.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            MQ-3 heater elements run warm to the touch (~50°C). This is normal and necessary for the tin-dioxide reduction reaction. Ensure your circuit wires are heat-resistant and away from direct contact with the metal mesh hood.
        </div>
    </div>
</div>

<!-- Assignment 17 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 17: 6-Axis Accelerometer Tilt Detector (MPU6050)</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">MPU6050 6-DOF IMU</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">I2C Bus (I2C1)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">4x Directional LEDs (N, S, E, W)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">4x 330Ω Resistors</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Interface an MPU6050 inertial measurement unit over I2C. Configure the accelerometer full-scale range to &plusmn;2g. Wake the device from sleep by writing <code>0x00</code> to register <code>PWR_MGMT_1 (0x6B)</code>. Read raw 16-bit 2's complement acceleration registers A<sub>X</sub>, A<sub>Y</sub>, A<sub>Z</sub> at 50 Hz. Convert raw values into pitch and roll tilt angles using trigonometric atan2 calculations:
                <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                    Pitch = atan2(A<sub>X</sub>, sqrt(A<sub>Y</sub><sup>2</sup> + A<sub>Z</sub><sup>2</sup>)) &times; (180&deg; / &pi;)<br>
                    Roll = atan2(A<sub>Y</sub>, sqrt(A<sub>X</sub><sup>2</sup> + A<sub>Z</sub><sup>2</sup>)) &times; (180&deg; / &pi;)
                </div>
                Illuminate four directional LEDs (North, South, East, West) corresponding to board inclination beyond a &plusmn;20&deg; threshold. When resting perfectly level (tilt &lt; 5&deg;), all directional LEDs remain dark.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Tilting board forward activates North LED; tilting backward activates South LED; tilting right activates East LED; tilting left activates West LED.</li>
                <li>UART telemetry reporting computed Pitch and Roll in degrees at 20 Hz.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>I2C address: <code>0xD0</code> (write) / <code>0xD1</code> (read) when AD0 pin is tied to GND.</li>
                <li>Perform burst read of 6 acceleration bytes starting from register <code>ACCEL_XOUT_H (0x3B)</code> in a single I2C transaction.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC17.1</td>
                        <td class="p-2 border">Rest board flat on level table.</td>
                        <td class="p-2 border">A<sub>Z</sub> &asymp; +1.0g, A<sub>X</sub> &asymp; 0, A<sub>Y</sub> &asymp; 0. Pitch and Roll &lt; 3&deg;. All LEDs OFF.</td>
                        <td class="p-2 border">Measured gravity vector matches 1.00 &plusmn; 0.05g.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC17.2</td>
                        <td class="p-2 border">Tilt board forward by +30° (Pitch).</td>
                        <td class="p-2 border">North LED turns ON; South, East, West remain OFF.</td>
                        <td class="p-2 border">Computed Pitch measures +30 &plusmn; 3&deg;.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC17.3</td>
                        <td class="p-2 border">Tilt board 45° diagonally (forward-right).</td>
                        <td class="p-2 border">Both North and East LEDs illuminate simultaneously.</td>
                        <td class="p-2 border">Multi-axis directional activation verified.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Remember that MPU6050 register bytes arrive High-Byte first (MSB) followed by Low-Byte (LSB). You must reconstruct the 16-bit signed integer using: <code>int16_t raw_x = (int16_t)((buffer[0] &lt;&lt; 8) | buffer[1]);</code>.
        </div>
    </div>
</div>

<!-- Assignment 18 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 18: Industrial Vibration Rate Monitor</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SW-420 Vibration Switch</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">GPIO External Interrupt (EXTI)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">1-Second Periodic Timer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Warning Buzzer &amp; LED</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Simulate an industrial machinery predictive maintenance monitor. An SW-420 piezoelectric vibration switch produces digital pulses upon mechanical shock and vibration. Connect the sensor's digital output to an EXTI interrupt pin. Increment an atomic <code>volatile uint32_t pulse_count</code> on every falling edge. A 1-second periodic hardware timer computes vibration event frequency in Hertz (events per second):
                If <strong>Frequency &lt; 5 Hz</strong>: Normal machinery operation.
                If <strong>5 Hz &le; Frequency &lt; 20 Hz</strong>: Mild imbalance / bearing wear (Yellow LED).
                If <strong>Frequency &ge; 20 Hz</strong>: Critical mechanical oscillation (Red LED + Pulsed Alarm Siren).
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>EXTI interrupt counter measures incoming vibration shocks.</li>
                <li>UART logs: <code>"Vibration Rate: 28 Hz | STATUS: CRITICAL FAULT"</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Double-buffered frequency calculation: Inside the 1-second timer callback, copy <code>pulse_count</code> to a local variable and reset <code>pulse_count = 0;</code> atomically.</li>
                <li>Ensure EXTI line priority does not block system timing base.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC18.1</td>
                        <td class="p-2 border">Stationary workbench.</td>
                        <td class="p-2 border">0 Hz measured; Green LED solid ON; Buzzer silent.</td>
                        <td class="p-2 border">Zero false vibration pulses counted.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC18.2</td>
                        <td class="p-2 border">Tap breadboard rhythmically ~10 times in 1 second.</td>
                        <td class="p-2 border">Frequency calculates to ~10 Hz; Yellow LED activates.</td>
                        <td class="p-2 border">Pulse count reflects actual physical taps &plusmn;2.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC18.3</td>
                        <td class="p-2 border">Apply continuous vibration motor against sensor chassis.</td>
                        <td class="p-2 border">Pulse rate exceeds 20 Hz; Red alarm LED trips; Buzzer sounds.</td>
                        <td class="p-2 border">Alarm trips within 1.0 second of vibration onset.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            SW-420 modules include an onboard potentiometer to adjust comparator threshold. If your sensor triggers continuously even when quiet, turn the onboard screw counter-clockwise to reduce sensitivity until the onboard DO LED just turns off.
        </div>
    </div>
</div>

<!-- Assignment 19 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 19: Capacitive Touch Light with Continuous Dimming</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">TTP223 Capacitive Touch Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">PWM LED Channel</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Hardware Timer</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a smart touch lamp controller supporting both momentary taps and continuous press-and-hold gestures.
                <strong>Short Touch (&lt; 400 ms):</strong> Toggles the LED ON and OFF. When turning ON, restore the previously remembered brightness level.
                <strong>Long Touch (&ge; 400 ms):</strong> Initiates continuous smooth PWM dimming. As the user maintains physical finger contact on the capacitive sensor pad, brightness ramps smoothly up to 100%, reverses direction upon hitting the ceiling, ramps down to 10%, reverses upon hitting the floor, and locks at whatever brightness is active when the finger lifts.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Short tap switches LED on/off cleanly with memory.</li>
                <li>Long press modulates PWM duty cycle smoothly at ~15% per second without stepping glitches.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Implement a gesture timing state machine: record <code>touch_start_time</code> on rising edge; evaluate duration on falling edge.</li>
                <li>While touch remains active beyond 400 ms, update PWM compare register in 20 ms non-blocking intervals.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC19.1</td>
                        <td class="p-2 border">Tap touch pad quickly (150 ms contact).</td>
                        <td class="p-2 border">LED toggles from OFF to ON at last stored brightness.</td>
                        <td class="p-2 border">Clean toggle without triggering dimming mode.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC19.2</td>
                        <td class="p-2 border">Hold touch pad continuously for 3.0 seconds.</td>
                        <td class="p-2 border">After 400 ms, LED brightness begins ramping up/down cyclically.</td>
                        <td class="p-2 border">Smooth PWM duty cycle sweep verified on oscilloscope.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC19.3</td>
                        <td class="p-2 border">Release finger at desired intermediate brightness (~60%).</td>
                        <td class="p-2 border">Brightness freezes at 60%; subsequent short tap turns it OFF and back ON to 60%.</td>
                        <td class="p-2 border">Memory retention verified.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            TTP223 breakout boards have solder jumpers (A and B) that configure active-high vs active-low and momentary vs toggle modes. Ensure the board is soldered for <strong>Active-High Momentary</strong> mode so the STM32 handles all logic states in software.
        </div>
    </div>
</div>

<!-- Assignment 20 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 20: Retriggerable PIR Motion Security Monitor</h3>
        <span class="px-2.5 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full">Level 2 · Beginner to Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">HC-SR501 PIR Motion Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Security Siren Buzzer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Alarm LED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Retrigger Timer</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build an automated security corridor lighting and alarm monitor. An HC-SR501 Passive Infrared (PIR) sensor monitors human body heat motion. When motion is detected (sensor pin goes HIGH), immediately illuminate a floodlight LED and sound an alert tone. Start a configurable 30-second inactivity countdown. If motion is detected again <em>while the countdown is running</em>, the timer re-arms back to 30.0 seconds (retriggerable operation). Only after 30 uninterrupted seconds of total silence and zero motion does the lighting and siren shut off.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Floodlight LED turns ON within &lt; 5 ms of PIR trigger pin rising HIGH.</li>
                <li>Countdown resets on repeated motion pulses; turns OFF only after complete inactivity.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>HC-SR501 requires a 60-second power-up stabilization window. Firmware must implement an initialization warm-up delay during which motion triggers are suppressed.</li>
                <li>Sensor outputs 3.3V logic (compatible with STM32 input pins), but requires 5V supply on its VCC pin.</li>
            </ul>
        </div>

        <div class="overflow-x-auto my-3">
            <table class="w-full text-xs text-left border border-gray-200">
                <thead class="bg-slate-50 text-brand-navy font-bold">
                    <tr>
                        <th class="p-2 border">Test Case ID</th>
                        <th class="p-2 border">Input / Stimulus</th>
                        <th class="p-2 border">Expected System Behavior</th>
                        <th class="p-2 border">Pass/Fail Criteria</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="p-2 border font-mono">TC20.1</td>
                        <td class="p-2 border">Initial power-on.</td>
                        <td class="p-2 border">Firmware waits 60s for PIR chamber stabilization; status LED indicates warm-up.</td>
                        <td class="p-2 border">Zero false alarms triggered during startup calibration.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC20.2</td>
                        <td class="p-2 border">Wave hand across sensor lens.</td>
                        <td class="p-2 border">PIR pin rises HIGH; Floodlight LED lights; 30s timer begins.</td>
                        <td class="p-2 border">Trigger response latency &lt; 10 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC20.3</td>
                        <td class="p-2 border">Move again at T = 20 s into the countdown.</td>
                        <td class="p-2 border">Timer resets to 30.0s; LED remains steadily ON.</td>
                        <td class="p-2 border">Total ON time extends to 50 seconds (20 + 30).</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC20.4</td>
                        <td class="p-2 border">Zero movement for 30 consecutive seconds.</td>
                        <td class="p-2 border">Timer expires at T = 30.0 s; Floodlight turns OFF.</td>
                        <td class="p-2 border">Clean return to armed standby state.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            The HC-SR501 has a physical jumper for 'H' (repeat trigger) vs 'L' (single trigger). Set the physical jumper on the PCB to <strong>H (repeat trigger)</strong>, and turn the onboard time-delay trimpot fully counter-clockwise to its minimum position (~3 seconds) so the STM32 handles all precision hold time in firmware.
        </div>
    </div>
</div>
"""
