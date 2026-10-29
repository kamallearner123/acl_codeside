"""
Level 1 Section: GPIO, Timers and Basic Control (Assignments 1–10)
"""

def get_level1_section():
    return r"""
<h2 id="level-1" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-toggle-on text-brand-coral mr-3"></i> 4. Level 1 — GPIO, Timers and Basic Control (Assignments 1–10 · Beginner)
</h2>

<p class="text-slate-700 leading-relaxed mb-6">
    Level 1 lays the bedrock of embedded software engineering: digital input/output register configuration, hardware timer timebases, non-blocking event loops, tactile debouncing, external interrupt servicing (EXTI), and pulse-width modulation (PWM).
</p>

<!-- Assignment 1 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 1: User Inactivity Detector</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">LED (with 330Ω resistor)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Hardware Timer / HAL_GetTick</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy01-inactivity-detector.svg" alt="STM32 Inactivity Detector Wiring Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.1: Push button input (PC13) and LED output (PA5) wiring diagram.</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                The firmware acts as a dead-man inactivity monitor. A countdown timer initializes to 60.0 seconds upon system boot. Whenever the user actuates the tactile push button, the inactivity timer instantly resets back to 60.0 seconds. If no valid button press is registered for 60 consecutive seconds, the system enters an "INACTIVE" warning state, oscillating the indicator LED at exactly 250 ms ON / 250 ms OFF (2.0 Hz blink frequency). As soon as the user presses the button during this flashing warning, the LED immediately turns OFF and the 60-second inactivity countdown restarts.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>GPIO Output Pin (e.g., PA5 / Green User LED): Inactive state: 2.0 Hz square wave (250 ms HIGH, 250 ms LOW, 50% duty cycle). Active state (&lt; 60 s): Constant LOW (0.0V).</li>
                <li>UART Telemetry (Optional 115200 baud): Periodic debug output logging seconds remaining every 5 seconds and an alert banner upon timeout.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Zero <code>HAL_Delay()</code> calls. The 60-second timeout and 250 ms flash interval must run simultaneously using non-blocking delta-time arithmetic (<code>HAL_GetTick()</code>) or a hardware timer.</li>
                <li>The push button input must incorporate software debouncing (minimum 30 ms threshold) to ignore contact bounce.</li>
                <li>Resetting the timer must occur on the <em>falling edge</em> (press event) of the debounced input.</li>
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
                        <td class="p-2 border font-mono">TC1.1</td>
                        <td class="p-2 border">Boot device and leave untouched.</td>
                        <td class="p-2 border">LED remains completely OFF for exactly 60 seconds; at T = 60.00 s, LED starts blinking at 250 ms toggle interval.</td>
                        <td class="p-2 border">Timer within &plusmn;100 ms accuracy on stopwatch; frequency 2.00 &plusmn; 0.05 Hz on oscilloscope.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC1.2</td>
                        <td class="p-2 border">Press button at T = 45 s.</td>
                        <td class="p-2 border">Inactivity timer resets; LED remains OFF until T = 105 s (45 + 60).</td>
                        <td class="p-2 border">No LED blink observed at original 60 s mark.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC1.3</td>
                        <td class="p-2 border">Allow LED to blink (inactive state), then press button.</td>
                        <td class="p-2 border">LED turns OFF immediately (&lt; 10 ms latency) and inactivity countdown resets to 60 s.</td>
                        <td class="p-2 border">LED instantly extinguished; clean return to active monitoring state.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC1.4</td>
                        <td class="p-2 border">Rapid chatter: button pressed 10 times in 1 second.</td>
                        <td class="p-2 border">Debouncer absorbs physical bounce; timer resets cleanly without spurious locks.</td>
                        <td class="p-2 border">No system crash, watchdog trip, or race condition.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            If your LED flashes erratically before 60 seconds, check if your button GPIO is configured as a floating input. If not using an external pull-up resistor, ensure the internal pull-up (<code>GPIO_PULLUP</code>) is enabled in STM32CubeMX. Verify your tick variables use <code>uint32_t</code> to avoid integer overflow issues during delta calculations.
        </div>
    </div>
</div>

<!-- Assignment 2 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 2: Multi-Mode LED Controller</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">LED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SysTick / TIM Base</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy02-multimode-led.svg" alt="STM32 Multi-Mode LED Controller Wiring Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.2: Deterministic multi-mode pattern sequencing on PA5 and PC13.</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a cyclostationary 4-mode lighting controller. Each successive valid push button depression advances the operational mode in a strict ring sequence:
                <strong>Mode 0: Constant OFF</strong> &rarr; <strong>Mode 1: Constant ON</strong> &rarr; <strong>Mode 2: 500 ms Slow Blink</strong> (1 Hz, 500 ms HIGH / 500 ms LOW) &rarr; <strong>Mode 3: 100 ms Fast Blink</strong> (5 Hz, 100 ms HIGH / 100 ms LOW) &rarr; <strong>Wrap to Mode 0</strong>.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Mode 0: 0.0V DC constant.</li>
                <li>Mode 1: 3.3V DC constant.</li>
                <li>Mode 2: 1.0 Hz square wave, 50% duty cycle on logic analyzer.</li>
                <li>Mode 3: 5.0 Hz square wave, 50% duty cycle on logic analyzer.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Structure the firmware around an explicit C <code>enum { MODE_OFF, MODE_ON, MODE_BLINK_500MS, MODE_BLINK_100MS }</code>.</li>
                <li>Button state transition must trigger strictly on the rising edge of the debounced press, advancing <code>mode = (mode + 1) % 4;</code>.</li>
                <li>Timing logic for Mode 2 and Mode 3 must be non-blocking. Switching modes while an LED is in the middle of a 500 ms blink phase must immediately transition to the new mode's output state without waiting for the prior timer interval to expire.</li>
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
                        <td class="p-2 border font-mono">TC2.1</td>
                        <td class="p-2 border">Power ON system (initial boot).</td>
                        <td class="p-2 border">System enters Mode 0; LED is completely OFF.</td>
                        <td class="p-2 border">Multimeter measures 0.0V on LED cathode/anode circuit.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC2.2</td>
                        <td class="p-2 border">Press button 1 time.</td>
                        <td class="p-2 border">System transitions to Mode 1; LED turns solid ON.</td>
                        <td class="p-2 border">3.3V measured on pin; no flickering observed.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC2.3</td>
                        <td class="p-2 border">Press button again (2nd press).</td>
                        <td class="p-2 border">System transitions to Mode 2; LED blinks at 1.0 Hz (500 ms ON / 500 ms OFF).</td>
                        <td class="p-2 border">Oscilloscope period measured at 1000 &plusmn; 2 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC2.4</td>
                        <td class="p-2 border">Press button again (3rd press).</td>
                        <td class="p-2 border">System transitions to Mode 3; LED blinks at 5.0 Hz (100 ms ON / 100 ms OFF).</td>
                        <td class="p-2 border">Oscilloscope period measured at 200 &plusmn; 1 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC2.5</td>
                        <td class="p-2 border">Press button a 4th time.</td>
                        <td class="p-2 border">System wraps back to Mode 0; LED shuts OFF instantly.</td>
                        <td class="p-2 border">LED turns OFF &lt; 5 ms following button release.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Watch out for button release bounces. If your code detects both press and release as separate mode triggers, you will skip modes (e.g. jumping from Mode 0 directly to Mode 2). Implement an edge detector: <code>if (current_debounced_state &amp;&amp; !prev_debounced_state)</code>.
        </div>
    </div>
</div>

<!-- Assignment 3 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 3: Configurable Electronic Buzzer</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Passive/Active Buzzer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2N2222 NPN Transistor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Timer PWM / Output Compare</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy03-transistor-buzzer.svg" alt="STM32 Configurable Electronic Buzzer Wiring Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.3: Transistor switching stage to buffer electromagnetic transducer loads (PB8 buzzer, PC13 button).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Construct an acoustic alerting module with 5 selectable tone frequencies or beep repetition rates. When using a passive piezoelectric transducer, each button press shifts the timer PWM output frequency:
                <strong>Setting 1: 440 Hz (Concert A)</strong> &rarr; <strong>Setting 2: 880 Hz</strong> &rarr; <strong>Setting 3: 1760 Hz</strong> &rarr; <strong>Setting 4: 2500 Hz</strong> &rarr; <strong>Setting 5: 3500 Hz</strong> &rarr; <strong>Wrap to Setting 1</strong>.
                If using an active buzzer, modulate the beep pulse interval from 1.0 s down to 100 ms in 5 distinct steps.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Timer PWM pin outputs a clean 50% duty cycle square wave at 440, 880, 1760, 2500, and 3500 Hz.</li>
                <li>Buzzer tone pitch visibly and audibly steps up on each button press.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li><strong>Mandatory Transistor Driver:</strong> Never connect the buzzer directly to an STM32 GPIO pin. Drive the base of a 2N2222 NPN transistor through a 1 k&Omega; resistor, with the buzzer coil tied to 5V/3.3V and a 1N4148 flyback diode connected in reverse-parallel.</li>
                <li>PWM frequency must be altered dynamically by adjusting the timer's Auto-Reload Register (<code>ARR</code>) on the fly without stopping the timer core:
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">f<sub>PWM</sub> = f<sub>TIM_CLK</sub> / ((PSC + 1) &times; (ARR + 1))</div>
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
                        <td class="p-2 border font-mono">TC3.1</td>
                        <td class="p-2 border">System boot / Setting 1.</td>
                        <td class="p-2 border">Buzzer emits stable 440 Hz tone.</td>
                        <td class="p-2 border">Oscilloscope frequency counter measures 440 &plusmn; 2 Hz.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC3.2</td>
                        <td class="p-2 border">Press button 1 time (Setting 2).</td>
                        <td class="p-2 border">Pitch doubles to 880 Hz.</td>
                        <td class="p-2 border">Frequency measures 880 &plusmn; 4 Hz.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC3.3</td>
                        <td class="p-2 border">Step through to Setting 5 and press again.</td>
                        <td class="p-2 border">From 3500 Hz, the setting wraps seamlessly back to 440 Hz.</td>
                        <td class="p-2 border">Audible pitch drops to low tone; no frequency glitches or timer crashes.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            When changing <code>ARR</code> dynamically, ensure the Capture/Compare Register (<code>CCR</code>) is also updated to maintain 50% duty cycle: <code>__HAL_TIM_SET_AUTORELOAD(&amp;htimX, new_arr); __HAL_TIM_SET_COMPARE(&amp;htimX, TIM_CHANNEL_1, new_arr / 2);</code>. Verify that TIM Prescaler is sized to ensure <code>ARR</code> stays within 16-bit bounds (0 to 65535).
        </div>
    </div>
</div>

<!-- Assignment 4 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 4: Binary Counter Display (4-Bit LEDs)</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">4x LEDs (Red/Green/Yellow/Blue)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">4x 330Ω Resistors</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">GPIO Output Port</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy04-binary-counter-leds.svg" alt="STM32 4-Bit Binary Counter LED Interface Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.4: 4-bit binary accumulator mapped to discrete GPIO pins (PB0-PB3).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a hardware 4-bit synchronous binary accumulator. Four discrete LEDs represent bits 0, 1, 2, and 3 (LED0 = LSB, LED3 = MSB). On initial boot, the display shows <code>0000</code> (0 decimal). Every valid button press increments the counter by 1. When the counter reaches <code>1111</code> (15 decimal), the subsequent press rolls over to <code>0000</code>.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>LED3..LED0 reflect the current binary state: 0 (<code>0000</code>) to 15 (<code>1111</code>).</li>
                <li>Clean atomic transitions across multiple bit flips (e.g., transition from 7 [<code>0111</code>] to 8 [<code>1000</code>] occurs with zero visual intermediate glitches).</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Use bitwise masking to update the output pins. If using contiguous pins on a single GPIO port (e.g., PB0..PB3), use the port Bit Set/Reset Register (<code>BSRR</code>) to write all four bits simultaneously in a single atomic instruction.</li>
                <li>Never update the bits sequentially with individual <code>HAL_GPIO_WritePin()</code> calls if it produces visible intermediate states under high-speed sampling.</li>
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
                        <td class="p-2 border font-mono">TC4.1</td>
                        <td class="p-2 border">Initial power-on reset.</td>
                        <td class="p-2 border">All 4 LEDs are completely OFF (Value = 0).</td>
                        <td class="p-2 border">Logic levels: [0, 0, 0, 0].</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC4.2</td>
                        <td class="p-2 border">Press button 5 times sequentially.</td>
                        <td class="p-2 border">LEDs display decimal 5: LED2 and LED0 ON, LED3 and LED1 OFF.</td>
                        <td class="p-2 border">Binary pattern matches [0, 1, 0, 1].</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC4.3</td>
                        <td class="p-2 border">Press button when count = 15 (<code>1111</code>).</td>
                        <td class="p-2 border">Counter rolls over to 0; all LEDs extinguish.</td>
                        <td class="p-2 border">Clean rollover without latching at 15.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Use STM32's <code>BSRR</code> register for atomic multi-pin updates:
            <code>GPIOB-&gt;BSRR = ((~count &amp; 0x0F) &lt;&lt; 16) | (count &amp; 0x0F);</code>. This writes the lower 4 bits and clears the upper bits in one clock cycle.
        </div>
    </div>
</div>

<!-- Assignment 5 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 5: Human Reaction-Time Measurement System</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Trigger LED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">High-Resolution Hardware Timer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART Virtual COM Port</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy05-reaction-timer.svg" alt="STM32 Reaction Time Measurement Wiring Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.5: Hardware timer capture and precision timing for reaction latency (PA5 LED, PC13 button).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Measure human reflex latency with 1-millisecond resolution. On round initiation, the system waits for an unpredictable pseudorandom delay between 2.0 and 6.0 seconds. At the exact expiration of this delay, the trigger LED illuminates and a high-resolution microsecond/millisecond hardware timer begins counting. The user must slap the push button as rapidly as possible. The elapsed time between LED illumination and the button press is calculated and transmitted over UART: <code>"Reaction Time: 238 ms"</code>.
                If the user presses the button <em>before</em> the LED lights up, declare a "FALSE START / CHEAT DETECTED" penalty and abort the trial.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Trigger LED on/off timing.</li>
                <li>UART output at 115200-8-N-1 reporting exact measured latency in milliseconds.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Timer counter must use a dedicated hardware timer (e.g. TIM2 configured with 1 MHz clock for 1 µs tick precision) or <code>HAL_GetTick()</code>.</li>
                <li>The pseudorandom delay can be seeded using the STM32 True Random Number Generator (RNG peripheral on F446/L476) or an unbuffered floating ADC channel read.</li>
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
                        <td class="p-2 border font-mono">TC5.1</td>
                        <td class="p-2 border">Start game, wait for LED, press button cleanly.</td>
                        <td class="p-2 border">Reaction latency calculated accurately and printed to UART terminal.</td>
                        <td class="p-2 border">Value matches logic analyzer pulse width between LED HIGH and Button LOW within &plusmn;1 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC5.2</td>
                        <td class="p-2 border">Press button 500 ms after round starts (while LED is still dark).</td>
                        <td class="p-2 border">Firmware detects preemptive press, prints <code>"FALSE START!"</code>, and turns on red penalty LED.</td>
                        <td class="p-2 border">Timer is cancelled; trial aborted.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            To eliminate software interrupt jitter, connect the push button to a Timer Input Capture channel. The LED activation can be triggered by Timer Output Compare, allowing the hardware timer to capture the response completely without CPU intervention!
        </div>
    </div>
</div>

<!-- Assignment 6 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 6: Interrupt-Driven Emergency E-Stop Button</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button (EXTI line)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Emergency Alarm LED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Buzzer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Cortex-M NVIC</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy06-estop-interrupt.svg" alt="STM32 Interrupt-Driven Emergency Button Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.6: Sub-microsecond emergency response via hardware EXTI vectoring (PA5 LED, PB8 buzzer, PC13 button).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Simulate an industrial machinery Emergency Stop (E-Stop). The main background loop executes heavy, continuous data processing tasks. When the emergency push button is depressed, an External Interrupt (EXTI) fires instantly, interrupting the main loop. Within <strong>less than 5 microseconds</strong> of switch contact closure, the firmware must trip an emergency shutdown relay, light a red alarm LED, activate an alarm siren, and latch the system into a safe state. Normal operation cannot resume until a dedicated software reset sequence is executed.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Oscilloscope dual-trace: Channel 1 on Button EXTI pin (falling edge), Channel 2 on Alarm Output Pin. Propagation delay measured between edge and pin high must be &lt; 5 &mu;s.</li>
                <li>System latches in SHUTDOWN mode; background worker loops cease execution immediately.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Configure the GPIO pin in CubeMX as <code>External Interrupt Mode with Falling edge trigger detection</code>.</li>
                <li>Assign the EXTI line highest NVIC priority (Preemption Priority = 0).</li>
                <li>Keep the ISR ultra-short: set a <code>volatile uint8_t estop_tripped = 1;</code> flag and trip the physical safety pins directly via register bit-set: <code>ALARM_GPIO_Port-&gt;BSRR = ALARM_Pin;</code>. Never call <code>printf</code> or blocking routines in the ISR!</li>
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
                        <td class="p-2 border font-mono">TC6.1</td>
                        <td class="p-2 border">Depress E-Stop button while main loop executes dummy load.</td>
                        <td class="p-2 border">Interrupt triggers immediately; alarm pin driven HIGH; main loop halted.</td>
                        <td class="p-2 border">Latency &lt; 5 &mu;s measured on digital storage oscilloscope.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC6.2</td>
                        <td class="p-2 border">Press normal operational buttons after E-stop trip.</td>
                        <td class="p-2 border">System ignores all normal inputs; safety interlock remains active.</td>
                        <td class="p-2 border">Zero motor/actuator activity permitted while in tripped state.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Remember to clear the pending interrupt bit in the EXTI Pending Register (handled automatically by <code>HAL_GPIO_EXTI_IRQHandler</code>). If the pending bit is not cleared, the Cortex-M will continuously re-enter the ISR, freezing the chip.
        </div>
    </div>
</div>

<!-- Assignment 7 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 7: Push-Button Debouncing Engine</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Periodic Hardware Timer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART Telemetry Output</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy07-debouncing-filter.svg" alt="STM32 Push Button Debouncing Circuit Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.7: Shift-register digital integrator for noise suppression and USART2 telemetry (PC13 button, PA2 TX).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a production-grade software debouncing engine using a periodic 5 ms timer interrupt or SysTick sampler. The algorithm must feed raw pin readings into an 8-bit shift register or integrator accumulator. The state is declared "PRESSED" only when the signal has remained stable for at least 6 consecutive samples (30 ms). Maintain an event counter that increments <strong>exactly by 1</strong> per physical press and outputs over UART: <code>"Press Count: 42"</code>. Even when intentionally tapping an old, noisy mechanical switch, zero false counts may occur.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>UART terminal logs consecutive incrementing integers (1, 2, 3, 4...).</li>
                <li>Zero double-counting or skipping across 100 rapid physical taps.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Do not use simple <code>HAL_Delay()</code> inside the button read function!</li>
                <li>Implement Jack Ganssle's debouncing algorithm or a 3-state FSM (<code>IDLE</code>, <code>DEBOUNCING</code>, <code>PRESSED_CONFIRMED</code>).</li>
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
                        <td class="p-2 border font-mono">TC7.1</td>
                        <td class="p-2 border">Actuate button 100 times firmly.</td>
                        <td class="p-2 border">Count increments from 0 to 100 cleanly.</td>
                        <td class="p-2 border">Exactly 100 counts logged; 0% error rate.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC7.2</td>
                        <td class="p-2 border">Inject 10 ms burst of electrical noise using wire scratching.</td>
                        <td class="p-2 border">Noise duration &lt; 30 ms stability threshold; rejected as glitch.</td>
                        <td class="p-2 border">Counter does not increment.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Inspect raw button signal on an oscilloscope while tapping vigorously. Observe the mechanical bounce train (often 5–15 ms). Ensure your debouncing threshold exceeds maximum observed bounce duration by at least a 2&times; safety margin.
        </div>
    </div>
</div>

<!-- Assignment 8 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 8: Non-Blocking Traffic Light with Pedestrian Crossing</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Red, Yellow, Green Traffic LEDs</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Red, Green Pedestrian LEDs</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Pedestrian Request Button</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Hardware Timer / FSM</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy08-traffic-light.svg" alt="STM32 Non-Blocking Traffic Light System Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.8: Deterministic multi-stage state machine with pedestrian override requests (PB0-PB4 LEDs, PC13 button).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Architect a municipal traffic junction controller as a deterministic Finite State Machine.
                Normal vehicular cycle: <strong>GREEN (10 s)</strong> &rarr; <strong>YELLOW (3 s)</strong> &rarr; <strong>RED (10 s)</strong>. Pedestrian lights remain RED during Green/Yellow vehicle phases.
                A pedestrian can press a "Cross Walk" request button at any time. When pressed, the controller latches a <code>pedestrian_request_pending</code> flag. The vehicle GREEN phase is curtailed to complete gracefully (guaranteeing at least 5 seconds of green before transitioning to yellow). Once vehicle traffic transitions to RED, the pedestrian light turns GREEN for 8 seconds, accompanied by a rapid acoustic beeper, followed by 3 seconds of flashing pedestrian green, before resuming vehicle traffic.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Vehicle LEDs (Red, Yellow, Green) and Pedestrian LEDs (Red, Green) transition strictly according to FSM rules.</li>
                <li>Zero illegal states (Vehicular GREEN and Pedestrian GREEN must never be illuminated simultaneously!).</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Strict non-blocking FSM using <code>switch (current_state)</code> evaluated in the main loop or timer callback.</li>
                <li>State durations timed with <code>HAL_GetTick()</code> delta values.</li>
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
                        <td class="p-2 border font-mono">TC8.1</td>
                        <td class="p-2 border">Uninterrupted operation (no pedestrian button).</td>
                        <td class="p-2 border">Vehicle cycle runs continuously: G(10s) -> Y(3s) -> R(10s). Pedestrian stays solid RED.</td>
                        <td class="p-2 border">Timings match &plusmn;50 ms precision.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC8.2</td>
                        <td class="p-2 border">Press pedestrian button during Vehicle GREEN at T = 2 s.</td>
                        <td class="p-2 border">System honors minimum 5s green, transitions to Yellow at T = 5 s, then vehicle Red + Pedestrian Green.</td>
                        <td class="p-2 border">Pedestrian request fulfilled safely; vehicle green not aborted dangerously early.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC8.3</td>
                        <td class="p-2 border">Safety Interlock Audit.</td>
                        <td class="p-2 border">Assert verification check: <code>!(Vehicle_Green &amp;&amp; Pedestrian_Green)</code>.</td>
                        <td class="p-2 border">Passes 100% of state space permutations.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Implement a software safety supervisor check inside your main loop: if both vehicle green and pedestrian green pins read HIGH simultaneously, immediately force all signals to flashing RED and lock the CPU in a fail-safe trap.
        </div>
    </div>
</div>

<!-- Assignment 9 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 9: PWM LED Dimmer with ADC Feedback</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">10kΩ Potentiometer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">LED (with resistor)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">ADC1 (12-bit)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Timer PWM Channel (e.g. TIM3_CH1)</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy09-pwm-dimmer.svg" alt="STM32 PWM LED Dimmer Wiring Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.9: Hardware PWM duty cycle generation modulated via 12-bit ADC conversion (PA0 wiper, PA5 PWM LED).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Create a flicker-free LED brightness dimmer. A 10 k&Omega; potentiometer connects across 3.3V and GND with its wiper tied to an analog ADC input pin (e.g., PA0 / ADC1_IN0). Configure a hardware timer to generate a 1 kHz PWM signal (above human flicker fusion threshold). The firmware periodically samples the 12-bit ADC (values 0 to 4095) and maps the analog reading linearly to the timer's PWM compare register (<code>CCR</code>), modulating the LED duty cycle seamlessly from 0.0% (completely off) to 100.0% (maximum brightness).
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Oscilloscope trace on PWM pin: Fixed 1.0 kHz frequency; pulse width modulates continuously from 0 µs to 1000 µs as the potentiometer dial rotates.</li>
                <li>Zero audible coil whine or optical pulsing.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Timer configuration: f<sub>CLK</sub> = 84 MHz, Prescaler = 83, ARR = 999 &rarr; f<sub>PWM</sub> = 84,000,000 / (84 &times; 1000) = 1000 Hz.</li>
                <li>Map 12-bit ADC range (0–4095) to ARR range (0–999):
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">CCR = (ADC_Value &times; 999) / 4095</div>
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
                        <td class="p-2 border font-mono">TC9.1</td>
                        <td class="p-2 border">Rotate potentiometer fully counter-clockwise (0.0V).</td>
                        <td class="p-2 border">ADC reads &lt; 10 counts; PWM duty cycle = 0.0%; LED completely dark.</td>
                        <td class="p-2 border">Oscilloscope shows flat 0.0V DC.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC9.2</td>
                        <td class="p-2 border">Rotate potentiometer to exact midpoint (~1.65V).</td>
                        <td class="p-2 border">ADC reads ~2048 counts; PWM duty cycle = 50.0%.</td>
                        <td class="p-2 border">Pulse width measured at 500 &plusmn; 10 &mu;s.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC9.3</td>
                        <td class="p-2 border">Rotate potentiometer fully clockwise (3.3V).</td>
                        <td class="p-2 border">ADC reads 4095 counts; PWM duty cycle = 100.0%; maximum brightness.</td>
                        <td class="p-2 border">Pulse width measured at 1000 µs (continuous 3.3V).</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Analog readings from breadboard potentiometers can pick up high-frequency noise. Implement a 4-sample or 8-sample moving average filter in software on the ADC buffer to prevent brightness jitter when the potentiometer is stationary.
        </div>
    </div>
</div>

<!-- Assignment 10 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 10: RGB LED Color Mixer with Master Dimmer</h3>
        <span class="px-2.5 py-1 bg-blue-100 text-blue-800 text-xs font-bold rounded-full">Level 1 · Beginner</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">RGB LED (Common Cathode)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">3x Current Resistors</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Push Button (Color Select)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Potentiometer (Master Brightness)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">3-Channel Timer PWM</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/diy10-rgb-color-mixer.svg" alt="STM32 RGB LED Color Mixer Wiring Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.10: Multi-channel timer PWM output synchronized with ADC brightness scaling (PB6/PB7/PB8 RGB, PA0 wiper, PC13 button).</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Synthesize arbitrary optical colors using a 3-channel PWM timer (e.g., TIM4 Channels 1, 2, and 3 driving the Red, Green, and Blue anodes through current-limiting resistors).
                Push button presses cycle through eight preset color palettes:
                <strong>Red</strong> &rarr; <strong>Green</strong> &rarr; <strong>Blue</strong> &rarr; <strong>Yellow</strong> (R+G) &rarr; <strong>Cyan</strong> (G+B) &rarr; <strong>Magenta</strong> (R+B) &rarr; <strong>White</strong> (R+G+B) &rarr; <strong>Warm Orange</strong>.
                Simultaneously, a 10 k&Omega; potentiometer on an ADC channel acts as a global master scaler, scaling all three RGB channels proportionally from 0% to 100% brightness while preserving color hue.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Three synchronized PWM channels with independent duty cycles.</li>
                <li>RGB module emits clear chromatic hues; potentiometer dial dims intensity smoothly without color shifts.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Math scaling equation for each color channel:
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        CCR<sub>x</sub> = (BaseColor<sub>x</sub> &times; MasterBrightness) / 255
                    </div>
                    <p class="text-xs text-slate-600 mt-1">where <code>BaseColor<sub>x</sub> &isin; [0, 255]</code> and <code>MasterBrightness &isin; [0, 255]</code>.</p>
                </li>
                <li>Use 32-bit integer arithmetic to prevent floating-point calculation overhead or precision loss in the periodic update loop.</li>
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
                        <td class="p-2 border font-mono">TC10.1</td>
                        <td class="p-2 border">Select Cyan preset (G=255, B=255, R=0). Master brightness = 100%.</td>
                        <td class="p-2 border">Green &amp; Blue channels output 100% PWM; Red channel outputs 0%.</td>
                        <td class="p-2 border">Vibrant cyan output observed; duty cycles verify on scope.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC10.2</td>
                        <td class="p-2 border">Dial potentiometer to 25% brightness on Yellow preset (R=255, G=255, B=0).</td>
                        <td class="p-2 border">Both Red &amp; Green PWM duty cycles scale down to exactly 25%; Blue remains 0%.</td>
                        <td class="p-2 border">Pure yellow hue maintained at low optical intensity.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Human eyes perceive brightness logarithmically, not linearly. If dimming looks abrupt near the bottom, implement a gamma correction lookup table: <code>const uint8_t gamma8[] = { ... };</code> to linearize perceived optical brightness.
        </div>
    </div>
</div>
"""
