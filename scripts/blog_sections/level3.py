"""
Level 3 Section: Displays, Distance and Motion (Assignments 21–30)
"""

def get_level3_section():
    return r"""
<h2 id="level-3" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-ruler-combined text-brand-coral mr-3"></i> 6. Level 3 — Displays, Distance and Motion (Assignments 21–30 · Intermediate)
</h2>

<p class="text-slate-700 leading-relaxed mb-6">
    Level 3 elevates your firmware into real-time robotics, mechatronics, and graphical instrumentation. You will master dual-edge timer input capture for microsecond sonic pulse measurement, H-bridge DC motor control with differential PWM, precise servo angle modulation, laser optics, hierarchical menu state machines, and battery-backed I2C RTC timekeeping.
</p>

<!-- Assignment 21 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 21: Precision Ultrasonic Distance Meter</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">HC-SR04 Ultrasonic Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">1kΩ + 2kΩ Resistors (5V->3.3V Divider)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Timer Input Capture (Dual-Edge)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 I2C OLED</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Measure acoustic time-of-flight from 2.0 cm to 400.0 cm with sub-millimeter precision. Every 100 ms, emit a 10 µs trigger pulse on the sensor's Trigger pin. When the HC-SR04 transmits its 40 kHz ultrasonic burst, its Echo pin rises HIGH and stays HIGH until the reflected echo returns. Capture the Echo pulse width using hardware <strong>Timer Input Capture</strong> on both rising and falling edges. Compute physical distance in centimeters and display on an OLED display: <code>"Distance: 45.2 cm"</code> along with an active distance bar graph.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED displays real-time distance updated at 10 Hz.</li>
                <li>Logic analyzer trace reveals 10 µs trigger pulse followed by Echo pulse width spanning 116 µs (2 cm) to 23.2 ms (400 cm).</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li><strong>Electrical Protection:</strong> HC-SR04 Echo pin outputs 5.0V TTL logic. <strong>You must place a 1 k&Omega; / 2 k&Omega; resistive voltage divider</strong> on the Echo pin to drop the signal to 3.3V before connecting to the STM32 timer pin!</li>
                <li>Sonic Distance Equation (at speed of sound c = 343 m/s = 0.0343 cm/&mu;s):
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        Distance (cm) = (&Delta;t (&mu;s) &times; 0.0343) / 2 = &Delta;t (&mu;s) / 58.3
                    </div>
                </li>
                <li>Zero polling delays! Use hardware input capture interrupt callbacks (<code>HAL_TIM_IC_CaptureCallback</code>).</li>
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
                        <td class="p-2 border font-mono">TC21.1</td>
                        <td class="p-2 border">Place flat cardboard target at exactly 10.0 cm (calibrated ruler).</td>
                        <td class="p-2 border">Echo pulse width captures at ~583 µs; OLED reads 10.0 cm.</td>
                        <td class="p-2 border">Measurement error within &plusmn;0.3 cm.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC21.2</td>
                        <td class="p-2 border">Move target to 100.0 cm.</td>
                        <td class="p-2 border">Echo pulse width captures at ~5830 µs; OLED reads 100.0 cm.</td>
                        <td class="p-2 border">Measurement error within &plusmn;1.0 cm.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC21.3</td>
                        <td class="p-2 border">Point sensor into open sky (no echo return).</td>
                        <td class="p-2 border">Timer capture times out at 30 ms; OLED displays <code>"OUT OF RANGE"</code>.</td>
                        <td class="p-2 border">System handles echo loss cleanly without locking timer.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Configure your capture timer prescaler so that the counter increments every 1.0 µs (e.g., Prescaler = 83 on an 84 MHz timer clock). Set the timer Auto-Reload Register (ARR) to <code>0xFFFF</code> (or <code>0xFFFFFFFF</code> for 32-bit timers like TIM2/TIM5) to prevent counter overflow during long echo returns.
        </div>
    </div>
</div>

<!-- Assignment 22 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 22: Automotive Parking Assistance System</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">HC-SR04 Ultrasonic Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Green, Yellow, Red LEDs</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Active/PWM Buzzer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Non-Blocking Rate Generator</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Simulate an automotive reverse parking sonar. As an obstacle approaches the rear bumper, modulate acoustic warning beeps and LED indicators through 4 distinct proximity zones:
                <strong>Zone 1 (Far: &gt; 80 cm):</strong> Green LED ON, Buzzer silent.
                <strong>Zone 2 (Caution: 40 cm to 80 cm):</strong> Yellow LED ON, Buzzer emits slow beeps (2 Hz repetition rate).
                <strong>Zone 3 (Warning: 15 cm to 40 cm):</strong> Yellow + Red LEDs ON, Buzzer emits fast beeps (6 Hz repetition rate).
                <strong>Zone 4 (Critical STOP: &lt; 15 cm):</strong> Red LED flashing, Buzzer sounds continuous solid tone (100% duty cycle).
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Buzzer audio pulse rate speeds up smoothly as distance shrinks.</li>
                <li>LED indicator bar transitions seamlessly between Green &rarr; Yellow &rarr; Red.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Buzzer pulsation must be driven by an asynchronous non-blocking software timer or timer PWM channel. Distance acquisition and beep toggling must never interfere with each other.</li>
                <li>Add a 2 cm hysteresis band to zone boundaries to eliminate auditory jitter when resting exactly at a zone transition (e.g. 40 cm).</li>
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
                        <td class="p-2 border font-mono">TC22.1</td>
                        <td class="p-2 border">Target at 120 cm (Zone 1).</td>
                        <td class="p-2 border">Green LED solid ON; Buzzer completely silent.</td>
                        <td class="p-2 border">Zero audible clicks or false pulses.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC22.2</td>
                        <td class="p-2 border">Target moves to 60 cm (Zone 2).</td>
                        <td class="p-2 border">Yellow LED lights; Buzzer beeps at exactly 2 Hz (250 ms on / 250 ms off).</td>
                        <td class="p-2 border">Beep period measured on scope at 500 &plusmn; 10 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC22.3</td>
                        <td class="p-2 border">Target moves to 25 cm (Zone 3).</td>
                        <td class="p-2 border">Buzzer beeps at 6 Hz (83 ms on / 83 ms off).</td>
                        <td class="p-2 border">Beep rate accelerates visibly.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC22.4</td>
                        <td class="p-2 border">Target moves to 8 cm (Zone 4).</td>
                        <td class="p-2 border">Red LED flashes; Buzzer sounds solid continuous tone without interruption.</td>
                        <td class="p-2 border">Continuous alarm tone maintained until obstacle backs away.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Ultrasonic sensors are susceptible to specular acoustic reflection at steep angles. Implement a 3-tap median filter on distance readings before feeding the zone selector to reject random lost-echo spikes.
        </div>
    </div>
</div>

<!-- Assignment 23 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 23: Multi-Channel IR Line Tracking Sensor Array</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">3x or 5x TCRT5000 IR Line Sensors</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">GPIO Input Bank</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">OLED / UART Visualizer</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Interface a 3-channel or 5-channel infrared reflective optical sensor array (TCRT5000). The sensors emit IR light and measure ground reflectance to detect high-contrast black tracks on white surfaces. Read the sensor states at 100 Hz, decode the spatial track alignment, and categorize the relative track position:
                <code>CENTERED</code> (middle sensor active), <code>SLIGHT_LEFT</code>, <code>HARD_LEFT</code>, <code>SLIGHT_RIGHT</code>, <code>HARD_RIGHT</code>, <code>CROSS_INTERSECTION</code> (all sensors black), and <code>TRACK_LOST</code> (all sensors white).
                Display the decoded position text and visual tracking cursor on an OLED display or serial console.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Real-time position string and graphical horizontal dot cursor indicating tape position.</li>
                <li>UART telemetry streaming decoded line status at 50 Hz.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Read all IR digital sensor pins in parallel using port bit masking (e.g., <code>uint8_t line_raw = (GPIOA-&gt;IDR &gt;&gt; 1) &amp; 0x07;</code>).</li>
                <li>Calculate an analog error weight for future PID line followers:
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        Error = &sum;((i - center) &times; S<sub>i</sub>) / &sum;(S<sub>i</sub>)
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
                        <td class="p-2 border font-mono">TC23.1</td>
                        <td class="p-2 border">Place track directly under center sensor.</td>
                        <td class="p-2 border">Sensors read <code>[0, 1, 0]</code>; Status reports <code>CENTERED (Error = 0)</code>.</td>
                        <td class="p-2 border">Cursor positioned in exact center of OLED.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC23.2</td>
                        <td class="p-2 border">Slide track to the left.</td>
                        <td class="p-2 border">Sensors read <code>[1, 0, 0]</code>; Status reports <code>HARD_LEFT</code>.</td>
                        <td class="p-2 border">Error weight evaluates to negative maximum.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC23.3</td>
                        <td class="p-2 border">Lift robot off track completely.</td>
                        <td class="p-2 border">Sensors read <code>[0, 0, 0]</code>; Status reports <code>TRACK_LOST</code>.</td>
                        <td class="p-2 border">System triggers track-lost fail-safe flag.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Calibrate each sensor's onboard trimpot under real ambient lighting conditions. Ensure that shiny black electrical tape is distinguishable from matte white printer paper, and maintain sensor height at 5 mm to 10 mm above the ground.
        </div>
    </div>
</div>

<!-- Assignment 24 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 24: Two-Sensor Autonomous Line-Following Robot</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x IR Line Sensors (Left, Right)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">L298N / TB6612 H-Bridge Motor Driver</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x Geared DC Motors (Chassis)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2-Channel PWM Timer (ENA, ENB)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">External 7.4V/9V Battery Pack</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Control a mobile robotic rover to track a closed loop black course. Two forward IR sensors straddle the black line:
                <strong>Both White (Line centered between sensors):</strong> Drive both Left and Right motors forward at baseline cruising speed (e.g., 60% PWM).
                <strong>Left Sensor on Black (Robot drifting right):</strong> Slow down or brake Left motor, boost Right motor PWM to pivot left.
                <strong>Right Sensor on Black (Robot drifting left):</strong> Slow down or brake Right motor, boost Left motor PWM to pivot right.
                <strong>Both Sensors on Black (Stop Mark / T-Junction):</strong> Cut motor drive to 0% PWM and stop robot smoothly.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Differential PWM duty cycles modulating dynamically between 30% and 80% on Timer channels.</li>
                <li>Vehicle tracks a curved figure-8 track without derailing or fishtailing uncontrollably.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li><strong>Electrical Isolation:</strong> Power the DC motors from an external battery pack through the H-bridge. Connect battery GND to STM32 GND. <strong>Never draw motor power from the STM32 board's 3.3V or 5V regulator!</strong></li>
                <li>PWM frequency set between 1 kHz and 10 kHz to ensure smooth motor torque without excessive acoustic switching whine.</li>
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
                        <td class="p-2 border font-mono">TC24.1</td>
                        <td class="p-2 border">Straight track: Both sensors detect white floor.</td>
                        <td class="p-2 border">Left PWM = 60%, Right PWM = 60%; Robot drives straight forward.</td>
                        <td class="p-2 border">Velocity vector is forward; deviation &lt; 2 cm per meter.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC24.2</td>
                        <td class="p-2 border">Sharp left curve: Left sensor hits black tape.</td>
                        <td class="p-2 border">Left PWM drops to 20%; Right PWM increases to 75%; Robot steers sharply left.</td>
                        <td class="p-2 border">Robot recovers track alignment within &lt; 200 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC24.3</td>
                        <td class="p-2 border">Cross horizontal black stop line: Both sensors detect black.</td>
                        <td class="p-2 border">Motor PWMs drop to 0%; H-bridge inputs configured for active dynamic brake.</td>
                        <td class="p-2 border">Robot halts within 5 cm of line crossing.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            If your robot shakes violently back and forth (hunting oscillation), the differential steering correction is too aggressive. Reduce the speed delta between inner and outer wheels, or implement smooth differential scaling rather than bang-bang on/off switching.
        </div>
    </div>
</div>

<!-- Assignment 25 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 25: Precision Servo Position Controller</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SG90 / MG996R Servomotor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">10kΩ Potentiometer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">12-bit ADC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Timer PWM (50 Hz / 20 ms Period)</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Synthesize standard RC servomotor control signals. Hobby servos require a strictly regulated 50 Hz PWM carrier (20.0 ms period), where the HIGH pulse duration dictates output horn shaft angle:
                <strong>1.0 ms HIGH:</strong> 0° (Full counter-clockwise).
                <strong>1.5 ms HIGH:</strong> 90° (Neutral / Center position).
                <strong>2.0 ms HIGH:</strong> 180° (Full clockwise).
                Read a 10 k&Omega; manual potentiometer via 12-bit ADC and map the readings linearly across the 0° to 180° range (1000 µs to 2000 µs pulse duration). The servo shaft must mirror the potentiometer knob position smoothly with zero hunting chatter.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Oscilloscope PWM trace: Fixed 20.0 ms period; pulse width smoothly varies from 1.0 ms to 2.0 ms.</li>
                <li>Servo horn tracks manual dial rotation smoothly through full 180° sweep.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Timer Clock Arithmetic: Prescale timer clock to 1 MHz (1 µs per tick). Set ARR = 19999 &rarr; Period = 20,000 &mu;s = 20 ms (50 Hz). Compare register (<code>CCR</code>) maps directly in microsecond units: <code>CCR &isin; [1000, 2000]</code>.</li>
                <li>Power the servo motor from an external 5V source (drawing up to 500 mA under stall); never supply servo power from the STM32 3.3V rail.</li>
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
                        <td class="p-2 border font-mono">TC25.1</td>
                        <td class="p-2 border">Turn pot fully counter-clockwise (ADC &asymp; 0).</td>
                        <td class="p-2 border">PWM pulse width = 1000 µs (1.00 ms); Servo positions at 0°.</td>
                        <td class="p-2 border">Angle measured at 0 &plusmn; 2&deg; on protractor.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC25.2</td>
                        <td class="p-2 border">Turn pot to center position (ADC &asymp; 2048).</td>
                        <td class="p-2 border">PWM pulse width = 1500 µs (1.50 ms); Servo positions at 90°.</td>
                        <td class="p-2 border">Angle measured at 90 &plusmn; 2&deg;.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC25.3</td>
                        <td class="p-2 border">Turn pot fully clockwise (ADC = 4095).</td>
                        <td class="p-2 border">PWM pulse width = 2000 µs (2.00 ms); Servo positions at 180°.</td>
                        <td class="p-2 border">Angle measured at 180 &plusmn; 2&deg;.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Budget servos often emit an annoying continuous buzzing sound when stationary. This is caused by noisy ADC readings jittering the pulse width between 1498 µs and 1502 µs. Implement a software deadband: only update <code>CCR</code> if the new pulse width differs from the current pulse width by more than 5 µs.
        </div>
    </div>
</div>

<!-- Assignment 26 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 26: Autonomous Obstacle-Avoidance Robot</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">HC-SR04 Ultrasonic Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SG90 Pan-Tilt Servo (Optional Radar)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Dual H-Bridge Motor Driver</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Chassis + 2x DC Motors</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Reactive Navigation FSM</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Architect an autonomous roving robot that roams an arena without colliding into walls. The robot advances forward in <code>STATE_FORWARD</code> while monitoring forward distance. When an obstacle is detected within 25.0 cm:
                (1) Immediately halt motors (<code>STATE_STOP</code> for 200 ms).
                (2) Back up slightly for 400 ms (<code>STATE_REVERSE</code>).
                (3) Pivot right or scan left/right to evaluate clearest escape corridor (<code>STATE_TURN</code>).
                (4) Resume forward propulsion once clear distance exceeds 40.0 cm.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Vehicle navigates a cluttered room continuously for 5 minutes with zero wall collisions.</li>
                <li>UART telemetry broadcasts FSM state transitions: <code>"CRUISING -&gt; OBSTACLE_DETECTED (18cm) -&gt; BACKUP -&gt; PIVOT_RIGHT -&gt; RESUME"</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Structured non-blocking FSM. Motor run-times during reverse and turn phases must be controlled via <code>HAL_GetTick()</code> timers.</li>
                <li>Fail-safe timeout: If trapped in a corner where all directions are blocked (&lt; 20 cm), execute a 180-degree emergency spin.</li>
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
                        <td class="p-2 border font-mono">TC26.1</td>
                        <td class="p-2 border">Open path (&gt; 50 cm clear).</td>
                        <td class="p-2 border">Robot moves forward steadily at cruising speed.</td>
                        <td class="p-2 border">Forward velocity maintained.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC26.2</td>
                        <td class="p-2 border">Place barrier 20 cm in front of advancing robot.</td>
                        <td class="p-2 border">Robot halts within 5 cm, reverses for 400 ms, turns 90° right, and moves forward.</td>
                        <td class="p-2 border">Zero physical contact with barrier.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Rapidly reversing DC motors creates severe inductive electrical spikes on power rails that can reset the STM32 microcontroller. Ensure a large electrolytic bulk decoupling capacitor (470 µF to 1000 µF) is soldered across the battery input terminals of your motor driver board.
        </div>
    </div>
</div>

<!-- Assignment 27 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 27: Laser Tripwire Perimeter Alarm</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">650nm 5mW Red Laser Diode</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Photodiode / LDR in Light Tube</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">ADC or Comparator GPIO</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Latching Alarm Siren + Disarm Button</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement an optical perimeter intrusion security system. A low-power eye-safe red laser module targets a photosensor located 1 to 5 meters away across a doorway. While the laser beam continuously illuminates the photosensor, the system is in <code>ARMED_SECURE</code> mode. If an intruder breaks the laser beam for even a fraction of a millisecond (e.g. t<sub>break</sub> &ge; 10 ms), the system immediately latches into <code>ALARM_TRIPPED</code> mode: pulsing a high-decibel siren and illuminating a strobe LED. The alarm must remain permanently latched even after the beam is restored, until an authenticated disarm sequence is entered via push button.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Beam break trips alarm siren in &lt; 2 milliseconds.</li>
                <li>Alarm remains actively sounding indefinitely until disarm button code is pressed.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li><strong>Laser Safety:</strong> Use strictly Class 1 or Class 2 low-power laser diodes (maximum optical output &lt; 5 mW at 650 nm). Never stare directly into the beam path or point the laser at human or animal eyes.</li>
                <li>Enclose the photo-receiver in a 3 cm opaque tube/shroud to shield it from ambient room lighting.</li>
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
                        <td class="p-2 border font-mono">TC27.1</td>
                        <td class="p-2 border">Align laser onto detector; arm system.</td>
                        <td class="p-2 border">System confirms alignment; status LED solid Green; Siren silent.</td>
                        <td class="p-2 border">Stable armed state.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC27.2</td>
                        <td class="p-2 border">Pass hand rapidly through laser beam (15 ms obstruction).</td>
                        <td class="p-2 border">Instant alarm trigger; Siren sounds; Red strobe flashes.</td>
                        <td class="p-2 border">Response time &lt; 5 ms; latched in alarm mode.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC27.3</td>
                        <td class="p-2 border">Beam restored to detector while siren is active.</td>
                        <td class="p-2 border">Siren continues sounding; intrusion state remains permanently latched.</td>
                        <td class="p-2 border">Verification of tamper latching.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC27.4</td>
                        <td class="p-2 border">Press disarm sequence.</td>
                        <td class="p-2 border">Alarm silenced; system returns to armed state if beam is intact.</td>
                        <td class="p-2 border">Disarm logic verified.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            To make the tripwire immune to optical bypass (where an intruder shines a pocket flashlight into the detector to fool it while breaking the laser), modulate the laser at a carrier frequency (e.g. 1 kHz) and verify the 1 kHz frequency signature on the receiver in software!
        </div>
    </div>
</div>

<!-- Assignment 28 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 28: Interactive OLED Hierarchical Menu System</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 128x64 I2C OLED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">3x Navigation Buttons (Up, Down, Select)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Hierarchical UI State Machine</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a multi-level graphical menu interface. Using three buttons (UP, DOWN, SELECT), navigate a structured setting hierarchy:
                <code>Root Menu</code> &rarr; <code>[1. Sensor Dashboard]</code>, <code>[2. Threshold Config]</code>, <code>[3. System Settings]</code>.
                Selecting <code>[2. Threshold Config]</code> opens sub-menus allowing the user to view and adjust operational parameters (e.g., Temperature Alarm Threshold from 20°C to 50°C, Buzzer Volume ON/OFF, Screen Brightness). Values are edited in real time on the screen and retained in SRAM memory structures.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED displays highlighted inverted selection bar moving smoothly between menu items.</li>
                <li>Entering edit mode displays inline value adjustment arrows: <code>"Temp Limit: [&lt; 38 C &gt;]"</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Architect the menu tree using a linked-struct or table-driven model:
                    <pre class="bg-slate-900 text-slate-100 p-2 rounded text-xs font-mono my-1"><code>typedef struct MenuItem {
    const char *title;
    struct MenuItem *parent;
    struct MenuItem *children;
    uint8_t child_count;
    void (*action_callback)(void);
} MenuItem_t;</code></pre>
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
                        <td class="p-2 border font-mono">TC28.1</td>
                        <td class="p-2 border">Press DOWN button on root menu.</td>
                        <td class="p-2 border">Selection cursor advances from Item 1 to Item 2.</td>
                        <td class="p-2 border">Cursor renders cleanly; no display tearing.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC28.2</td>
                        <td class="p-2 border">Press SELECT on "Threshold Config".</td>
                        <td class="p-2 border">Screen redraws with sub-menu list.</td>
                        <td class="p-2 border">Submenu navigation depth tracked properly.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC28.3</td>
                        <td class="p-2 border">Edit value from 35 to 40 and press SELECT to save.</td>
                        <td class="p-2 border">Global parameter updates to 40; system returns to parent menu.</td>
                        <td class="p-2 border">New threshold is active in firmware control loops.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Redraw the OLED buffer only when an input event actually occurs or a dynamic telemetry value updates. Continuously redrawing static text at full speed wastes CPU cycles and saturates the I2C bus unnecessarily.
        </div>
    </div>
</div>

<!-- Assignment 29 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 29: RTC-Based Alarm Clock (DS3231)</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">DS3231 High-Precision I2C RTC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Active Alarm Buzzer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x Push Buttons (Set Time / Snooze)</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build a standalone digital desk clock with a battery-backed real-time clock. Communicate with the DS3231 RTC over I2C to read timestamp registers (Seconds, Minutes, Hours, Day, Date, Month, Year in BCD format). Display real-world clock time on the OLED: <code>"14:35:08"</code> and date <code>"2026-10-29"</code>. Provide user buttons to set an acoustic alarm time (e.g. <code>"Alarm: 07:00:00"</code>). When the RTC time matches the programmed alarm registers, trigger the buzzer. A snooze button silences the alarm for 5 minutes.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED renders crisp digital clock updating every 1.0 second.</li>
                <li>Buzzer sounds immediately on the designated second of alarm match.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Convert between Binary Coded Decimal (BCD) and binary integers:
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        bin = ((bcd &gt;&gt; 4) * 10) + (bcd &amp; 0x0F)<br>
                        bcd = ((bin / 10) &lt;&lt; 4) | (bin % 10)
                    </div>
                </li>
                <li>Leverage the DS3231 hardware interrupt pin (<code>/INT/SQW</code>) configured to fire on internal alarm match, saving the STM32 from continuous polling.</li>
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
                        <td class="p-2 border font-mono">TC29.1</td>
                        <td class="p-2 border">Run clock for 60 seconds.</td>
                        <td class="p-2 border">Clock seconds advance cleanly from 00 to 59, rolling over to next minute.</td>
                        <td class="p-2 border">Seconds match reference atomic clock &plusmn;0 s.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC29.2</td>
                        <td class="p-2 border">Set alarm for current time + 1 minute. Wait.</td>
                        <td class="p-2 border">At the exact second match, buzzer trips and "ALARM WAKE UP" flashes on OLED.</td>
                        <td class="p-2 border">Trip occurs on exact matching second.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC29.3</td>
                        <td class="p-2 border">Unplug USB power from STM32 for 10 seconds, then reconnect.</td>
                        <td class="p-2 border">DS3231 coin-cell preserves time; clock resumes with correct current time.</td>
                        <td class="p-2 border">Zero time loss across microcontroller reboot.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            The DS3231 I2C address is fixed at <code>0xD0</code> (7-bit address <code>0x68</code>). Remember that many DS3231 modules include an onboard 24C32 EEPROM chip on the same I2C lines at address <code>0x57</code>.
        </div>
    </div>
</div>

<!-- Assignment 30 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 30: Autonomous Environmental Data Logger</h3>
        <span class="px-2.5 py-1 bg-purple-100 text-purple-800 text-xs font-bold rounded-full">Level 3 · Intermediate</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Temperature Sensor (LM35 / DS18B20)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">LDR Light Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">DS3231 RTC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">External SPI Flash (W25Qxx) or MicroSD</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART CLI</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build a self-contained environmental blackbox data logger. Every 10.0 seconds, sample ambient temperature, light level, and timestamp from the RTC. Package the measurement into a structured 16-byte binary record or CSV line:
                <code>"2026-10-29 14:35:10, 24.5C, 680 Lux"</code>. Append the record into external non-volatile SPI NOR Flash or an SD Card FAT32 filesystem. Provide a UART serial command line interface: typing <code>"DUMP"</code> dumps all logged records to the PC terminal; typing <code>"CLEAR"</code> safely erases the log memory sector.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Data records reliably logged and preserved across system power down and hard reset.</li>
                <li>UART command <code>"DUMP"</code> outputs clean tabular logs suitable for Excel/Python plotting.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Handle flash write boundaries: NOR flash must be erased in 4 KB sectors before writing, and programmed in 256-byte pages. Maintain an in-RAM write buffer.</li>
                <li>Format corrupt-proof records: Include a leading synchronization magic word (<code>0xAA55</code>) and a trailing 16-bit CRC checksum on every log packet.</li>
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
                        <td class="p-2 border font-mono">TC30.1</td>
                        <td class="p-2 border">Log data for 1 minute (6 records).</td>
                        <td class="p-2 border">Records written with sequential timestamps spaced exactly 10s apart.</td>
                        <td class="p-2 border">Timestamps advance by 10s &plusmn;0 s.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC30.2</td>
                        <td class="p-2 border">Power down device; wait 30 seconds; power up. Send <code>"DUMP"</code>.</td>
                        <td class="p-2 border">Pre-shutdown records are fully retrieved intact with valid CRCs.</td>
                        <td class="p-2 border">Zero data loss across power interruption.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC30.3</td>
                        <td class="p-2 border">Send <code>"CLEAR"</code> command over UART.</td>
                        <td class="p-2 border">Flash sector erased; record pointer resets to index 0.</td>
                        <td class="p-2 border">Log memory wiped cleanly.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            When buffering records in RAM before writing to external flash, ensure that the write triggers when the buffer reaches page size (256 bytes) or when an explicit flush timeout expires, so you never lose the trailing records when logging ends.
        </div>
    </div>
</div>
"""
