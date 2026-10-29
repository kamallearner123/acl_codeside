"""
Level 5 Section: Integrated Embedded Systems (Assignments 41–50)
"""

def get_level5_section():
    return r"""
<h2 id="level-5" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-industry text-brand-coral mr-3"></i> 8. Level 5 — Integrated Embedded Systems (Assignments 41–50 · Advanced)
</h2>

<p class="text-slate-700 leading-relaxed mb-6">
    Level 5 represents capstone systems engineering. Individual peripheral drivers are orchestrated into full-scale integrated embedded products featuring multi-sensor data fusion, circular DMA pipelines, safety-critical watchdog supervision, industrial fault logging, fail-safe state machines, and vehicular CAN clusters.
</p>

<!-- Assignment 41 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 41: Autonomous Smart Room Environmental &amp; Occupancy Controller</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">PIR Motion Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">LDR Ambient Light Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Temperature Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">DS3231 RTC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Lighting Relay Driver</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Architect a commercial building automation controller. Integrate occupancy detection (PIR), ambient light (LDR), thermal metrics (LM35/DS18B20), and time-of-day scheduling (DS3231 RTC).
                <strong>Operational Rules:</strong>
                (1) Lighting is actuated only if human occupancy is detected AND ambient light falls below the twilight threshold.
                (2) During scheduled night hours (23:00 to 06:00), lighting is forced into low-intensity night-light mode with a 1-minute timeout.
                (3) If temperature exceeds 28.0°C during occupied periods, activate a ventilation cooling fan.
                (4) When unoccupied for &gt; 10 minutes, enter energy-saving standby (extinguishing lighting and climate systems).
                Display system status, current time, occupancy countdown, and environmental readings on the OLED dashboard.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Automated relay and PWM outputs modulating in accordance with environmental and schedule rules.</li>
                <li>OLED status dashboard updating smoothly at 2 Hz.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Structure the application using a hierarchical state machine (<code>ROOM_UNOCCUPIED</code>, <code>ROOM_ACTIVE_DAY</code>, <code>ROOM_ACTIVE_NIGHT</code>, <code>EMERGENCY_OVERHEAT</code>).</li>
                <li>Non-blocking design: no delay calls anywhere in execution tasks.</li>
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
                        <td class="p-2 border font-mono">TC41.1</td>
                        <td class="p-2 border">Daytime (14:00), Room dark (curtains closed), PIR motion detected.</td>
                        <td class="p-2 border">Relay activates full lighting; 10-minute occupancy timer begins.</td>
                        <td class="p-2 border">Lighting turns ON within &lt; 50 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC41.2</td>
                        <td class="p-2 border">Night schedule (02:00), PIR motion detected.</td>
                        <td class="p-2 border">Low-intensity night-light PWM activates; timeout set to 60s.</td>
                        <td class="p-2 border">Full lighting inhibited; night schedule enforced.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC41.3</td>
                        <td class="p-2 border">Room vacated; no motion for 10 minutes.</td>
                        <td class="p-2 border">Occupancy timer expires; lighting and climate fans shut down cleanly.</td>
                        <td class="p-2 border">Energy savings mode confirmed.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Separate timebase generation from display updates. Update sensor acquisition at 10 Hz, FSM logic at 5 Hz, and OLED rendering at 2 Hz using delta-tick counters to maintain balanced CPU utilization.
        </div>
    </div>
</div>

<!-- Assignment 42 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 42: Industrial Machine Health &amp; Condition Monitoring Unit</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">MPU6050 3-Axis Accelerometer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Temperature Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">W25Qxx SPI Flash / EEPROM</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">OLED + Alert Siren</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Simulate an industrial rotating machinery predictive health monitor (ISO 10816 vibration severity standard). Sample 3-axis acceleration at 200 Hz. Compute Root-Mean-Square (RMS) vibration velocity and peak acceleration shock:
                <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                    A<sub>RMS</sub> = sqrt( (1 / N) &times; &sum;(a<sub>i</sub> - a<sub>mean</sub>)<sup>2</sup> )
                </div>
                Simultaneously monitor bearing surface temperature. If vibration RMS exceeds 4.5 mm/s or temperature crosses 65°C, log a timestamped fault event packet into external SPI flash (including timestamp, peak vibration vector, and temperature), sound a warning buzzer, and display an industrial maintenance alert on the OLED.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED displays real-time RMS vibration and bearing temperature metrics.</li>
                <li>Flash memory stores permanent blackbox event log of all fault excursions.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Compute RMS across a 128-sample sliding window buffer. Use the hardware Floating Point Unit (FPU) on Cortex-M4F (NUCLEO-F446RE) for fast hardware square root: <code>arm_sqrt_f32()</code>.</li>
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
                        <td class="p-2 border font-mono">TC42.1</td>
                        <td class="p-2 border">Healthy baseline machine vibration (A<sub>RMS</sub> &lt; 1.5 mm/s).</td>
                        <td class="p-2 border">Status reads "HEALTHY / ZONE A"; Green LED active; No flash writes.</td>
                        <td class="p-2 border">Nominal operation confirmed.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC42.2</td>
                        <td class="p-2 border">Inject mechanical shake causing RMS to exceed 4.5 mm/s.</td>
                        <td class="p-2 border">Alarm trips; Red LED flashes; Event record appended to SPI flash.</td>
                        <td class="p-2 border">Fault packet logged with valid timestamp &lt; 100 ms.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Enable the Cortex-M4 hardware FPU in your compiler settings (<code>-mfloat-abi=hard -mfpu=fpv4-sp-d16</code>). Hardware single-precision floating-point instructions execute math in single clock cycles, preventing mathematical calculations from starving interrupt service routines.
        </div>
    </div>
</div>

<!-- Assignment 43 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 43: Smart Door Access Security System</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">TTP223 Touch Pad / 4x4 Keypad</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">PIR Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">DS3231 RTC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Door Solenoid / Relay</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Security Siren</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a secure physical access control panel. Users authenticate by entering a 4-digit PIN code via keypad or sequential touch button patterns.
                <strong>Security Rules:</strong>
                (1) If PIN is correct: Unlock door solenoid for 5.0 seconds (Green LED + Welcome Chime), then auto-relock.
                (2) If incorrect PIN entered 3 consecutive times: Lock panel for a 60-second penalty lockout; sound security warning beep.
                (3) If PIR motion detects approach while armed after curfew hours (22:00 to 05:00 via RTC) without PIN entry within 15 seconds, sound an unauthorized loitering alert.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Solenoid lock activates and releases precisely on schedule.</li>
                <li>Penalty lockout timer visibly counts down on OLED display.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Store default PIN in non-volatile flash or backup registers to resist power cuts.</li>
                <li>Implement constant-time PIN comparison to protect against timing-attack side-channel probing.</li>
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
                        <td class="p-2 border font-mono">TC43.1</td>
                        <td class="p-2 border">Enter correct PIN (<code>1-2-3-4</code>).</td>
                        <td class="p-2 border">Solenoid unlocks; Green LED lights; Relocks after exactly 5.0s.</td>
                        <td class="p-2 border">Unlock duration 5.00 &plusmn; 0.05 s.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC43.2</td>
                        <td class="p-2 border">Enter wrong PIN 3 times sequentially.</td>
                        <td class="p-2 border">System enters LOCKOUT; keypad ignored; 60s countdown begins.</td>
                        <td class="p-2 border">Zero access permitted during 60s lockout.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Always use a flyback diode across the door solenoid coil. Electric door strikes generate massive 200V+ inductive back-EMF spikes when de-energized that can instantly reboot or fry the microcontroller.
        </div>
    </div>
</div>

<!-- Assignment 44 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 44: Automated Water Reservoir Pump Controller</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">HC-SR04 Ultrasonic Sensor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">OLED Display</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">5V Relay Module (Pump Driver)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Safety Float Switch / Dry-Run Sensor</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build an automated industrial water tank filling controller. An overhead ultrasonic sensor measures the air gap down to the water surface, calculating current liquid volume percentage (0% to 100%).
                <strong>Automated Pump Rules:</strong>
                (1) When water level drops below 20.0% (Low Limit), turn the water pump relay ON.
                (2) Keep the pump pumping continuously until the water level reaches 90.0% (High Limit), then turn the pump OFF.
                (3) <strong>Dry-Run Protection:</strong> If the pump runs for more than 5 minutes without water level increasing by at least 2%, declare <code>"DRY RUN FAULT: SUPPLY EMPTY"</code>, cut pump power instantly, and latch an audible fault siren.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED displays tank level bar graph, percentage, and pump status (IDLE / PUMPING / FAULT).</li>
                <li>Relay contacts click firmly to switch pump motor.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Implement a 5-second sensor low-pass filter to reject water ripples and splashing surface disturbances.</li>
                <li>Dry-run safety timer must be backed by a hardware watchdog.</li>
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
                        <td class="p-2 border font-mono">TC44.1</td>
                        <td class="p-2 border">Water level drops to 19% (below 20% mark).</td>
                        <td class="p-2 border">Pump relay turns ON; status updates to PUMPING.</td>
                        <td class="p-2 border">Relay activates &lt; 500 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC44.2</td>
                        <td class="p-2 border">Water rises through 50% (intermediate band).</td>
                        <td class="p-2 border">Pump remains ON; hysteresis band honored.</td>
                        <td class="p-2 border">No short-cycling.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC44.3</td>
                        <td class="p-2 border">Water level reaches 90%.</td>
                        <td class="p-2 border">Pump relay cuts OFF immediately; status updates to FULL.</td>
                        <td class="p-2 border">Relay de-energizes smoothly.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Standard ultrasonic sensors cannot measure closer than 2 cm (blind zone). Position the HC-SR04 transducer at least 10 cm above the absolute maximum overflow lip of the tank.
        </div>
    </div>
</div>

<!-- Assignment 45 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 45: Precision Smart Energy &amp; Power Profiler</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">INA219 I2C Current/Power Sensor (or Dual ADC Shunt)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Load Resistors / Small DC Motor</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART Telemetry Stream</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build a real-time DC energy meter. Sample bus voltage V (Volts) and shunt current I (Milliamperes) at 100 Hz. Integrate energy consumption over time:
                <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                    P = V &times; I  (Milliwatts)<br>
                    E = &sum;(P<sub>i</sub> &times; &Delta;t)  (Milliwatt-hours or Joules)
                </div>
                Render real-time Voltage, Current, Instantaneous Power, and Cumulative Energy on the OLED display. Transmit a high-speed telemetry packet over UART at 10 Hz for live plotting on a PC oscilloscope / graphing utility.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED shows: <code>"V: 5.02V | I: 235mA | P: 1.18W | E: 0.42Wh"</code>.</li>
                <li>UART outputs comma-separated values: <code>timestamp_ms,voltage_v,current_ma,power_mw</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Maintain numerical accuracy: Accumulate energy using 64-bit integer arithmetic (microjoules) to prevent floating-point accumulation truncation over long logging sessions.</li>
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
                        <td class="p-2 border font-mono">TC45.1</td>
                        <td class="p-2 border">Connect precision 10 &Omega; load on 5.0V rail (I = 500 mA).</td>
                        <td class="p-2 border">Measured voltage = 5.0V &plusmn;0.05V; current = 500 mA &plusmn;5 mA.</td>
                        <td class="p-2 border">Error &lt; 1.0% compared to bench reference meter.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC45.2</td>
                        <td class="p-2 border">Run load for exactly 1.00 hour.</td>
                        <td class="p-2 border">Accumulated energy calculates to 2.50 Wh (5V &times; 0.5A &times; 1h).</td>
                        <td class="p-2 border">Energy accumulator matches within &plusmn;1.5%.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Calibrate the INA219 shunt calibration register correctly for your specific shunt resistor (typically 0.1 &Omega;). Verify that the common-mode voltage on the shunt does not exceed 26V (the maximum input rating of the INA219).
        </div>
    </div>
</div>

<!-- Assignment 46 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 46: Integrated Automotive Reverse-Assist Cockpit</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">HC-SR04 Ultrasonic Sonar</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED (Rearview Graphic)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">PWM Buzzer (Progressive Tone)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">3x Proximity LEDs (G, Y, R)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Reverse Gear Switch Input</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Deliver a full production prototype of an automotive reverse parking assistant.
                (1) The subsystem activates only when a "Reverse Gear" toggle switch is engaged (simulating vehicle gear interlock).
                (2) When in reverse, pulse the ultrasonic sonar at 15 Hz.
                (3) Render a bird's-eye vehicle graphic on the OLED with animated curved distance arcs closing in toward the bumper as the vehicle approaches a wall.
                (4) Modulate acoustic warning beeps continuously from 1 Hz (at 150 cm) to 12 Hz (at 20 cm), transitioning to a continuous solid 2.5 kHz alarm tone when d &lt; 15 cm.
                (5) When reverse gear is disengaged, shut down display, extinguish LEDs, and mute the buzzer immediately.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Animated OLED graphical dashboard with dynamic distance arcs.</li>
                <li>Seamless acoustic feedback scaling synchronously with obstacle proximity.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Reverse engage/disengage handled via EXTI or debounced input, enabling/disabling the sonar task instantly.</li>
                <li>Graphic rendering must use double-buffered RAM canvas and DMA to prevent screen flickering.</li>
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
                        <td class="p-2 border font-mono">TC46.1</td>
                        <td class="p-2 border">Reverse switch OFF.</td>
                        <td class="p-2 border">System in standby; OLED dark; Buzzer silent; Sonar inactive.</td>
                        <td class="p-2 border">Zero power dissipation on transducers.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC46.2</td>
                        <td class="p-2 border">Engage Reverse switch; obstacle at 45 cm.</td>
                        <td class="p-2 border">OLED lights instantly; displays vehicle graphic; Buzzer pulses at 5 Hz; Yellow LED active.</td>
                        <td class="p-2 border">System boots &lt; 100 ms after gear engagement.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC46.3</td>
                        <td class="p-2 border">Approach to 10 cm; then disengage Reverse switch mid-alarm.</td>
                        <td class="p-2 border">Solid alarm sounds at 10 cm; instantly silenced when switch flipped OFF.</td>
                        <td class="p-2 border">Immediate muting &lt; 5 ms.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            To create an authentic automotive sound, modulate the buzzer tone with a dual-frequency sweep (e.g. 2400 Hz for 40 ms, followed by silence) rather than a crude square wave.
        </div>
    </div>
</div>

<!-- Assignment 47 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 47: CAN-Based Vehicle Digital Instrument Cluster</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">STM32 CAN1 (500 kbps)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">CAN Transceiver (SN65HVD230)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED (Dashboard UI)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Warning LEDs (MIL / Check Engine, Oil)</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Build an automotive electronic instrument cluster. Connect to a simulated CAN vehicle powertrain bus. Receive and parse high-speed CAN frames:
                <strong>CAN ID 0x100:</strong> Vehicle Speed (0–240 km/h) and Engine RPM (0–8000 RPM).
                <strong>CAN ID 0x101:</strong> Engine Coolant Temperature (C) and Fuel Level (0–100%).
                <strong>CAN ID 0x102:</strong> Fault Warning Bits (Check Engine MIL, Low Oil Pressure, Door Ajar).
                Render a realistic dashboard cluster on the OLED: digital speedometer digits, an arc gauge for Engine RPM, and warning icon banners. Drive external physical warning LEDs when critical error bits are asserted in CAN frames.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED speedometer and tachometer needle animate smoothly at 20 frames per second.</li>
                <li>Check Engine LED illuminates within &lt; 10 ms of receiving fault frame <code>0x102</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Handle message loss: If vehicle speed frames (ID 0x100) cease arriving for &gt; 1000 ms, flash a "CAN BUS COMM FAILURE" warning and set speedometer to <code>"--- km/h"</code>.</li>
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
                        <td class="p-2 border font-mono">TC47.1</td>
                        <td class="p-2 border">Inject CAN ID 0x100 with Speed = 85 km/h, RPM = 3200.</td>
                        <td class="p-2 border">Speedometer renders "85 KM/H"; RPM bar extends to 40% full scale.</td>
                        <td class="p-2 border">Values accurately decoded.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC47.2</td>
                        <td class="p-2 border">Inject CAN ID 0x102 with Bit 0 set (Check Engine MIL).</td>
                        <td class="p-2 border">Physical Yellow MIL LED lights up; "CHECK ENGINE" icon renders on OLED.</td>
                        <td class="p-2 border">Warning assertion verified.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Automotive signals often use scaling factors and offsets (e.g. Speed = Raw &times; 0.1 km/h; Temp = Raw - 40&deg;C). Ensure your decoding formulas match DBC database specifications exactly.
        </div>
    </div>
</div>

<!-- Assignment 48 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 48: High-Throughput Event-Driven DMA Data Acquisition (DAQ)</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Multi-Channel ADC (3 Analog Channels)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Hardware Timer TRGO Trigger</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Direct Memory Access (DMA2 Stream 0)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Zero-Copy Double Buffering</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART DMA Telemetry</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a deterministic data acquisition pipeline with <strong>zero CPU intervention during sampling</strong>. Configure a hardware timer (TIM2) to trigger ADC conversions automatically via Trigger Output (TRGO) at an exact 10.0 kHz rate across three scan channels (potentiometer, analog sensor, internal temperature). Direct Memory Access (DMA) transfers converted 16-bit values into a circular RAM ping-pong buffer (512 samples per buffer).
                When the first half-buffer fills (<code>HAL_ADC_ConvHalfCpltCallback</code>), the CPU processes and packetizes Buffer A. While the CPU processes Buffer A, DMA silently fills Buffer B. When the full buffer completes (<code>HAL_ADC_ConvCpltCallback</code>), process Buffer B while DMA wraps to Buffer A. Stream the processed blocks out via UART DMA without ever stalling sampling.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Deterministic 10 kHz multi-channel sampling with 0% CPU overhead spent waiting on ADC conversion registers.</li>
                <li>Zero sample drops or buffer overwrites verified across continuous 1-hour stress testing.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>ADC configured in <code>Scan Conversion Mode</code> and <code>Continuous Mode DISABLED</code> (paced strictly by timer TRGO).</li>
                <li>DMA configured in <code>Circular Mode</code> with <code>Data Width: Half Word (16-bit)</code>.</li>
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
                        <td class="p-2 border font-mono">TC48.1</td>
                        <td class="p-2 border">Run DAQ pipeline; observe DMA interrupt rate.</td>
                        <td class="p-2 border">Half-transfer and full-transfer callbacks fire with mathematical regularity.</td>
                        <td class="p-2 border">Callback interval matches exact calculated buffer filling period &plusmn;0.1%.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC48.2</td>
                        <td class="p-2 border">Measure CPU loading during 10 kHz 3-channel acquisition.</td>
                        <td class="p-2 border">CPU utilization stays under 15% (measured via idle-task counter).</td>
                        <td class="p-2 border">85%+ CPU cycles available for foreground processing.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Ensure your buffer array is declared with the <code>volatile</code> qualifier or appropriate cache-cleaning operations on Cortex-M7 platforms with D-Cache enabled, so the CPU reads updated DMA RAM contents rather than stale cached copies!
        </div>
    </div>
</div>

<!-- Assignment 49 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 49: Mission-Critical Fault Detection, Watchdog &amp; Fail-Safe Recovery</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Independent Watchdog (IWDG)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Window Watchdog (WWDG)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Flash Blackbox Memory</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Fault Injection Buttons</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Safe-State Relay Actuator</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Architect a high-reliability firmware supervisor conforming to IEC 61508 / ISO 26262 functional safety concepts. Implement:
                (1) <strong>Independent Hardware Watchdog (IWDG):</strong> Driven by its dedicated Low-Speed Internal (LSI) RC oscillator (32 kHz). The main task must refresh ("kick") the IWDG within a 1000 ms window. If a software freeze or infinite loop occurs, the hardware watchdog forces a cold chip reset.
                (2) <strong>Reset Reason Identification:</strong> On boot, inspect the Reset and Clock Control (<code>RCC-&gt;CSR</code>) register to identify whether the reboot was caused by a Power-On Reset (POR), External Pin Reset (PINRST), or Watchdog Timeout (IWDGRST). Log the fault code into non-volatile memory.
                (3) <strong>Fail-Safe State:</strong> If a task hangs or a critical sensor disconnects, immediately trip all actuator relays to their de-energized, safe resting state.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Firmware reliably survives deliberate software locks by auto-rebooting and resuming safe operations within 1.0 second.</li>
                <li>UART terminal prints exact boot cause: <code>"BOOT: RESET REASON = IWDG TIMEOUT FAULT (CODE: 0x20)"</code>.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Never kick the watchdog from an interrupt service routine (ISR)! Kicking the watchdog inside an ISR will keep the watchdog satisfied even if the main application thread has completely locked up.</li>
                <li>Clear reset flags using <code>__HAL_RCC_CLEAR_RESET_FLAGS()</code> after inspecting <code>RCC-&gt;CSR</code>.</li>
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
                        <td class="p-2 border font-mono">TC49.1</td>
                        <td class="p-2 border">Normal execution.</td>
                        <td class="p-2 border">Watchdog kicked periodically every 200 ms; system runs indefinitely.</td>
                        <td class="p-2 border">No unexpected resets over 24 hours.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC49.2</td>
                        <td class="p-2 border">Press Fault Injection Button to enter infinite loop: <code>while(1);</code>.</td>
                        <td class="p-2 border">Watchdog expires at exactly 1000 ms; hardware resets chip; boots logging IWDG reset.</td>
                        <td class="p-2 border">Hardware reset confirmed at 1000 &plusmn; 50 ms.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            During GDB debugging, the debugger halts the CPU, which would normally trigger an annoying IWDG reset immediately. Use the STM32 debug freeze register: <code>__HAL_DBGMCU_FREEZE_IWDG()</code> so the watchdog pauses when your debugger hits a breakpoint!
        </div>
    </div>
</div>

<!-- Assignment 50 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 50: Mini Industrial IoT Edge Controller</h3>
        <span class="px-2.5 py-1 bg-red-100 text-red-800 text-xs font-bold rounded-full">Level 5 · Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Multi-Sensor Suite (Temp, Light, Vibration)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">DS3231 RTC</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">W25Qxx SPI Flash Memory</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">CAN Bus &amp; UART Egress</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Config Menu Interface</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                The ultimate capstone assignment: Build an industrial-grade edge controller consolidating all five curriculum levels.
                <strong>System Capabilities:</strong>
                (1) <strong>Sensor Ingestion Engine:</strong> Ingest ambient temperature, illumination, and vibration shock metrics concurrently via I2C, ADC, and EXTI interrupts.
                (2) <strong>Hierarchical Menu GUI:</strong> OLED display renders a navigable configuration menu, allowing on-site technicians to configure alarm limits, calibrate sensor offsets, and view live telemetry using 3 navigation buttons.
                (3) <strong>Non-Volatile Persistence:</strong> All user-configured thresholds and critical fault events are stored in external SPI Flash memory with CRC integrity validation.
                (4) <strong>Dual Fieldbus Gateway:</strong> Concurrently streams telemetry frames onto an industrial CAN bus network (500 kbps) and an ASCII JSON command interface over UART (115200 baud).
                (5) <strong>Safety Supervisor:</strong> Governed by a non-blocking Finite State Machine, hardware watchdog supervision, and fail-safe alarm relay actuation.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Full multi-screen graphical OLED UI with responsive menu navigation.</li>
                <li>Simultaneous verified data delivery across CAN, UART, and SPI Flash blackbox.</li>
                <li>Zero memory leaks, stack overflows, or watchdog trips under full 48-hour continuous bench stress testing.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Cooperative round-robin event loop or FreeRTOS task partition:
                    Task 1 (Sensor Ingestion @ 50 Hz), Task 2 (FSM &amp; Alarm Manager @ 20 Hz), Task 3 (OLED UI @ 5 Hz), Task 4 (CAN/UART Gateway @ 10 Hz), Task 5 (Watchdog Supervisor @ 1 Hz).
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
                        <td class="p-2 border font-mono">TC50.1</td>
                        <td class="p-2 border">Modify temperature alarm threshold via OLED menu from 35°C to 45°C.</td>
                        <td class="p-2 border">New value saved to SPI flash; persists across reboot; active in monitoring.</td>
                        <td class="p-2 border">Value retained after hard power cycle.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC50.2</td>
                        <td class="p-2 border">Exceed vibration threshold during heavy UART streaming.</td>
                        <td class="p-2 border">Emergency CAN frame dispatched immediately; alarm siren trips; event logged to flash.</td>
                        <td class="p-2 border">End-to-end response &lt; 25 ms; no frame drops.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC50.3</td>
                        <td class="p-2 border">48-Hour Continuous Soak Test.</td>
                        <td class="p-2 border">Zero memory leaks; stack high-water mark stable; zero watchdog resets.</td>
                        <td class="p-2 border">100% uptime confirmed.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Congratulations on completing Assignment 50! Profile your firmware stack depth using the linker map file and monitor stack high-water marks (by filling the stack with <code>0xA5A5A5A5</code> at startup) to guarantee that nested interrupts and task contexts never collide with the heap.
        </div>
    </div>
</div>
"""
