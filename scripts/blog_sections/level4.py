"""
Level 4 Section: Communication Protocols (Assignments 31–40)
"""

def get_level4_section():
    return r"""
<h2 id="level-4" class="text-3xl font-bold text-brand-navy mt-12 mb-6 border-b-2 border-brand-coral pb-3 flex items-center">
    <i class="fas fa-network-wired text-brand-coral mr-3"></i> 7. Level 4 — Communication Protocols (Assignments 31–40 · Intermediate to Advanced)
</h2>

<p class="text-slate-700 leading-relaxed mb-6">
    Level 4 is where embedded firmware transitions into distributed industrial networking. You will master standard wired physical layers: multi-device I2C with bus-lock recovery, high-speed SPI NOR Flash memory drivers, asynchronous DMA-driven UART command line interfaces, and Controller Area Network (CAN 2.0B) automotive fieldbus architectures.
</p>

<!-- Assignment 31 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 31: I2C Bus Diagnostic Scanner</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">I2C1 Peripheral</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Multiple I2C Devices (OLED, RTC, MPU6050)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x 4.7kΩ Pull-Up Resistors</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART CLI</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement an automated I2C bus diagnostic scanner. Sweep through all standard 7-bit I2C addresses (from <code>0x01</code> to <code>0x7F</code>). For each address, issue an I2C START condition followed by the address byte configured for WRITE mode. Listen for the hardware Acknowledge (ACK) or Not-Acknowledge (NACK) response bit. Format and output a 16x8 diagnostic matrix over UART terminal: printing the hexadecimal address when a device responds with ACK, and a dot (<code>"."</code>) when NACK is received. Cross-reference discovered addresses against a built-in peripheral lookup table:
                <code>"0x3C: SSD1306 OLED Display"</code>, <code>"0x68: DS3231 RTC / MPU6050 IMU"</code>.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>UART terminal renders a complete 128-address diagnostic matrix within &lt; 200 ms.</li>
                <li>Identifies exact peripheral hardware present on the bus.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Use <code>HAL_I2C_IsDeviceReady(&amp;hi2c1, (address &lt;&lt; 1), 2, 5)</code> with a short 5 ms timeout per address.</li>
                <li>Configure I2C GPIO pins as <code>Alternate Function Open Drain</code> with external 4.7 k&Omega; pull-up resistors to 3.3V.</li>
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
                        <td class="p-2 border font-mono">TC31.1</td>
                        <td class="p-2 border">Connect SSD1306 OLED (address 0x3C) and run scan.</td>
                        <td class="p-2 border">Matrix outputs ACK at <code>0x3C</code>; all other positions report NACK.</td>
                        <td class="p-2 border">Exact match on 0x3C.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC31.2</td>
                        <td class="p-2 border">Add DS3231 RTC (0x68) to same bus lines and scan.</td>
                        <td class="p-2 border">Matrix reports both <code>0x3C</code> and <code>0x68</code> active.</td>
                        <td class="p-2 border">Both devices detected simultaneously.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC31.3</td>
                        <td class="p-2 border">Short SDA line to GND (bus fault test).</td>
                        <td class="p-2 border">Driver detects bus lock; reports <code>"I2C_ERROR_BUSY"</code>; attempts software clock reset sequence.</td>
                        <td class="p-2 border">Firmware recovers without hanging.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            ST's HAL I2C driver accepts an 8-bit shifted address (where bit 0 is R/W): <code>(7bit_addr &lt;&lt; 1)</code>. If your scan fails to find an address <code>0x3C</code> device, ensure you are passing <code>0x78</code> (which is 0x3C &times; 2) to the HAL function!
        </div>
    </div>
</div>

<!-- Assignment 32 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 32: High-Speed SPI Master Peripheral Interface</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SPI1 Peripheral (Master)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SPI Display or MAX6675 Thermocouple / Flash</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Dedicated GPIO Chip Select (/CS)</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Configure hardware SPI in Master full-duplex mode. Synchronize clock polarity (CPOL) and clock phase (CPHA) to match peripheral specifications (e.g. SPI Mode 0: CPOL=0, CPHA=0). Drive a dedicated GPIO Chip Select line (/CS): pulling /CS LOW before transmission, transmitting a multi-byte command/data sequence at high clock speed (e.g. 10 MHz to 20 MHz), and releasing /CS HIGH. Read back peripheral telemetry or sensor data and display the result.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Oscilloscope / logic analyzer trace reveals SCK, MOSI, MISO, and /CS lines with zero setup/hold timing violations.</li>
                <li>Reliable byte transactions verified at up to 20 Mbps.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Never leave /CS floating or permanently grounded. Always manage /CS explicitly in software to frame transactions.</li>
                <li>Ensure SPI prescaler does not exceed maximum allowable peripheral clock frequency specified in the target device's datasheet.</li>
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
                        <td class="p-2 border font-mono">TC32.1</td>
                        <td class="p-2 border">Transmit 4-byte query payload over SPI.</td>
                        <td class="p-2 border">/CS asserts LOW; 32 clock cycles generated; /CS returns HIGH.</td>
                        <td class="p-2 border">Waveform timings match Mode 0 on logic analyzer.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC32.2</td>
                        <td class="p-2 border">Loopback Test: Wire MOSI directly to MISO.</td>
                        <td class="p-2 border">Bytes transmitted out MOSI are received identically on MISO buffer.</td>
                        <td class="p-2 border">100,000 random bytes transferred with 0 bit errors.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Check whether your peripheral requires MSB-first or LSB-first bit order. Most SPI peripherals (like displays and flash memories) mandate <code>SPI_FIRSTBIT_MSB</code>.
        </div>
    </div>
</div>

<!-- Assignment 33 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 33: Full-Duplex UART Command Line Interface (CLI)</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">USART2 (ST-LINK VCP / External FTDI)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART Circular Ring Buffer</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Status LED</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">String Tokenizer Parser</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement an interactive command shell over UART at 115200 baud. Ingest incoming keystrokes into a circular ring buffer via DMA or RX interrupt. Echo characters back to the terminal. When a newline (<code>\r</code> or <code>\n</code>) is received, parse and execute commands:
                <code>"LED ON"</code>: Sets output LED HIGH, replies <code>"OK: LED illuminated\r\n"</code>.
                <code>"LED OFF"</code>: Resets output LED LOW, replies <code>"OK: LED extinguished\r\n"</code>.
                <code>"STATUS"</code>: Prints system uptime, supply voltage, and sensor readings.
                <code>"HELP"</code>: Lists all available commands.
                Any unknown command replies with: <code>"ERR: Unknown command '%s'. Type HELP for syntax.\r\n"</code>.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Interactive terminal shell prompt (<code>"stm32&gt; "</code>) with responsive character echo and command parsing.</li>
                <li>LED switches states immediately in response to valid serial commands.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Zero polling reads! Use <code>HAL_UART_Receive_IT()</code> or Circular DMA with UART IDLE Line Detection (<code>IDLEIE</code>).</li>
                <li>Prevent buffer overflows: Handle cases where the user pastes large text streams exceeding line buffer length.</li>
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
                        <td class="p-2 border font-mono">TC33.1</td>
                        <td class="p-2 border">Send <code>"LED ON\r"</code>.</td>
                        <td class="p-2 border">LED turns ON; terminal prints <code>"OK: LED illuminated\r\nstm32&gt; "</code>.</td>
                        <td class="p-2 border">GPIO pin rises to 3.3V; prompt returns.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC33.2</td>
                        <td class="p-2 border">Send <code>"STATUS\r"</code>.</td>
                        <td class="p-2 border">System prints uptime in seconds and system status metrics.</td>
                        <td class="p-2 border">Formatted telemetry matches internal tick counters.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC33.3</td>
                        <td class="p-2 border">Send invalid text: <code>"MOTOR_START\r"</code>.</td>
                        <td class="p-2 border">System rejects input; prints <code>"ERR: Unknown command..."</code>.</td>
                        <td class="p-2 border">No unhandled fault or crash.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Different serial terminals (PuTTY, TeraTerm, screen) send different line ending sequences (<code>\r</code>, <code>\n</code>, or <code>\r\n</code>). Ensure your line parser treats both <code>\r</code> and <code>\n</code> as command terminators and filters out duplicate empty lines.
        </div>
    </div>
</div>

<!-- Assignment 34 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 34: CAN Bus Node Transmitter</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">STM32 CAN1 Peripheral</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SN65HVD230 3.3V CAN Transceiver</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">120Ω Terminating Resistors</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Second CAN Node / USB Analyzer</span>
    </div>

    <figure class="my-4">
        <img src="/static/blogs/stm32-diy/can-bus-transceiver-circuit.svg" alt="CAN Bus Transceiver and Fieldbus Circuit Diagram" class="rounded-lg shadow border border-gray-200 max-h-64 object-cover mx-auto">
        <figcaption class="text-xs text-center text-slate-500 mt-2">Figure 1.11: Differential physical layer (CANH/CANL) with 120Ω terminating resistors.</figcaption>
    </figure>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Configure the on-chip bxCAN (CAN 2.0B) peripheral for 500 kbps nominal baud rate. Construct standard 11-bit CAN data frames with Identifier <code>0x301</code> containing sensor readings (e.g. 2 bytes temperature, 2 bytes light intensity, 1 byte button state, 1 byte rolling sequence counter). Broadcast this telemetry packet every 100 ms across the differential CANH/CANL bus. Verify that transmitted frames are successfully acknowledged (ACK) by an active second node on the bus.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Oscilloscope differential trace across CANH and CANL: Recessive state = 2.5V differential (0V delta); Dominant state = CANH @ 3.5V, CANL @ 1.5V (2.0V delta).</li>
                <li>Second node captures frames cleanly with ID 0x301 and DLC = 6.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Baud Rate Timing Calculation (for 500 kbps on 45 MHz APB1 clock):
                    <div class="my-2 p-2 bg-slate-50 border border-slate-200 rounded font-mono text-xs text-slate-800">
                        Prescaler = 5, Time Quanta (TQ) = 1 + BS1(15) + BS2(2) = 18<br>
                        Baud = 45 MHz / (5 &times; 18) = 500 kbps
                    </div>
                </li>
                <li>Both physical ends of the twisted-pair cable must have a 120 &Omega; terminating resistor connected across CANH and CANL.</li>
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
                        <td class="p-2 border font-mono">TC34.1</td>
                        <td class="p-2 border">Transmit frames with second node connected.</td>
                        <td class="p-2 border">Frame is acknowledged; Tx mailbox clears; no CAN error flags set.</td>
                        <td class="p-2 border">Frame visible in second node's packet capture.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC34.2</td>
                        <td class="p-2 border">Disconnect second node (no ACK on bus).</td>
                        <td class="p-2 border">CAN hardware attempts retransmissions; sets ACK Error flag (<code>CAN_ESR_ACKERR</code>).</td>
                        <td class="p-2 border">Transmitter enters error passive if not resolved; no CPU lockup.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            CAN is a broadcast bus that <strong>strictly mandates at least one other active receiving node to generate the ACK bit</strong>. If you run a CAN transmitter with no second node connected, it will endlessly retry and enter Bus-Off state!
        </div>
    </div>
</div>

<!-- Assignment 35 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 35: CAN Bus Node Receiver &amp; Telemetry Display</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">STM32 CAN1 (RX Mode)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">CAN Transceiver (SN65HVD230)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">CAN Hardware Acceptance Filter</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SSD1306 OLED Display</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Configure CAN hardware acceptance filters (Filter Bank 0) in Identifier Mask Mode to accept only messages with Standard IDs in the range <code>0x300</code> to <code>0x30F</code>, automatically discarding all unrelated network traffic in hardware without interrupting the CPU. When a matching frame arrives, service the RX FIFO interrupt (<code>HAL_CAN_RxFifo0MsgPendingCallback</code>), extract the payload, decode the sensor data, and render the live values on an OLED display in real time.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>OLED renders incoming CAN telemetry: <code>"CAN ID: 0x301 | Temp: 26.4 C | Status: OK"</code>.</li>
                <li>Zero dropped frames under 100 Hz network traffic bursts.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Filter Configuration: Filter ID = <code>0x300 &lt;&lt; 5</code>, Filter Mask = <code>0x7F0 &lt;&lt; 5</code> (passes <code>0x300</code> through <code>0x30F</code>).</li>
                <li>Always read frames using <code>HAL_CAN_GetRxMessage()</code> inside the FIFO callback to prevent FIFO overrun errors.</li>
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
                        <td class="p-2 border font-mono">TC35.1</td>
                        <td class="p-2 border">Send CAN frame with ID <code>0x301</code>.</td>
                        <td class="p-2 border">Hardware filter accepts frame; RX interrupt fires; OLED updates.</td>
                        <td class="p-2 border">Data decoded with 100% accuracy.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC35.2</td>
                        <td class="p-2 border">Send CAN frame with ID <code>0x123</code> (outside filter range).</td>
                        <td class="p-2 border">Hardware filter rejects frame; RX FIFO remains empty; no interrupt triggered.</td>
                        <td class="p-2 border">CPU receives zero interrupts for rejected IDs.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Remember that the STM32 CAN filter register requires standard 11-bit identifiers to be left-aligned into the upper bits of the 16-bit register: <code>sFilterConfig.FilterIdHigh = (0x300 &lt;&lt; 5);</code>.
        </div>
    </div>
</div>

<!-- Assignment 36 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 36: Distributed CAN-Based Temperature Alert System</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x STM32 Boards (Node A &amp; Node B)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x CAN Transceivers</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Temperature Sensor (Node A)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Alarm Relay / Buzzer (Node B)</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a distributed industrial monitoring system across two independent STM32 boards.
                <strong>Node A (Sensor Node):</strong> Samples temperature every 250 ms, packages reading into CAN ID <code>0x210</code>, and broadcasts it across the bus.
                <strong>Node B (Actuator Node):</strong> Listens for ID <code>0x210</code>. When the received temperature crosses 40.0°C, Node B trips a high-power alarm siren and transmits a high-priority emergency CAN frame (ID <code>0x050</code>) back across the network to notify Node A, which illuminates its warning LED.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Bi-directional CAN traffic verified across bus.</li>
                <li>Heating sensor on Node A causes Node B alarm to sound in &lt; 20 ms.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Emergency broadcast uses high-priority identifier <code>0x050</code> (lower numerical ID has natural arbitration priority on the CAN bus).</li>
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
                        <td class="p-2 border font-mono">TC36.1</td>
                        <td class="p-2 border">Normal operation (T &lt; 40&deg;C).</td>
                        <td class="p-2 border">Node A broadcasts ID 0x210; Node B receives data; Alarm stays OFF.</td>
                        <td class="p-2 border">Steady heartbeat transmission.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC36.2</td>
                        <td class="p-2 border">Warm sensor on Node A above 40°C.</td>
                        <td class="p-2 border">Node B trips siren; Node B transmits ID 0x050; Node A lights alarm LED.</td>
                        <td class="p-2 border">End-to-end network response latency &lt; 20 ms.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Ensure both boards share a common reference ground (GND) wire alongside the CANH/CANL differential pair to keep the common-mode voltage within the transceivers' operational range (-7V to +12V).
        </div>
    </div>
</div>

<!-- Assignment 37 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 37: CAN Bus Network Heartbeat &amp; Node Guarding</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">2x CAN Nodes</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">CAN Transceivers</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Watchdog Timer / SysTick</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Node Status LEDs</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Implement a safety-critical CANopen-style heartbeat protocol. Node 1 broadcasts a periodic "Heartbeat" frame (CAN ID <code>0x701</code>, DLC = 1, byte 0 = <code>0x05</code> for Operational) every 500 ms. Node 2 acts as a network supervisor: it reloads an internal software watchdog timer whenever a valid heartbeat arrives. If no heartbeat frame is detected for <strong>more than 1500 ms</strong> (missing 3 consecutive pulses), Node 2 declares <code>"NODE 1 OFFLINE / TIMEOUT FAULT"</code>, illuminates an amber warning LED, and switches into safe fallback operation.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Green "Node Online" LED illuminated while periodic heartbeat is received.</li>
                <li>Amber "Node Missing" warning LED illuminates exactly 1.50 seconds after disconnecting Node 1.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Watchdog check evaluated using non-blocking delta time: <code>if (HAL_GetTick() - last_heartbeat_time &gt; 1500)</code>.</li>
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
                        <td class="p-2 border font-mono">TC37.1</td>
                        <td class="p-2 border">Continuous heartbeat broadcast from Node 1.</td>
                        <td class="p-2 border">Node 2 pet watchdog periodically; Green LED solid ON.</td>
                        <td class="p-2 border">Healthy network state verified.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC37.2</td>
                        <td class="p-2 border">Unplug power from Node 1.</td>
                        <td class="p-2 border">At T = 1.50 s following last frame, Node 2 trips timeout fault.</td>
                        <td class="p-2 border">Timeout detected within 1500 &plusmn; 20 ms.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC37.3</td>
                        <td class="p-2 border">Power up Node 1 again.</td>
                        <td class="p-2 border">Heartbeat frames resume; Node 2 clears fault and restores Green LED.</td>
                        <td class="p-2 border">Automatic fault recovery verified.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Heartbeat timeout multipliers are typically set to 2.5&times; to 3&times; the broadcast interval in industrial systems (ISO 11898 / CANopen DS301) to prevent false-positive alarms caused by transient network arbitration delays.
        </div>
    </div>
</div>

<!-- Assignment 38 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 38: SPI NOR Flash Memory Driver &amp; Wear Manager</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">W25Q64 / W25Q128 SPI NOR Flash</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">SPI Peripheral (SPI1 or SPI2)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">UART Diagnostic Console</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Author a complete device driver for a Winbond W25Q64/128 SPI NOR Flash memory chip. Implement:
                (1) <code>Read JEDEC ID (0x9F)</code> to verify manufacturer ID (<code>0xEF</code>) and device capacity,
                (2) <code>Write Enable (0x06)</code> and status register polling for Write-In-Progress (WIP bit in SR1),
                (3) <code>Sector Erase (0x20)</code> to erase a 4 KB sector to <code>0xFF</code>,
                (4) <code>Page Program (0x02)</code> to write up to 256 bytes, and
                (5) <code>Fast Read (0x0B)</code> to read data back.
                Write a verification pattern, perform a hardware reset of the microcontroller, read back the flash contents, and confirm 100% data integrity.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>UART terminal prints valid JEDEC ID (e.g. <code>"Winbond W25Q64BV Detected (8MB)"</code>).</li>
                <li>Read-back data matches written buffer byte-for-byte across reboots.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Flash memory bits can only be transitioned from <code>1</code> to <code>0</code> during programming. You must erase the sector (transitioning all bits to <code>1</code>) prior to reprogramming!</li>
                <li>Handle page programming boundary wrapping: writing beyond a 256-byte page boundary wraps back to the start of the same page rather than advancing to the next page.</li>
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
                        <td class="p-2 border font-mono">TC38.1</td>
                        <td class="p-2 border">Query JEDEC ID.</td>
                        <td class="p-2 border">Returns 3 bytes: <code>0xEF</code>, <code>0x40</code>, <code>0x17</code> (W25Q64).</td>
                        <td class="p-2 border">Manufacturer and capacity match datasheet.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC38.2</td>
                        <td class="p-2 border">Erase Sector 0; read all 4096 bytes.</td>
                        <td class="p-2 border">Every single byte reads <code>0xFF</code>.</td>
                        <td class="p-2 border">Clean erase confirmed.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC38.3</td>
                        <td class="p-2 border">Program 128 bytes of test data; trigger MCU Reset; Read back.</td>
                        <td class="p-2 border">Data buffer read back matches original buffer identically.</td>
                        <td class="p-2 border">Zero checksum mismatches.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            After issuing an Erase or Page Program command, you must poll the Status Register WIP bit (<code>while (W25Q_ReadStatus() &amp; 0x01);</code>) until the internal memory write cycle completes. Erasing a 4 KB sector typically takes 40 ms to 200 ms!
        </div>
    </div>
</div>

<!-- Assignment 39 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 39: Multi-Device I2C Telemetry Hub with Bus Recovery</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">MPU6050 Accelerometer (0x68)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">BMP280 / AHT10 Temp/Pressure (0x76 or 0x38)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Single I2C Bus</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">I2C Bus Lock Recovery Algorithm</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Interface two distinct digital sensors with different internal register architectures on a shared physical I2C bus. Sample both sensors interleaved at 20 Hz. If one sensor fails to acknowledge or locks the bus (e.g., slave device pulls SDA low and hangs), the firmware must not freeze. Implement a <strong>Bus Lock Recovery Routine</strong>: reconfigure the SCL/SDA pins as software GPIO outputs, pulse SCL up to 9 times to force the slave device to release the stuck SDA line, issue an I2C STOP condition, and re-initialize the I2C peripheral hardware.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Simultaneous display of MPU6050 accelerometer and environmental sensor data.</li>
                <li>Simulated bus lock recovers automatically within &lt; 5 ms.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Handle individual device timeouts independently: if the pressure sensor fails, continue acquiring accelerometer data without crashing.</li>
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
                        <td class="p-2 border font-mono">TC39.1</td>
                        <td class="p-2 border">Normal multi-sensor operation.</td>
                        <td class="p-2 border">Both devices polled seamlessly on shared bus lines.</td>
                        <td class="p-2 border">Zero bus contention or data corruption.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC39.2</td>
                        <td class="p-2 border">Simulate stuck SDA: pull SDA LOW for 100 ms during transaction.</td>
                        <td class="p-2 border">Driver detects timeout, executes 9-clock recovery sequence, clears bus lock.</td>
                        <td class="p-2 border">Subsequent I2C transactions succeed normally.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            Bus lock happens when an STM32 resets mid-byte while an I2C slave is holding SDA low to transmit a '0'. The slave waits forever for SCL clocks that never arrive. Clocking SCL manually 9 times allows the slave to finish sending the byte and release SDA!
        </div>
    </div>
</div>

<!-- Assignment 40 -->
<div class="assignment-card bg-white border border-gray-200 rounded-xl p-6 mb-8 shadow-sm">
    <div class="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h3 class="text-2xl font-bold text-brand-navy">Assignment 40: Heterogeneous Multi-Protocol Sensor Gateway</h3>
        <span class="px-2.5 py-1 bg-indigo-100 text-indigo-800 text-xs font-bold rounded-full">Level 4 · Intermediate to Advanced</span>
    </div>
    <div class="flex items-center gap-2 mb-4">
        <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Components:</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">I2C Sensor (MPU6050 / Temp)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">Analog ADC Channel</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">USART2 (ASCII Telemetry Stream)</span>
        <span class="bg-slate-100 text-slate-800 text-xs font-mono px-2 py-0.5 rounded">CAN1 (Industrial Fieldbus Broadcast)</span>
    </div>

    <div class="space-y-3 text-sm text-slate-700">
        <div>
            <strong class="text-brand-navy">Expectations:</strong>
            <p>
                Architect an industrial edge translation gateway. Collect heterogeneous inputs:
                (1) I2C digital accelerometer / environmental data,
                (2) Analog ADC potentiometer voltage, and
                (3) Push button states.
                Normalize all inputs into an internal telemetry struct. Concurrently publish the aggregated data across two egress interfaces:
                (A) <strong>UART Channel:</strong> Formatted JSON string at 115200 baud for cloud/PC dashboards:
                <code>{"t_ms": 124500, "temp_c": 24.8, "pot_v": 1.65, "btn": 1}\r\n</code>.
                (B) <strong>CAN Bus:</strong> Packed 8-byte binary frame (CAN ID <code>0x450</code>) broadcast onto the automotive vehicle bus at 500 kbps.
            </p>
        </div>
        <div>
            <strong class="text-brand-navy">Observable Outputs:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Simultaneous valid JSON transmission on UART and binary CAN frame broadcast.</li>
                <li>Zero jitter or cross-protocol data starvation.</li>
            </ul>
        </div>
        <div>
            <strong class="text-brand-navy">Architectural Rules:</strong>
            <ul class="list-disc pl-5 space-y-1">
                <li>Non-blocking pipelined design: Ingest sensors at 50 Hz; publish UART JSON at 10 Hz; broadcast CAN at 20 Hz.</li>
                <li>Use DMA for UART transmission (<code>HAL_UART_Transmit_DMA</code>) so string serialization does not stall CAN transmission mailboxes.</li>
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
                        <td class="p-2 border font-mono">TC40.1</td>
                        <td class="p-2 border">Rotate analog pot while monitoring both UART and CAN.</td>
                        <td class="p-2 border">JSON string updates voltage; CAN frame byte 2/3 reflect identical ADC value.</td>
                        <td class="p-2 border">Protocol synchronization verified.</td>
                    </tr>
                    <tr>
                        <td class="p-2 border font-mono">TC40.2</td>
                        <td class="p-2 border">Saturate UART with continuous streaming.</td>
                        <td class="p-2 border">CAN transmission timing remains rock-solid at 20 Hz with zero dropped frames.</td>
                        <td class="p-2 border">DMA offloads CPU; CAN periodicity preserved &plusmn;1 ms.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded text-xs text-slate-800">
            <strong>Suggestions to Check:</strong>
            When formatting JSON strings with <code>snprintf</code>, keep buffer sizes strictly bounded to prevent stack overflow. Avoid <code>sprintf</code> without length checks.
        </div>
    </div>
</div>
"""
