# STM32L476RG Internal Data-Flow Animation Specification

## Purpose

Create a set of interactive HTML animation pages that explain how data and control flow through an STM32L476RG microcontroller.

The pages are intended for teaching Embedded C and STM32 architecture.

## Scope and Accuracy Notes

- Use STM32L476RG-specific reference information for peripheral names, registers, clock routing, IRQs, DMA requests, pin alternate functions, and memory sizes. Treat the diagrams as conceptual until those device details are checked against the STM32L4 reference manual and datasheet.
- Identify the exact development board whenever showing a pin, button, or LED. MCU pin capabilities do not by themselves establish how a board is wired.
- On Cortex-M reset, the first vector-table word supplies MSP and the second supplies the reset-handler address. Startup code then performs toolchain-specific runtime setup; it does not initialize a stack before reset-handler execution.
- The integrated bxCAN peripheral is the CAN controller; a separate external transceiver supplies the CAN physical layer. Do not draw a second controller block.
- DMA is a data-movement engine, not another CPU. Explain the configured transfer, arbitration, buffer ownership, and completion/error notification without promising a fixed CPU-load reduction.
- Keep Sleep, Stop, and Standby distinct. Their clocking, retained state, wake sources, and return behavior are not interchangeable.
- Distinguish generic protocol behavior from HAL-specific callback names and register-clear sequences. Link device-specific implementation details to the relevant manual.

The animation must support two modes:

1. **Automatic mode** — the flow progresses through predefined steps automatically.
2. **Manual mode** — the learner advances one step at a time using Next/Previous/Play/Pause/Reset controls.

The visual language should be inspired by a modern semiconductor architecture diagram, but it must represent STM32 concepts rather than GPU concepts.

---

# 1. Overall Design Requirements

## Visual Style

Use a dark technical architecture style.

Recommended visual hierarchy:

- MCU boundary: large dark panel
- CPU: prominent central block
- FLASH and SRAM: memory blocks
- Bus matrix/interconnect: horizontal or vertical data path
- Peripheral blocks: GPIO, TIM, UART, SPI, I2C, ADC, CAN
- NVIC: interrupt routing block
- DMA: independent data-movement block
- RCC: clock/reset source
- External devices: sensors, LED, CAN transceiver, PC
- Active data path: bright animated line
- Selected flow route: thick, high-contrast line with a clear arrowhead in both
  the persistent architecture map and the detailed flow diagram; keep the
  upcoming segment emphasized before playback starts.
- Inactive components: dimmed
- Current component: highlighted
- Completed path: slightly brighter but less prominent than current step

Use consistent colors by data type:

| Flow | Suggested visual treatment |
|---|---|
| Instruction fetch | Blue |
| Normal data | Cyan |
| Interrupt | Orange |
| DMA | Green |
| Clock | Purple |
| Reset | Red |
| Serial communication | Yellow |
| CAN traffic | Magenta |
| Error/Fault | Red |

Do not make the animation dependent on color alone. Also use labels, arrows, and state indicators.

---

# 2. Common Page Layout

Every animation page should contain:

```text
+-------------------------------------------------------------+
| Title                                                       |
| Short explanation                                           |
+-------------------------------------------------------------+
|                                                             |
|                    ARCHITECTURE DIAGRAM                     |
|                                                             |
|                                                             |
+-------------------------------------------------------------+
| Current Step: 4                                             |
| Step Description                                            |
+-------------------------------------------------------------+
| [Reset] [Previous] [Play/Pause] [Next] [Auto/Manual]       |
| Speed: [0.5x] [1x] [2x]                                    |
+-------------------------------------------------------------+
```

The learner must always be able to see:

- Current step number
- Current step title
- Current action
- Source
- Destination
- What data is moving
- Why the movement occurs

---

# 3. Master STM32L476RG Architecture

Create a persistent, clickable, spatial block diagram for the STM32L476RG and
show it as shared architecture context on every detailed flow. Highlight the
relevant components and bus/interrupt/clock routes for the selected flow. This
is an on-chip logical architecture view, not a package-pin diagram. Use the supplied
STM32L432xx diagram as visual inspiration only: the target device for this
course is STM32L476RG, so device-specific peripherals, buses, and capacities
must match the L476 datasheet and reference manual.

```text
                         STM32L476RG
+--------------------------------------------------------------------+
| Cortex-M4F core                                                    |
| Registers · ALU · FPU · PC/SP/LR · NVIC                            |
|                                                                    |
| I-CODE: instruction fetch  ------ address/control ------+           |
| D-CODE: code-space data    ------ address/control --+   |           |
| SYSTEM: SRAM/peripherals  ------ address/control -+ |   |           |
|                                                     v v   v           |
|                                                  AHB BUS MATRIX      |
|                                       arbitration · address decode   |
|                                         /         |         \        |
|                   instruction/data ---+          |          +-- DMA1/2
|                                                SRAM1/2                 |
| Flash interface                                      |                 |
|                                                      |                 |
|                                   AHB2: GPIO / EXTI  |                 |
|                                                      |                 |
|                                   AHB-to-APB bridge  |                 |
|                                     /          \                       |
|                                APB1            APB2                    |
|                         TIM/USART/I2C/      TIM/SPI/ADC                 |
|                            bxCAN                                     |
|                                                                    |
| RCC: clock sources -> SYSCLK / AHB / APB / peripheral clock enables |
| Reset control -> core and peripheral reset domains                  |
| Peripheral IRQ requests -> NVIC -> Cortex-M exception entry         |
+--------------------------------------------------------------------+
```

Draw DMA1/2 as additional bus masters entering the matrix, not as a block
downstream of the CPU. Draw the NVIC beside/within the core boundary and
interrupt requests returning from peripherals as a distinct event path.
Show a separate dotted RCC clock/reset network. Do not draw clock or IRQ
signals as payload data flowing over the address/data bus.

## Make each bus transaction understandable

Show a labeled logical bus bundle and explain its roles:

```text
MASTER                                      TARGET
  |                                             ^
  +-- ADDRESS -------------------------------> |
  +-- CONTROL: read/write, transfer attributes |
  +<-- READ DATA ------------------------------+
  +-- WRITE DATA ----------------------------->+
```

An address identifies the memory/peripheral location; control qualifies the
access; write data travels to the target and read data returns to the master.
The AHB matrix arbitrates and decodes requests. An AHB/APB bridge adapts
transactions to APB peripheral buses. These are internal MCU interconnects,
not external parallel address/data pins.

For the Cortex-M4, draw and label the three core interfaces accurately:

- **I-CODE**: instruction fetch path.
- **D-CODE**: data access to the code region.
- **SYSTEM**: access to SRAM, peripherals, and the system address space.

The exact path to a target depends on its memory-map region and the device
interconnect; do not imply every access travels over one shared CPU bus. Show
instruction/data paths as logical connections to the matrix, not as physical
package pins.

Include separate target blocks for Flash, SRAM1, SRAM2, AHB-connected GPIO,
the AHB/APB bridge, APB1/APB2 peripheral groups, DMA, RCC/reset, NVIC, and
package pins. Provide click-to-inspect detail for every block. Add step modes:

1. I-CODE instruction address/fetch control to Flash, instruction data back.
2. D-CODE and SYSTEM address/control to the bus matrix.
3. SRAM read data back to the master and write data to SRAM.
4. Memory-mapped peripheral access through AHB/APB bridge.
5. DMA-master transfers to/from supported memory/peripheral endpoints.
6. Peripheral interrupt request through NVIC to the CPU.
7. RCC clock distribution and reset control, visually distinct from data.

The page must state that the attached reference depicts STM32L432xx. Do not
copy its peripheral inventory, memory sizes, maximum clock values, or pin
mapping into this L476RG course without checking the L476-specific manuals.

---

# 4. Animation 01 — Power-On and Reset

## Objective

Explain what happens immediately after power-on or reset.

## Flow

```text
POWER ON
   |
   v
RESET
   |
   v
Cortex-M4 reset sequence
   |
   v
Vector Table
   |
   +----> Initial Stack Pointer
   |
   +----> Reset_Handler
             |
             v
        Startup Code
             |
             +----> Initialize .data
             |
             +----> Clear .bss
             |
             +----> Use initial stack pointer loaded from vector table
             |
             v
          SystemInit
             |
             v
            main()
```

## Animation Steps

### Step 1
Highlight POWER ON.

Text:

> The MCU receives power and starts the reset sequence.

### Step 2
Highlight RESET.

Text:

> Reset places the processor and peripherals into their reset state.

### Step 3
Highlight Vector Table.

Text:

> The processor loads the initial Main Stack Pointer from the first vector-table entry and obtains Reset_Handler from the second entry.

### Step 4
Highlight Initial Stack Pointer.

Text:

> The processor loads the initial Main Stack Pointer before executing Reset_Handler. Startup code does not need to initialize the stack before its first C function call.

### Step 5
Highlight Reset_Handler.

Text:

> Execution begins at Reset_Handler.

### Step 6
Highlight startup code.

Text:

> Startup code uses the stack selected by reset, copies initialized writable data into SRAM, clears .bss, and prepares the C runtime. Exact startup order depends on the toolchain and device startup file.

### Step 7
Animate .data from FLASH to SRAM.

### Step 8
Animate clearing .bss in SRAM to zero.

### Step 9
Highlight SystemInit.

### Step 10
Highlight main().

Final state:

```text
Reset -> Startup -> Runtime Initialization -> main()
```

---

# 5. Animation 02 — Flash, SRAM and Memory Layout

## Objective

Explain where program instructions and variables live.

Show:

```text
FLASH
+--------------------------+
| Vector Table             |
+--------------------------+
| .text                    |
| Program instructions    |
+--------------------------+
| .rodata                  |
| Constants                |
+--------------------------+
| Initial .data values     |
+--------------------------+

SRAM
+--------------------------+
| Stack                    |
|        down              |
+--------------------------+
|                          |
| Free space               |
|                          |
+--------------------------+
| Heap                     |
|        up                |
+--------------------------+
| .bss                     |
+--------------------------+
| .data                    |
+--------------------------+
```

Use example code:

```c
int global = 10;
int counter;
const int limit = 100;

void foo(void)
{
    int local = 5;
}
```

Map:

```text
global  -> .data
counter -> .bss
limit   -> .rodata
local   -> stack
foo()   -> .text
```

Important teaching point:

> The CPU normally executes instructions directly from Flash. It does not copy the complete program into SRAM before execution.

Animate:

```text
FLASH
  |
  +---- instructions ---> CPU

FLASH
  |
  +---- initial .data ---> SRAM
```

---

# 6. Animation 03 — CPU Instruction Fetch and Execute

## Objective

Connect a C statement to CPU execution.

Example:

```c
int x = a + b;
```

Animation:

```text
FLASH
 |
 | instruction fetch
 v
CPU
 |
 +--> load a
 |
 +--> load b
 |
 +--> ADD
 |
 +--> store x
 |
 v
SRAM
```

Show the Program Counter moving through instructions.

Display:

```text
PC
SP
LR
R0-R12
xPSR
```

Do not attempt to reproduce the exact internal transistor-level CPU implementation. The animation should represent the architectural data flow.

---

# 7. Animation 04 — Memory-Mapped GPIO

## Objective

Explain how C code controls hardware.

Example:

```c
GPIOA->ODR |= (1U << 5);
```

Flow:

```text
C Statement
    |
    v
CPU executes load/store instruction
    |
    v
Bus transaction
    |
    v
GPIO register
    |
    v
GPIO output logic
    |
    v
GPIO pin
    |
    v
LED
```

Explain:

> GPIO registers occupy addresses in the MCU memory map. A CPU access to such an address becomes a peripheral register access.

---

# 8. Animation 05 — GPIO Input

Example:

```c
if (GPIOC->IDR & (1U << 13))
{
    ...
}
```

Flow:

```text
Push Button
    |
    v
GPIO Pin
    |
    v
GPIO Input Circuit
    |
    v
IDR Register
    |
    v
Bus
    |
    v
CPU
    |
    v
C Program
```

Show the bit value changing:

```text
IDR[13] = 0
IDR[13] = 1
```

---

# 9. Animation 06 — RCC and Clock Flow

## Objective

Explain why peripherals require clocks.

Architecture:

```text
HSI / HSE / MSI / LSI / LSE
             |
             v
            RCC
             |
       +-----+------+
       |            |
       v            v
     CPU clock   Peripheral clocks
                    |
        +-----------+-----------+
        |           |           |
       GPIO        TIM         UART
```

Animation:

1. Clock source starts.
2. RCC selects/configures clock.
3. Clock reaches CPU.
4. Peripheral clock is enabled.
5. Peripheral becomes operational.

Highlight the clock path separately from data paths.

---

# 10. Animation 07 — Timer

Example objective:

> Generate an event every 1 ms.

Flow:

```text
Clock
  |
  v
Prescaler
  |
  v
Counter
  |
  v
Auto Reload / Compare
  |
  v
Timer Event
```

Then show two variants.

### Polling

```text
CPU
 |
 +--> check timer flag
 |
 +--> check timer flag
 |
 +--> check timer flag
```

### Interrupt

```text
Timer
  |
  v
Interrupt Request
  |
  v
NVIC
  |
  v
CPU
  |
  v
ISR
```

---

# 11. Animation 08 — Interrupt and Callback

Use GPIO or Timer as the source.

```text
Peripheral
    |
    v
Interrupt Request
    |
    v
NVIC
    |
    v
CPU
    |
    v
Vector Table
    |
    v
ISR
    |
    v
HAL Handler
    |
    v
User Callback
```

Example:

```c
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    // application code
}
```

Explain that the callback is not magic.

The actual conceptual path is:

```text
Hardware Event
 -> Interrupt Request
 -> NVIC
 -> CPU Exception Entry
 -> ISR
 -> HAL Handler
 -> User Callback
```

---

# 12. Animation 09 — DMA

## Objective

Show why DMA exists.

### CPU-driven transfer

```text
UART
 |
 v
CPU
 |
 v
SRAM
```

CPU performs every transfer.

### DMA-driven transfer

```text
UART
 |
 v
DMA
 |
 v
SRAM
 |
 v
CPU notified after transfer
```

Show:

```text
CPU utilization:
HIGH -> LOW

DMA activity:
LOW -> HIGH
```

Do not imply that DMA is a magical independent processor. Explain it as a hardware data-movement engine. It can reduce per-item CPU servicing, but configuration, arbitration, memory bandwidth, and completion handling still have costs; avoid promising a fixed CPU-utilization reduction.

---

# 13. Animation 10 — UART TX

Flow:

```text
CPU
 |
 v
UART TX Register
 |
 v
UART Transmitter
 |
 v
Shift Register
 |
 v
TX Pin
 |
 v
External Device
```

Show a byte such as:

```text
'A' = 0x41
```

Animate the serial frame:

```text
Idle | Start | Data bits (LSB first) | Stop
  1  |   0   |  10000010            |  1
```

Also show:

```text
TXE
TC
RXNE
```

as peripheral status events. Explain that TXE means the transmit data register can accept data, while TC means the full frame has completed; RXNE indicates received data is ready. Check exact flag naming and clearing behavior against the selected STM32L4 USART instance and reference manual.

---

# 14. Animation 11 — UART RX with Interrupt

Flow:

```text
External Device
       |
       v
RX Pin
       |
       v
UART Receiver
       |
       v
RX Register
       |
       v
RXNE
       |
       v
NVIC
       |
       v
CPU
       |
       v
ISR
```

---

# 15. Animation 12 — UART RX with DMA

Flow:

```text
External Device
       |
       v
UART RX
       |
       v
DMA
       |
       v
SRAM Buffer
       |
       v
DMA Complete Interrupt
       |
       v
CPU
```

Show a buffer:

```text
SRAM
+----+----+----+----+----+----+
| H  | e  | l  | l  | o  | .. |
+----+----+----+----+----+----+
```

Treat received data as bytes with an explicit length. Do not imply that UART reception automatically adds a C string terminator.

---

# 16. Animation 13 — SPI

Architecture:

```text
STM32
 |
 v
SPI Peripheral
 |
 +---- SCK ----> Device
 +---- MOSI ---> Device
 +<--- MISO ---- Device
 +---- CS -----> Device
```

Animation:

```text
CPU
 |
 v
SPI TX Register
 |
 v
SPI Shift Register
 |
 v
MOSI
 |
 v
External Device
```

For reception:

```text
External Device
 |
 v
MISO
 |
 v
SPI Shift Register
 |
 v
SPI RX Register
 |
 v
CPU / DMA
```

---

# 17. Animation 14 — I2C

Architecture:

```text
STM32
 |
 v
I2C Peripheral
 |
 +---- SDA <--------> Sensor
 |
 +---- SCL ---------> Sensor
```

Animate a transaction:

```text
START
  |
  v
7-bit Address
  |
  v
R/W Bit
  |
  v
ACK
  |
  v
Register Address
  |
  v
ACK
  |
  v
Data
  |
  v
ACK
  |
  v
STOP
```

Use a BME280 as an example sensor. Show its 7-bit address (commonly 0x76 or 0x77, depending on the SDO pin), ACK/NACK, and a repeated START for a combined register read. The sensor's register map is separate from the generic I2C bus protocol.

---

# 18. Animation 15 — ADC

Flow:

```text
Analog Sensor
     |
     v
Analog Voltage
     |
     v
ADC
     |
     v
Digital Sample
     |
     +------> CPU
     |
     +------> DMA
                 |
                 v
                SRAM
```

Show:

```text
Analog waveform
      |
      | sampling
      v
Discrete samples
      |
      v
Digital values
```

---

# 19. Animation 16 — CAN

Use this as the automotive-specific flow.

```text
Application
     |
     v
bxCAN peripheral
     |
     +--> TX mailbox
     |
     +--> RX FIFO
     |
     v
CAN Transceiver
     |
     +---- CANH
     |
     +---- CANL
             |
             v
          CAN Bus
             |
             v
          Other ECU
```

Explicitly distinguish:

```text
bxCAN controller != CAN transceiver
```

The STM32L476RG bxCAN controller is integrated into the MCU. The external transceiver provides the physical-layer electrical interface. Do not show another separate CAN controller block between bxCAN and the transceiver.

For STM32L476RG, identify the CAN peripheral as **bxCAN** rather than FDCAN.

---

# 20. Animation 17 — CAN Receive Interrupt

Flow:

```text
CAN Bus
   |
   v
CAN Transceiver
   |
   v
CAN Peripheral
   |
   v
RX FIFO
   |
   v
Interrupt
   |
   v
NVIC
   |
   v
CPU
   |
   v
CAN ISR / HAL
   |
   v
Application Callback
```

Show the CAN frame fields:

```text
+------+-----+------+-----+------+
| ID   | RTR | DLC  | DATA       |
+------+-----+------+-----+------+
```

---

# 21. Animation 18 — ADC + DMA Sensor Pipeline

Create a realistic embedded sensor pipeline.

```text
Sensor
  |
  v
ADC
  |
  v
DMA
  |
  v
Circular SRAM Buffer
  |
  v
CPU
  |
  v
Signal Processing
  |
  v
Application
```

Example:

```text
ADC samples:
[1200, 1212, 1198, 1220, ...]
```

Show the CPU processing blocks of data rather than handling every sample.

---

# 22. Animation 19 — Watchdog

Flow:

```text
Application
    |
    v
Watchdog Timer
    |
    +---- regularly refreshed ----> Continue
    |
    +---- timeout ----------------> Reset
                                      |
                                      v
                                    Boot
```

Show a fault scenario:

```text
main loop
    |
    v
Software hangs
    |
    X
Watchdog not refreshed
    |
    v
MCU RESET
    |
    v
Boot sequence
```

---

# 23. Animation 20 — Fault Handling

Show:

```text
CPU
 |
 +---- invalid memory access
 |
 +---- illegal instruction
 |
 +---- bus fault
 |
 +---- usage fault
 |
 v
Exception
 |
 v
Fault Handler
 |
 +---- diagnose
 |
 +---- record error
 |
 +---- recover/reset
```

Include:

- HardFault
- MemManage
- BusFault
- UsageFault

---

# 24. Animation 21 — Low-Power Flow

Show:

```text
Application
    |
    v
Prepare peripherals
    |
    v
Enter a named mode: Sleep, Stop, or Standby
    |
    v
Clock and retained-state behavior depends on the mode
    |
    v
Interrupt/Event
    |
    v
Wake-up
    |
    v
CPU resumes
```

Use a visual power-state indicator.

Keep Sleep, Stop, and Standby as distinct states: wake sources, retained state, clock restart behavior, and whether execution resumes or restarts differ.

---

# 25. Final Animation — Complete STM32 Data Journey

Create a final page that combines the major flows.

Example:

```text
                 STM32L476RG

 Sensor
   |
   v
 ADC ---> DMA ---> SRAM
                 |
                 v
                CPU
                 |
       +---------+----------+
       |                    |
       v                    v
    Algorithm             UART
       |                    |
       v                    v
    Decision             PC/Debug
       |
       v
     GPIO
       |
       v
      LED


CAN Bus
   |
   v
Transceiver
   |
   v
CAN Peripheral
   |
   v
RX FIFO
   |
   v
NVIC
   |
   v
CPU


Timer
  |
  v
NVIC
  |
  v
CPU


RCC
 |
 +----> CPU Clock
 +----> Bus Clock
 +----> Peripheral Clock
```

The final page should allow the learner to select individual paths.

---

# 26. Universal Animation Model

All pages should use the same conceptual animation engine.

Each flow should have:

```text
Flow
 |
 +-- Step 0: Initial state
 +-- Step 1: Highlight source
 +-- Step 2: Create data/event
 +-- Step 3: Move data
 +-- Step 4: Highlight destination
 +-- Step 5: Explain result
 +-- Step 6: Hold final state
```

Represent each step conceptually as:

```json
{
  "step": 1,
  "source": "CPU",
  "destination": "GPIO",
  "event": "WRITE_REGISTER",
  "data": "GPIOA->ODR",
  "description": "CPU performs a memory-mapped write to the GPIO output register."
}
```

The HTML implementation may use JavaScript objects with the same structure.

---

# 27. Automatic Mode

Automatic mode should:

1. Start at Step 0.
2. Highlight the active component.
3. Animate the data/event path.
4. Update the explanation panel.
5. Wait for a configurable delay.
6. Move to the next step.
7. Continue until the final step.
8. Keep the final architecture visible.

Controls:

```text
[Play]
[Pause]
[Reset]
[Previous]
[Next]
Speed: 0.5x / 1x / 2x
```

---

# 28. Manual Mode

Manual mode should never advance automatically.

When the learner presses Next:

```text
Step N
   |
   v
Highlight source
   |
   v
Animate transfer
   |
   v
Highlight destination
   |
   v
Update explanation
```

Previous should restore the previous visual state.

Reset should return to Step 0.

---

# 29. Interactive Component Inspection

Every major component should be clickable.

Examples:

### CPU

Show:

- Cortex-M4F
- Program Counter
- Stack Pointer
- Link Register
- General-purpose registers
- ALU
- FPU
- NVIC

### FLASH

Show:

- Vector table
- `.text`
- `.rodata`
- initial `.data`

### SRAM

Show:

- `.data`
- `.bss`
- heap
- stack

### GPIO

Show:

- MODER
- IDR
- ODR
- BSRR
- PUPDR

### Timer

Show:

- clock
- prescaler
- counter
- ARR
- compare
- status
- interrupt

### UART

Show:

- TX register
- RX register
- status
- baud-rate configuration
- interrupt
- DMA

### DMA

Show:

- source
- destination
- transfer size
- direction
- interrupt
- circular mode

---

# 30. Important Teaching Rules

The generated pages must avoid these misconceptions.

## Rule 1

Do not say:

> "The entire program is loaded into SRAM."

Instead:

> Program instructions normally execute from Flash. Startup code copies initialized writable data into SRAM and clears `.bss`.

## Rule 2

Do not say:

> "GPIO is memory."

Instead:

> GPIO registers are memory-mapped peripheral registers.

## Rule 3

Do not say:

> "Interrupts directly call a callback."

Instead:

```text
Peripheral
 -> interrupt request
 -> NVIC
 -> CPU exception entry
 -> ISR
 -> framework/HAL handler
 -> callback
```

## Rule 4

Do not describe DMA as a second CPU.

DMA is a hardware engine that performs programmed data transfers.

## Rule 5

Do not merge CAN controller and CAN transceiver.

They perform different functions.

## Rule 6

Do not present HAL APIs as the hardware architecture.

For example:

```c
HAL_UART_Transmit(...)
```

is software abstraction.

The underlying architecture is:

```text
Application
 -> HAL
 -> UART Driver
 -> UART Registers
 -> UART Peripheral
 -> Pin
```

---

# 31. Recommended Navigation

Create the HTML pages with this navigation:

```text
STM32 Architecture
|
+-- Persistent context: Cortex-M4 STM32 block diagram
+-- 02 Power-on and reset
+-- 03 Flash and SRAM layout
+-- 04 CPU instruction execution
+-- 05 GPIO output: register to pin
+-- 06 GPIO input: pin to program
+-- 07 RCC and clock flow
+-- PERIPHERALS
|   +-- 08 Timer: clock to event
|   +-- 09 Interrupt: event to callback
|   +-- 10 DMA: hardware data movement
+-- SERIAL I/O
|   +-- 11 UART transmit
|   +-- 12 UART receive with interrupt
|   +-- 13 UART receive with DMA
|   +-- 14 SPI full-duplex transfer
|   +-- 15 I2C register read
+-- SENSING & NETWORKING
|   +-- 16 ADC: analog to digital
|   +-- 17 CAN: application to bus
|   +-- 18 CAN receive interrupt
|   +-- 19 ADC + DMA sensor pipeline
+-- FAULTS & POWER
|   +-- 20 Watchdog: timeout to reset
|   +-- 21 Fault: exception to diagnosis
|   +-- 22 Low-power entry and wake-up
+-- SYSTEM VIEW
    +-- 23 Complete data journey
```

The persistent architecture context is labelled "01" but is not a separate
flow. It remains visible with the relevant route highlighted while the learner
selects any of the 22 detailed flows, numbered 02 through 23.

Each detailed flow has a separately clickable number. Selecting that number
opens an on-page learning prompt for the selected flow, showing its explanation
and core idea; selecting the flow title continues directly to the walkthrough.

---

# 32. Recommended Learning Pattern

Every animation page should answer five questions:

```text
1. WHO generates the data/event?
2. WHERE does it enter the STM32?
3. WHICH hardware block handles it?
4. WHERE does the data go?
5. HOW does the CPU become involved?
```

For each peripheral, explicitly classify the transfer as:

```text
Polling
Interrupt
DMA
```

This classification should be visible on the page.

---

# 33. Example Final Teaching Narrative

The entire course should build toward this mental model:

```text
                 C PROGRAM
                     |
                     v
             CPU INSTRUCTIONS
                     |
                     v
             MEMORY-MAPPED BUS
                     |
        +------------+-------------+
        |            |             |
        v            v             v
      GPIO         TIMER          UART
        |            |             |
        v            v             v
      PIN          EVENT        SERIAL DATA
                     |
                     v
                    NVIC
                     |
                     v
                    CPU


             SENSOR DATA
                  |
                  v
                 ADC
                  |
                  v
                 DMA
                  |
                  v
                 SRAM
                  |
                  v
                 CPU
                  |
                  v
              APPLICATION


              CAN BUS
                  |
                  v
             TRANSCEIVER
                  |
                  v
             CAN CONTROLLER
                  |
                  v
               RX FIFO
                  |
                  v
                 NVIC
                  |
                  v
                 CPU
```

The central message of the animation series is:

> **C statement → CPU instruction → bus transaction → peripheral register → hardware → interrupt/DMA → memory → application**

This should be the recurring visual language across all generated HTML pages.

## Review Summary

The proposed flow list gives the course a strong progression from reset and memory through peripheral transfers and system behavior. Before publishing, validate device-specific examples (board pin wiring, timer clock calculations, USART flags, bxCAN filtering/IRQ names, DMA request mapping, ADC configuration, and low-power wake behavior) against the exact STM32L476RG reference manual and board schematic. Keep the common animation engine separate from these flow definitions so the same controls and explanatory fields are applied consistently.
