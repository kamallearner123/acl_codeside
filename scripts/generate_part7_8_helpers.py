"""
generate_part7_8_helpers.py - Masterclass articles for Modules 33, 34, 35, 37, 38, 39
"""

# -------------------------------------------------------------
# MODULE 33: Hardware Interfacing & MMIO
# -------------------------------------------------------------
MOD33_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 7 • Embedded C &amp; Firmware</span>
          <h1 class="chapter-h1">Module 33: Memory-Mapped I/O (MMIO) &amp; Register Control</h1>
          <p class="chapter-lead">Master bare-metal peripheral manipulation on ARM Cortex-M processors. Deep dive into Advanced Microcontroller Bus Architecture (AHB/APB), memory barriers (DMB, DSB, ISB), atomic register modification via Bit Set/Reset Registers (BSRR), CMSIS register struct mappings, and optimizer traps with volatile.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 65m</div>
            <div class="meta-item">🎯 Standard: ARMv7-M Architecture / CMSIS-Core / C11</div>
            <div class="meta-item">⚡ Target: ARM Cortex-M4 / STM32F4 Bare-Metal</div>
          </div>
        </div>

        <h2 id="sec-33-1">33.1 ARM Cortex-M Bus Architecture & MMIO</h2>
        <p>
          Unlike x86 processors that feature dedicated I/O ports (<code>in</code>/<code>out</code> assembly instructions), ARM microcontrollers utilize <strong>Memory-Mapped I/O (MMIO)</strong>. All on-chip peripherals (GPIO, Timers, UART, I2C, DMA) share the unified 4-gigabyte 32-bit linear address space alongside Flash ROM and SRAM.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">ARM Cortex-M Internal Bus Matrix &amp; Address Map</div>
          <pre style="margin:0;color:#94A3B8;">
0xFFFFFFFF ┌────────────────────────────────────────────────────────┐
           │ System Control Space (NVIC, SysTick, MPU: 0xE000E000)  │ &lt;-- Internal Core Bus
0xE0000000 ├────────────────────────────────────────────────────────┤
           │ External Peripherals / Devices (0xA0000000)            │
0x60000000 ├────────────────────────────────────────────────────────┤
           │ Peripherals Bus (APB1: 0x40000000, APB2: 0x40010000,   │
           │                  AHB1: 0x40020000 - GPIO, RCC, DMA)    │ &lt;-- AHB/APB Bridges
0x40000000 ├────────────────────────────────────────────────────────┤
           │ SRAM (0x20000000 - 0x2001FFFF: 128 KB SRAM1/SRAM2)     │ &lt;-- System Bus (S-Bus)
0x20000000 ├────────────────────────────────────────────────────────┤
           │ Flash ROM (0x08000000 - 0x0807FFFF: 512 KB Flash)      │ &lt;-- I-Code &amp; D-Code Bus
0x00000000 └────────────────────────────────────────────────────────┘
          </pre>
        </div>

        <h2 id="sec-33-2">33.2 The Critical Role of the volatile Qualifier</h2>
        <p>
          In standard C, the optimizing compiler assumes that a memory location only changes if modified by explicit program statements. For hardware peripheral registers (such as an ADC data register or UART status register), the value can change asynchronously due to external physical events or silicon hardware logic!
        </p>

        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">The Deadlock Optimization Bug:</strong> If you omit <code>volatile</code> on a hardware register read:
          <pre><code class="language-c">/* FATAL ERROR: Omission of volatile */
uint32_t *status_reg = (uint32_t *)0x40011000;
while ((*status_reg &amp; 0x01) == 0) { /* Wait for hardware flag */ }</code></pre>
          Under <code>-O2</code> optimization, the compiler hoists the read outside the loop:
          <pre><code class="language-assembly">ldr r2, [r3]       ; Read register once into register R2
.L2:
tst r2, #1         ; Test register R2
beq .L2            ; Infinite loop! CPU never reads hardware memory again!</code></pre>
        </div>

        <p>
          Marking pointers as <code>volatile uint32_t *</code> forces the compiler to emit a physical load/store bus transaction on <strong>every single access</strong>, respecting sequence points.
        </p>

        <h2 id="sec-33-3">33.3 Atomic GPIO Toggling via BSRR Registers</h2>
        <p>
          A common beginner mistake when toggling a GPIO pin is using read-modify-write on the output data register:
        </p>
        <pre><code class="language-c">/* Non-Atomic Read-Modify-Write (Race Hazard!) */
GPIOA-&gt;ODR |= (1 &lt;&lt; 5); /* Read ODR, Modify in ALU, Write back to ODR */</code></pre>
        <p>
          If a high-priority interrupt preempts execution between the Read and the Write instructions and modifies another pin on <code>GPIOA</code>, the mainline code will overwrite and destroy the interrupt’s pin state upon resuming!
        </p>
        <p>
          <strong>The Silicon Solution: Bit Set/Reset Register (BSRR)</strong>. Writing a <code>1</code> to the lower 16 bits atomically sets the pin; writing a <code>1</code> to the upper 16 bits atomically resets the pin. Writes of <code>0</code> have no effect. <strong>Zero read-modify-write! Zero interrupt disabling needed!</strong>
        </p>
        <pre><code class="language-c">/* Atomic 1-cycle Pin Manipulation */
GPIOA-&gt;BSRR = (1U &lt;&lt; 5);       /* Atomically SET Pin 5 high */
GPIOA-&gt;BSRR = (1U &lt;&lt; (5 + 16)); /* Atomically RESET Pin 5 low */</code></pre>

        <h2 id="sec-33-4">33.4 Memory Barriers: DMB, DSB, and ISB</h2>
        <p>
          Modern out-of-order execution pipelines and write buffers can reorder memory transactions. Cortex-M provides dedicated architectural barriers:
        </p>
        <ul>
          <li><strong>DMB (Data Memory Barrier):</strong> Ensures that all explicit memory accesses before the barrier complete before any memory accesses after the barrier. Critical for lock-free ring buffers and DMA transfers.</li>
          <li><strong>DSB (Data Synchronization Barrier):</strong> Halts instruction execution until all outstanding memory bus operations have physically finished. Mandatory after configuring clock registers or disabling interrupts before entering sleep (<code>WFI</code>).</li>
          <li><strong>ISB (Instruction Synchronization Barrier):</strong> Flushes the CPU instruction pipeline, forcing instructions to be refetched from cache/flash. Mandatory after reconfiguring the Memory Protection Unit (MPU) or modifying vector table base addresses (VTOR).</li>
        </ul>

        <h2 id="sec-33-5">33.5 Industrial Case Study: The Medical Infusion Pump Motor Stalling Bug</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Disappearing Motor Pulse Under Compiler Optimization</h4>
          <p>
            An infusion pump stepper motor driver missed 10% of motor steps when the firmware was compiled with <code>-O2</code> release flags, though it operated flawlessly under <code>-O0</code> debug builds.
          </p>
          <p>
            <strong>Investigation:</strong> The pulse generation routine generated a 2-microsecond high pulse by writing directly to GPIO:
          </p>
          <pre><code class="language-c">/* Defective Pulse Generation */
GPIO_TypeDef *gpio = (GPIO_TypeDef *)0x40020000;
gpio-&gt;ODR |= (1 &lt;&lt; 2); // Motor step HIGH
delay_us(2);
gpio-&gt;ODR &amp;= ~(1 &lt;&lt; 2); // Motor step LOW</code></pre>
          <p>
            Because the vendor header file mistakenly omitted <code>volatile</code> on the <code>ODR</code> member in a custom HAL typedef, the compiler optimizer merged the two stores and eliminated the high pulse completely as dead code!
          </p>
          <p>
            <strong>Resolution:</strong> The typedef was corrected to enforce <code>__IO uint32_t ODR;</code> (where <code>__IO</code> expands to <code>volatile</code> in CMSIS), and pin toggling was refactored to use the atomic BSRR register with a compiler memory barrier.
          </p>
        </div>

        <h2 id="sec-33-6">33.6 Hands-On Lab: Writing a Register-Accurate Bare-Metal GPIO Driver</h2>
        <p>
          In this laboratory, you will write a complete, MISRA-compliant bare-metal C driver for the STM32F4 GPIO port from scratch, defining CMSIS register structs, configuring clock gating, and toggling an LED using BSRR.
        </p>

        <h3>Complete Bare-Metal Driver (<code>gpio_driver.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdint.h&gt;

/* Base Addresses from STM32F4 Reference Manual */
#define RCC_BASE        (0x40023800UL)
#define GPIOA_BASE      (0x40020000UL)

/* RCC Register Map */
typedef struct {
    volatile uint32_t CR;
    volatile uint32_t PLLCFGR;
    volatile uint32_t CFGR;
    volatile uint32_t CIR;
    volatile uint32_t AHB1RSTR;
    volatile uint32_t AHB2RSTR;
    volatile uint32_t AHB3RSTR;
    uint32_t          RESERVED0;
    volatile uint32_t APB1RSTR;
    volatile uint32_t APB2RSTR;
    uint32_t          RESERVED1[2];
    volatile uint32_t AHB1ENR; /* GPIOA clock enable bit 0 */
} RCC_TypeDef;

/* GPIO Register Map */
typedef struct {
    volatile uint32_t MODER;   /* Mode register */
    volatile uint32_t OTYPER;  /* Output type register */
    volatile uint32_t OSPEEDR; /* Output speed register */
    volatile uint32_t PUPDR;   /* Pull-up/pull-down register */
    volatile uint32_t IDR;     /* Input data register */
    volatile uint32_t ODR;     /* Output data register */
    volatile uint32_t BSRR;    /* Bit set/reset register (Atomic) */
    volatile uint32_t LCKR;    /* Configuration lock register */
    volatile uint32_t AFR[2];  /* Alternate function registers */
} GPIO_TypeDef;

#define RCC   ((RCC_TypeDef *)RCC_BASE)
#define GPIOA ((GPIO_TypeDef *)GPIOA_BASE)

void led_init(void) {
    /* 1. Enable Clock to GPIOA (Bit 0 of AHB1ENR) */
    RCC-&gt;AHB1ENR |= (1U &lt;&lt; 0);

    /* Memory barrier ensuring clock is active before touching GPIO registers */
    __asm__ volatile("dsb" ::: "memory");

    /* 2. Configure Pin PA5 as General Purpose Output (Mode 01) */
    GPIOA-&gt;MODER &amp;= ~(3U &lt;&lt; (5 * 2)); // Clear mode bits 10 and 11
    GPIOA-&gt;MODER |=  (1U &lt;&lt; (5 * 2)); // Set mode to 01 (Output)

    /* 3. Configure Push-Pull Output (OTYPER bit 5 = 0) */
    GPIOA-&gt;OTYPER &amp;= ~(1U &lt;&lt; 5);

    /* 4. Configure High Speed (OSPEEDR bits = 10) */
    GPIOA-&gt;OSPEEDR |= (2U &lt;&lt; (5 * 2));
}

void led_set(void) {
    /* Atomically set PA5 high */
    GPIOA-&gt;BSRR = (1U &lt;&lt; 5);
}

void led_clear(void) {
    /* Atomically reset PA5 low (Bit 5 + 16 = 21) */
    GPIOA-&gt;BSRR = (1U &lt;&lt; 21);
}</code></pre>

        <h2 id="sec-33-quiz">33.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why is writing to the Bit Set/Reset Register (BSRR) preferred over modifying the Output Data Register (ODR)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) BSRR uses less power.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) BSRR performs atomic single-cycle bit modification without read-modify-write race hazards with concurrent interrupts. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) ODR cannot drive LEDs.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> BSRR allows setting or clearing specific pins atomically via a single 32-bit write, avoiding the hazard where an interrupt preempts a read-modify-write on ODR.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What happens if you omit the <code>volatile</code> qualifier when defining pointers to hardware registers?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) The code generates a compiler syntax error.</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) The optimizer may cache the register value in a CPU register or eliminate polling loops, resulting in infinite loops or skipped writes. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) The microcontroller executes at double speed.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>volatile</code> informs the compiler that the value may change outside program control, preventing the optimizer from eliding memory reads or caching values.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Which ARM architectural barrier instruction flushes the instruction pipeline and forces refetching?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>DMB</code> (Data Memory Barrier)</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>ISB</code> (Instruction Synchronization Barrier) (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>NOP</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>ISB</code> flushes the processor pipeline, ensuring that subsequent instructions are fetched afresh following MPU, cache, or memory mapping reconfigurations.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What bus bridge on the STM32 connects high-speed CPU peripherals (such as DMA and GPIO) to the system crossbar?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) APB1</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) AHB (Advanced High-performance Bus) (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) I2C</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> AHB connects high-bandwidth peripherals (GPIO, DMA, Flash memory, Ethernet) directly to the multi-layer bus matrix at full CPU clock speeds.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What is the primary risk of using C bitfields (e.g. <code>struct { uint32_t pin0:1; }</code>) for memory-mapped hardware registers?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) The ISO C standard does not specify bitfield allocation ordering (MSB to LSB vs LSB to MSB), and the compiler emits non-atomic read-modify-write instructions. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) Bitfields are unsupported by ARM Cortex-M.</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) Bitfields force the code into Thumb-1 mode.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Bitfield layout in memory is implementation-defined by ISO C; moreover, updating a bitfield requires multi-cycle read-modify-write instructions that introduce race conditions on hardware registers.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 34: Linker Scripts & Memory Maps
# -------------------------------------------------------------
MOD34_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 7 • Embedded C &amp; Firmware</span>
          <h1 class="chapter-h1">Module 34: Linker Scripts &amp; Memory Maps</h1>
          <p class="chapter-lead">Master bare-metal GNU ld linker script architecture for embedded microcontrollers. Deep dive into MEMORY and SECTIONS directives, VMA vs LMA address translation, copying .data from Flash to SRAM (AT>FLASH), zero-initializing .bss, stack placement (_estack), and custom memory regions.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: GNU ld Script Syntax / ELF Object Format</div>
            <div class="meta-item">⚡ Target: ARM Cortex-M Flash &amp; SRAM Layouts</div>
          </div>
        </div>

        <h2 id="sec-34-1">34.1 The Linker's Role in Bare-Metal Systems</h2>
        <p>
          In a hosted operating system (such as Linux), the OS kernel’s ELF loader (<code>ld-linux.so</code>) allocates virtual memory pages, loads binary sections, and dynamically relocates symbols at runtime.
        </p>
        <p>
          In a <strong>bare-metal microcontroller</strong>, there is no operating system and no loader. The silicon powers up directly into physical Flash ROM. The <strong>GNU Linker Script (<code>.ld</code>)</strong> is the master architectural blueprint that binds compiled object files into physical silicon addresses.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Linker Script Section Mapping: Flash to SRAM</div>
          <pre style="margin:0;color:#94A3B8;">
Physical Flash ROM (0x08000000 - Non-Volatile)
┌────────────────────────────────────────────────────────┐
│ Vector Table (.isr_vector)                             │
├────────────────────────────────────────────────────────┤
│ Executable Machine Code (.text)                        │
├────────────────────────────────────────────────────────┤
│ Constants &amp; String Literals (.rodata)                  │
├────────────────────────────────────────────────────────┤
│ Initial values for .data (LMA: Load Memory Address)    │ ──(Copied to SRAM during startup!)──┐
└────────────────────────────────────────────────────────┘                                     │
                                                                                               │
Physical SRAM (0x20000000 - Volatile RAM)                                                      │
┌────────────────────────────────────────────────────────┐ &lt;──────────────────────────────────┘
│ Initialized Variables (.data - VMA: Execution Address) │ (Holds non-zero globals: int g_val = 42;)
├────────────────────────────────────────────────────────┤
│ Zero-Initialized Variables (.bss)                      │ (Cleared to 0x00 by Reset_Handler!)
├────────────────────────────────────────────────────────┤
│ Dynamic Heap (.heap) ──▶ (Grows upward)               │
│                                                        │
│                          ▲                             │
│                          └── (Stack Collision Guard)   │
│                                                        │
│ Main Stack (.stack)  ◀── (Grows downward)              │
└────────────────────────────────────────────────────────┘ &lt;-- Top of RAM (_estack = 0x20020000)
          </pre>
        </div>

        <h2 id="sec-34-2">34.2 VMA vs. LMA: The AT>FLASH Directive</h2>
        <p>
          One of the most crucial concepts in embedded linker scripts is the distinction between:
        </p>
        <ul>
          <li><strong>VMA (Virtual / Execution Memory Address):</strong> The memory address where the code expects the data to reside when running (e.g. SRAM: <code>0x20000000</code>).</li>
          <li><strong>LMA (Load Memory Address):</strong> The physical address where the data is permanently stored in the flashed binary image (e.g. Flash ROM: <code>0x08040000</code>).</li>
        </ul>
        <p>
          Initialized global variables (such as <code>uint32_t g_timeout = 5000;</code>) must retain their initial value even after power loss. Therefore, their values must be stored in Flash ROM (LMA). However, because they are mutable variables, they must execute in volatile SRAM (VMA). The linker script expresses this dual-address relationship using <code>AT &gt; FLASH</code>:
        </p>
        <pre><code class="language-ld">.data :
{
  . = ALIGN(4);
  _sdata = .;        /* Create symbol for start of .data in RAM */
  *(.data*)
  . = ALIGN(4);
  _edata = .;        /* Create symbol for end of .data in RAM */
} &gt;RAM AT&gt; FLASH     /* VMA is RAM; LMA is FLASH! */

_sidata = LOADADDR(.data); /* LMA address of .data in Flash */</code></pre>

        <h2 id="sec-34-3">34.3 Production STM32F4 Linker Script</h2>
        <pre><code class="language-ld">/* ==============================================================================
 * Production GNU ld Linker Script for STM32F401RETx (512 KB Flash, 96 KB SRAM)
 * ============================================================================== */

ENTRY(Reset_Handler)

/* Highest address of the user mode stack (Top of RAM) */
_estack = ORIGIN(RAM) + LENGTH(RAM);

/* Minimum stack and heap sizes */
_Min_Heap_Size = 0x200;  /* 512 bytes */
_Min_Stack_Size = 0x400; /* 1024 bytes */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 96K
}

SECTIONS
{
  /* The startup vector table goes first into FLASH */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } &gt;FLASH

  /* Program code and read-only data */
  .text :
  {
    . = ALIGN(4);
    *(.text*)
    *(.glue_7)         /* Glue ARM to Thumb code */
    *(.glue_7t)
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))
    . = ALIGN(4);
    _etext = .;        /* End of text section */
  } &gt;FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata*)
    . = ALIGN(4);
  } &gt;FLASH

  /* Initialized data sections copied into RAM at startup */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } &gt;RAM AT&gt; FLASH

  _sidata = LOADADDR(.data);

  /* Uninitialized data section (.bss) zeroed at startup */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } &gt;RAM

  /* Check that there is enough space left for Stack and Heap */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } &gt;RAM
}</code></pre>

        <h2 id="sec-34-4">34.4 Industrial Case Study: The Misaligned Flash DMA HardFault</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The 4-Byte Unaligned Flash Sector Corruption</h4>
          <p>
            An automotive electronic control unit (ECU) experienced random HardFault crashes immediately following an Over-The-Air (OTA) firmware update in cold weather testing.
          </p>
          <p>
            <strong>Investigation:</strong> The developers added a raw binary cryptographic certificate into Flash using a custom linker section:
          </p>
          <pre><code class="language-ld">.crypto_cert :
{
  *(.crypto_cert)
} &gt;FLASH</code></pre>
          <p>
            The certificate file was 387 bytes (not a multiple of 4). Because the linker section lacked <code>. = ALIGN(4);</code>, the subsequent <code>.text</code> section began on an <strong>odd 2-byte boundary (<code>0x08010183</code>)</strong>.
          </p>
          <p>
            When the CPU attempted to fetch a 32-bit Thumb-2 instruction from an unaligned address across flash word boundaries, the flash memory controller’s 128-bit prefetch buffer generated a bus error, crashing the ECU!
          </p>
          <p>
            <strong>Resolution:</strong> Mandated explicit <code>. = ALIGN(4);</code> before and after every single section declaration in all linker scripts.
          </p>
        </div>

        <h2 id="sec-34-5">34.5 Hands-On Lab: Custom Section for Non-Volatile Bootloader Calibration</h2>
        <p>
          In this lab, you will define a custom flash memory region (<code>CALIBRATION_ROM</code>) in a linker script, and place a C configuration structure directly into it using GCC <code>__attribute__((section(".cal_data")))</code>.
        </p>

        <h3>1. Linker Script Snippet</h3>
        <pre><code class="language-ld">MEMORY
{
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 480K
  CAL_ROM (r)     : ORIGIN = 0x08078000, LENGTH = 32K  /* Dedicated calibration sector */
  RAM (rwx)       : ORIGIN = 0x20000000, LENGTH = 96K
}

SECTIONS
{
  /* Standard sections ... */

  .cal_data :
  {
    . = ALIGN(4);
    _scal_data = .;
    KEEP(*(.cal_data))
    . = ALIGN(4);
    _ecal_data = .;
  } &gt;CAL_ROM
}</code></pre>

        <h3>2. C Source Code (<code>calibration.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdint.h&gt;

typedef struct {
    uint32_t magic_header;
    float    adc_voltage_scale;
    float    temperature_offset;
    uint32_t serial_number;
} device_calibration_t;

/* Explicitly locate this struct into the CAL_ROM sector */
__attribute__((section(".cal_data"), used))
const device_calibration_t g_factory_calibration = {
    .magic_header = 0xC001CAFE,
    .adc_voltage_scale = 3.3005f,
    .temperature_offset = -0.42f,
    .serial_number = 981245
};</code></pre>

        <h2 id="sec-34-quiz">34.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What does the <code>LOADADDR(.data)</code> linker script built-in function return?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) The RAM execution address (VMA) of .data.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) The Flash ROM load address (LMA) where the initial values of .data are permanently stored. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) The size of the RAM section.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>LOADADDR()</code> returns the Load Memory Address (LMA), which the startup code reads to copy initial values from Flash into RAM.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: Why must the <code>KEEP()</code> directive be applied to the vector table in the linker script (<code>KEEP(*(.isr_vector))</code>)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) It compresses the vector table.</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) It prevents the linker from stripping the vector table when link-time garbage collection (<code>--gc-sections</code>) is enabled. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) It encrypts the reset vector.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Because the vector table is referenced by hardware silicon rather than by explicit C code calls, the linker's dead-code elimination (<code>--gc-sections</code>) would discard it without <code>KEEP()</code>.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What represents the initial value of the Main Stack Pointer (MSP) in ARM Cortex-M microcontrollers?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) The first 32-bit word at offset 0x00000000 in Flash (typically <code>_estack</code> at top of RAM). (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Address 0x00000004.</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) Hardcoded register R13.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> On reset, the ARM core hardware loads SP from memory address 0x00000000 and the initial Program Counter (PC) from address 0x00000004.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What symbol in a linker script represents the current memory location counter?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>$</code></label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>.</code> (period / dot) (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>@</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The dot (<code>.</code>) represents the location counter; modifying it advances or aligns memory allocation addresses.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What section contains global variables that are uninitialized or initialized to zero?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>.text</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>.bss</code> (Block Started by Symbol) (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>.rodata</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The <code>.bss</code> section occupies RAM but zero bytes in the flash binary image; it is zeroed out in RAM by the startup code.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 35: Startup Code & Reset Sequence
# -------------------------------------------------------------
MOD35_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 7 • Embedded C &amp; Firmware</span>
          <h1 class="chapter-h1">Module 35: Startup Code &amp; Reset Sequence (crt0)</h1>
          <p class="chapter-lead">Deep dive into the silicon reset lifecycle of ARM Cortex-M microcontrollers. Master the hardware reset sequence, writing the pure C Reset_Handler, copying .data from Flash to RAM, zero-filling .bss, hardware FPU coprocessor initialization, and calling main().</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: ARMv7-M Architecture / C11 Runtime Init</div>
            <div class="meta-item">⚡ Target: Bare-Metal ARM Cortex-M4F / STM32F4</div>
          </div>
        </div>

        <h2 id="sec-35-1">35.1 The Hardware Reset Sequence</h2>
        <p>
          When power is applied to an ARM Cortex-M processor (or when the hardware NRST pin is released), the core executes a strict, unalterable hardware sequence hardwired into the silicon:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Cortex-M Hardware Power-On Execution Sequence</div>
          <pre style="margin:0;color:#94A3B8;">
Power On / Reset Pin Released
          │
          ▼
Hardware samples BOOT pins (determines whether vector table alias is Flash, System ROM, or SRAM)
          │
          ▼
Step 1: Hardware reads 32-bit word from address 0x00000000 ──▶ Loads into MSP (Main Stack Pointer %sp)
          │
          ▼
Step 2: Hardware reads 32-bit word from address 0x00000004 ──▶ Loads into PC (Program Counter %pc)
          │                                                    (LSB must be 1 to indicate Thumb mode!)
          ▼
Step 3: CPU begins executing machine instructions at Reset_Handler()
          </pre>
        </div>

        <h2 id="sec-35-2">35.2 The 4 Mandatory Steps of Reset_Handler (crt0)</h2>
        <p>
          Before any high-level C code (like <code>main()</code>) can execute, the C runtime environment (<code>crt0</code>) must establish memory invariants guaranteed by the ISO C standard:
        </p>

        <ol>
          <li><strong>Copy <code>.data</code> from Flash to RAM:</strong> Initialized global and static variables have their initial values stored in Flash ROM. The startup code copies them word-by-word into SRAM.</li>
          <li><strong>Zero-Fill <code>.bss</code> in RAM:</strong> The ISO C standard guarantees that all uninitialized global/static variables start with value zero. The startup code loops through the <code>.bss</code> memory region and fills it with <code>0x00000000</code>.</li>
          <li><strong>Enable the Hardware Floating-Point Unit (FPU):</strong> On Cortex-M4F, the hardware FPU coprocessor is powered down by default on reset. Any floating-point instruction executed before enabling CP10/CP11 triggers an immediate <code>NOCP (No Coprocessor) UsageFault</code>!</li>
          <li><strong>Branch to <code>main()</code>:</strong> With stacks initialized and global memory established, execution branches to <code>main()</code>.</li>
        </ol>

        <h2 id="sec-35-3">35.3 Writing a Pure C <code>startup_stm32f4.c</code> from Scratch</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;

/* Symbols defined in Linker Script */
extern uint32_t _estack;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

/* Function prototypes */
int main(void);
void Reset_Handler(void);
void Default_Handler(void);

/* Weak aliases for Exception Handlers */
void NMI_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)     __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)    __attribute__((weak, alias("Default_Handler")));

/* Vector Table placed into .isr_vector section */
__attribute__((section(".isr_vector"), used))
uint32_t * const g_pfnVectors[] = {
    (uint32_t *)&amp;_estack,            /* 0x00: Initial Main Stack Pointer */
    (uint32_t *)Reset_Handler,        /* 0x04: Reset Handler */
    (uint32_t *)NMI_Handler,          /* 0x08: Non-Maskable Interrupt */
    (uint32_t *)HardFault_Handler,    /* 0x0C: HardFault Handler */
    (uint32_t *)MemManage_Handler,    /* 0x10: Memory Management Fault */
    (uint32_t *)BusFault_Handler,     /* 0x14: Bus Fault */
    (uint32_t *)UsageFault_Handler,   /* 0x18: Usage Fault */
    0, 0, 0, 0,                       /* Reserved */
    (uint32_t *)SVC_Handler,          /* 0x2C: SVCall */
    0, 0,                             /* Reserved */
    (uint32_t *)PendSV_Handler,       /* 0x38: PendSV */
    (uint32_t *)SysTick_Handler       /* 0x3C: SysTick */
};

/* The C Runtime Reset Entry Point */
void Reset_Handler(void) {
    /* Step 1: Copy .data from Flash ROM to SRAM */
    uint32_t *src = &amp;_sidata;
    uint32_t *dst = &amp;_sdata;
    while (dst &lt; &amp;_edata) {
        *dst++ = *src++;
    }

    /* Step 2: Zero-fill .bss section in SRAM */
    dst = &amp;_sbss;
    while (dst &lt; &amp;_ebss) {
        *dst++ = 0U;
    }

    /* Step 3: Enable Hardware FPU Coprocessor (CP10 and CP11) */
    /* System Control Space: CPACR register at 0xE000ED88 */
    volatile uint32_t *cpacr = (volatile uint32_t *)0xE000ED88;
    *cpacr |= ((3UL &lt;&lt; (10 * 2)) | (3UL &lt;&lt; (11 * 2))); /* Full access */
    __asm__ volatile("dsb\nisb" ::: "memory");

    /* Step 4: Branch to Application Entry Point */
    main();

    /* Trap if main ever returns */
    while (1) {
        __asm__ volatile("wfi");
    }
}

void Default_Handler(void) {
    while (1) {
        /* Infinite loop for unhandled interrupts */
    }
}</code></pre>

        <h2 id="sec-35-4">35.4 Industrial Case Study: Sub-Zero Smart Meter Reset Failure</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Uninitialized Pointer Winter Blackout Incident</h4>
          <p>
            An electrical utility deployed 100,000 smart grid electricity meters. During a severe polar vortex (-30°C), thousands of meters experienced power glitches and rebooted into a corrupted state with spinning fans and disabled metering.
          </p>
          <p>
            <strong>Investigation:</strong> The developers optimized boot time by commenting out the <code>.bss</code> zeroing loop in the startup file to shave 3 milliseconds off reset!
          </p>
          <p>
            They reasoned: <em>"We explicitly initialize all critical variables in C anyway."</em>
          </p>
          <p>
            However, a secondary flash write lock flag was declared as a global:
          </p>
          <pre><code class="language-c">static bool g_flash_write_in_progress; /* Depended on ISO C zero-init! */</code></pre>
          <p>
            During a cold power glitch, unpowered SRAM cells decayed to arbitrary random patterns (often <code>0xFF</code> or <code>0xAA</code>). When the meter rebooted without <code>.bss</code> zeroing, <code>g_flash_write_in_progress</code> woke up with value <code>true</code>! The metering task assumed an ongoing write and refused to log power readings forever.
          </p>
          <p>
            <strong>Corrective Action:</strong> Strict compliance with the ISO C <code>crt0</code> specification was mandated. Zeroing <code>.bss</code> was permanently re-enabled across all firmware projects.
          </p>
        </div>

        <h2 id="sec-35-5">35.5 Hands-On Lab: Verifying BSS and DATA Initialization via GDB</h2>
        <p>
          In this lab, you will compile the custom startup file, set a breakpoint at the entry of <code>Reset_Handler</code> in GDB, inspect SRAM before and after the copy/zero loops, and prove that <code>main()</code> receives pristine memory.
        </p>

        <h3>Verification Script</h3>
        <pre><code class="language-bash"># Connect GDB to running QEMU or OpenOCD target
arm-none-eabi-gdb -q build/firmware.elf

# Set breakpoint at Reset_Handler
(gdb) target remote localhost:3333
(gdb) break Reset_Handler
(gdb) continue

# Inspect uninitialized SRAM at _sdata before copy loop
(gdb) print &amp;_sdata
$1 = (uint32_t *) 0x20000000
(gdb) x/4xw 0x20000000
0x20000000: 0x00000000 0x00000000 0x00000000 0x00000000

# Step past copy loop
(gdb) until main
(gdb) x/4xw 0x20000000
# Displays populated initialized data copied faithfully from Flash!</code></pre>

        <h2 id="sec-35-quiz">35.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why must the least-significant bit (LSB) of function pointers in the ARM Cortex-M vector table be set to 1?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) To indicate parity.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) To instruct the ARM CPU to execute in Thumb instruction mode; an LSB of 0 triggers an immediate UsageFault exception. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) It represents 32-bit alignment.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Cortex-M processors exclusively support Thumb-2 instructions; setting LSB=1 signals Thumb mode to the processor core.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What happens if an application executes a hardware floating-point instruction on Cortex-M4F without first enabling CP10/CP11 in the CPACR register?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) The CPU converts the float to an integer.</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) An immediate NOCP UsageFault exception is generated. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) The calculation returns NaN.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The FPU coprocessor is powered down on reset; executing FPU opcodes without enabling access triggers a No Coprocessor UsageFault.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What memory operations are performed by the Reset_Handler loop handling the <code>.data</code> section?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) Zeroing out memory in RAM.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Copying initialized variable values from Flash ROM (LMA) into volatile SRAM (VMA). (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) Decrypting firmware.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Initialized variables must be copied from non-volatile flash into writeable SRAM before <code>main()</code> executes.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What attribute is used to provide default fallback implementations for interrupt service routines that can be overridden by user drivers?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>__attribute__((weak, alias("Default_Handler")))</code> (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>__attribute__((naked))</code></label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>__attribute__((packed))</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Weak symbols allow the linker to bind to <code>Default_Handler</code> unless a strong definition (such as a custom <code>USART1_IRQHandler</code>) is supplied elsewhere.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What instruction should be placed in an infinite loop after <code>main()</code> in bare-metal startup code to conserve electrical power?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>NOP</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>WFI</code> (Wait For Interrupt) (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>RESET</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>WFI</code> suspends CPU core clocks until an interrupt arrives, drastically reducing power consumption compared to a busy spin loop.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 37: RTOS Fundamentals & FreeRTOS
# -------------------------------------------------------------
MOD37_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 7 • Embedded C &amp; Firmware</span>
          <h1 class="chapter-h1">Module 37: RTOS Fundamentals &amp; FreeRTOS Internals</h1>
          <p class="chapter-lead">Deep dive into real-time operating system kernel architecture on ARM Cortex-M. Master SysTick, PendSV, and SVC context switching mechanics, dual-stack pointers (MSP vs PSP), task control blocks (TCBs), FreeRTOS queues, semaphores, mutex priority inheritance, and the critical FromISR rules.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 70m</div>
            <div class="meta-item">🎯 Standard: FreeRTOS V10.x+ / ARM CMSIS-RTOS2</div>
            <div class="meta-item">⚡ Target: Hard Real-Time ARM Cortex-M Firmware</div>
          </div>
        </div>

        <h2 id="sec-37-1">37.1 Hard vs. Soft Real-Time & Determinism</h2>
        <p>
          A common misconception is that a Real-Time Operating System (RTOS) makes code execute faster. <strong>An RTOS does not guarantee high speed; it guarantees mathematical determinism!</strong>
        </p>
        <ul>
          <li><strong>Hard Real-Time:</strong> Missing a deadline constitutes a total system failure (e.g., deploying an automotive airbag within 15 ms, pacemaker cardiac pacing).</li>
          <li><strong>Soft Real-Time:</strong> Missing a deadline degrades quality of service but does not cause catastrophic damage (e.g., video streaming frame drop).</li>
        </ul>

        <h2 id="sec-37-2">37.2 Preemptive Context Switching on ARM Cortex-M</h2>
        <p>
          How does FreeRTOS switch execution between two tasks without an MMU? Cortex-M silicon provides three dedicated architectural features designed explicitly for RTOS kernels:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Cortex-M Dual Stack &amp; Context Switch Pipeline</div>
          <pre style="margin:0;color:#94A3B8;">
1. Dual Stack Pointers:
   MSP (Main Stack Pointer): Used exclusively by OS Kernel &amp; Hardware Interrupt ISRs.
   PSP (Process Stack Pointer): Used exclusively by User RTOS Tasks (Thread Mode).

2. Context Switch Trigger (SysTick -&gt; PendSV):
   [ Running Task A (PSP) ]
             │
             ├──▶ [ SysTick Interrupt (1 kHz) ] ──▶ Decides Task B should run
             │                                       Pends PendSV exception (lowest priority!)
             │                                       Returns to Task A immediately
             ▼
   [ PendSV Exception Executes ] (Runs when NO other ISRs are active!)
     ├── Hardware automatically pushed: R0-R3, R12, LR, PC, xPSR onto Task A's stack
     ├── PendSV software handler pushes: R4-R11 onto Task A's stack
     ├── Saves Task A's SP into Task A's TCB (Task Control Block)
     ├── Loads Task B's SP from Task B's TCB
     ├── Pops: R4-R11 from Task B's stack
     └── Returns from exception (EXC_RETURN) ──▶ Hardware pops R0-R3, R12, LR, PC, xPSR
                                                 Execution resumes seamlessly in Task B!
          </pre>
        </div>

        <h2 id="sec-37-3">37.3 The Critical FromISR API Rules</h2>
        <p>
          In FreeRTOS, every queue, semaphore, and notification API comes in two distinct variants:
        </p>
        <ul>
          <li>Standard API (e.g. <code>xQueueSend()</code>, <code>xSemaphoreGive()</code>): May only be called from <strong>Task context</strong>. These functions can block and put the calling task to sleep.</li>
          <li>FromISR API (e.g. <code>xQueueSendFromISR()</code>, <code>xSemaphoreGiveFromISR()</code>): Must be used inside <strong>Interrupt Service Routines</strong>. These functions can <strong>NEVER block</strong>!</li>
        </ul>

        <h3>Immediate Preemption upon ISR Exit</h3>
        <p>
          If an ISR wakes up a higher-priority task, the ISR must request a context switch before returning using <code>portYIELD_FROM_ISR()</code>:
        </p>
        <pre><code class="language-c">void USART1_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    uint8_t byte = (uint8_t)USART1-&gt;DR;

    /* Push byte to queue; do NOT block! */
    xQueueSendFromISR(g_uart_queue, &amp;byte, &amp;xHigherPriorityTaskWoken);

    /* If consumer task has higher priority, trigger immediate context switch! */
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}</code></pre>

        <h2 id="sec-37-4">37.4 Industrial Case Study: Autonomous AGV Steering Jitter</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The 50 ms Mutex Deadlock on Factory Floor Vehicles</h4>
          <p>
            An autonomous guided vehicle (AGV) in a distribution warehouse experienced periodic violent steering twitching when navigating optical floor markers.
          </p>
          <p>
            <strong>Investigation:</strong> The AGV executed FreeRTOS. The LiDAR obstacle detection task (Priority 2) and the Motor Control task (Priority 5) shared an internal CAN bus transmission mutex.
          </p>
          <p>
            The developers created the mutex using:
          </p>
          <pre><code class="language-c">g_can_lock = xSemaphoreCreateBinary(); /* Missing Priority Inheritance! */</code></pre>
          <p>
            Because a binary semaphore was used instead of a true Mutex (<code>xSemaphoreCreateMutex()</code>), the system suffered classic Priority Inversion. A telemetry WiFi logging task (Priority 3) preempted the LiDAR task while it held the CAN lock, delaying the Motor Control task by 50 ms.
          </p>
          <p>
            <strong>Resolution:</strong> Converted the synchronization object to <code>xSemaphoreCreateMutex()</code>, enabling FreeRTOS priority inheritance and eliminating steering jitter.
          </p>
        </div>

        <h2 id="sec-37-5">37.5 Hands-On Lab: FreeRTOS Producer-Consumer Sensor Pipeline</h2>
        <p>
          In this laboratory, you will implement a multi-task FreeRTOS architecture featuring a sensor polling task, a thread-safe message queue, and a high-priority telemetry transmitter task.
        </p>

        <h3>Complete FreeRTOS Application (<code>freertos_app.c</code>)</h3>
        <pre><code class="language-c">#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include &lt;stdio.h&gt;

typedef struct {
    uint32_t sample_id;
    int16_t  temperature;
} sensor_msg_t;

static QueueHandle_t g_sensor_queue = NULL;

void vSensorTask(void *pvParameters) {
    sensor_msg_t msg = {0};
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(100); // 10 Hz exact

    for (;;) {
        vTaskDelayUntil(&amp;xLastWakeTime, xFrequency); // Strict periodic execution
        msg.sample_id++;
        msg.temperature = 245; // 24.5 C

        /* Send to queue, wait up to 10 ms if full */
        xQueueSend(g_sensor_queue, &amp;msg, pdMS_TO_TICKS(10));
    }
}

void vTelemetryTask(void *pvParameters) {
    sensor_msg_t received_msg;
    for (;;) {
        /* Block indefinitely until a sensor reading arrives */
        if (xQueueReceive(g_sensor_queue, &amp;received_msg, portMAX_DELAY) == pdPASS) {
            printf("[TELEMETRY] Transmitted Sample #%u: Temp=%d\n",
                   received_msg.sample_id, received_msg.temperature);
        }
    }
}

int main(void) {
    g_sensor_queue = xQueueCreate(8, sizeof(sensor_msg_t));

    xTaskCreate(vSensorTask, "Sensor", configMINIMAL_STACK_SIZE + 128, NULL, 2, NULL);
    xTaskCreate(vTelemetryTask, "Telem", configMINIMAL_STACK_SIZE + 128, NULL, 3, NULL);

    vTaskStartScheduler(); // Starts FreeRTOS Kernel!

    for (;;); // Should never reach here
}</code></pre>

        <h2 id="sec-37-quiz">37.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why does FreeRTOS execute context switches inside the PendSV exception handler rather than directly inside the SysTick handler?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) SysTick has no stack.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) PendSV is configured to the lowest possible interrupt priority, ensuring context switches are deferred until all higher-priority physical ISRs have finished. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) PendSV runs in hardware floating-point mode.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Deferring context switching to PendSV prevents preempting active interrupt service routines, avoiding nested stack corruption.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the primary functional difference between a FreeRTOS Binary Semaphore and a Mutex?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) Mutexes include automatic priority inheritance to eliminate priority inversion; binary semaphores do not. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) Binary semaphores can hold up to 256 items.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) Mutexes can be unlocked by any task.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> FreeRTOS mutexes implement priority inheritance and strict ownership semantics; binary semaphores are intended for task synchronization and lack priority inheritance.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What stack pointer register is actively used when a Cortex-M core is executing in Thread mode running a FreeRTOS user task?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) MSP (Main Stack Pointer)</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) PSP (Process Stack Pointer) (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) USP (User Stack Pointer)</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> FreeRTOS configures user tasks to run with PSP; interrupts and OS kernel startup execute with MSP.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What happens if an engineer calls standard <code>xQueueSend()</code> with a non-zero timeout inside an Interrupt Service Routine?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) The queue sends the item immediately.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) System crash or assertion failure: hardware interrupts cannot block or sleep. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) The interrupt frequency doubles.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> ISRs must never block; only non-blocking <code>FromISR</code> variants may be called from interrupt handlers.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What function guarantees strict, drift-free periodic execution of an RTOS task?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>vTaskDelay()</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>vTaskDelayUntil()</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>sleep()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>vTaskDelayUntil()</code> calculates delays relative to the previous wake time, avoiding cumulative timing drift caused by task execution duration.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 38: MISRA C:2012 / MISRA C:2023 Compliance
# -------------------------------------------------------------
MOD38_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 8 • Secure C Programming</span>
          <h1 class="chapter-h1">Module 38: MISRA C:2012 &amp; MISRA C:2023 Compliance</h1>
          <p class="chapter-lead">Master safety-critical C standards mandated across automotive (ISO 26262), aerospace (DO-178C), and medical devices (IEC 62304). Deep dive into MISRA rule classifications (Mandatory, Required, Advisory), essential type models, pointer conversion restrictions, dynamic allocation bans, and formal deviation permits.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: MISRA C:2012 (AMD 1-3) / MISRA C:2023</div>
            <div class="meta-item">⚡ Target: ISO 26262 ASIL-D &amp; DO-178C Level A Systems</div>
          </div>
        </div>

        <h2 id="sec-38-1">38.1 Why MISRA C is Mandatory in High-Reliability Systems</h2>
        <p>
          The ISO C standard is renowned for its expressive power and runtime performance. However, ISO C contains <strong>hundreds of undefined, unspecified, and implementation-defined behaviors</strong> (such as integer promotion traps, signed overflow, and unsequenced side-effects).
        </p>
        <p>
          The <strong>Motor Industry Software Reliability Association (MISRA)</strong> established MISRA C to define a disciplined, mathematically verifiable subset of C that eliminates dangerous language ambiguities without sacrificing low-level hardware control.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">MISRA Rule Categorization Hierarchy</div>
          <pre style="margin:0;color:#94A3B8;">
1. Mandatory Rules:
   Zero tolerance. Cannot be deviated under any circumstances!
   Example: Rule 9.1 (Uninitialized variables), Rule 22.1 (Freeing reserved memory).

2. Required Rules:
   Must be obeyed unless a formal, documented Engineering Deviation Permit is approved.
   Example: Rule 11.4 (Pointer to integer casting for MMIO registers).

3. Advisory Rules:
   Best-practice engineering recommendations; deviations do not require formal customer sign-off.
   Example: Rule 16.4 (Every switch statement should have a default clause).
          </pre>
        </div>

        <h2 id="sec-38-2">38.2 Core Safety Rules in MISRA C:2012</h2>

        <h3>1. Rule 17.7 (Required): Non-Void Return Values Must Be Tested</h3>
        <p>
          Every non-void function return value must be checked. If deliberately ignored, it must be explicitly cast to <code>(void)</code>:
        </p>
        <pre><code class="language-c">/* VIOLATION */
memcpy(dest, src, 16);

/* COMPLIANT */
(void)memcpy(dest, src, 16);</code></pre>

        <h3>2. Rule 21.3 (Required): Banning Dynamic Memory Allocation</h3>
        <p>
          Functions <code>malloc()</code>, <code>calloc()</code>, <code>realloc()</code>, and <code>free()</code> are strictly prohibited after system initialization. Heap fragmentation, unbounded allocation latency, and out-of-memory states cannot be tolerated in hard real-time controllers. All buffers must be statically allocated.
        </p>

        <h3>3. Rule 14.4 (Required): Boolean Expressions in Control Flow</h3>
        <p>
          The controlling expression of an <code>if</code>, <code>while</code>, or <code>for</code> statement must have explicit boolean type:
        </p>
        <pre><code class="language-c">/* VIOLATION: Using integer as boolean */
if (ptr) { ... }
if (count) { ... }

/* COMPLIANT */
if (ptr != NULL) { ... }
if (count &gt; 0U) { ... }</code></pre>

        <h3>4. Rule 10.4 (Required): Essential Type Model</h3>
        <p>
          Both operands of a binary operator must belong to the exact same <em>Essential Type Category</em>:
        </p>
        <pre><code class="language-c">uint8_t a = 5U;
uint16_t b = 100U;
/* VIOLATION: Mismatched unsigned types */
uint16_t c = a + b;

/* COMPLIANT: Explicit cast to identical type */
uint16_t c = (uint16_t)a + b;</code></pre>

        <h2 id="sec-38-3">38.3 Writing Formal Deviation Permits for MMIO</h2>
        <p>
          In bare-metal embedded systems, hardware registers exist at physical addresses. Casting an integer literal like <code>0x40020000UL</code> to a pointer violates <strong>Rule 11.4</strong> (conversion between integer and pointer).
        </p>
        <p>
          To write valid firmware, engineers document a formal <strong>MISRA Deviation Record (MDR)</strong>:
        </p>
        <pre><code class="language-c">/*
 * DEVIATION PERMIT: MDR-GPIO-001
 * MISRA Rule: 11.4 (A conversion should not be performed between a pointer to object and an integer)
 * Justification: Hardware peripheral registers on STM32F4 are memory-mapped to fixed physical addresses.
 * Safety Case: Register definitions are verified against the STM32F4 Reference Manual and tested in HIL.
 */
#define GPIOA ((GPIO_TypeDef *)0x40020000UL) /* Deviation Permitted */</code></pre>

        <h2 id="sec-38-4">38.4 Industrial Case Study: The Toyota Unintended Acceleration Assessment</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The NASA Software Analysis of Electronic Throttle Control</h4>
          <p>
            Following lawsuits alleging unintended acceleration in consumer vehicles, NASA and independent software safety experts audited the vehicle’s Electronic Throttle Control System (ETCS) C codebase.
          </p>
          <p>
            <strong>Audit Findings:</strong>
            The software failed to adhere to MISRA coding guidelines:
          </p>
          <ul>
            <li>Over <strong>2,200 global variables</strong> shared across concurrent tasks with inadequate synchronization.</li>
            <li>Cyclomatic complexity of key throttle angle routines exceeded <strong>50</strong> (safety guidelines mandate &le; 10).</li>
            <li>Ignoring function return codes for safety-critical checks.</li>
            <li>Recursion and unbounded task stack depth risking silent stack overflow into global memory.</li>
          </ul>
          <p>
            The trial resulted in a $1.2 billion settlement, cementing MISRA C compliance as a non-negotiable legal and engineering requirement across the global automotive and aerospace supply chain.
          </p>
        </div>

        <h2 id="sec-38-5">38.5 Hands-On Lab: Refactoring Non-Compliant C Code to 100% MISRA Compliance</h2>
        <p>
          In this lab, you will audit a buggy sensor reading function violating 5 MISRA C rules, and refactor it into clean, compliant production code.
        </p>

        <h3>1. Non-Compliant Code (<code>dirty_code.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;

int read_voltage(int *out) {
    if (out) { // MISRA 14.4 Violation: not explicit boolean
        *out = 3300;
        return 1;
    }
    // MISRA 17.4 Violation: function does not exit through single point
}

void process(void) {
    int v;
    read_voltage(&amp;v); // MISRA 17.7 Violation: Return value ignored
}</code></pre>

        <h3>2. Fully Refactored MISRA C:2012 Compliant Code</h3>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

typedef enum {
    STATUS_OK = 0,
    STATUS_ERR_NULL_PARAM = 1
} status_code_t;

status_code_t read_voltage(int32_t * const out_milli_volts) {
    status_code_t status = STATUS_ERR_NULL_PARAM;

    if (out_milli_volts != (void *)0) {
        *out_milli_volts = 3300;
        status = STATUS_OK;
    }

    return status; /* Single exit point */
}

void process(void) {
    int32_t voltage_mV = 0;
    status_code_t result = read_voltage(&amp;voltage_mV);
    if (result != STATUS_OK) {
        /* Handled error status defensively */
        voltage_mV = 0;
    }
}</code></pre>

        <h2 id="sec-38-quiz">38.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Under MISRA C:2012, which classification of rule can NEVER be deviated under any circumstances?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Advisory</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) Mandatory (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) Required</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Mandatory rules permit zero deviations; any codebase violating a Mandatory rule fails compliance automatically.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: Why does MISRA C Rule 21.3 prohibit dynamic memory allocation (<code>malloc</code>/<code>free</code>) in safety-critical systems?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) Dynamic allocation can cause unbounded allocation latency, heap fragmentation, and unhandled out-of-memory faults. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) Microcontrollers lack heap memory.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) C11 removed malloc.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Heap allocation is non-deterministic and can fail unpredictably at runtime; deterministic safety requires static memory sizing.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: How must a function call be structured under MISRA Rule 17.7 if its return value is intentionally ignored?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) Prefix with <code>ignore</code>.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Explicitly cast the function call to <code>(void)</code>. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) Place inside comments.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Explicitly casting to <code>(void)</code> documents to reviewers and static analyzers that discarding the return value was deliberate.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: Under MISRA Rule 14.4, is <code>if (status)</code> legal if <code>status</code> is an <code>int</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) Yes, because in C non-zero is true.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) No; controlling expressions must have an explicit boolean type (e.g. <code>if (status != 0)</code>). (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) Only if compiled with <code>-O3</code>.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> MISRA mandates that control expressions evaluate an essentially boolean expression to prevent subtle logical errors.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What document justifies an intentional deviation from a Required MISRA rule in safety-critical projects?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) Git commit message</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) Formal Deviation Permit / Record with technical safety rationale. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) Makefile comment</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Required rules can be deviated only with formal, signed engineering documentation detailing why the deviation is necessary and why it is safe.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 39: CERT C & Defensive Coding Practices
# -------------------------------------------------------------
MOD39_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 8 • Secure C Programming</span>
          <h1 class="chapter-h1">Module 39: CERT C &amp; Defensive Coding Practices</h1>
          <p class="chapter-lead">Master the SEI CERT C Secure Coding Standard. Deep dive into integer wrap-around vulnerabilities (INT30-C/INT32-C), string boundary truncation hazards (STR31-C), pointer sanitization post-free (MEM30-C), branchless overflow detection, and defensive contract programming in C.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: SEI CERT C Coding Standard / Common Weakness Enumeration (CWE)</div>
            <div class="meta-item">⚡ Target: High-Security Networked &amp; Embedded C Software</div>
          </div>
        </div>

        <h2 id="sec-39-1">39.1 MISRA C vs. CERT C: Functional Safety vs. Security</h2>
        <p>
          Engineers frequently ask: <em>"If our codebase is MISRA-compliant, is it secure?"</em>
        </p>
        <p>
          Not necessarily! <strong>MISRA C</strong> was designed primarily for <strong>Functional Safety</strong> (ensuring software does not fail randomly in closed automotive/avionics environments). <strong>CERT C</strong> was designed by Carnegie Mellon’s Software Engineering Institute explicitly for <strong>Cybersecurity</strong> (preventing active malicious adversaries from exploiting software vulnerabilities).
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Safety vs. Security Standards Spectrum</div>
          <pre style="margin:0;color:#94A3B8;">
MISRA C:2012 / 2023 ──▶ Focus: Reliability, determinism, banning risky C syntax.
                        Audience: Closed embedded ECUs, braking, steering, avionics.

CERT C Standard     ──▶ Focus: Defeating malicious exploits, buffer overflows, tainted data.
                        Audience: Connected devices, IoT, networking stacks, Linux services.
          </pre>
        </div>

        <h2 id="sec-39-2">39.2 Top CERT C Vulnerabilities & Mitigations</h2>

        <h3>1. INT30-C & INT32-C: Integer Overflow & Wrap-Around</h3>
        <p>
          In C, unsigned integer overflow wraps around modulo <code>2^N</code>. Signed integer overflow is <strong>Undefined Behavior</strong>!
        </p>
        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">Exploitable Integer Wrap-Around:</strong>
          <pre><code class="language-c">/* VULNERABLE: If num_items = 0x40000001 on 32-bit:
   num_items * sizeof(int) wraps to 4 bytes!
   malloc allocates 4 bytes, but loop writes 1 GB of data, corrupting heap! */
void *allocate_array(size_t num_items) {
    return malloc(num_items * sizeof(int));
}</code></pre>
        </div>

        <h3>Secure Multiplication Bounds Check (CERT INT30-C Compliant)</h3>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stddef.h&gt;

void *secure_allocate_array(size_t num_items, size_t elem_size) {
    if (elem_size == 0 || num_items == 0) return NULL;

    /* Check for overflow before multiplying */
    if (num_items &gt; (SIZE_MAX / elem_size)) {
        return NULL; /* Overflow prevented! */
    }

    return malloc(num_items * elem_size);
}</code></pre>

        <h3>2. STR31-C: String Copying & Truncation</h3>
        <p>
          Legacy C string functions (<code>strcpy</code>, <code>strcat</code>, <code>sprintf</code>) do not perform bounds checking and cause catastrophic stack buffer overflows.
        </p>
        <p>
          Even <code>strncpy</code> is hazardous: if the source string exceeds the destination buffer length, <code>strncpy</code> <strong>does not null-terminate the destination!</strong>
        </p>
        <pre><code class="language-c">/* SECURE STRING COPY PATTERN */
void secure_copy(char *dest, size_t dest_size, const char *src) {
    if (!dest || dest_size == 0 || !src) return;
    snprintf(dest, dest_size, "%s", src);
    dest[dest_size - 1] = '\0'; /* Guaranteed null-termination */
}</code></pre>

        <h3>3. MEM30-C: Dangling Pointers & Use-After-Free</h3>
        <p>
          When memory is freed, the pointer variable still holds the physical address of the deallocated memory. Accessing it is undefined behavior.
        </p>
        <pre><code class="language-c">/* DEFENSIVE POINTER SCRUBBING MACRO */
#define SAFE_FREE(ptr) do { \
    free(ptr);              \
    (ptr) = NULL;           \
} while(0)</code></pre>

        <h2 id="sec-39-3">39.3 Compiler Built-in Safe Arithmetic</h2>
        <p>
          Modern GCC and Clang provide hardware-accelerated integer overflow builtins that check CPU carry and overflow flags directly in a single assembly instruction:
        </p>
        <pre><code class="language-c">uint32_t a = 0xFFFFFFFF;
uint32_t b = 10;
uint32_t result;

/* Branchless, Hardware-Assisted Overflow Detection */
if (__builtin_add_overflow(a, b, &amp;result)) {
    /* Overflow occurred! Handle defensively */
} else {
    /* Valid result */
}</code></pre>

        <h2 id="sec-39-4">39.4 Industrial Case Study: Automotive Remote Keyless Entry Exploit</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Rolling Code 16-Bit Wrap-Around Vehicle Theft Bug</h4>
          <p>
            Security researchers discovered that a popular line of vehicles could be unlocked by replaying intercepted RF key fob transmissions.
          </p>
          <p>
            <strong>Vulnerability Analysis:</strong> The vehicle verified rolling codes using a 16-bit sequence counter. To prevent denial-of-service when buttons were pressed away from the car, the car accepted sequence numbers within a future window of 256 counts:
          </p>
          <pre><code class="language-c">/* Vulnerable Sequence Check */
if ((received_seq - stored_seq) &lt; 256) {
    unlock_doors();
}</code></pre>
          <p>
            Due to implicit integer promotion to 32-bit signed integers, when <code>received_seq = 0x0005</code> and <code>stored_seq = 0xFFF0</code> (normal counter wrap-around), the expression evaluated as <code>5 - 65520 = -65515</code>. Because <code>-65515 &lt; 256</code> is true, the car unlocked!
          </p>
          <p>
            Worse: an adversary transmitting <code>received_seq = 0</code> always produced a negative number, unlocking the vehicle instantly at any time!
          </p>
          <p>
            <strong>The Fix:</strong> Explicit unsigned arithmetic masking adhering to CERT INT30-C.
          </p>
        </div>

        <h2 id="sec-39-5">39.5 Hands-On Lab: Writing a Secure Integer Arithmetic Library</h2>
        <p>
          In this lab, you will write a complete, MISRA and CERT compliant safe arithmetic library in C providing overflow-checked addition, subtraction, and multiplication.
        </p>

        <h3>Safe Math Library (<code>safe_math.h</code> & <code>safe_math.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

bool safe_add_u32(uint32_t a, uint32_t b, uint32_t *out) {
    if (!out) return false;
    #if defined(__GNUC__) || defined(__clang__)
        return !__builtin_add_overflow(a, b, out);
    #else
        if (UINT32_MAX - a &lt; b) return false;
        *out = a + b;
        return true;
    #endif
}

bool safe_mul_u32(uint32_t a, uint32_t b, uint32_t *out) {
    if (!out) return false;
    #if defined(__GNUC__) || defined(__clang__)
        return !__builtin_mul_overflow(a, b, out);
    #else
        if (a &gt; 0 &amp;&amp; b &gt; (UINT32_MAX / a)) return false;
        *out = a * b;
        return true;
    #endif
}</code></pre>

        <h2 id="sec-39-quiz">39.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: According to ISO C standards, what is the consequence of signed integer overflow?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Guaranteed wrap-around modulo 2^32.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) Undefined Behavior (UB), allowing the compiler to optimize away safety checks. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) An automatic hardware exception.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Signed integer overflow is undefined behavior in ISO C; compilers assume overflow can never occur and may delete downstream bounds checks.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What CERT C rule mandates setting pointer variables to NULL immediately following <code>free()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) MEM30-C (Do not access freed memory) (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) FIO37-C</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) STR31-C</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Setting pointers to NULL after freeing prevents dangling pointer dereferences and double-free vulnerabilities.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What dangerous behavior occurs with standard <code>strncpy(dest, src, n)</code> if <code>strlen(src) &gt;= n</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) It truncates with a null byte.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) It does NOT null-terminate the destination buffer at all. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) It returns NULL.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>strncpy</code> fails to append a null terminator if the source length equals or exceeds <code>n</code>, leaving <code>dest</code> as an unterminated character array.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What GCC builtin performs hardware-checked integer addition and returns true on overflow?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>__builtin_add_overflow()</code> (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>__builtin_safe_add()</code></label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>__builtin_carry()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>__builtin_add_overflow(a, b, &amp;res)</code> executes single-cycle arithmetic and tests the hardware overflow flag directly.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: How can multiplication overflow be checked prior to executing <code>a * b</code> without builtins?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>if (a &gt; (MAX / b))</code> (assuming b &gt; 0) (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>if (a * b &lt; a)</code></label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>if (a + b &gt; MAX)</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Precondition checking using division (<code>a &gt; MAX / b</code>) detects overflow <em>before</em> the arithmetic operation executes.
            </div>
          </div>
        </div>'''
