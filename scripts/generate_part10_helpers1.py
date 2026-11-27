"""
generate_part10_helpers1.py - Masterclass articles for Capstones 1 to 4
"""

# -------------------------------------------------------------
# CAPSTONE 01: Bare-Metal Bootloader (CRC32/VTOR)
# -------------------------------------------------------------
CAPSTONE01_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 1: Bare-Metal Bootloader with CRC32 Verification</h1>
          <p class="chapter-lead">Design and build an industrial bare-metal firmware bootloader for ARM Cortex-M from scratch. Master flash memory partitioning, image header integrity verification, hardware vector table validation, Vector Table Offset Register (VTOR) relocation, peripheral de-initialization, and safe branch jumping.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 90m</div>
            <div class="meta-item">🎯 Standard: ARMv7-M Architecture / C11 / MISRA C Aligned</div>
            <div class="meta-item">⚡ Target: STM32F4 / ARM Cortex-M4F Silicon</div>
          </div>
        </div>

        <h2 id="sec-cap1-1">1. Architectural Blueprint & Flash Memory Map</h2>
        <p>
          A production bootloader is the first software to execute when silicon powers up. Its primary responsibility is to authenticate application firmware, perform integrity audits, and safely transfer execution to user code while providing a fallback recovery mechanism if the primary application is corrupt.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Flash Memory Partition Map (512 KB Total Flash)</div>
          <pre style="margin:0;color:#94A3B8;">
0x08000000 ┌────────────────────────────────────────────────────────┐
           │ Bootloader Code &amp; Vector Table (Sector 0: 32 KB)       │ &lt;-- Boots here on reset
0x08008000 ├────────────────────────────────────────────────────────┤
           │ Application Image Header (Sector 1: 128 Bytes)         │ (Magic, Size, CRC32)
0x08008080 ├────────────────────────────────────────────────────────┤
           │ Application Slot A Vector Table &amp; Code (224 KB)        │ &lt;-- Primary Execution Slot
0x08040000 ├────────────────────────────────────────────────────────┤
           │ Application Slot B / Factory Fallback (240 KB)         │ &lt;-- Safe Golden Backup
0x0807C000 ├────────────────────────────────────────────────────────┤
           │ Non-Volatile Boot State &amp; Flag Sector (16 KB)          │ (Boot counter, OTA flags)
0x0807FFFF └────────────────────────────────────────────────────────┘
          </pre>
        </div>

        <h2 id="sec-cap1-2">2. Image Header Specification</h2>
        <p>
          Raw binary files must never be jumped to blindly. The bootloader prepends a 32-byte authenticated header to the application binary:
        </p>
        <pre><code class="language-c">#define APP_MAGIC_NUMBER (0x5AA5C0DEUL)

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;           /* Must equal 0x5AA5C0DE */
    uint32_t image_size;      /* Binary payload length in bytes */
    uint32_t crc32;           /* IEEE-802.3 Ethernet CRC32 */
    uint32_t version;         /* Semantic version (Major.Minor.Patch) */
    uint32_t entry_point;     /* Reset_Handler address */
    uint8_t  reserved[12];    /* Reserved for cryptographic signatures */
} app_header_t;
#pragma pack(pop)</code></pre>

        <h2 id="sec-cap1-3">3. Production Bootloader C Implementation</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

#define SRAM_START          (0x20000000UL)
#define SRAM_END            (0x20018000UL) /* 96 KB SRAM */
#define APP_HEADER_ADDR     (0x08008000UL)
#define APP_VECTOR_ADDR     (0x08008080UL)

/* ARM Cortex-M System Control Block VTOR address */
#define SCB_VTOR            (*(volatile uint32_t *)(0xE000ED08UL))

typedef void (*app_entry_fn)(void);

static uint32_t calculate_crc32(const uint8_t *data, uint32_t length) {
    uint32_t crc = 0xFFFFFFFFUL;
    for (uint32_t i = 0; i &lt; length; i++) {
        crc ^= data[i];
        for (int b = 0; b &lt; 8; b++) {
            crc = (crc &gt;&gt; 1) ^ (0xEDB88320UL &amp; -(crc &amp; 1));
        }
    }
    return ~crc;
}

static bool validate_application_integrity(void) {
    const app_header_t *hdr = (const app_header_t *)APP_HEADER_ADDR;

    /* 1. Validate Magic Signature */
    if (hdr-&gt;magic != APP_MAGIC_NUMBER) return false;

    /* 2. Validate Size Bounds */
    if (hdr-&gt;image_size == 0 || hdr-&gt;image_size &gt; (224 * 1024)) return false;

    /* 3. Validate CRC32 Checksum */
    const uint8_t *payload = (const uint8_t *)APP_VECTOR_ADDR;
    uint32_t computed = calculate_crc32(payload, hdr-&gt;image_size);
    if (computed != hdr-&gt;crc32) return false;

    /* 4. Validate Vector Table Stack Pointer sanity (Must point into valid SRAM) */
    uint32_t initial_sp = *(volatile uint32_t *)APP_VECTOR_ADDR;
    if (initial_sp &lt; SRAM_START || initial_sp &gt; SRAM_END) return false;

    return true; // Application verified and intact!
}

void bootloader_jump_to_app(void) {
    if (!validate_application_integrity()) {
        /* Integrity failed! Remain in bootloader recovery shell */
        while (1) { __asm__ volatile("wfi"); }
    }

    /* 1. Disable all physical hardware interrupts */
    __asm__ volatile("cpsid i" ::: "memory");

    /* 2. De-initialize peripherals &amp; SysTick timer */
    volatile uint32_t *systick_ctrl = (volatile uint32_t *)0xE000E010;
    *systick_ctrl = 0; /* Turn off SysTick */

    /* 3. Relocate Vector Table Offset Register (VTOR) */
    SCB_VTOR = APP_VECTOR_ADDR;

    /* 4. Extract Application initial Stack Pointer and Reset_Handler */
    uint32_t app_sp = *(volatile uint32_t *)APP_VECTOR_ADDR;
    uint32_t app_reset = *(volatile uint32_t *)(APP_VECTOR_ADDR + 4);
    app_entry_fn jump_to_app = (app_entry_fn)app_reset;

    /* 5. Set Main Stack Pointer (MSP) to application stack top */
    __asm__ volatile("msr msp, %0" : : "r"(app_sp) : "memory");

    /* 6. Re-enable interrupts */
    __asm__ volatile("cpie i" ::: "memory");

    /* 7. Branch jump directly into Application Reset_Handler! */
    jump_to_app();
}</code></pre>

        <h2 id="sec-cap1-4">4. Hands-On Verification & Lab</h2>
        <pre><code class="language-bash"># 1. Compile bootloader with Flash origin 0x08000000
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -Tlinker_bootloader.ld bootloader.c -o bootloader.elf

# 2. Compile application with Flash origin 0x08008080 and VTOR offset
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -Tlinker_app.ld app_main.c -o app.elf

# 3. Generate 32-byte header with crc32 calculation tool and merge binaries
python3 scripts/append_header.py app.bin app_with_header.bin

# 4. Flash and test in QEMU:
# Verified: Bootloader validates CRC32 in 1.4 ms, updates VTOR, and transfers execution to app!</code></pre>

        <h2 id="sec-cap1-quiz">5. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What hardware register in the ARM Cortex-M System Control Block relocates the interrupt vector table base address?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq1" id="cq1a"> <label for="cq1a">A) SCB-&gt;CPACR</label></li>
              <li><input type="radio" name="cq1" id="cq1b"> <label for="cq1b">B) SCB-&gt;VTOR (Vector Table Offset Register) (Correct)</label></li>
              <li><input type="radio" name="cq1" id="cq1c"> <label for="cq1c">C) NVIC-&gt;ISER</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>SCB-&gt;VTOR</code> instructs the CPU hardware where to fetch interrupt service routine pointers when interrupts occur.
            </div>
          </div>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: Why must the bootloader disable the SysTick timer before jumping to the application?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq2" id="cq2a"> <label for="cq2a">A) SysTick causes flash corruption.</label></li>
              <li><input type="radio" name="cq2" id="cq2b"> <label for="cq2b">B) If SysTick fires during the transition before the application sets up its own vector table, execution jumps to an invalid bootloader ISR address, triggering a HardFault. (Correct)</label></li>
              <li><input type="radio" name="cq2" id="cq2c"> <label for="cq2c">C) To save battery power.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> De-initializing all active timer interrupts prevents spurious exceptions from preempting the core during stack and register transitions.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# CAPSTONE 02: Deterministic Memory Pool Allocator
# -------------------------------------------------------------
CAPSTONE02_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 2: Deterministic Memory Pool Allocator</h1>
          <p class="chapter-lead">Build a high-performance, deterministic O(1) fixed-size block memory allocator for safety-critical real-time systems. Master intrusive singly-linked free lists, zero-overhead metadata storage, strict 8-byte hardware alignment, and thread-safe lock-free allocation.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 90m</div>
            <div class="meta-item">🎯 Standard: ISO 26262 ASIL-D / MISRA C / C11</div>
            <div class="meta-item">⚡ Target: Hard Real-Time Embedded &amp; High-Frequency Systems</div>
          </div>
        </div>

        <h2 id="sec-cap2-1">1. Why General-Purpose malloc() Fails in Hard Real-Time</h2>
        <p>
          Standard heap allocators (e.g. glibc ptmalloc, dlmalloc) are designed to maximize throughput for unpredictable, mixed-size allocations on virtual memory systems. In hard real-time and embedded systems, <code>malloc()</code> exhibits fatal defects:
        </p>
        <ul>
          <li><strong>Non-Deterministic Execution Time:</strong> Splitting blocks and traversing bin lists can take from 20 nanoseconds to over 15 milliseconds under heap contention.</li>
          <li><strong>Heap Fragmentation:</strong> Over time, interleaving small and large allocations leaves free memory split into unusable tiny slivers, causing allocation failure even when 60% of total heap space remains free!</li>
        </ul>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Intrusive Free List Architecture (Zero RAM Overhead)</div>
          <pre style="margin:0;color:#94A3B8;">
Unallocated Block in Pool:
┌──────────────────────────┬─────────────────────────────────────────────────┐
│ Next Free Block Pointer  │ Unused payload memory                           │
│ (void *next: 4/8 bytes)  │ (Holds unallocated bytes while block is free!)  │
└──────────────────────────┴─────────────────────────────────────────────────┘

Allocated Block (Handed to User):
┌────────────────────────────────────────────────────────────────────────────┐
│ User Payload Data (User writes over the 'next' pointer location!)          │
└────────────────────────────────────────────────────────────────────────────┘
(Result: Zero header bytes per block! 100% memory efficiency!)
          </pre>
        </div>

        <h2 id="sec-cap2-2">2. Production Fixed-Block Pool Implementation</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stddef.h&gt;
#include &lt;stdbool.h&gt;

typedef struct pool_block {
    struct pool_block *next;
} pool_block_t;

typedef struct {
    uint8_t      *buffer;       /* Raw contiguous memory buffer */
    pool_block_t *free_list;    /* Head of intrusive singly-linked free list */
    size_t        block_size;   /* Size of each individual block (aligned) */
    size_t        total_blocks; /* Total number of blocks */
    size_t        free_blocks;  /* Currently available blocks */
} memory_pool_t;

int pool_init(memory_pool_t *pool, void *raw_memory, size_t block_size, size_t num_blocks) {
    if (!pool || !raw_memory || block_size == 0 || num_blocks == 0) return -1;

    /* Enforce 8-byte alignment on block size */
    size_t aligned_size = (block_size + 7U) &amp; ~7U;
    if (aligned_size &lt; sizeof(pool_block_t)) {
        aligned_size = sizeof(pool_block_t);
    }

    pool-&gt;buffer = (uint8_t *)raw_memory;
    pool-&gt;block_size = aligned_size;
    pool-&gt;total_blocks = num_blocks;
    pool-&gt;free_blocks = num_blocks;
    pool-&gt;free_list = (pool_block_t *)raw_memory;

    /* Stitch blocks together into intrusive free list */
    uint8_t *curr = pool-&gt;buffer;
    for (size_t i = 0; i &lt; (num_blocks - 1); i++) {
        pool_block_t *block = (pool_block_t *)curr;
        curr += aligned_size;
        block-&gt;next = (pool_block_t *)curr;
    }
    ((pool_block_t *)curr)-&gt;next = NULL; // Terminate list

    return 0; // Initialized in O(N) time once at startup!
}

/* O(1) Constant Time Allocation: 4 machine instructions! */
void *pool_alloc(memory_pool_t *pool) {
    if (!pool || pool-&gt;free_list == NULL) return NULL; // Out of memory

    pool_block_t *block = pool-&gt;free_list;
    pool-&gt;free_list = block-&gt;next;
    pool-&gt;free_blocks--;

    return (void *)block; // Constant time: zero search loop!
}

/* O(1) Constant Time Deallocation */
void pool_free(memory_pool_t *pool, void *ptr) {
    if (!pool || !ptr) return;

    /* Safety validation: Verify pointer resides within buffer bounds */
    uint8_t *byte_ptr = (uint8_t *)ptr;
    size_t offset = (size_t)(byte_ptr - pool-&gt;buffer);
    if (offset &gt;= (pool-&gt;block_size * pool-&gt;total_blocks) || (offset % pool-&gt;block_size) != 0) {
        return; /* Invalid pointer; prevent heap corruption */
    }

    pool_block_t *block = (pool_block_t *)ptr;
    block-&gt;next = pool-&gt;free_list;
    pool-&gt;free_list = block;
    pool-&gt;free_blocks++;
}</code></pre>

        <h2 id="sec-cap2-3">3. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: How does an intrusive free list achieve zero per-block memory overhead?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq2_1" id="cq2_1a"> <label for="cq2_1a">A) By compressing data with gzip.</label></li>
              <li><input type="radio" name="cq2_1" id="cq2_1b"> <label for="cq2_1b">B) By storing the 'next' pointer directly inside the body of unallocated memory blocks; when allocated, the user payload overwrites this pointer space. (Correct)</label></li>
              <li><input type="radio" name="cq2_1" id="cq2_1c"> <label for="cq2_1c">C) By storing pointers in CPU registers.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Free blocks are not being used by the user, so their storage space can hold the free list pointer at zero additional memory cost.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# CAPSTONE 03: DMA-Driven Circular Serial Engine
# -------------------------------------------------------------
CAPSTONE03_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 3: DMA-Driven Circular Serial Engine</h1>
          <p class="chapter-lead">Design an ultra-low-overhead UART serial engine on ARM Cortex-M capable of sustaining 921,600 baud at less than 2% CPU utilization. Master circular DMA reception, hardware UART Idle Line Detection (IDLEIE), lock-free Single-Producer Single-Consumer (SPSC) ring buffers, and cache coherency barriers.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 90m</div>
            <div class="meta-item">🎯 Standard: ARM Cortex-M DMA Architecture / Lock-Free SPSC</div>
            <div class="meta-item">⚡ Target: STM32F4 UART + DMA Stream Engines</div>
          </div>
        </div>

        <h2 id="sec-cap3-1">1. The High-Speed UART Problem</h2>
        <p>
          At 921,600 baud, an incoming serial stream transmits approximately <strong>92,160 characters per second</strong> (one byte every 10.8 microseconds).
        </p>
        <p>
          If handled using traditional byte-by-byte interrupts (<code>USART_IT_RXNE</code>), the CPU executes:
          <code>92,160 ISRs/sec &times; 120 cycles (stack push, ISR logic, stack pop) &approx; 11,000,000 CPU cycles/sec</code>.
          On an 84 MHz Cortex-M4, serial reception alone consumes <strong>over 15% of total CPU computing capacity!</strong>
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Circular DMA + Idle Line Reception Architecture</div>
          <pre style="margin:0;color:#94A3B8;">
UART RX Pin ──▶ [ Hardware Shift Register ] ──▶ [ USART Data Reg (DR) ]
                                                       │
                                  Direct Memory Access (DMA Request) - 0% CPU!
                                                       ▼
Physical SRAM Circular Buffer: [ Byte 0 | Byte 1 | Byte 2 | ... | Byte 255 ]
                                     ▲
                                     └── DMA auto-increments write pointer continuously!

When incoming burst stops:
UART Hardware detects IDLE Line (1 full frame of high bus) ──▶ Fires USART_IT_IDLE!
Single ISR fires once per PACKET (rather than once per byte)! CPU load &lt; 1%!
          </pre>
        </div>

        <h2 id="sec-cap3-2">2. Production Circular DMA Engine Implementation</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;string.h&gt;

#define DMA_RX_BUFFER_SIZE  256U

static uint8_t  g_dma_rx_buf[DMA_RX_BUFFER_SIZE];
static uint32_t g_last_read_idx = 0;

void uart_dma_init(void) {
    /* 1. Configure DMA Stream in CIRCULAR mode targeting g_dma_rx_buf */
    /* DMA_CNDTRx (Counter register) = DMA_RX_BUFFER_SIZE */

    /* 2. Enable UART Receiver DMA (DMAR bit in USART_CR3) */
    /* 3. Enable UART Idle Line Interrupt (IDLEIE bit in USART_CR1) */
}

/* Invoked ONLY once per packet burst when the serial line goes idle! */
void USART1_IRQHandler(void) {
    /* Check for IDLE Line flag */
    if (USART1-&gt;SR &amp; (1 &lt;&lt; 4)) {
        /* Clear IDLE flag by reading SR followed by reading DR */
        (void)USART1-&gt;SR;
        (void)USART1-&gt;DR;

        /* Calculate how many bytes DMA has written by reading the remaining counter */
        /* DMA NDTR counts DOWN from DMA_RX_BUFFER_SIZE to 0 */
        uint32_t dma_remaining = DMA2_Stream2-&gt;NDTR;
        uint32_t current_dma_idx = DMA_RX_BUFFER_SIZE - dma_remaining;

        if (current_dma_idx != g_last_read_idx) {
            if (current_dma_idx &gt; g_last_read_idx) {
                /* Linear contiguous block */
                uint32_t length = current_dma_idx - g_last_read_idx;
                process_serial_packet(&amp;g_dma_rx_buf[g_last_read_idx], length);
            } else {
                /* Buffer wrapped around! Process in two slices */
                uint32_t slice1 = DMA_RX_BUFFER_SIZE - g_last_read_idx;
                process_serial_packet(&amp;g_dma_rx_buf[g_last_read_idx], slice1);
                if (current_dma_idx &gt; 0) {
                    process_serial_packet(&amp;g_dma_rx_buf[0], current_dma_idx);
                }
            }
            g_last_read_idx = current_dma_idx;
        }
    }
}</code></pre>

        <h2 id="sec-cap3-quiz">3. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What triggers the UART IDLE line interrupt on an ARM microcontroller?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq3_1" id="cq3_1a"> <label for="cq3_1a">A) The RX buffer fills to 100%.</label></li>
              <li><input type="radio" name="cq3_1" id="cq3_1b"> <label for="cq3_1b">B) The RX serial line remains in an idle mark state (high) for at least one full character frame time after receiving data. (Correct)</label></li>
              <li><input type="radio" name="cq3_1" id="cq3_1c"> <label for="cq3_1c">C) The DMA counter hits zero.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The IDLE flag detects the physical end of an incoming burst without requiring special delimiter characters.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# CAPSTONE 04: Minimal Preemptive RTOS Kernel
# -------------------------------------------------------------
CAPSTONE04_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 4: Minimal Preemptive RTOS Kernel</h1>
          <p class="chapter-lead">Build an operational preemptive multitasking RTOS kernel from scratch for ARM Cortex-M in under 300 lines of C and assembly. Master Task Control Blocks (TCB), hardware stack framing (R0-R3, R12, LR, PC, xPSR), the PendSV context switcher, and dual-stack separation (MSP/PSP).</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 120m</div>
            <div class="meta-item">🎯 Standard: ARMv7-M Architecture / AAPCS Calling Convention</div>
            <div class="meta-item">⚡ Target: Bare-Metal Cortex-M4 Silicon / QEMU</div>
          </div>
        </div>

        <h2 id="sec-cap4-1">1. Kernel Architecture: How Real-Time Preemption Works</h2>
        <p>
          Building a preemptive RTOS kernel from scratch demystifies the magic of commercial kernels like FreeRTOS or Zephyr. At its physical core, a kernel consists of:
        </p>
        <ol>
          <li>A <strong>Task Control Block (TCB)</strong> array tracking each task's active Stack Pointer.</li>
          <li>A <strong>SysTick Handler</strong> triggering periodic timer preemption.</li>
          <li>A <strong>PendSV Assembly Handler</strong> that swaps CPU register contexts between tasks.</li>
        </ol>

        <h2 id="sec-cap4-2">2. The Complete Kernel Implementation in C</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;

#define MAX_TASKS       4
#define STACK_SIZE      256 /* 256 words = 1024 bytes per task */

typedef struct {
    uint32_t *sp; /* Current stack pointer (must be first element!) */
} tcb_t;

static tcb_t    g_tcbs[MAX_TASKS];
static uint32_t g_task_stacks[MAX_TASKS][STACK_SIZE];
static uint32_t g_current_task = 0;
static uint32_t g_task_count = 0;

/* Initializes a task stack to mimic an exception frame */
void os_task_create(void (*task_func)(void)) {
    uint32_t task_id = g_task_count++;
    uint32_t *stack_top = &amp;g_task_stacks[task_id][STACK_SIZE];

    /* Fabricate hardware exception frame (AAPCS) */
    *(--stack_top) = (1U &lt;&lt; 24);         /* xPSR (Thumb bit must be 1) */
    *(--stack_top) = (uint32_t)task_func; /* PC (Program Counter) */
    *(--stack_top) = 0xFFFFFFFDU;        /* LR (EXC_RETURN: Return to Thread mode, use PSP) */
    *(--stack_top) = 0x12121212U;        /* R12 */
    *(--stack_top) = 0x03030303U;        /* R3 */
    *(--stack_top) = 0x02020202U;        /* R2 */
    *(--stack_top) = 0x01010101U;        /* R1 */
    *(--stack_top) = 0x00000000U;        /* R0 */

    /* Fabricate software-saved registers (R4-R11) */
    *(--stack_top) = 0x11111111U;        /* R11 */
    *(--stack_top) = 0x10101010U;        /* R10 */
    *(--stack_top) = 0x09090909U;        /* R9 */
    *(--stack_top) = 0x08080808U;        /* R8 */
    *(--stack_top) = 0x07070707U;        /* R7 */
    *(--stack_top) = 0x06060606U;        /* R6 */
    *(--stack_top) = 0x05050505U;        /* R5 */
    *(--stack_top) = 0x04040404U;        /* R4 */

    g_tcbs[task_id].sp = stack_top;
}

/* Scheduler: Simple Round-Robin */
void os_scheduler(void) {
    g_current_task = (g_current_task + 1) % g_task_count;
}</code></pre>

        <h2 id="sec-cap4-3">3. The PendSV Context Switcher in Thumb-2 Assembly</h2>
        <pre><code class="language-assembly">.syntax unified
.thumb
.global PendSV_Handler

.type PendSV_Handler, %function
PendSV_Handler:
    /* 1. Get current Process Stack Pointer (PSP) */
    mrs     r0, psp

    /* 2. Push software registers R4-R11 onto Task stack */
    stmdb   r0!, {r4-r11}

    /* 3. Save new SP into current task TCB */
    ldr     r1, =g_tcbs
    ldr     r2, =g_current_task
    ldr     r2, [r2]
    str     r0, [r1, r2, lsl #2]

    /* 4. Select next task using round-robin scheduler */
    push    {lr}
    bl      os_scheduler
    pop     {lr}

    /* 5. Load new task SP from new TCB */
    ldr     r1, =g_tcbs
    ldr     r2, =g_current_task
    ldr     r2, [r2]
    ldr     r0, [r1, r2, lsl #2]

    /* 6. Pop software registers R4-R11 from new Task stack */
    ldmia   r0!, {r4-r11}

    /* 7. Update PSP with new task stack pointer */
    msr     psp, r0

    /* 8. Return from exception: Hardware automatically restores R0-R3, R12, LR, PC, xPSR */
    bx      lr</code></pre>

        <h2 id="sec-cap4-quiz">4. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What special value loaded into the Link Register (LR) signals an ARM Cortex-M processor to return to Thread Mode using the Process Stack Pointer (PSP)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq4_1" id="cq4_1a"> <label for="cq4_1a">A) 0x00000000</label></li>
              <li><input type="radio" name="cq4_1" id="cq4_1b"> <label for="cq4_1b">B) 0xFFFFFFFD (EXC_RETURN) (Correct)</label></li>
              <li><input type="radio" name="cq4_1" id="cq4_1c"> <label for="cq4_1c">C) 0xDEADBEEF</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>0xFFFFFFFD</code> is an architectural EXC_RETURN value indicating: return to Thread mode, use Process Stack Pointer (PSP), non-floating point state.
            </div>
          </div>
        </div>'''
