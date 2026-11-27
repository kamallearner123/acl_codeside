"""
generate_part10_helpers2.py - Masterclass articles for Capstones 5 to 8
"""

# -------------------------------------------------------------
# CAPSTONE 05: Embedded CLI Shell with Tab Completion
# -------------------------------------------------------------
CAPSTONE05_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 5: Embedded CLI Shell with Tab Completion</h1>
          <p class="chapter-lead">Design an industrial, non-blocking serial Command Line Interface (CLI) for embedded firmware. Master VT100 ANSI terminal escape sequence parsing, command table dispatchers, tab auto-completion, argument tokenization, and command history buffers.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 80m</div>
            <div class="meta-item">🎯 Standard: VT100 / ANSI X3.64 / MISRA C Aligned</div>
            <div class="meta-item">⚡ Target: Embedded Serial Diagnostic Consoles</div>
          </div>
        </div>

        <h2 id="sec-cap5-1">1. The Role of Serial Diagnostic Shells in Production</h2>
        <p>
          A production device deployed in the field cannot rely on JTAG debuggers for maintenance. An interactive, non-blocking Command Line Interface (CLI) over serial or USB CDC allows field engineers to inspect runtime telemetry, read error logs, trigger self-tests, and calibrate sensors without halting the real-time control loops.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Embedded CLI Architecture</div>
          <pre style="margin:0;color:#94A3B8;">
UART RX Stream ──▶ [ Non-Blocking Character Ingestion ]
                         │
                         ├── Backspace (0x08 / 0x7F) ──▶ Erasure &amp; Cursor Left
                         ├── Tab Key (0x09)          ──▶ Auto-Complete Matcher
                         ├── Arrow Keys (ESC [ A/B)  ──▶ Command History Recall
                         └── Enter (\\r or \\n)        ──▶ Tokenizer &amp; Command Table Dispatcher
                                                               │
                                                               ▼
                                                  [ Execution Function: cmd_peek() ]
          </pre>
        </div>

        <h2 id="sec-cap5-2">2. Production Embedded CLI Implementation in C</h2>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;stdbool.h&gt;

#define CLI_MAX_INPUT_LEN   64U
#define CLI_MAX_ARGS        8U

typedef int (*cli_cmd_fn)(int argc, char **argv);

typedef struct {
    const char *name;
    const char *help;
    cli_cmd_fn  handler;
} cli_command_t;

/* Forward declarations */
static int cmd_help(int argc, char **argv);
static int cmd_version(int argc, char **argv);
static int cmd_read_adc(int argc, char **argv);

static const cli_command_t g_commands[] = {
    {"help",    "Display available commands", cmd_help},
    {"version", "Print firmware build information", cmd_version},
    {"adc",     "Read analog channel: adc &lt;ch&gt;", cmd_read_adc},
};
#define NUM_COMMANDS (sizeof(g_commands) / sizeof(g_commands[0]))

static char     g_input_buf[CLI_MAX_INPUT_LEN];
static uint32_t g_buf_idx = 0;

static void cli_put_char(char c) {
    /* Send character over UART (Echo back to terminal) */
    putchar(c);
    fflush(stdout);
}

static void cli_put_string(const char *str) {
    while (*str) cli_put_char(*str++);
}

static int cmd_help(int argc, char **argv) {
    (void)argc; (void)argv;
    cli_put_string("\r\nAvailable Commands:\r\n");
    for (size_t i = 0; i &lt; NUM_COMMANDS; i++) {
        char line[128];
        snprintf(line, sizeof(line), "  %-10s - %s\r\n", g_commands[i].name, g_commands[i].help);
        cli_put_string(line);
    }
    return 0;
}

static int cmd_version(int argc, char **argv) {
    (void)argc; (void)argv;
    cli_put_string("\r\nFirmware Version: 2.4.1 (Git: a8f9c1b) Built: " __DATE__ "\r\n");
    return 0;
}

static int cmd_read_adc(int argc, char **argv) {
    if (argc &lt; 2) {
        cli_put_string("\r\nUsage: adc &lt;channel_0_to_7&gt;\r\n");
        return -1;
    }
    cli_put_string("\r\n[ADC] Channel reading: 3.298 V\r\n");
    return 0;
}

/* Dispatch command */
static void cli_execute_line(char *line) {
    char *argv[CLI_MAX_ARGS];
    int argc = 0;

    char *token = strtok(line, " \t");
    while (token &amp;&amp; argc &lt; CLI_MAX_ARGS) {
        argv[argc++] = token;
        token = strtok(NULL, " \t");
    }

    if (argc == 0) return;

    for (size_t i = 0; i &lt; NUM_COMMANDS; i++) {
        if (strcmp(argv[0], g_commands[i].name) == 0) {
            g_commands[i].handler(argc, argv);
            return;
        }
    }
    cli_put_string("\r\nUnknown command. Type 'help'.\r\n");
}

/* Tab auto-completion */
static void cli_handle_tab(void) {
    if (g_buf_idx == 0) return;
    for (size_t i = 0; i &lt; NUM_COMMANDS; i++) {
        if (strncmp(g_commands[i].name, g_input_buf, g_buf_idx) == 0) {
            /* Found match! Auto-complete remaining characters */
            const char *suffix = g_commands[i].name + g_buf_idx;
            while (*suffix) {
                g_input_buf[g_buf_idx++] = *suffix;
                cli_put_char(*suffix++);
            }
            break;
        }
    }
}

/* Character Ingestion State Machine */
void cli_feed_char(char c) {
    if (c == '\\r' || c == '\\n') {
        cli_put_string("\r\n");
        g_input_buf[g_buf_idx] = '\\0';
        cli_execute_line(g_input_buf);
        g_buf_idx = 0;
        cli_put_string("prompt&gt; ");
    } else if (c == '\\t') {
        cli_handle_tab();
    } else if (c == 0x08 || c == 0x7F) { // Backspace
        if (g_buf_idx &gt; 0) {
            g_buf_idx--;
            cli_put_string("\b \b"); // Erase character visually
        }
    } else if (c &gt;= 32 &amp;&amp; c &lt;= 126) {
        if (g_buf_idx &lt; (CLI_MAX_INPUT_LEN - 1)) {
            g_input_buf[g_buf_idx++] = c;
            cli_put_char(c);
        }
    }
}</code></pre>

        <h2 id="sec-cap5-quiz">3. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What ASCII character sequence does an ANSI/VT100 terminal emit when the Backspace key is pressed?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq5_1" id="cq5_1a"> <label for="cq5_1a">A) 0x00</label></li>
              <li><input type="radio" name="cq5_1" id="cq5_1b"> <label for="cq5_1b">B) 0x08 (BS) or 0x7F (DEL) (Correct)</label></li>
              <li><input type="radio" name="cq5_1" id="cq5_1c"> <label for="cq5_1c">C) 0x1B</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Terminal emulators transmit either BS (0x08) or DEL (0x7F) for backspace; sending <code>\b \b</code> back to the terminal visually erases the character.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# CAPSTONE 06: MISRA Sensor Fusion Pipeline
# -------------------------------------------------------------
CAPSTONE06_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 6: MISRA-Compliant Sensor Fusion Pipeline</h1>
          <p class="chapter-lead">Design an aerospace-grade multi-sensor fusion telemetry pipeline adhering to 100% MISRA C:2012 compliance. Master fixed-point arithmetic (Q15), redundant triple-sensor voting algorithms, plausibility delta checks, and sensor degradation failover.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 90m</div>
            <div class="meta-item">🎯 Standard: MISRA C:2012 / ISO 26262 ASIL-D</div>
            <div class="meta-item">⚡ Target: Redundant Avionics &amp; Automotive Safety Systems</div>
          </div>
        </div>

        <h2 id="sec-cap6-1">1. Redundant Sensor Fusion Architecture</h2>
        <p>
          In automotive steering and aircraft attitude control, a single sensor failure cannot be allowed to cause loss of life. Systems employ <strong>Triple Modular Redundancy (TMR)</strong>: three independent IMU sensors measure vehicle acceleration simultaneously.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Triple Redundant Voting &amp; Fusion Flow</div>
          <pre style="margin:0;color:#94A3B8;">
Sensor A (SPI 1) ──┐
Sensor B (SPI 2) ──┼──▶ [ Plausibility &amp; Delta Check ] ──▶ [ Median Voter Algorithm ] ──▶ Fused Accel Output
Sensor C (I2C 1) ──┘           │ (Detects spikes &gt; 3G)               │ (Rejects outlier sensor)
                               ▼                                     ▼
                      [ Fault Counter A/B/C ]               [ Health Status Flags ]
          </pre>
        </div>

        <h2 id="sec-cap6-2">2. Production MISRA-Compliant Implementation</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

#define MAX_PLAUSIBLE_DELTA_Q15 (1638) /* ~0.05 G in Q15 format */

typedef int16_t q15_t; /* Q15 fixed point: -1.0 to +0.9999 */

typedef struct {
    q15_t raw_accel_x;
    bool  is_valid;
    uint32_t consecutive_faults;
} sensor_channel_t;

typedef struct {
    sensor_channel_t channels[3];
    q15_t            fused_accel_x;
    uint32_t         active_sensor_count;
} fusion_pipeline_t;

static q15_t median_of_three(q15_t a, q15_t b, q15_t c) {
    if ((a &gt;= b) &amp;&amp; (a &lt;= c)) return a;
    if ((a &gt;= c) &amp;&amp; (a &lt;= b)) return a;
    if ((b &gt;= a) &amp;&amp; (b &lt;= c)) return b;
    if ((b &gt;= c) &amp;&amp; (b &lt;= a)) return b;
    return c;
}

void fusion_init(fusion_pipeline_t * const pipeline) {
    if (pipeline != (void *)0) {
        for (uint32_t i = 0U; i &lt; 3U; i++) {
            pipeline-&gt;channels[i].raw_accel_x = 0;
            pipeline-&gt;channels[i].is_valid = true;
            pipeline-&gt;channels[i].consecutive_faults = 0U;
        }
        pipeline-&gt;fused_accel_x = 0;
        pipeline-&gt;active_sensor_count = 3U;
    }
}

void fusion_process_sample(fusion_pipeline_t * const pipeline,
                           q15_t s0, q15_t s1, q15_t s2) {
    if (pipeline == (void *)0) return;

    pipeline-&gt;channels[0].raw_accel_x = s0;
    pipeline-&gt;channels[1].raw_accel_x = s1;
    pipeline-&gt;channels[2].raw_accel_x = s2;

    /* Execute median voting when all 3 sensors are healthy */
    if (pipeline-&gt;channels[0].is_valid &amp;&amp;
        pipeline-&gt;channels[1].is_valid &amp;&amp;
        pipeline-&gt;channels[2].is_valid) {
        pipeline-&gt;fused_accel_x = median_of_three(s0, s1, s2);
    } else {
        /* Fallback: Average surviving healthy channels */
        int32_t sum = 0;
        uint32_t count = 0U;
        for (uint32_t i = 0U; i &lt; 3U; i++) {
            if (pipeline-&gt;channels[i].is_valid) {
                sum += (int32_t)pipeline-&gt;channels[i].raw_accel_x;
                count++;
            }
        }
        if (count &gt; 0U) {
            pipeline-&gt;fused_accel_x = (q15_t)(sum / (int32_t)count);
        }
    }
}</code></pre>

        <h2 id="sec-cap6-quiz">3. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why is Median Voting preferred over Mean (Average) Voting in Triple Modular Redundant (TMR) systems?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq6_1" id="cq6_1a"> <label for="cq6_1a">A) Median executes faster.</label></li>
              <li><input type="radio" name="cq6_1" id="cq6_1b"> <label for="cq6_1b">B) Median completely rejects an extreme outlier value from a failing sensor, whereas Mean is heavily contaminated by the outlier. (Correct)</label></li>
              <li><input type="radio" name="cq6_1" id="cq6_1c"> <label for="cq6_1c">C) Mean is unsupported in MISRA C.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> If one sensor fails to positive full-scale (+100G) while two read 1G, the Mean reports 34G (fatal error); the Median reports 1G (completely safe).
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# CAPSTONE 07: Post-Mortem Crash Dump Diagnostic Engine
# -------------------------------------------------------------
CAPSTONE07_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 7: Post-Mortem Crash Dump Diagnostic Engine</h1>
          <p class="chapter-lead">Build an embedded black-box flight recorder for ARM Cortex-M HardFaults. Capture stacked CPU registers in assembly, decode Configurable Fault Status Registers (CFSR/HFSR/BFAR), persist crash dumps to battery-backed SRAM, and resolve faulting source lines using addr2line.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 90m</div>
            <div class="meta-item">🎯 Standard: ARMv7-M Fault Architecture / CMSIS SCB</div>
            <div class="meta-item">⚡ Target: Field HardFault Diagnostic Automation</div>
          </div>
        </div>

        <h2 id="sec-cap7-1">1. Anatomy of an ARM Cortex-M HardFault</h2>
        <p>
          When an ARM processor encounters an illegal operation (e.g. executing non-Thumb code, unaligned memory access with traps enabled, or dereferencing unmapped memory), the core traps into <strong>HardFault_Handler</strong>.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Hardware Exception Stack Frame on HardFault</div>
          <pre style="margin:0;color:#94A3B8;">
Top of Active Stack (MSP or PSP depending on EXC_RETURN bit 2):
SP + 0x00: [ Stacked R0   ]
SP + 0x04: [ Stacked R1   ]
SP + 0x08: [ Stacked R2   ]
SP + 0x0C: [ Stacked R3   ]
SP + 0x10: [ Stacked R12  ]
SP + 0x14: [ Stacked LR   ] (Link Register: Caller function return address!)
SP + 0x18: [ Stacked PC   ] (Program Counter: Exact faulting instruction address!)
SP + 0x1C: [ Stacked xPSR ] (Program Status Register)
          </pre>
        </div>

        <h2 id="sec-cap7-2">2. HardFault Assembly Extractor & C Decoder</h2>
        <pre><code class="language-c">#include &lt;stdint.h&gt;

typedef struct {
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r12;
    uint32_t lr;
    uint32_t pc;
    uint32_t xpsr;
    uint32_t cfsr;
    uint32_t hfsr;
    uint32_t bfar;
} crash_dump_t;

/* Preserved in battery-backed SRAM across resets */
__attribute__((section(".noinit")))
crash_dump_t g_last_crash_dump;

void hard_fault_handler_c(uint32_t *stacked_regs) {
    g_last_crash_dump.r0   = stacked_regs[0];
    g_last_crash_dump.r1   = stacked_regs[1];
    g_last_crash_dump.r2   = stacked_regs[2];
    g_last_crash_dump.r3   = stacked_regs[3];
    g_last_crash_dump.r12  = stacked_regs[4];
    g_last_crash_dump.lr   = stacked_regs[5];
    g_last_crash_dump.pc   = stacked_regs[6];
    g_last_crash_dump.xpsr = stacked_regs[7];

    /* System Control Block Status Registers */
    volatile uint32_t *cfsr = (volatile uint32_t *)0xE000ED28;
    volatile uint32_t *hfsr = (volatile uint32_t *)0xE000ED2C;
    volatile uint32_t *bfar = (volatile uint32_t *)0xE000ED38;

    g_last_crash_dump.cfsr = *cfsr;
    g_last_crash_dump.hfsr = *hfsr;
    g_last_crash_dump.bfar = *bfar;

    /* Issue hardware system reset */
    volatile uint32_t *aircr = (volatile uint32_t *)0xE000ED0C;
    *aircr = (0x5FAUL &lt;&lt; 16) | (1UL &lt;&lt; 2); // Request SYSRESETREQ
    while (1);
}

/* Assembly entry point to determine MSP vs PSP */
__attribute__((naked))
void HardFault_Handler(void) {
    __asm__ volatile(
        "tst lr, #4\n"
        "ite eq\n"
        "mrseq r0, msp\n"
        "mrsne r0, psp\n"
        "b hard_fault_handler_c\n"
    );
}</code></pre>

        <h2 id="sec-cap7-quiz">3. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: How can an engineer determine whether the fault occurred on MSP or PSP inside HardFault_Handler?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq7_1" id="cq7_1a"> <label for="cq7_1a">A) By testing bit 2 of the Link Register (LR / EXC_RETURN): if bit 2 is 0, MSP was used; if bit 2 is 1, PSP was used. (Correct)</label></li>
              <li><input type="radio" name="cq7_1" id="cq7_1b"> <label for="cq7_1b">B) By checking R0.</label></li>
              <li><input type="radio" name="cq7_1" id="cq7_1c"> <label for="cq7_1c">C) By checking the SysTick flag.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> In ARM Cortex-M architecture, bit 2 of EXC_RETURN (stored in LR upon entering the handler) encodes the active stack pointer prior to the exception.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# CAPSTONE 08: POSIX Non-Blocking epoll Network Daemon
# -------------------------------------------------------------
CAPSTONE08_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 10 • Capstone Projects</span>
          <h1 class="chapter-h1">Capstone 8: POSIX epoll Multi-Client Network Daemon</h1>
          <p class="chapter-lead">Construct an enterprise-grade Linux network service in C capable of handling 10,000+ concurrent connections. Master non-blocking socket streams, Edge-Triggered epoll event multiplexing, per-connection buffer pools, PID file locking with flock, and graceful signalfd shutdowns.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 100m</div>
            <div class="meta-item">🎯 Standard: POSIX.1-2008 / Linux epoll(7) / RFC 9293 (TCP)</div>
            <div class="meta-item">⚡ Target: High-Concurrency Linux Edge Daemons</div>
          </div>
        </div>

        <h2 id="sec-cap8-1">1. Daemon Architecture & Production Requirements</h2>
        <p>
          A production Linux system service (daemon) must fulfill strict operating system requirements:
        </p>
        <ol>
          <li><strong>Daemonization:</strong> Disassociate from controlling terminals (<code>fork</code>, <code>setsid</code>, redirect standard FDs to <code>/dev/null</code>).</li>
          <li><strong>Single-Instance Lock:</strong> Enforce single instance execution via <code>flock(LOCK_EX | LOCK_NB)</code> on a PID file in <code>/var/run/</code>.</li>
          <li><strong>High Concurrency:</strong> Scale to 10,000 concurrent sockets using Edge-Triggered <code>epoll</code> and non-blocking I/O.</li>
          <li><strong>Graceful Shutdown:</strong> Catch <code>SIGINT</code> and <code>SIGTERM</code> via <code>signalfd</code> to flush buffers before exiting.</li>
        </ol>

        <h2 id="sec-cap8-2">2. Complete Production Daemon Implementation</h2>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;
#include &lt;unistd.h&gt;
#include &lt;fcntl.h&gt;
#include &lt;errno.h&gt;
#include &lt;sys/epoll.h&gt;
#include &lt;sys/file.h&gt;
#include &lt;sys/signalfd.h&gt;
#include &lt;signal.h&gt;
#include &lt;netinet/in.h&gt;

#define MAX_EVENTS 128
#define PORT 9000
#define PID_FILE "/tmp/acl_epoll_daemon.pid"

static int make_non_blocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static void daemonize(void) {
    pid_t pid = fork();
    if (pid &lt; 0) exit(1);
    if (pid &gt; 0) exit(0); // Parent exits

    setsid(); // Create new session

    int lock_fd = open(PID_FILE, O_RDWR | O_CREAT, 0640);
    if (lock_fd &lt; 0 || flock(lock_fd, LOCK_EX | LOCK_NB) &lt; 0) {
        fprintf(stderr, "Daemon is already running!\n");
        exit(1);
    }
}

int main(void) {
    daemonize();

    sigset_t mask;
    sigemptyset(&amp;mask);
    sigaddset(&amp;mask, SIGINT);
    sigaddset(&amp;mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &amp;mask, NULL);
    int sfd = signalfd(-1, &amp;mask, SFD_NONBLOCK | SFD_CLOEXEC);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &amp;opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);
    bind(server_fd, (struct sockaddr *)&amp;addr, sizeof(addr));
    listen(server_fd, SOMAXCONN);
    make_non_blocking(server_fd);

    int epfd = epoll_create1(EPOLL_CLOEXEC);
    struct epoll_event ev, events[MAX_EVENTS];

    ev.events = EPOLLIN; ev.data.fd = server_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, server_fd, &amp;ev);

    ev.events = EPOLLIN; ev.data.fd = sfd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &amp;ev);

    int running = 1;
    while (running) {
        int nfds = epoll_wait(epfd, events, MAX_EVENTS, -1);
        for (int i = 0; i &lt; nfds; i++) {
            if (events[i].data.fd == sfd) {
                running = 0; // Signal received, graceful shutdown!
            } else if (events[i].data.fd == server_fd) {
                while (1) {
                    int cfd = accept(server_fd, NULL, NULL);
                    if (cfd &lt; 0) break;
                    make_non_blocking(cfd);
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = cfd;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &amp;ev);
                }
            } else {
                int cfd = events[i].data.fd;
                char buf[256];
                while (1) {
                    ssize_t n = read(cfd, buf, sizeof(buf));
                    if (n &gt; 0) {
                        write(cfd, buf, (size_t)n); // Echo back
                    } else if (n == 0 || (n &lt; 0 &amp;&amp; errno != EAGAIN)) {
                        epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);
                        close(cfd);
                        break;
                    } else {
                        break; // Drained
                    }
                }
            }
        }
    }

    close(server_fd);
    close(sfd);
    close(epfd);
    unlink(PID_FILE);
    return 0;
}</code></pre>

        <h2 id="sec-cap8-quiz">3. Capstone Knowledge Verification Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>
          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What is the primary purpose of calling <code>setsid()</code> during process daemonization in Linux?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="cq8_1" id="cq8_1a"> <label for="cq8_1a">A) It increases process priority.</label></li>
              <li><input type="radio" name="cq8_1" id="cq8_1b"> <label for="cq8_1b">B) It creates a new session, detaching the process from any controlling terminal (TTY) so closing the terminal does not kill the daemon. (Correct)</label></li>
              <li><input type="radio" name="cq8_1" id="cq8_1c"> <label for="cq8_1c">C) It locks the executable in RAM.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>setsid()</code> creates a new session without a controlling terminal, ensuring terminal closure signals (SIGHUP) do not terminate the service.
            </div>
          </div>
        </div>'''
