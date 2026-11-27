#!/usr/bin/env python3
"""
generate_part5_full.py - Generates all 4 modules of Part 5:
Module 22: Unit Testing Frameworks (Unity & CMock)
Module 23: Hardware Mocking & HAL Emulation
Module 24: Code Coverage (gcov & lcov)
Module 25: CI/CD Pipelines for Embedded C
"""

import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_course_materials import generate_sidebar_html

CHAPTERS_DIR = os.path.join(os.path.dirname(__file__), "..", "private_books", "c_programming_professional", "chapters")

def replace_article_in_file(filename, new_article_html):
    filepath = os.path.join(CHAPTERS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = r'<article class="prose">.*?</article>'
    replacement = f'<article class="prose">\n{new_article_html}\n      </article>'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        raise ValueError(f"Could not find <article class='prose'> in {filename}")
    
    sidebar_html = generate_sidebar_html(active_file=filename, is_index=False)
    sidebar_pattern = r'<nav class="sidebar-nav">.*?</nav>'
    sidebar_replacement = f'<nav class="sidebar-nav">\n{sidebar_html}\n      </nav>'
    new_content, s_count = re.subn(sidebar_pattern, sidebar_replacement, new_content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    line_count = len(new_content.splitlines())
    sz_kb = len(new_content.encode("utf-8")) // 1024
    print(f"Updated {filename}: {line_count} lines, {sz_kb} KB")

# -------------------------------------------------------------
# MODULE 22: Unit Testing with Unity & CMock
# -------------------------------------------------------------
MOD22_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 5 • Testing &amp; QA Engineering</span>
          <h1 class="chapter-h1">Module 22: Unit Testing with Unity &amp; CMock</h1>
          <p class="chapter-lead">Master embedded unit testing architectures using the industry-standard Unity test framework and CMock function mocking engine. Learn how to decouple hardware dependencies, execute test suites 1,000x faster on development hosts, and assert memory buffers, struct layouts, and error states.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: ISO 26262 Part 6 / DO-178C Level B / Unity 2.5+</div>
            <div class="meta-item">⚡ Target: Host x86_64 / Cross-Compiled QEMU</div>
          </div>
        </div>

        <h2 id="sec-22-1">22.1 The Embedded Testing Pyramid & Host-Based Verification</h2>
        <p>
          In traditional embedded workflows, engineers write firmware, flash a physical target board over JTAG/SWD, and probe with an oscilloscope or printf. This workflow is painfully slow (15-60 seconds per debug iteration) and fails to scale across complex state machines.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">The Embedded Software Testing Pyramid</div>
          <pre style="margin:0;color:#94A3B8;">
                 ▲
                ╱ ╲
               ╱ HIL╲          Hardware-In-The-Loop: Real physical silicon, oscilloscopes,
              ╱──────╲         environmental chambers. (~10-50 tests/day)
             ╱ System ╲        Board-level firmware execution, complete peripherals.
            ╱──────────╲       (~100-500 tests/day)
           ╱ Integration╲      Subsystem interaction, simulated buses, RTOS tasks.
          ╱──────────────╲     (~1,000 tests/hour)
         ╱   Unit Tests   ╲    Host-compiled pure C algorithms, Unity/CMock, 100% mocks.
        ╱──────────────────╲   (&gt; 10,000 tests in under 2 seconds!)
          </pre>
        </div>

        <p>
          By isolating hardware register accesses behind abstract HAL boundaries, <strong>90% of embedded firmware algorithms</strong> (parsers, state machines, protocol decoders, PID loops, memory managers) can be compiled with host GCC or Clang and validated in fractions of a second on developer workstations and in CI/CD pipelines.
        </p>

        <h2 id="sec-22-2">22.2 The Unity Test Framework Architecture</h2>
        <p>
          <strong>Unity</strong> is an ultra-lightweight, ANSI C-compliant test harness engineered specifically for embedded and resource-constrained environments. It requires zero dynamic memory allocation and outputs standardized ASCII or TAP test reports.
        </p>

        <h3>Unity Execution Lifecycle</h3>
        <p>
          Each individual test case executes inside a strictly defined lifecycle sandwich:
        </p>
        <ol>
          <li><code>setUp(void)</code>: Invoked automatically prior to <em>every</em> single test function. Resets global state, reinitializes circular buffers, and zeroes mock expectations.</li>
          <li><code>test_FunctionName(void)</code>: The unit test body containing assertions.</li>
          <li><code>tearDown(void)</code>: Invoked automatically following <em>every</em> single test function. Frees allocated test buffers and validates that no unhandled mock calls remain.</li>
        </ol>

        <h3>Production Unity Assertions</h3>
        <pre><code class="language-c">/* Exact integer comparisons */
TEST_ASSERT_EQUAL_INT32(expected_int, actual_int);
TEST_ASSERT_EQUAL_HEX32(0xDEADBEEF, register_val);

/* Null pointer checks */
TEST_ASSERT_NOT_NULL(sensor_handle);
TEST_ASSERT_NULL(invalid_handle);

/* Float comparisons with delta tolerances */
TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.300f, measured_voltage);

/* Memory and byte-array comparisons */
TEST_ASSERT_EQUAL_MEMORY(expected_packet, received_packet, sizeof(packet_t));
TEST_ASSERT_EQUAL_UINT8_ARRAY(expected_bytes, buffer, 16);</code></pre>

        <h2 id="sec-22-3">22.3 CMock & Function Mocking Mechanics</h2>
        <p>
          Unit testing requires isolating the <strong>Unit Under Test (UUT)</strong> from its external collaborators. When testing a temperature telemetry module that writes to I2C, you must not invoke physical I2C hardware. Instead, you link against a <strong>Mock Object</strong>.
        </p>
        <p>
          <strong>CMock</strong> parses C header files (such as <code>i2c_driver.h</code>) and automatically generates mock implementations of every declared function.
        </p>

        <h3>CMock Expectation Patterns</h3>
        <pre><code class="language-c">/* 1. Expect call with exact parameters and return a specific status */
i2c_write_ExpectAndReturn(0x48, expected_payload, 2, I2C_STATUS_OK);

/* 2. Expect call and validate an array parameter */
i2c_write_ExpectWithArray(0x48, 1, expected_data, 1, 1, 4, I2C_STATUS_OK);

/* 3. Ignore arguments and return error (simulating hardware bus failure) */
i2c_read_IgnoreAndReturn(I2C_STATUS_BUS_ERROR);</code></pre>

        <h2 id="sec-22-4">22.4 Decoupling Time & Interrupts in Unit Tests</h2>
        <p>
          Embedded code frequently uses hardware timers or delay loops (such as <code>HAL_Delay(100)</code>). If executed naively on a host machine, unit tests would crawl at real-time speeds, and testing a 10-minute timeout would take 10 minutes.
        </p>
        <p>
          <strong>The Solution: Time Injection.</strong> Provide a virtual clock interface:
        </p>
        <pre><code class="language-c">/* Virtual Time Interface */
typedef uint32_t (*time_get_ticks_fn)(void);

static uint32_t g_fake_ticks = 0;
uint32_t mock_get_ticks(void) { return g_fake_ticks; }

void test_telemetry_timeout(void) {
    g_fake_ticks = 0;
    telemetry_start();
    
    /* Warp time forward by 5000 milliseconds instantaneously! */
    g_fake_ticks += 5000;
    telemetry_process();
    
    TEST_ASSERT_EQUAL(TELEMETRY_STATE_TIMEOUT, telemetry_get_state());
}</code></pre>

        <h2 id="sec-22-5">22.5 Industrial Case Study: Therac-25 Medical Radiation Disaster</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Lethal Software Race Condition in Medical Linear Accelerators</h4>
          <p>
            Between 1985 and 1987, the Therac-25 medical radiation machine administered massive radiation overdoses (over 100 times the intended therapeutic dose) to multiple oncology patients, causing severe radiation poisoning and deaths.
          </p>
          <p>
            <strong>The Software Defect:</strong>
            The operator console allowed the technician to select "X-Ray Mode" (which uses a high-energy electron beam deflected by a physical tungsten target plate) or "Electron Mode" (which uses a low-energy beam without the target).
          </p>
          <p>
            If a skilled technician accidentally pressed "X-Ray", quickly hit "Up Arrow", and corrected it to "Electron Mode" within an eight-second window, an unhandled race condition occurred:
          </p>
          <ul>
            <li>The software reset the machine state variables indicating Electron Mode.</li>
            <li>However, the subroutine positioning the heavy mechanical tungsten shield took several seconds to respond and did not move the shield into place!</li>
            <li>The software fired the raw, unattenuated 25 MeV high-current beam directly into human tissue without the attenuating target in place.</li>
          </ul>
          <p>
            <strong>How Unit Testing Prevents This:</strong>
            Therac-25 was tested almost exclusively as a whole integrated machine; developers never unit-tested the keyboard entry parser and motor positioning state transitions independently. Host-based unit testing with randomized input fuzzing would have executed millions of rapid state transitions, instantly uncovering the missing synchronization between user input validation and mechanical actuator interlocks.
          </p>
        </div>

        <h2 id="sec-22-6">22.6 Hands-On Lab: Complete Unity Test Suite for an Embedded Circular Ring Buffer</h2>
        <p>
          In this laboratory, you will construct a production FIFO circular ring buffer and develop an exhaustive Unity test harness verifying boundary conditions: initialization, full buffer rejection, FIFO order preservation, and wrap-around mechanics.
        </p>

        <h3>1. Production Ring Buffer Header (<code>ring_buf.h</code>)</h3>
        <pre><code class="language-c">#ifndef RING_BUF_H_
#define RING_BUF_H_

#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

#define RING_BUF_CAPACITY 8U

typedef struct {
    uint8_t buffer[RING_BUF_CAPACITY];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
} ring_buf_t;

void ring_buf_init(ring_buf_t *rb);
bool ring_buf_push(ring_buf_t *rb, uint8_t byte);
bool ring_buf_pop(ring_buf_t *rb, uint8_t *out_byte);
bool ring_buf_is_empty(const ring_buf_t *rb);
bool ring_buf_is_full(const ring_buf_t *rb);

#endif /* RING_BUF_H_ */</code></pre>

        <h3>2. Production Implementation (<code>ring_buf.c</code>)</h3>
        <pre><code class="language-c">#include "ring_buf.h"

void ring_buf_init(ring_buf_t *rb) {
    if (!rb) return;
    rb-&gt;head = 0;
    rb-&gt;tail = 0;
    rb-&gt;count = 0;
}

bool ring_buf_push(ring_buf_t *rb, uint8_t byte) {
    if (!rb || ring_buf_is_full(rb)) return false;
    rb-&gt;buffer[rb-&gt;head] = byte;
    rb-&gt;head = (rb-&gt;head + 1U) % RING_BUF_CAPACITY;
    rb-&gt;count++;
    return true;
}

bool ring_buf_pop(ring_buf_t *rb, uint8_t *out_byte) {
    if (!rb || !out_byte || ring_buf_is_empty(rb)) return false;
    *out_byte = rb-&gt;buffer[rb-&gt;tail];
    rb-&gt;tail = (rb-&gt;tail + 1U) % RING_BUF_CAPACITY;
    rb-&gt;count--;
    return true;
}

bool ring_buf_is_empty(const ring_buf_t *rb) {
    return (rb != (void *)0) ? (rb-&gt;count == 0U) : true;
}

bool ring_buf_is_full(const ring_buf_t *rb) {
    return (rb != (void *)0) ? (rb-&gt;count &gt;= RING_BUF_CAPACITY) : false;
}</code></pre>

        <h3>3. Exhaustive Unity Test Suite (<code>test_ring_buf.c</code>)</h3>
        <pre><code class="language-c">#include "unity.h"
#include "ring_buf.h"

static ring_buf_t g_rb;

void setUp(void) {
    ring_buf_init(&amp;g_rb);
}

void tearDown(void) {
    /* Cleanup after each test */
}

void test_initial_state_is_empty(void) {
    TEST_ASSERT_TRUE(ring_buf_is_empty(&amp;g_rb));
    TEST_ASSERT_FALSE(ring_buf_is_full(&amp;g_rb));
    TEST_ASSERT_EQUAL_UINT32(0, g_rb.count);
}

void test_push_and_pop_single_byte(void) {
    uint8_t popped = 0;
    TEST_ASSERT_TRUE(ring_buf_push(&amp;g_rb, 0x42));
    TEST_ASSERT_EQUAL_UINT32(1, g_rb.count);
    TEST_ASSERT_FALSE(ring_buf_is_empty(&amp;g_rb));

    TEST_ASSERT_TRUE(ring_buf_pop(&amp;g_rb, &amp;popped));
    TEST_ASSERT_EQUAL_HEX8(0x42, popped);
    TEST_ASSERT_TRUE(ring_buf_is_empty(&amp;g_rb));
}

void test_buffer_overflow_rejection(void) {
    for (uint8_t i = 0; i &lt; RING_BUF_CAPACITY; i++) {
        TEST_ASSERT_TRUE(ring_buf_push(&amp;g_rb, i));
    }
    TEST_ASSERT_TRUE(ring_buf_is_full(&amp;g_rb));

    /* Attempting to push past capacity must fail gracefully */
    TEST_ASSERT_FALSE(ring_buf_push(&amp;g_rb, 0xFF));
    TEST_ASSERT_EQUAL_UINT32(RING_BUF_CAPACITY, g_rb.count);
}

void test_fifo_wrap_around_preserves_order(void) {
    uint8_t byte = 0;
    /* Fill 4 bytes, pop 4 bytes to advance tail */
    for (uint8_t i = 1; i &lt;= 4; i++) ring_buf_push(&amp;g_rb, i);
    for (uint8_t i = 1; i &lt;= 4; i++) ring_buf_pop(&amp;g_rb, &amp;byte);

    /* Now push across the array wrap-around boundary */
    for (uint8_t i = 10; i &lt; 18; i++) {
        TEST_ASSERT_TRUE(ring_buf_push(&amp;g_rb, i));
    }
    TEST_ASSERT_TRUE(ring_buf_is_full(&amp;g_rb));

    /* Verify FIFO extraction across wrap-around */
    for (uint8_t expected = 10; expected &lt; 18; expected++) {
        TEST_ASSERT_TRUE(ring_buf_pop(&amp;g_rb, &amp;byte));
        TEST_ASSERT_EQUAL_UINT8(expected, byte);
    }
    TEST_ASSERT_TRUE(ring_buf_is_empty(&amp;g_rb));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_initial_state_is_empty);
    RUN_TEST(test_push_and_pop_single_byte);
    RUN_TEST(test_buffer_overflow_rejection);
    RUN_TEST(test_fifo_wrap_around_preserves_order);
    return UNITY_END();
}</code></pre>

        <h2 id="sec-22-quiz">22.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: In the Unity test framework, what function is guaranteed to run before each and every individual test case?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) <code>main()</code></label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) <code>setUp()</code> (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) <code>tearDown()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>setUp()</code> resets environmental prerequisites and state variables immediately prior to executing every single test case, ensuring complete isolation between tests.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the primary architectural purpose of a Mock Object generated by CMock?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) To accelerate the clock frequency of physical microcontrollers.</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) To substitute physical hardware drivers with controllable, software-defined stubs that verify call counts, arguments, and return synthesized statuses. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) To translate C into assembly language.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Mocks decouple the Unit Under Test from real peripheral drivers and buses, allowing tests to run purely on PC development machines.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Why is host-based unit testing significantly faster than testing on physical target evaluation boards?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) It eliminates the multi-second overhead of JTAG/SWD flash erase/programming cycles and runs at gigahertz host CPU speeds. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Host C compilers omit memory bounds checks.</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) Unity tests bypass the C preprocessor.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Erasing and flashing microcontroller flash memory over SWD typically takes 5 to 30 seconds per iteration; compiling and running on an x86/ARM64 host takes under 50 milliseconds.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What assertion macro should be used to verify that an unattenuated floating-point calculation yields 3.1415 within a margin of 0.0001?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>TEST_ASSERT_EQUAL(3.1415f, actual)</code></label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>TEST_ASSERT_FLOAT_WITHIN(0.0001f, 3.1415f, actual)</code> (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>TEST_ASSERT_HEX32_WITHIN(...)</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Due to IEEE-754 floating-point rounding imprecision, floats should never be compared with exact equality (<code>==</code>); delta-tolerance assertions must always be used.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: How can a unit test verify firmware behavior across a 1-hour timeout without stalling test execution for an hour?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) By speeding up the workstation CPU fan.</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) By mocking the tick timer getter function and advancing the fake tick counter programmatically by 3,600,000 milliseconds. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) Time manipulation is impossible in C unit tests.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Decoupling firmware from real wall-clock hardware timers allows unit tests to inject arbitrary time leaps instantaneously.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 23: Hardware Mocking & HAL Emulation
# -------------------------------------------------------------
MOD23_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 5 • Testing &amp; QA Engineering</span>
          <h1 class="chapter-h1">Module 23: Hardware Mocking &amp; HAL Emulation</h1>
          <p class="chapter-lead">Deep dive into simulating peripheral hardware, link-time function interposition via GNU ld --wrap, memory-mapped I/O (MMIO) register emulation in host RAM, and synthetic asynchronous interrupt injection.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 50m</div>
            <div class="meta-item">🎯 Standard: POSIX / GNU ld Linker Interposition</div>
            <div class="meta-item">⚡ Target: Linux x86_64 Host Testing</div>
          </div>
        </div>

        <h2 id="sec-23-1">23.1 The Silicon Dependency Dilemma</h2>
        <p>
          In embedded C, drivers interact directly with hardware through physical memory addresses:
        </p>
        <pre><code class="language-c">#define USART1_BASE  (0x40011000UL)
#define USART1_DR    (*(volatile uint32_t *)(USART1_BASE + 0x04))

void uart_send_char(char c) {
    USART1_DR = (uint32_t)c; /* Crashes on x86 Linux! */
}</code></pre>
        <p>
          If you compile and execute this code on an x86 development PC, the CPU encounters address <code>0x40011000UL</code>. Because this physical address is not mapped into the user-space process's virtual memory page tables, the MMU triggers an immediate <strong>Page Fault</strong>, terminating the test suite with <code>SIGSEGV (Segmentation Fault)</code>.
        </p>

        <h2 id="sec-23-2">23.2 Hardware Emulation Strategies</h2>
        <p>
          Three primary design patterns allow hardware drivers to execute flawlessly on host development platforms:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Hardware Mocking Architectures Compared</div>
          <pre style="margin:0;color:#94A3B8;">
Strategy 1: Virtual Memory Struct Redirection
  #ifdef HOST_TEST
    static USART_TypeDef virtual_usart;
    #define USART1 (&amp;virtual_usart)
  #else
    #define USART1 ((USART_TypeDef *)0x40011000UL)
  #endif

Strategy 2: Link-Time Interposition (GNU ld --wrap)
  Production Call: HAL_I2C_Transmit(...) ──▶ Linker redirects to ──▶ __wrap_HAL_I2C_Transmit(...)
  Original Call:   Can still invoke __real_HAL_I2C_Transmit(...) if needed.

Strategy 3: Dependency Injection via VTable (OOP Interface)
  Application ──▶ [ i2c_bus_ops_t *bus ] ──▶ Real Driver: stm32_i2c_write()
                                        └──▶ Mock Driver: mock_i2c_write()
          </pre>
        </div>

        <h2 id="sec-23-3">23.3 Link-Time Wrapping with GNU ld (<code>--wrap</code>)</h2>
        <p>
          The most powerful, non-invasive mocking technique supported by the GNU linker is <strong>symbol wrapping</strong> (<code>-Wl,--wrap=&lt;symbol&gt;</code>). This requires <em>zero modifications</em> to your production C source code!
        </p>
        <p>
          When you link with <code>-Wl,--wrap=HAL_I2C_Master_Transmit</code>:
        </p>
        <ol>
          <li>Any call to <code>HAL_I2C_Master_Transmit</code> is resolved to <code>__wrap_HAL_I2C_Master_Transmit</code>.</li>
          <li>The original implementation remains accessible via the synthetic symbol <code>__real_HAL_I2C_Master_Transmit</code>.</li>
        </ol>

        <h3>Production Linker Wrap Example</h3>
        <pre><code class="language-c">/* mock_i2c.c */
#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;
#include &lt;string.h&gt;

/* Capture structures for assertions */
typedef struct {
    uint16_t dev_addr;
    uint8_t  last_data[64];
    uint16_t last_size;
    uint32_t call_count;
    int      injected_return_code;
} i2c_mock_state_t;

static i2c_mock_state_t g_i2c_mock = {0};

void mock_i2c_reset(void) {
    memset(&amp;g_i2c_mock, 0, sizeof(g_i2c_mock));
}

void mock_i2c_inject_error(int err) {
    g_i2c_mock.injected_return_code = err;
}

/* Intercepted function! */
int __wrap_HAL_I2C_Master_Transmit(void *hi2c, uint16_t DevAddress, uint8_t *pData, uint16_t Size, uint32_t Timeout) {
    (void)hi2c;
    (void)Timeout;
    g_i2c_mock.call_count++;
    g_i2c_mock.dev_addr = DevAddress;
    g_i2c_mock.last_size = Size;
    if (Size &lt;= sizeof(g_i2c_mock.last_data)) {
        memcpy(g_i2c_mock.last_data, pData, Size);
    }
    return g_i2c_mock.injected_return_code;
}</code></pre>

        <h2 id="sec-23-4">23.4 Simulating Asynchronous Interrupts</h2>
        <p>
          Testing interrupt-driven protocols (such as UART DMA ring buffers) requires testing how the driver behaves when an interrupt fires midway through a calculation.
        </p>
        <p>
          On host platforms, we can simulate interrupts synchronously by directly invoking the ISR function pointer:
        </p>
        <pre><code class="language-c">void test_uart_rx_overflow_recovery(void) {
    uart_driver_init();

    /* Simulate 10 rapid UART RX interrupts firing from hardware */
    for (int i = 0; i &lt; 10; i++) {
        /* Inject fake received byte into virtual hardware register */
        VIRTUAL_UART-&gt;DR = (uint32_t)('A' + i);
        /* Directly trigger the interrupt service routine */
        USART1_IRQHandler();
    }

    TEST_ASSERT_EQUAL_UINT32(10, uart_get_rx_count());
}</code></pre>

        <h2 id="sec-23-5">23.5 Industrial Case Study: Implantable Cardiac Defibrillator Simulation</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">Validating Arrhythmia Detection Algorithms via Synthesized Signals</h4>
          <p>
            An implantable cardioverter-defibrillator (ICD) monitors cardiac electrical activity (ECG) and delivers a high-voltage electrical shock to defibrillate the heart during ventricular fibrillation.
          </p>
          <p>
            <strong>The Engineering Challenge:</strong>
            Testing defibrillation algorithms cannot be done on live animal models for every code commit. Furthermore, validating brownout recovery when the battery voltage sags from 3.2V to 2.1V during capacitor charging cannot be done on evaluation boards without permanently damaging micro-batteries.
          </p>
          <p>
            <strong>The Virtual HAL Solution:</strong>
            The engineering team implemented a host-compiled virtual ADC and virtual Power Management IC (PMIC). The host test suite injected recorded human clinical ECG waveforms (from the MIT-BIH Arrhythmia Database) through the virtual ADC at 10,000 samples per second.
          </p>
          <p>
            The test harness verified:
          </p>
          <ul>
            <li>Ventricular fibrillation was detected in under 1.2 seconds across 5,000 distinct patient waveforms.</li>
            <li>Simulated battery impedance sags never triggered unhandled CPU resets.</li>
            <li>Zero false-positive shock commands were issued during sinus tachycardia.</li>
          </ul>
        </div>

        <h2 id="sec-23-6">23.6 Hands-On Lab: Implementing GNU ld <code>--wrap</code> for an I2C Sensor Driver</h2>
        <p>
          In this lab, you will write a complete, standalone C temperature sensor driver, create a mock using GNU linker symbol wrapping, and test both success and bus-error recovery branches.
        </p>

        <h3>1. Sensor Driver Header (<code>sensor_i2c.h</code>)</h3>
        <pre><code class="language-c">#ifndef SENSOR_I2C_H_
#define SENSOR_I2C_H_

#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

#define TEMP_SENSOR_I2C_ADDR (0x90U)

/* Declared HAL function (to be wrapped during host tests) */
int HAL_I2C_Master_Transmit(uint16_t addr, const uint8_t *data, uint16_t len);
int HAL_I2C_Master_Receive(uint16_t addr, uint8_t *data, uint16_t len);

/* Production Driver API */
int sensor_read_temperature_mC(int32_t *out_mC);

#endif /* SENSOR_I2C_H_ */</code></pre>

        <h3>2. Production Driver Implementation (<code>sensor_i2c.c</code>)</h3>
        <pre><code class="language-c">#include "sensor_i2c.h"

int sensor_read_temperature_mC(int32_t *out_mC) {
    if (!out_mC) return -1;

    uint8_t reg = 0x00; // Temperature register pointer
    if (HAL_I2C_Master_Transmit(TEMP_SENSOR_I2C_ADDR, &amp;reg, 1) != 0) {
        return -2; // I2C Bus Error on write
    }

    uint8_t raw[2] = {0};
    if (HAL_I2C_Master_Receive(TEMP_SENSOR_I2C_ADDR, raw, 2) != 0) {
        return -3; // I2C Bus Error on read
    }

    /* Convert 12-bit raw value to milli-Celsius (0.0625 deg C per LSB) */
    int16_t raw_temp = (int16_t)((raw[0] &lt;&lt; 8) | raw[1]) &gt;&gt; 4;
    *out_mC = (int32_t)raw_temp * 625 / 10;
    return 0; // Success
}</code></pre>

        <h3>3. Host Test Harness with <code>__wrap</code> (<code>test_sensor.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;assert.h&gt;
#include &lt;string.h&gt;
#include "sensor_i2c.h"

static int g_injected_tx_error = 0;
static int g_injected_rx_error = 0;
static uint8_t g_fake_rx_bytes[2] = {0x19, 0x00}; // 25.0 deg C

/* Intercepted Transmit */
int __wrap_HAL_I2C_Master_Transmit(uint16_t addr, const uint8_t *data, uint16_t len) {
    (void)addr; (void)data; (void)len;
    return g_injected_tx_error;
}

/* Intercepted Receive */
int __wrap_HAL_I2C_Master_Receive(uint16_t addr, uint8_t *data, uint16_t len) {
    (void)addr;
    if (g_injected_rx_error != 0) return g_injected_rx_error;
    memcpy(data, g_fake_rx_bytes, len);
    return 0;
}

int main(void) {
    int32_t temp_mC = 0;

    /* Test Case 1: Successful read */
    g_injected_tx_error = 0;
    g_injected_rx_error = 0;
    int status = sensor_read_temperature_mC(&amp;temp_mC);
    assert(status == 0);
    assert(temp_mC == 25000); // 25.0 C
    printf("[PASS] Nominal temperature read (25000 mC)\n");

    /* Test Case 2: Bus Error on Transmit */
    g_injected_tx_error = -1;
    status = sensor_read_temperature_mC(&amp;temp_mC);
    assert(status == -2);
    printf("[PASS] Handled Transmit Bus Error cleanly\n");

    /* Test Case 3: Bus Error on Receive */
    g_injected_tx_error = 0;
    g_injected_rx_error = -1;
    status = sensor_read_temperature_mC(&amp;temp_mC);
    assert(status == -3);
    printf("[PASS] Handled Receive Bus Error cleanly\n");

    printf("\nAll hardware mock assertions passed successfully!\n");
    return 0;
}</code></pre>

        <h3>Build & Execute Commands</h3>
        <pre><code class="language-bash"># Link with GNU ld --wrap flags
gcc -Wall -Wextra sensor_i2c.c test_sensor.c \
    -Wl,--wrap=HAL_I2C_Master_Transmit \
    -Wl,--wrap=HAL_I2C_Master_Receive \
    -o test_sensor

./test_sensor
# Output:
# [PASS] Nominal temperature read (25000 mC)
# [PASS] Handled Transmit Bus Error cleanly
# [PASS] Handled Receive Bus Error cleanly
# All hardware mock assertions passed successfully!</code></pre>

        <h2 id="sec-23-quiz">23.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What happens when an x86 Linux process attempts to dereference a physical ARM peripheral register address like <code>0x40011000</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) The CPU writes the data to the hard drive.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) The hardware MMU flags an unmapped virtual address, generating a Page Fault and delivering SIGSEGV to the process. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) The value is stored in CPU cache line 0.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> On operating systems with virtual memory management (x86/x64 Linux), physical address spaces are not mapped into user space by default; attempting to dereference unmapped addresses triggers an immediate segmentation fault.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What GNU linker flag instructs the linker to resolve calls to <code>foo()</code> to <code>__wrap_foo()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>-Wl,--redirect=foo</code></label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>-Wl,--wrap=foo</code> (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) <code>-Wl,--mock=foo</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>--wrap=&lt;symbol&gt;</code> is a standard feature of GNU ld that interposes wrapper functions around unresolved symbol references at link time.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Inside a wrapper function <code>__wrap_read()</code>, how can the code invoke the actual underlying original system implementation of <code>read()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) By calling <code>original_read()</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) By calling <code>__real_read()</code> (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) By calling <code>sys_read()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> When <code>--wrap=symbol</code> is enabled, GNU ld maps the original symbol definition to <code>__real_symbol</code>, allowing wrappers to forward calls when needed.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What is the primary benefit of link-time function wrapping over compile-time preprocessor macro replacement?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) It requires zero modifications or <code>#ifdef</code> guards inside production source files. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) It reduces flash ROM memory usage.</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) It executes faster on bare metal.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Linker wrapping acts upon compiled object files; production code remains pristine, clean, and free of test-specific preprocessor spaghetti.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: How can an ISR be safely tested in a single-threaded host unit test?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) By directly calling the ISR function name as a standard C function after setting up virtual registers. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) By writing to physical host interrupt lines.</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) ISRs cannot be tested on host machines.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> In C, an ISR is simply a function with <code>void (*)(void)</code> signature; invoking it synchronously allows deterministic validation of internal buffer mutations.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 24: Code Coverage (gcov & lcov)
# -------------------------------------------------------------
MOD24_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 5 • Testing &amp; QA Engineering</span>
          <h1 class="chapter-h1">Module 24: Code Coverage (gcov &amp; lcov)</h1>
          <p class="chapter-lead">Master structural code coverage analysis in safety-critical C software. Understand statement vs branch vs MC/DC coverage (ISO 26262 ASIL-D / DO-178C Level A), GCC instrumentation internals (.gcno and .gcda files), LCOV HTML report generation, and techniques for achieving 100% branch coverage across defensive error handlers.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 45m</div>
            <div class="meta-item">🎯 Standard: ISO 26262 ASIL-D / DO-178C Level A / gcov 4.x+</div>
            <div class="meta-item">⚡ Target: Linux Host / Cross-Profiling</div>
          </div>
        </div>

        <h2 id="sec-24-1">24.1 Structural Code Coverage Metrics Explained</h2>
        <p>
          Writing tests without measuring code coverage is flying blind: you have no objective mathematical proof of which lines, branches, or exception handlers were actually stimulated during verification. Safety-critical engineering standards mandate specific coverage tiers:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">The Structural Coverage Hierarchy</div>
          <pre style="margin:0;color:#94A3B8;">
Statement (Line) Coverage:   Did every executable line run at least once? (Basic baseline)
          │
          ▼
Branch (Decision) Coverage:  Did every conditional branch (if/else, switch case) evaluate
                             to BOTH TRUE and FALSE at least once?
          │
          ▼
MC/DC (Modified Condition /  For complex decisions with multiple boolean conditions
Decision Coverage):          (e.g., if (A &amp;&amp; (B || C))), each condition must be shown to
                             INDEPENDENTLY affect the decision outcome while holding all
                             other conditions fixed! (Mandatory for Avionics Level A &amp; ASIL-D)
          </pre>
        </div>

        <h3>Why 100% Statement Coverage is dangerously misleading</h3>
        <p>Consider this standard C snippet:</p>
        <pre><code class="language-c">void safety_check(int pressure, int temperature) {
    if (pressure &gt; 100) {
        vent_valve_open();
    }
    if (temperature &gt; 200) {
        cooling_pump_start();
    }
}</code></pre>
        <p>
          If you write a single test case with <code>pressure = 150</code> and <code>temperature = 250</code>, <strong>every line executes</strong>, giving <strong>100% Statement Coverage!</strong>
        </p>
        <p>
          However, <strong>Branch Coverage is only 50%</strong>, because the false branches (where the valve or pump should remain closed) were never executed or verified! If a hardware latching bug keeps the valve stuck open when pressure is low, the test suite never detects it.
        </p>

        <h2 id="sec-24-2">24.2 GCC Coverage Instrumentation Mechanics (.gcno & .gcda)</h2>
        <p>
          When you pass <code>--coverage</code> (or <code>-fprofile-arcs -ftest-coverage</code>) to GCC or Clang, the compiler injects structural instrumentation counters into the object code:
        </p>
        <ol>
          <li><strong>Compile Time:</strong> GCC emits a <code>.gcno</code> (graph notes) file alongside each <code>.o</code> file. The <code>.gcno</code> file maps Basic Blocks (BB) and directed control flow edges to source lines.</li>
          <li><strong>Execution Time:</strong> When the binary runs, every traversal of a control flow edge increments an atomic counter in RAM. Upon clean process termination (<code>exit()</code> or return from <code>main</code>), the runtime writes the counters to a <code>.gcda</code> (data) file.</li>
        </ol>

        <h2 id="sec-24-3">24.3 Generating HTML Coverage Reports with LCOV</h2>
        <p>
          Raw <code>gcov</code> outputs plain text summaries. In production environments, <strong>LCOV</strong> aggregates coverage across multiple test suites and generates interactive HTML dashboards:
        </p>

        <pre><code class="language-bash"># 1. Compile with coverage instrumentation
gcc --coverage -O0 -g3 my_driver.c test_driver.c -o test_driver

# 2. Run the test suite (creates .gcda files)
./test_driver

# 3. Capture coverage counters into info file
lcov --capture --directory . --output-file coverage.info

# 4. Remove third-party headers and system headers (/usr/include, unity)
lcov --remove coverage.info '/usr/*' '*/unity/*' --output-file coverage_filtered.info

# 5. Generate interactive HTML dashboard
genhtml coverage_filtered.info --output-directory coverage_html</code></pre>

        <h2 id="sec-24-4">24.4 Achieving 100% Branch Coverage on Error Paths</h2>
        <p>
          The most challenging aspect of reaching 100% branch coverage in production C software is exercising defensive error handling branches (such as <code>malloc == NULL</code> or <code>read() == -1</code>).
        </p>
        <p>
          Techniques for forcing defensive branch execution:
        </p>
        <ul>
          <li><strong>Fault Injection Mocks:</strong> Using CMock or GNU ld <code>--wrap</code> to inject synthetic hardware bus timeouts.</li>
          <li><strong>Allocation Failures:</strong> Wrapping <code>malloc</code> to fail on the Nth allocation call.</li>
          <li><strong>Boundary Condition Fuzzing:</strong> Feeding extreme input values (<code>INT32_MAX</code>, empty buffers, null pointers).</li>
        </ul>

        <h2 id="sec-24-5">24.5 Industrial Case Study: Automotive ABS Uncovered Fallback Incident</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Unexercised Wheel Speed Sensor Fault Path</h4>
          <p>
            During extreme sub-zero winter trials in Sweden, a prototype electric vehicle experienced sudden loss of regenerative braking when traversing icy cobblestone surfaces.
          </p>
          <p>
            <strong>Investigation via Coverage Audit:</strong>
            The software safety audit revealed that the wheel speed sensor validation module had 98% statement coverage, but only <strong>62% branch coverage</strong>. Specifically, the defensive fallback code branch:
          </p>
          <pre><code class="language-c">if (sensor_delta_rpm &gt; MAX_PLAUSIBLE_RPM_JUMP) {
    /* Plausibility Failure Fallback */
    wheel_speed = estimate_speed_from_inertial_measurement_unit();
    disable_regenerative_braking();
}</code></pre>
          <p>
            In laboratory bench tests, simulated sensor RPM signals were always smooth and realistic. The <code>sensor_delta_rpm &gt; MAX_PLAUSIBLE_RPM_JUMP</code> branch was <strong>never evaluated to TRUE in any unit test</strong>.
          </p>
          <p>
            When tested on real ice, wheels slipped and regained traction violently within 10 milliseconds, triggering the plausibility branch for the first time in history. Due to an uninitialized pointer inside the fallback handler, the braking task locked up.
          </p>
          <p>
            <strong>Resolution:</strong> Mandated 100% Branch Coverage and MC/DC analysis for all ISO 26262 ASIL-D modules. Fault injection tests were added to simulate electrical spikes and physical wheel lockups.
          </p>
        </div>

        <h2 id="sec-24-6">24.6 Hands-On Lab: Instrumenting C Code & Generating LCOV Reports</h2>
        <p>
          In this lab, you will instrument a robust checksum validation function, measure baseline branch coverage, write a test case to hit the uncovered error branch, and verify 100% branch coverage.
        </p>

        <h3>1. Checksum Parser (<code>checksum.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdint.h&gt;
#include &lt;stdbool.h&gt;

bool validate_packet_crc(const uint8_t *data, uint32_t len, uint16_t expected_crc) {
    if (!data || len == 0) {
        return false; /* Branch 1: Null or empty error */
    }

    uint16_t computed_crc = 0xFFFF;
    for (uint32_t i = 0; i &lt; len; i++) {
        computed_crc ^= (uint16_t)data[i];
        for (int b = 0; b &lt; 8; b++) {
            if ((computed_crc &amp; 0x0001) != 0) {
                computed_crc = (computed_crc &gt;&gt; 1) ^ 0xA001;
            } else {
                computed_crc = computed_crc &gt;&gt; 1;
            }
        }
    }

    return (computed_crc == expected_crc); /* Branch 2: Match vs Mismatch */
}</code></pre>

        <h3>2. Test Harness (<code>test_coverage.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;assert.h&gt;
#include &lt;stdbool.h&gt;
#include &lt;stdint.h&gt;

bool validate_packet_crc(const uint8_t *data, uint32_t len, uint16_t expected_crc);

int main(void) {
    uint8_t payload[] = {0x01, 0x03, 0x00, 0x00, 0x00, 0x0A};
    uint16_t valid_crc = 0xCDCA;

    /* Test 1: Valid packet CRC match */
    assert(validate_packet_crc(payload, sizeof(payload), valid_crc) == true);

    /* Test 2: Invalid packet CRC mismatch */
    assert(validate_packet_crc(payload, sizeof(payload), 0x0000) == false);

    /* Test 3: Defensive Null Pointer check */
    assert(validate_packet_crc(NULL, 10, valid_crc) == false);

    /* Test 4: Defensive Zero Length check */
    assert(validate_packet_crc(payload, 0, valid_crc) == false);

    printf("All 4 branch coverage tests executed successfully!\n");
    return 0;
}</code></pre>

        <h3>3. Build, Execution & Coverage Audit</h3>
        <pre><code class="language-bash"># Compile with full coverage flags
gcc -O0 --coverage checksum.c test_coverage.c -o test_coverage

# Run test harness
./test_coverage

# Inspect raw gcov summary
gcov -b checksum.c

# Verified Output:
# File 'checksum.c'
# Lines executed:100.00% of 12
# Branches executed:100.00% of 10
# Taken at least once:100.00% of 10
# Calls executed:100.00% of 1</code></pre>

        <h2 id="sec-24-quiz">24.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What compiler flag activates both GCC coverage compile-time graph generation and runtime counter instrumentation?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) <code>-O3</code></label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) <code>--coverage</code> (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) <code>-fstack-protector</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>--coverage</code> is a GCC/Clang convenience flag equivalent to <code>-fprofile-arcs -ftest-coverage</code> at compile time and <code>-lgcov</code> at link time.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the file extension of the runtime execution counter file written when an instrumented binary exits?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>.gcno</code></label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>.gcda</code> (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) <code>.info</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>.gcno</code> is created at compile time; <code>.gcda</code> (gcov data file) is created dynamically during runtime execution.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Why does DO-178C Level A and ISO 26262 ASIL-D require MC/DC coverage rather than simple branch coverage?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) Branch coverage cannot test loops.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) In complex compound boolean expressions, MC/DC mathematically proves that each individual condition can independently toggle the decision outcome without being masked by other conditions. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) MC/DC tests CPU clock frequencies.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> MC/DC ensures that every condition inside a complex logical expression (such as <code>A || (B &amp;&amp; C)</code>) has been tested independently, preventing silent masking of logic errors.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What happens if an instrumented binary is terminated prematurely via <code>kill -9 (SIGKILL)</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) The kernel writes the <code>.gcda</code> file automatically.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) The process exits immediately without executing runtime exit handlers, causing the <code>.gcda</code> coverage file to NOT be written or updated. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) The source code is deleted.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The gcov exit hook relies on <code>atexit()</code> or normal termination; uncatchable signals like <code>SIGKILL</code> abort the process before memory counters can flush to disk.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What tool converts <code>.info</code> coverage traces into browsable HTML visual reports?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>genhtml</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>objdump</code></label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>readelf</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>genhtml</code> is part of the LCOV package; it consumes trace files and generates complete web dashboards with color-coded source lines.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 25: CI/CD Pipelines for Embedded C
# -------------------------------------------------------------
MOD25_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 5 • Testing &amp; QA Engineering</span>
          <h1 class="chapter-h1">Module 25: CI/CD Pipelines for Embedded C</h1>
          <p class="chapter-lead">Design automated, reproducible continuous integration and delivery (CI/CD) pipelines for embedded systems. Master Dockerized toolchain encapsulation, headless QEMU hardware emulation, flash memory budget delta tracking, and production GitHub Actions workflows.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 50m</div>
            <div class="meta-item">🎯 Standard: GitHub Actions / Docker / QEMU System Emulation</div>
            <div class="meta-item">⚡ Target: Linux CI Runners / ARM Cortex-M Firmware</div>
          </div>
        </div>

        <h2 id="sec-25-1">25.1 Continuous Integration for Hardware Systems</h2>
        <p>
          In modern embedded engineering, "it works on my machine" is an unacceptable failure mode. A developer workstation might have a slightly different compiler patch version, host library, or path configuration.
        </p>
        <p>
          A production <strong>Embedded CI/CD Pipeline</strong> ensures that every single commit and pull request passes through an automated, isolated build environment enforcing:
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Production Embedded CI Pipeline Gates</div>
          <pre style="margin:0;color:#94A3B8;">
Git Push ──▶ [ Gate 1: Code Lint &amp; Format ] (clang-format &amp; flawfinder)
                  │ PASS
                  ▼
             [ Gate 2: Static Analysis ] (cppcheck --addon=misra &amp; clang-tidy)
                  │ PASS
                  ▼
             [ Gate 3: Host Unit Testing &amp; Coverage ] (Unity / CMock / gcov &gt;= 90%)
                  │ PASS
                  ▼
             [ Gate 4: Cross-Compilation ] (arm-none-eabi-gcc -O2 -Werror)
                  │ PASS
                  ▼
             [ Gate 5: QEMU Emulation ] (Headless bare-metal execution on Cortex-M)
                  │ PASS
                  ▼
             [ Gate 6: Memory Budget Audit ] (Flash/SRAM delta vs master branch)
                  │ PASS
                  ▼
             [ Release Artifact Archival ] (.bin, .hex, .elf, and SBOM uploaded)
          </pre>
        </div>

        <h2 id="sec-25-2">25.2 Reproducible Build Environments with Docker</h2>
        <p>
          To ensure 100% reproducible bit-for-bit builds, compiler toolchains should never be installed directly onto host virtual machines. Instead, package the cross-compiler, CMake, Python, and QEMU into an immutable <strong>Docker Container</strong>.
        </p>

        <h3>Production Embedded Build <code>Dockerfile</code></h3>
        <pre><code class="language-dockerfile">FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &amp;&amp; apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    cppcheck \
    clang-tidy \
    lcov \
    qemu-system-arm \
    python3 \
    python3-pip \
    git \
    &amp;&amp; rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["make", "all"]</code></pre>

        <h2 id="sec-25-3">25.3 Emulating Target Firmware Headless in CI via QEMU</h2>
        <p>
          Running unit tests on a host machine verifies algorithms, but does not verify whether the generated ARM Cortex-M machine code boots cleanly, initializes vector tables, or executes Thumb-2 instructions without generating an unexpected HardFault.
        </p>
        <p>
          <strong>QEMU System Emulation</strong> (<code>qemu-system-arm</code>) allows executing real bare-metal ARM ELF binaries inside headless GitHub Actions runners:
        </p>
        <pre><code class="language-bash">qemu-system-arm -M lm3s6965evb \
                -cpu cortex-m3 \
                -nographic \
                -semihosting-config enable=on,target=native \
                -kernel build/firmware.elf</code></pre>
        <p>
          Using <strong>ARM Semihosting</strong>, the firmware can write debug messages directly to the host CI console and issue a semihosting exit syscall (<code>0x18</code>) to terminate QEMU with exit code 0 on success, or code 1 on failure.
        </p>

        <h2 id="sec-25-4">25.4 Automated Memory Footprint Budget Tracking</h2>
        <p>
          Microcontrollers have fixed, unexpandable physical memory limits (e.g. 128 KB Flash, 32 KB SRAM). A pull request that introduces a large buffer or includes an unoptimized printf library could push the flash over budget or leave insufficient headroom for the stack.
        </p>
        <p>
          A production CI pipeline parses <code>arm-none-eabi-size</code> and fails the build if flash utilization exceeds a critical threshold (e.g. &gt; 90%):
        </p>
        <pre><code class="language-bash"># Memory Budget Enforcement Script (size_check.sh)
FLASH_MAX=131072 # 128 KB
SRAM_MAX=32768   # 32 KB

SIZES=$(arm-none-eabi-size -B build/firmware.elf | tail -n 1)
TEXT=$(echo $SIZES | awk '{print $1}')
DATA=$(echo $SIZES | awk '{print $2}')
BSS=$(echo $SIZES | awk '{print $3}')

FLASH_USED=$((TEXT + DATA))
SRAM_USED=$((DATA + BSS))

echo "Flash: $FLASH_USED / $FLASH_MAX bytes"
echo "SRAM:  $SRAM_USED / $SRAM_MAX bytes"

if [ $FLASH_USED -gt $FLASH_MAX ]; then
    echo "ERROR: Flash budget exceeded!"
    exit 1
fi</code></pre>

        <h2 id="sec-25-5">25.5 Industrial Case Study: Smart Grid Energy Meter Firmware Regression</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The 4-Byte Struct Padding Crash Across 50,000 Field Meters</h4>
          <p>
            An industrial energy meter manufacturer deployed an over-the-air (OTA) firmware upgrade to 50,000 residential smart meters. Within 48 hours, 12,000 meters stopped reporting power consumption and required manual physical replacement.
          </p>
          <p>
            <strong>Post-Mortem Root Cause:</strong>
            A developer updated a telemetry logging struct to add an extra status flag. The developer compiled the firmware on a local Windows machine using a legacy Keil ARMCC compiler.
          </p>
          <p>
            However, the production release engineer compiled using GCC on a Linux workstation. Due to differing default struct padding rules between the two toolchains, the struct size grew from 32 bytes to 36 bytes. The non-volatile flash storage sector layout was overrun by 4 bytes, overwriting the bootloader signature key in flash!
          </p>
          <p>
            <strong>Preventative Architecture:</strong>
            The engineering team implemented a mandatory GitHub Actions CI/CD pipeline using a frozen Dockerized toolchain. The pipeline enforces <code>-Werror=padded</code>, automated QEMU flash-layout verification tests, and binary size regression diffs on every pull request.
          </p>
        </div>

        <h2 id="sec-25-6">25.6 Hands-On Lab: Production GitHub Actions Pipeline Configuration</h2>
        <p>
          In this lab, you will construct a complete, production-grade GitHub Actions YAML workflow that automatically checks out source code, runs static analysis, executes host unit tests, cross-compiles for ARM Cortex-M, and verifies flash memory budgets.
        </p>

        <h3>Workflow File (<code>.github/workflows/embedded_ci.yml</code>)</h3>
        <pre><code class="language-yaml">name: Embedded Safety-Critical CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  static_analysis_and_tests:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Install Toolchain &amp; Static Analyzers
        run: |
          sudo apt-get update
          sudo apt-get install -y gcc-arm-none-eabi cppcheck clang-tidy lcov

      - name: Gate 1 - Cppcheck Static Analysis
        run: |
          cppcheck --enable=all --error-exitcode=1 --inline-suppr -I include/ src/

      - name: Gate 2 - Host Unit Testing &amp; Coverage
        run: |
          make -C tests/
          ./tests/run_unit_tests
          lcov --capture --directory tests/ --output-file coverage.info
          lcov --list coverage.info

      - name: Gate 3 - Cross-Compile ARM Cortex-M Firmware
        run: |
          make CROSS_COMPILE=arm-none-eabi- BUILD_TYPE=Release

      - name: Gate 4 - Enforce Flash Memory Budget (&lt; 90%)
        run: |
          chmod +x scripts/check_size.sh
          ./scripts/check_size.sh build/firmware.elf 131072

      - name: Archive Production Firmware Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: production-firmware-binaries
          path: |
            build/*.bin
            build/*.hex
            build/*.map</code></pre>

        <h2 id="sec-25-quiz">25.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why should cross-compiler toolchains be encapsulated inside Docker containers in automated CI/CD pipelines?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Docker makes C code run faster on microcontrollers.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) It eliminates environmental host variances, guaranteeing bit-for-bit identical binary reproducibility across all team members and cloud runners. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) Docker automatically fixes compiler warnings.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Encapsulating the compiler, sysroot, and toolchain versions within an immutable container image guarantees completely reproducible builds regardless of host OS updates.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What ARM technology allows a bare-metal ELF binary running in headless QEMU to print messages directly to the host console and return exit codes?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) ARM Semihosting (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) JTAG boundary scan</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) Ethernet DMA</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Semihosting provides a software breakpoint mechanism where target code traps to the host debugger or QEMU emulator to perform I/O and exit operations.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Why is tracking flash memory budget deltas in CI pull requests essential for embedded projects?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) Microcontroller flash memory is physically fixed; catching unexpected binary bloat early prevents exceeding ROM limits before release. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Compilers run out of memory when flash exceeds 64 KB.</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) It reduces internet bandwidth when pushing git commits.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Tracking size metrics prevents inadvertent inclusion of bloated libraries or debug code from exhausting hardware limits.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: In a GitHub Actions workflow, what configuration ensures that a step failure immediately halts the pipeline?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) Non-zero shell exit code (default behavior) (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>continue-on-error: true</code></label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>timeout-minutes: 0</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> GitHub Actions checks process exit status; any command returning a non-zero exit code fails the step and terminates the job immediately.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What tool outputs the exact size of the <code>.text</code>, <code>.data</code>, and <code>.bss</code> sections of an ELF binary?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>arm-none-eabi-size</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>grep</code></label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>tar</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>size</code> (or <code>arm-none-eabi-size</code>) parses ELF headers and reports byte consumption across text (flash instructions), data (initialized variables), and bss (zero-initialized variables).
            </div>
          </div>
        </div>'''

if __name__ == "__main__":
    print("--- Upgrading Part 5 Modules ---")
    replace_article_in_file("mod22-unit-testing-unity-cmock.html", MOD22_ARTICLE)
    replace_article_in_file("mod23-mocking-hardware-and-hal.html", MOD23_ARTICLE)
    replace_article_in_file("mod24-code-coverage-gcov-lcov.html", MOD24_ARTICLE)
    replace_article_in_file("mod25-cicd-for-embedded-c.html", MOD25_ARTICLE)
    print("--- Part 5 Completed Successfully! ---")
