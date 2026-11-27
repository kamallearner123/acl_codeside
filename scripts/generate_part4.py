#!/usr/bin/env python3
"""
generate_part4.py - Overhaul Part 4: Linux & Engineering Tools (Modules 17 to 21)
into comprehensive 10-page masterclass textbook chapters.
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
    
    # Replace the article content
    pattern = r'<article class="prose">.*?</article>'
    replacement = f'<article class="prose">\n{new_article_html}\n      </article>'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        raise ValueError(f"Could not find <article class='prose'> in {filename}")
    
    # Also ensure the sidebar is fresh
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
# MODULE 17: Build Systems (Make & CMake)
# -------------------------------------------------------------
MOD17_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 4 • Linux & Engineering Tools</span>
          <h1 class="chapter-h1">Module 17: Build Systems (Make & Modern CMake)</h1>
          <p class="chapter-lead">Master production Makefile architecture with automatic header dependency generation (-MMD -MP), out-of-tree builds, recursive vs non-recursive build paradigms, and modern target-based CMake 3.20+ cross-compilation for embedded systems and safety-critical runtimes.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: POSIX Make / GNU Make 4.x / CMake 3.20+</div>
            <div class="meta-item">⚡ Target: ARM Cortex-M / Linux x86_64</div>
          </div>
        </div>

        <h2 id="sec-17-1">17.1 GNU Make Internals & Dependency Resolution Graphs</h2>
        <p>
          At its computational core, <strong>GNU Make</strong> is an engine that evaluates a <em>Directed Acyclic Graph (DAG)</em> of target files, prerequisite files, and shell execution rules. Make determines whether a target needs to be updated by comparing the target's filesystem modification timestamp (<code>mtime</code>) against the modification timestamps of all its prerequisites.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">DAG Dependency Resolution &amp; Timestamp Traversal</div>
          <pre style="margin:0;color:#94A3B8;">
                 ┌──────────────────┐
                 │    firmware.elf  │ (mtime: 10:04:15)
                 └────────┬─────────┘
          Prerequisites:  │  Linker stage (arm-none-eabi-gcc)
            ┌─────────────┴─────────────┐
            ▼                           ▼
     ┌──────────────┐            ┌──────────────┐
     │    main.o    │            │   sensor.o   │ (mtime: 10:04:10)
     └──────┬───────┘            └──────┬───────┘
  Prereq:   │                     Prereq:   │
   ┌────────┴────────┐             ┌────────┴────────┐
   ▼                 ▼             ▼                 ▼
┌───────┐       ┌──────────┐    ┌──────────┐    ┌──────────┐
│main.c │       │ sensor.h │    │ sensor.c │    │ sensor.h │ (mtime: 10:05:00 - TOUCHED!)
└───────┘       └──────────┘    └──────────┘    └──────────┘
  (Make detects: sensor.h mtime > sensor.o &amp; main.o -&gt; triggers recompilation of both)
          </pre>
        </div>

        <h3>The Fatal Flaw of Naive Makefiles</h3>
        <p>
          Consider the traditional, naive rule found in beginner embedded projects:
        </p>
        <pre><code class="language-makefile">%.o: %.c
	$(CC) $(CFLAGS) -c $&lt; -o $@</code></pre>
        <p>
          In this naive rule, the prerequisite list contains only <code>%.c</code>. If an engineer modifies <code>sensor.h</code> (for example, adding a struct member or modifying an enum value), Make inspects <code>main.o</code> and <code>main.c</code>. Because <code>main.c</code>'s timestamp has not changed, Make assumes <code>main.o</code> is completely up-to-date and <strong>skips recompilation</strong>.
        </p>
        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">Catastrophic Failure Mode: Struct Layout Skew</strong><br>
          When header dependencies are ignored, module <code>sensor.c</code> might recompile against the new struct layout while <code>main.c</code> retains object code compiled against the old struct layout. When linked, the binary executes with conflicting memory offsets, resulting in silent memory corruption, hard faults, or telemetry field swapping in production hardware.
        </div>

        <h3>GCC Automatic Dependency Generation (-MMD, -MP)</h3>
        <p>
          The robust, production-grade solution is to instruct the compiler preprocessor to emit Makefile-compatible dependency rules during compilation using GCC flags:
        </p>
        <ul>
          <li><code>-MD</code>: Emits dependency rules into a <code>.d</code> file for all headers, including system headers (<code>&lt;stdio.h&gt;</code>, <code>&lt;stdint.h&gt;</code>).</li>
          <li><code>-MMD</code>: Emits dependency rules into a <code>.d</code> file for <em>user headers only</em> (omitting system headers), which speeds up Make parsing significantly.</li>
          <li><code>-MP</code>: Generates an empty phony target for every header file encountered in the dependency tree.</li>
        </ul>

        <p>
          <strong>Why is <code>-MP</code> essential?</strong> Without <code>-MP</code>, if you delete or rename a header file (e.g., refactoring <code>legacy_adc.h</code> into <code>adc_driver.h</code>), Make will read the existing <code>.d</code> file from the previous build, see that <code>legacy_adc.h</code> is a prerequisite, search for it on disk, discover it is missing, and crash with:
        </p>
        <pre><code class="language-text">make: *** No rule to make target 'legacy_adc.h', needed by 'main.o'. Stop.</code></pre>
        <p>
          With <code>-MP</code>, the compiler adds dummy phony targets like <code>legacy_adc.h:</code> into the <code>.d</code> file. When Make discovers the file is gone, it treats the prerequisite as satisfied and proceeds to invoke the compiler, which cleanly updates the object and dependency files.
        </p>

        <h2 id="sec-17-2">17.2 Production ARM Cortex-M Makefile Architecture</h2>
        <p>
          Below is a complete, industrial-grade Makefile engineered for embedded ARM Cortex-M firmware (such as STM32, NXP LPC, or TI MSPM0). It enforces:
        </p>
        <ol>
          <li>Complete out-of-tree build artifact isolation in <code>build/</code>.</li>
          <li>Automated <code>-MMD -MP</code> header dependency tracking via <code>-include $(DEPS)</code>.</li>
          <li>Dual target configurations (Debug with <code>-Og -g3</code>, Release with <code>-O2 -flto -DNDEBUG</code>).</li>
          <li>Automated memory footprint reporting via <code>size</code>.</li>
        </ol>

        <pre><code class="language-makefile"># ==============================================================================
# Professional Embedded C Makefile Architecture
# Target: ARM Cortex-M4F (STM32F401RE / STM32F411RE)
# ==============================================================================
PROJECT         ?= telemetry_firmware
BUILD_DIR       ?= build
CROSS_COMPILE   ?= arm-none-eabi-

CC              := $(CROSS_COMPILE)gcc
CXX             := $(CROSS_COMPILE)g++
OBJCOPY         := $(CROSS_COMPILE)objcopy
OBJDUMP         := $(CROSS_COMPILE)objdump
SIZE            := $(CROSS_COMPILE)size

# ------------------------------------------------------------------------------
# Target Architecture Flags (ARM Cortex-M4F, Hard VFPv4)
# ------------------------------------------------------------------------------
MCU_FLAGS       := -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard

# ------------------------------------------------------------------------------
# Source &amp; Include Directories
# ------------------------------------------------------------------------------
C_SOURCES       := src/main.c \
                   src/bsp/system_clock.c \
                   src/drivers/uart_driver.c \
                   src/drivers/sensor_fusion.c

ASM_SOURCES     := startup/startup_stm32f401xe.s

INCLUDES        := -Iinclude \
                   -Iinclude/bsp \
                   -Iinclude/drivers \
                   -Icmsis/include

LDSCRIPT        := linker/STM32F401RETx_FLASH.ld

# ------------------------------------------------------------------------------
# Compiler &amp; Preprocessor Flags
# ------------------------------------------------------------------------------
# -MMD -MP: Generate dependency files automatically
# -Wall -Wextra -Wpedantic: Maximum warning coverage
# -Wconversion -Wshadow -Wdouble-promotion: Safety-critical static checks
# -fdata-sections -ffunction-sections: Strip dead code during link
CPPFLAGS        := $(INCLUDES) -DSTM32F401xE -MMD -MP

CFLAGS          := $(MCU_FLAGS) \
                   -std=c11 \
                   -Wall -Wextra -Wpedantic \
                   -Wconversion -Wshadow -Wdouble-promotion -Wstrict-prototypes \
                   -fdata-sections -ffunction-sections

# Build Configuration: Debug vs Release
BUILD_TYPE      ?= Debug
ifeq ($(BUILD_TYPE),Debug)
  CFLAGS        += -Og -g3 -DDEBUG
else
  CFLAGS        += -O2 -g -flto -DNDEBUG
endif

# Linker Flags: Garbage-collect unused code/data (--gc-sections)
LDFLAGS         := $(MCU_FLAGS) \
                   -T$(LDSCRIPT) \
                   -Wl,-Map=$(BUILD_DIR)/$(PROJECT).map,--cref \
                   -Wl,--gc-sections \
                   -specs=nano.specs -specs=nosys.specs

# ------------------------------------------------------------------------------
# Object File Lists
# ------------------------------------------------------------------------------
OBJS            := $(addprefix $(BUILD_DIR)/, $(C_SOURCES:.c=.o))
OBJS            += $(addprefix $(BUILD_DIR)/, $(ASM_SOURCES:.s=.o))
DEPS            := $(OBJS:.o=.d)

# Phony Targets
.PHONY: all clean flash size disasm

all: $(BUILD_DIR)/$(PROJECT).elf $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex size

# Linking
$(BUILD_DIR)/$(PROJECT).elf: $(OBJS) | $(BUILD_DIR)
	@echo " [LD]  $@"
	@$(CC) $(OBJS) $(LDFLAGS) -o $@

# Binary generation
$(BUILD_DIR)/$(PROJECT).bin: $(BUILD_DIR)/$(PROJECT).elf
	@echo " [BIN] $@"
	@$(OBJCOPY) -O binary $&lt; $@

$(BUILD_DIR)/$(PROJECT).hex: $(BUILD_DIR)/$(PROJECT).elf
	@echo " [HEX] $@"
	@$(OBJCOPY) -O ihex $&lt; $@

# Compile C Sources
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo " [CC]  $&lt;"
	@$(CC) $(CPPFLAGS) $(CFLAGS) -c $&lt; -o $@

# Assemble Assembly Sources
$(BUILD_DIR)/%.o: %.s
	@mkdir -p $(dir $@)
	@echo " [AS]  $&lt;"
	@$(CC) $(MCU_FLAGS) -c $&lt; -o $@

$(BUILD_DIR):
	@mkdir -p $@

size: $(BUILD_DIR)/$(PROJECT).elf
	@echo "--- Target Memory Utilization ---"
	@$(SIZE) -B $&lt;

disasm: $(BUILD_DIR)/$(PROJECT).elf
	@$(OBJDUMP) -d -S $&lt; &gt; $(BUILD_DIR)/$(PROJECT).asm
	@echo "Disassembly written to $(BUILD_DIR)/$(PROJECT).asm"

clean:
	@echo " [CLEAN] Removing build directory..."
	@rm -rf $(BUILD_DIR)

# Include automatically generated header dependencies (-MMD -MP)
# The minus sign suppresses errors when .d files do not yet exist
-include $(DEPS)</code></pre>

        <h2 id="sec-17-3">17.3 Modern Target-Based CMake (3.20+) Architecture</h2>
        <p>
          Legacy CMake (version 2.8 and early 3.x) relied on directory-scoped global variables such as <code>include_directories()</code>, <code>add_definitions()</code>, and <code>link_libraries()</code>. In complex multi-subsystem embedded projects, this legacy pattern caused global variable pollution and unexpected flag leakage between unit tests and firmware targets.
        </p>
        <p>
          <strong>Modern CMake (3.15+)</strong> introduces an Object-Oriented target model. Every library, driver, and executable is an independent <em>target</em> possessing encapsulated properties. Dependencies between targets are declared explicitly, and transitive properties propagate strictly through visibility modifiers:
        </p>

        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Visibility Scope</th>
              <th style="padding:10px;text-align:left;">Target Itself Consumes?</th>
              <th style="padding:10px;text-align:left;">Downstream Dependents Inherit?</th>
              <th style="padding:10px;text-align:left;">Embedded Typical Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong style="color:#38BDF8;">PRIVATE</strong></td>
              <td style="padding:10px;">Yes</td>
              <td style="padding:10px;">No</td>
              <td style="padding:10px;">Internal driver implementation headers (e.g. <code>src/drivers/i2c_reg_map.h</code>).</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong style="color:#10B981;">INTERFACE</strong></td>
              <td style="padding:10px;">No</td>
              <td style="padding:10px;">Yes</td>
              <td style="padding:10px;">Header-only abstractions, CMSIS headers, interface contracts (pure virtual vtables).</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong style="color:#F59E0B;">PUBLIC</strong></td>
              <td style="padding:10px;">Yes</td>
              <td style="padding:10px;">Yes</td>
              <td style="padding:10px;">Driver API header exposed to both the driver source and application callers.</td>
            </tr>
          </tbody>
        </table>

        <h3>Transitive Target Dependency Graph in CMake</h3>
        <pre><code class="language-cmake"># Modern CMake Target Configuration
add_library(bsp_uart STATIC
    src/uart.c
)

target_include_directories(bsp_uart
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # Consumers need uart.h
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src         # Only uart.c needs internal register maps
)

target_compile_options(bsp_uart
    PRIVATE
        -Wall -Wextra -Wconversion              # Strict warnings on internal compilation
)

# Application Executable
add_executable(firmware src/main.c)
target_link_libraries(firmware PRIVATE bsp_uart)
# 'firmware' automatically inherits bsp_uart's PUBLIC include directories!</code></pre>

        <h2 id="sec-17-4">17.4 Cross-Compiling with CMake Toolchain Files</h2>
        <p>
          CMake requires an external <strong>toolchain file</strong> to instruct it that it is targeting a bare-metal platform rather than the host Linux or macOS environment. Without a toolchain file, CMake will attempt to test-compile host binaries, query host glibc libraries, and fail when searching for embedded libraries.
        </p>

        <h3>Production Toolchain File: <code>arm-none-eabi.cmake</code></h3>
        <pre><code class="language-cmake"># ------------------------------------------------------------------------------
# arm-none-eabi.cmake - Bare-Metal Cross-Compilation Toolchain
# ------------------------------------------------------------------------------
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Specify Cross-Compiler Executables
set(TOOLCHAIN_PREFIX arm-none-eabi-)
find_program(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc     REQUIRED)
find_program(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++     REQUIRED)
find_program(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}gcc     REQUIRED)
find_program(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}objcopy REQUIRED)
find_program(CMAKE_SIZE_UTIL    ${TOOLCHAIN_PREFIX}size    REQUIRED)

# Inhibit CMake from trying to link an executable during initial compiler check
# (Bare-metal linking fails without a valid linker script)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Prevent CMake from searching for host libraries and headers
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)</code></pre>

        <h3>Root <code>CMakeLists.txt</code> with Linker Script & Post-Build Size Rule</h3>
        <pre><code class="language-cmake">cmake_minimum_required(VERSION 3.20)
project(cortex_m_firmware C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Hardware Architecture Compilation Flags
set(ARM_FLAGS -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard)
add_compile_options(${ARM_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -ffunction-sections -fdata-sections)

# Target Linker Script
set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/linker/STM32F401RETx_FLASH.ld)
add_link_options(${ARM_FLAGS} -T${LINKER_SCRIPT} -Wl,--gc-sections -specs=nano.specs -specs=nosys.specs)

# Define Executable
add_executable(${PROJECT_NAME}.elf
    src/main.c
    src/sensor.c
    startup/startup_stm32f401xe.s
)

# Custom Command: Generate .bin and print size after successful build
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary ${PROJECT_NAME}.elf ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_SIZE_UTIL} -B ${PROJECT_NAME}.elf
    COMMENT "Building binary and analyzing memory utilization"
)</code></pre>

        <h2 id="sec-17-5">17.5 Industrial Case Study: The Out-of-Date Header Flight Computer Glitch</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Telemetry Packet Shift Incident (Aerospace UAV Subsystem)</h4>
          <p>
            During pre-flight testing of an unmanned aerial system (UAS), ground telemetry reported corrupt attitude values (pitch = 180°, roll = NaN) even though the on-board AHRS unit was verified operational on an oscilloscope.
          </p>
          <p>
            <strong>Root Cause Analysis:</strong>
            The communication protocol header <code>telemetry_packet.h</code> was updated by the navigation team to insert a 4-byte <code>uint32_t flight_mode_status</code> field before the <code>float pitch</code> and <code>float roll</code> fields. The developers ran <code>make</code> on their local workstation.
          </p>
          <p>
            Because the project’s Makefile lacked automated dependency tracking (<code>-MMD -MP</code>), Make noted that <code>nav_task.c</code> had been edited and recompiled it, but left <code>telemetry_transmitter.c</code> untouched because its <code>.c</code> file was older than its <code>.o</code> file.
          </p>
          <div class="diagram-box" style="background:#0F172A;padding:1rem;border-radius:6px;font-family:monospace;font-size:0.8rem;color:#E2E8F0;margin:1rem 0;">
Nav Task compiled with NEW struct (size = 28 bytes):
[0..3: hdr] [4..7: mode] [8..11: pitch] [12..15: roll] [16..27: payload]

Transmitter compiled with OLD struct (size = 24 bytes):
[0..3: hdr] [4..7: pitch] [8..11: roll] [12..23: payload]
          </div>
          <p>
            When <code>telemetry_transmitter.c</code> read <code>pitch</code> from offset 4, it was actually broadcasting the integer bits of <code>flight_mode_status</code> interpreted as an IEEE-754 floating-point number, corrupting all downlinked data.
          </p>
          <p>
            <strong>Corrective Action:</strong> All project Makefiles were refactored to enforce <code>-MMD -MP</code> and <code>-include $(DEPS)</code>. The CI/CD pipeline was configured to build with strict dependency checks and enforce out-of-tree builds.
          </p>
        </div>

        <h2 id="sec-17-6">17.6 Hands-On Lab: Multi-Module C Project with Dependency Tracking</h2>
        <p>
          In this laboratory, you will create a multi-module C project with complete automated dependency tracking. You will prove that editing a header file automatically triggers the recompilation of all dependent translation units without touching the <code>.c</code> files.
        </p>

        <h3>Step 1: File Structure</h3>
        <pre><code class="language-text">lab_build/
├── include/
│   ├── config.h
│   └── sensor.h
├── src/
│   ├── main.c
│   └── sensor.c
└── Makefile</code></pre>

        <h3>Step 2: Source Files</h3>
        <p><strong><code>include/config.h</code>:</strong></p>
        <pre><code class="language-c">#ifndef CONFIG_H_
#define CONFIG_H_

#define SENSOR_SAMPLING_RATE_HZ  (100U)
#define TELEMETRY_MAGIC_CODE     (0xDEADBEEFU)

#endif /* CONFIG_H_ */</code></pre>

        <p><strong><code>include/sensor.h</code>:</strong></p>
        <pre><code class="language-c">#ifndef SENSOR_H_
#define SENSOR_H_

#include &lt;stdint.h&gt;
#include "config.h"

typedef struct {
    uint32_t sample_id;
    int32_t  temperature_mC;
    uint32_t status_flags;
} sensor_reading_t;

int sensor_init(void);
int sensor_poll(sensor_reading_t *out_reading);

#endif /* SENSOR_H_ */</code></pre>

        <p><strong><code>src/sensor.c</code>:</strong></p>
        <pre><code class="language-c">#include "sensor.h"
#include &lt;stdio.h&gt;

static uint32_t g_sample_counter = 0;

int sensor_init(void) {
    printf("[SENSOR] Initialized at rate %u Hz\n", SENSOR_SAMPLING_RATE_HZ);
    return 0;
}

int sensor_poll(sensor_reading_t *out_reading) {
    if (!out_reading) return -1;
    out_reading-&gt;sample_id = ++g_sample_counter;
    out_reading-&gt;temperature_mC = 25400; // 25.4 deg C
    out_reading-&gt;status_flags = TELEMETRY_MAGIC_CODE;
    return 0;
}</code></pre>

        <p><strong><code>src/main.c</code>:</strong></p>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include "sensor.h"

int main(void) {
    sensor_reading_t reading = {0};
    sensor_init();
    sensor_poll(&amp;reading);
    printf("[MAIN] Sample #%u: Temp=%d mC, Status=0x%08X\n",
           reading.sample_id, reading.temperature_mC, reading.status_flags);
    return 0;
}</code></pre>

        <h3>Step 3: Verification Commands</h3>
        <pre><code class="language-bash"># 1. First build
make
# Output shows both sensor.c and main.c being compiled

# 2. Touch header file only
touch include/config.h
make
# Verified: Both sensor.o and main.o recompile because both depend on config.h!

# 3. Inspect generated dependency file
cat build/src/main.d
# Output:
# build/src/main.o: src/main.c include/sensor.h include/config.h
# include/sensor.h:
# include/config.h:</code></pre>

        <h2 id="sec-17-quiz">17.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What is the primary purpose of the GCC <code>-MP</code> flag when generating Make dependencies?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) It parallelizes compilation across all available CPU cores.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) It emits an empty phony target for each header, preventing Make errors when a header is deleted or renamed. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) It minimizes the generated binary size by stripping unused debugging symbols.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Without <code>-MP</code>, removing an obsolete header referenced in a cached <code>.d</code> file causes Make to crash with <code>No rule to make target 'deleted_header.h'</code>.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the difference between <code>=</code> and <code>:=</code> variable assignment in GNU Make?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>=</code> performs recursively-expanded evaluation upon every variable use, while <code>:=</code> evaluates simply once at declaration time. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>:=</code> defines environment variables, while <code>=</code> defines local variables.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) There is no difference; they are interchangeable aliases in POSIX Make.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Recursively-expanded variables (<code>=</code>) re-evaluate their right-hand expressions (including shell commands) every time the variable is referenced, which can cause significant build slowdowns.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: In Modern CMake (3.15+), if target B links to target A with <code>target_link_libraries(B PRIVATE A)</code>, which properties does B inherit from A?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) B inherits both PRIVATE and PUBLIC properties of A.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) B inherits PUBLIC and INTERFACE properties of A, but will NOT propagate them to targets that link against B. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) B inherits only the binary file path of A with no include directories.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>PRIVATE</code> linking consumes the upstream target's <code>PUBLIC</code> and <code>INTERFACE</code> requirements internally, but stops transitive forwarding to downstream dependents.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: Why is <code>CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY</code> critical in bare-metal CMake toolchain files?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) It bypasses the test-linking step during compiler detection, preventing failure before a linker script is supplied. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) It forces all C libraries to compile as shared objects (<code>.so</code>).</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) It instructs the host GCC to run in 32-bit compatibility mode.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> By default, CMake compiles and links a test executable to verify the compiler. On bare-metal targets, linking fails without a device-specific linker script and reset vector, causing CMake to abort configuration.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What automatic Make variable represents the filename of the target being built?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>$&lt;</code> (First prerequisite)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>$^</code> (All prerequisites)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>$@</code> (Target name) (Correct)</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>$@</code> evaluates to the target being generated, <code>$&lt;</code> evaluates to the first prerequisite (e.g. <code>%.c</code>), and <code>$^</code> evaluates to all prerequisites.
            </div>
          </div>
        </div>'''

if __name__ == "__main__":
    replace_article_in_file("mod17-build-systems-make-and-cmake.html", MOD17_ARTICLE)

# -------------------------------------------------------------
# MODULE 18: GDB Debugging & Core Dumps
# -------------------------------------------------------------
MOD18_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 4 • Linux & Engineering Tools</span>
          <h1 class="chapter-h1">Module 18: GDB Debugging & Core Dump Analysis</h1>
          <p class="chapter-lead">Deep dive into debugger architecture, the Linux ptrace system call, ARM Remote Serial Protocol (RSP) hardware debugging, hardware watchpoints vs software breakpoints, post-mortem Linux core dump analysis, and automated GDB Python scripting.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 65m</div>
            <div class="meta-item">🎯 Standard: POSIX ptrace / GDB RSP / DWARF 4 &amp; 5</div>
            <div class="meta-item">⚡ Target: Linux x86_64 / ARM Cortex-M (OpenOCD/JTAG)</div>
          </div>
        </div>

        <h2 id="sec-18-1">18.1 Debugger Architecture: ptrace & Remote Serial Protocol</h2>
        <p>
          To master debugging, an engineer must understand how the debugger manipulates the target program. On Linux, GDB relies on the <strong>ptrace(2)</strong> system call. On bare-metal microcontrollers (ARM Cortex-M), GDB communicates across a TCP socket or UART with a debug agent (OpenOCD, J-Link, or pyOCD) using the <strong>GDB Remote Serial Protocol (RSP)</strong>.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Debugger Architectural Paradigms</div>
          <pre style="margin:0;color:#94A3B8;">
1. Linux Host Debugging (ptrace):
   [ GDB User Process ] ──ptrace(PTRACE_PEEKTEXT/POKEDATA, pid)──▶ [ Linux Kernel ]
                                                                          │
                                                                   Controls CPU context
                                                                          ▼
                                                                [ Target Process (SIGTRAP) ]

2. Embedded Bare-Metal Remote Debugging (RSP):
   [ arm-none-eabi-gdb ] ──TCP Port 3333 ($vCont?#49)──▶ [ OpenOCD / GDB Server ]
                                                                  │
                                                           SWD / JTAG Bus
                                                                  ▼
                                                      [ ARM Cortex-M Core (DWT/FPB) ]
          </pre>
        </div>

        <h3>Linux ptrace Mechanics</h3>
        <p>
          When GDB attaches to a target process via <code>ptrace(PTRACE_ATTACH, pid, ...)</code>, the kernel suspends the target thread and delivers a <code>SIGSTOP</code> signal. GDB then reads or writes target memory using <code>PTRACE_PEEKDATA</code> and <code>PTRACE_POKEDATA</code>, and inspects or alters CPU register state using <code>PTRACE_GETREGS</code> and <code>PTRACE_SETREGS</code>.
        </p>

        <h3>ARM GDB Remote Serial Protocol (RSP)</h3>
        <p>
          In embedded systems, the target hardware cannot run a full Linux kernel. Instead, a host-side GDB server (such as OpenOCD) translates GDB's ASCII packets into low-level Serial Wire Debug (SWD) or JTAG register access commands:
        </p>
        <ul>
          <li><code>$g#67</code>: Read all general-purpose registers (R0-R15, xPSR).</li>
          <li><code>$m20000000,4#XX</code>: Read 4 bytes from SRAM address <code>0x20000000</code>.</li>
          <li><code>$Z0,08000100,2#XX</code>: Insert a software breakpoint at flash address <code>0x08000100</code>.</li>
        </ul>

        <h2 id="sec-18-2">18.2 Hardware Watchpoints vs. Software Breakpoints</h2>
        <p>
          One of the most critical low-level distinctions is the difference between a <strong>Software Breakpoint</strong> and a <strong>Hardware Breakpoint / Watchpoint</strong>.
        </p>

        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Feature</th>
              <th style="padding:10px;text-align:left;">Software Breakpoint (<code>break</code>)</th>
              <th style="padding:10px;text-align:left;">Hardware Breakpoint (<code>hbreak</code>)</th>
              <th style="padding:10px;text-align:left;">Hardware Watchpoint (<code>watch</code>)</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Mechanism</strong></td>
              <td style="padding:10px;">Replaces instruction opcode with a trap instruction: <code>INT 3</code> (0xCC) on x86, <code>BKPT</code> (0xBE00) on ARM.</td>
              <td style="padding:10px;">Configures silicon comparator registers (ARM FPB: Flash Patch &amp; Breakpoint unit).</td>
              <td style="padding:10px;">Configures silicon memory comparators (ARM DWT: Data Watchpoint and Trace).</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Flash / ROM Compatible?</strong></td>
              <td style="padding:10px;"><span style="color:#EF4444;font-weight:bold;">No</span> (Flash cannot be overwritten without an erase cycle).</td>
              <td style="padding:10px;"><span style="color:#10B981;font-weight:bold;">Yes</span> (Silicon unit intercepts instruction fetch address on bus).</td>
              <td style="padding:10px;"><span style="color:#10B981;font-weight:bold;">Yes</span> (Intercepts data read/write bus transactions).</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Quantity Available</strong></td>
              <td style="padding:10px;">Virtually unlimited (in RAM).</td>
              <td style="padding:10px;">Limited by silicon (Typically 4 to 8 on Cortex-M4).</td>
              <td style="padding:10px;">Limited by silicon (Typically 2 to 4 on Cortex-M4).</td>
            </tr>
          </tbody>
        </table>

        <h3>Using Watchpoints to Catch Silent Memory Corruption</h3>
        <p>
          When an unknown bug randomly overwrites a global variable or buffer, stepping through thousands of lines of code is useless. Instead, set a hardware data watchpoint:
        </p>
        <pre><code class="language-gdb">(gdb) watch g_system_state.telemetry_counter
Hardware watchpoint 1: g_system_state.telemetry_counter
(gdb) continue
Continuing.

Hardware watchpoint 1: g_system_state.telemetry_counter
Old value = 14
New value = 3735928559
0x08001a42 in faulty_buffer_copy (src=0x20000100, len=128) at drivers/dma.c:45
45          dest[i] = src[i]; /* Overflow stepped on g_system_state! */</code></pre>

        <h2 id="sec-18-3">18.3 Memory Inspection, Stack Unwinding & Disassembly</h2>
        <p>
          Mastery of GDB requires instant fluency in reading memory and disassembly:
        </p>

        <h3>1. The <code>x</code> (Examine Memory) Command</h3>
        <p>
          Syntax: <code>x / [RepeatCount] [Format] [UnitSize] [Address]</code><br>
          Format specifiers: <code>x</code> (hex), <code>d</code> (signed dec), <code>u</code> (unsigned dec), <code>s</code> (string), <code>i</code> (instruction).<br>
          Unit size: <code>b</code> (byte), <code>h</code> (halfword/16-bit), <code>w</code> (word/32-bit), <code>g</code> (giant/64-bit).
        </p>
        <pre><code class="language-gdb"># Dump 16 32-bit words in hex from peripheral base address
(gdb) x/16xw 0x40020000
0x40020000: 0x00000000 0x00000000 0x00000000 0x00000000
0x40020010: 0x00000000 0x00000000 0x00000000 0x00000000

# Disassemble the next 8 instructions at the Program Counter
(gdb) x/8i $pc
=&gt; 0x8000450 &lt;HardFault_Handler&gt;:    tst.w   lr, #4
   0x8000454 &lt;HardFault_Handler+4&gt;:  ite     eq
   0x8000456 &lt;HardFault_Handler+6&gt;:  mrseq   r0, msp
   0x800045a &lt;HardFault_Handler+10&gt;: mrsne   r0, psp
   0x800045e &lt;HardFault_Handler+14&gt;: b.w     0x80005a0 &lt;prvGetRegistersFromStack&gt;</code></pre>

        <h3>2. Stack Unwinding with AAPCS Calling Conventions</h3>
        <p>
          When an execution halts, <code>backtrace full</code> (or <code>bt full</code>) decodes the DWARF call frame information (CFI) to reconstruct the active stack frames, arguments, and local variables.
        </p>
        <pre><code class="language-gdb">(gdb) bt full
#0  ring_buffer_pop (rb=0x20000400, out_byte=0x0) at src/ring_buffer.c:52
        tail = 4
        head = 12
#1  0x080003b4 in uart_rx_isr () at src/uart.c:120
        received_char = 65 'A'
#2  &lt;signal handler called&gt;
#3  0x080001f0 in main () at src/main.c:88
        i = 100</code></pre>

        <h2 id="sec-18-4">18.4 Linux Core Dumps & Post-Mortem Analysis</h2>
        <p>
          In production Linux deployments (such as embedded gateways, robotics controllers, or telecom boxes), applications run unattended without an attached interactive debugger. When an application crashes via <code>SIGSEGV</code>, <code>SIGBUS</code>, or <code>SIGABRT</code>, the Linux kernel can dump the process’s complete memory address space, register state, and stack traces into an ELF file known as a <strong>Core Dump</strong>.
        </p>

        <h3>Enabling Core Dumps in Linux</h3>
        <pre><code class="language-bash"># Set maximum core dump size to unlimited (for current shell session)
ulimit -c unlimited

# Configure kernel core dump naming pattern (PID, executable name, timestamp)
sudo sysctl -w kernel.core_pattern="/var/crash/core-%e-%p-%t"</code></pre>

        <h3>Post-Mortem Investigation Workflow</h3>
        <p>
          Once a crash occurs, load the binary executable alongside the generated core dump into GDB:
        </p>
        <pre><code class="language-bash">gdb ./bin/telemetry_daemon /var/crash/core-telemetry_daemon-4512-1718002000</code></pre>
        <pre><code class="language-gdb">[New LWP 4512]
Core was generated by `./bin/telemetry_daemon --config /etc/telemetry.conf`.
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x0000555555555234 in process_packet (pkt=0x0) at src/packet.c:78
78          uint32_t magic = pkt-&gt;magic_number; /* Crash: dereferencing NULL pkt! */
(gdb) info registers
rax            0x0                 0
rbx            0x7fffffffe200      140737488347648
rcx            0x555555556010      93824992239632
rdx            0x0                 0
rsp            0x7fffffffe1a0      0x7fffffffe1a0
rip            0x555555555234      0x555555555234 &lt;process_packet+20&gt;</code></pre>

        <h2 id="sec-18-5">18.5 GDB Automation & Python Scripting</h2>
        <p>
          Modern GDB includes an integrated Python 3 interpreter. Engineers can write Python scripts to inspect complex embedded data structures, automate register validation, and produce clean visualizations of circular ring buffers or RTOS task lists.
        </p>

        <h3>Custom Ring Buffer Visualizer in GDB Python</h3>
        <pre><code class="language-python"># save as print_ringbuf.py
import gdb

class PrintRingBuffer(gdb.Command):
    """Inspect and pretty-print an embedded circular ring buffer."""
    
    def __init__(self):
        super(PrintRingBuffer, self).__init__("print-ringbuf", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        val = gdb.parse_and_eval(arg)
        head = int(val['head'])
        tail = int(val['tail'])
        capacity = int(val['capacity'])
        buffer_ptr = val['buffer']
        
        count = (head - tail + capacity) % capacity
        print(f"--- RingBuffer [{arg}] ---")
        print(f" Capacity: {capacity} | Head: {head} | Tail: {tail} | Items Pending: {count}")
        
        items = []
        idx = tail
        while idx != head:
            byte_val = int(buffer_ptr[idx])
            items.append(f"0x{byte_val:02X}")
            idx = (idx + 1) % capacity
        print(" Contents: [" + ", ".join(items) + "]")

PrintRingBuffer()</code></pre>
        <p>Load and invoke the script directly inside GDB:</p>
        <pre><code class="language-gdb">(gdb) source print_ringbuf.py
(gdb) print-ringbuf g_uart_rx_buf
--- RingBuffer [g_uart_rx_buf] ---
 Capacity: 128 | Head: 5 | Tail: 2 | Items Pending: 3
 Contents: [0x53, 0x59, 0x4E]</code></pre>

        <h2 id="sec-18-6">18.6 Industrial Case Study: Subsea Sensor Watchdog HardFault</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Deep-Sea Oceanographic Acoustic Logger Failure</h4>
          <p>
            An acoustic hydrophone array operating at 2,000 meters depth suffered sporadic system reboots every 18 to 36 hours. The hardware watchdog triggered an emergency reset, preventing acoustic data loss but corrupting the current flash sector.
          </p>
          <p>
            <strong>Investigation via Embedded Crash Dump:</strong>
            The firmware was equipped with a custom HardFault handler that saved the CPU register frame (R0-R3, R12, LR, PC, xPSR) and Configurable Fault Status Register (CFSR) into a reserved battery-backed SRAM region.
          </p>
          <p>
            Upon recovery, engineers dumped the fault registers:
          </p>
          <pre><code class="language-text">CFSR: 0x00008200 (BFARVALID | PRECISERR - Precise Data Bus Error)
BFAR: 0x20008004 (Bus Fault Address Register - Attempted access at 0x20008004)
PC:   0x0800318C
LR:   0x08003107</code></pre>
          <p>
            Loading the ELF file into <code>arm-none-eabi-gdb</code>:
          </p>
          <pre><code class="language-gdb">(gdb) info line *0x0800318C
Line 142 of "src/hydrophone_dma.c" starts at address 0x8003184 &lt;dma_ping_pong_swap+16&gt;
(gdb) disassemble 0x8003184
0x08003188 &lt;+20&gt;: ldr   r3, [r0, #4]
0x0800318a &lt;+22&gt;: ldr   r2, [r3, #0]
0x0800318c &lt;+24&gt;: str   r1, [r2, #4]  &lt;-- CRASH HERE (BFAR = 0x20008004)</code></pre>
          <p>
            <strong>Root Cause:</strong> The hydrophone SRAM size was 32 KB (ending at <code>0x20007FFF</code>). A DMA ping-pong buffer pointer calculation suffered an off-by-one pointer arithmetic overflow during sample rates exceeding 96 kHz, writing to <code>0x20008004</code>. Because that address was unmapped physical memory, the Cortex-M bus matrix generated an immediate <code>PRECISERR</code> Data Bus Fault.
          </p>
        </div>

        <h2 id="sec-18-7">18.7 Hands-On Lab: Debugging a Real Segfaulting C Daemon</h2>
        <p>
          In this lab, you will compile a multi-threaded C program with a latent pointer corruption defect, generate a core dump, and isolate the exact faulting line using GDB.
        </p>

        <h3>Faulty C Code (<code>lab_crash.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;

typedef struct node {
    int id;
    char tag[16];
    struct node *next;
} node_t;

static node_t *create_node(int id, const char *tag) {
    node_t *n = (node_t *)malloc(sizeof(node_t));
    if (!n) return NULL;
    n-&gt;id = id;
    strncpy(n-&gt;tag, tag, sizeof(n-&gt;tag) - 1);
    n-&gt;next = NULL;
    return n;
}

static void print_list(node_t *head) {
    node_t *curr = head;
    while (curr != NULL) {
        printf("Node [%d]: %s\n", curr-&gt;id, curr-&gt;tag);
        curr = curr-&gt;next; /* Will crash if curr pointer was corrupted */
    }
}

int main(void) {
    node_t *first = create_node(1, "Alpha");
    node_t *second = create_node(2, "Beta");
    first-&gt;next = second;

    /* Simulating a silent heap corruption bug */
    char *corrupter = (char *)second;
    memset(corrupter + sizeof(int) + 16, 0xAA, sizeof(void *)); /* Overwrites second-&gt;next with 0xAAAAAAAAAAAAAAAA */

    printf("Traversing list...\n");
    print_list(first);

    return 0;
}</code></pre>

        <h3>Build & Debug Script</h3>
        <pre><code class="language-bash"># 1. Compile with debugging symbols (-g3) and frame pointers
gcc -O0 -g3 -fno-omit-frame-pointer lab_crash.c -o lab_crash

# 2. Enable core dumps
ulimit -c unlimited

# 3. Execute and trigger crash
./lab_crash
# Output:
# Traversing list...
# Node [1]: Alpha
# Node [2]: Beta
# Segmentation fault (core dumped)

# 4. Load core dump in GDB
gdb -q ./lab_crash core

# 5. Inside GDB:
# (gdb) bt
# #0  print_list (head=0x5555555592a0) at lab_crash.c:23
# #1  main () at lab_crash.c:37
# (gdb) print curr
# $1 = (node_t *) 0xaaaaaaaaaaaaaaaa
# (gdb) x/16xb second
# Displays exact bytes showing how second-&gt;next was overwritten with 0xAA!</code></pre>

        <h2 id="sec-18-quiz">18.8 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why cannot a standard software breakpoint be set in microcontroller NOR Flash memory without special hardware assistance?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Flash memory cannot execute Thumb-2 instructions.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) Software breakpoints require overwriting the instruction with a BKPT opcode (0xBE), which fails because NOR flash requires sector erases prior to writes. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) The NVIC vector table disables breakpoints in flash memory.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Software breakpoints require dynamically patching the opcode at the target address. NOR flash bits cannot be toggled from 0 to 1 without a block erase operation, requiring dedicated hardware comparators (ARM Flash Patch & Breakpoint unit).
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the purpose of setting <code>ulimit -c unlimited</code> in Linux?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) It removes execution time limits for CPU-intensive algorithms.</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) It allows the kernel to generate core dump files of unrestricted size upon abnormal program termination. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) It allocates unlimited virtual heap memory for malloc.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> By default on many Linux distributions, the maximum core dump size is set to 0, which silently prevents core file creation when crashes occur.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What GDB command sequence inspects 8 contiguous 32-bit memory words in hexadecimal starting at pointer <code>ptr</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>dump 8 ptr</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>x/8xw ptr</code> (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>read memory 32 8 ptr</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>x/8xw</code> specifies: examine (<code>x</code>), 8 elements (<code>8</code>), in hexadecimal format (<code>x</code>), of word unit size (<code>w</code> = 4 bytes).
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What silicon block provides hardware data watchpoints on an ARM Cortex-M4 microcontroller?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) SysTick timer</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) DWT (Data Watchpoint and Trace) unit (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) RCC (Reset and Clock Control)</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The Cortex-M DWT contains address comparator registers that continuously monitor data address bus transactions and halt the core when a match occurs.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: When analyzing a multithreaded crash dump in GDB, how do you switch the execution context to thread 3?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>thread 3</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>switch 3</code></label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>pid 3</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>info threads</code> lists all active threads, and <code>thread &lt;id&gt;</code> switches GDB's inspection context to examine that specific thread's registers and stack trace.
            </div>
          </div>
        </div>'''

if __name__ == "__main__":
    replace_article_in_file("mod17-build-systems-make-and-cmake.html", MOD17_ARTICLE)
    replace_article_in_file("mod18-gdb-debugging-and-core-dumps.html", MOD18_ARTICLE)
