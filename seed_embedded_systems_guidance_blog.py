"""
Seed the "Guidance on Embedded Systems Software Development" blog post.

This script is idempotent and can be run safely more than once.

Usage:
    python seed_embedded_systems_guidance_blog.py
"""
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from django.utils import timezone

from blogs.models import Post

SLUG = "guidance-on-embedded-systems-software-development"
TITLE = "Guidance on Embedded Systems Software Development"
AUTHOR = "Apt Computing Labs"
EXCERPT = (
    "A practical guide to designing dependable embedded software, from requirements "
    "and hardware boundaries through testing, security, and field updates."
)

LEGACY_HTML_CONTENT = """
<p class="lead" style="font-size: 1.25rem; font-weight: 500; color: #1E3557;">Embedded systems software sits between strict hardware limits and software that must remain dependable for years. Good engineering therefore starts with disciplined requirements, clear interfaces, and a verification strategy that reflects the real device.</p>

<div class="toc-box mt-8 mb-10 p-6 bg-brand-light border border-gray-100 rounded-2xl">
    <p class="font-bold text-brand-navy mb-3">In this article</p>
    <ol style="margin:0; padding-left: 1.25rem; line-height: 1.9;">
        <li><a href="#start-with-requirements">Start with Requirements and Constraints</a></li>
        <li><a href="#design-boundaries">Design Clear Hardware and Software Boundaries</a></li>
        <li><a href="#choose-architecture">Choose an Appropriate Architecture</a></li>
        <li><a href="#make-time-visible">Make Time, Memory, and Failure Behaviour Visible</a></li>
        <li><a href="#test-in-layers">Test in Layers</a></li>
        <li><a href="#build-security-in">Build Security into the Lifecycle</a></li>
        <li><a href="#release-maintain">Plan for Release and Maintenance</a></li>
        <li><a href="#conclusion">Conclusion</a></li>
    </ol>
</div>

<h2 id="start-with-requirements">1. Start with Requirements and Constraints</h2>
<p>Before selecting a microcontroller, RTOS, or programming language, describe what the system must do and the conditions in which it must do it. Requirements should cover functional behaviour, timing, power consumption, memory limits, startup time, environmental conditions, safety goals, and communication interfaces.</p>
<p>Turn vague expectations into measurable acceptance criteria. For example, replace “the controller responds quickly” with a maximum response time and define what happens when the deadline cannot be met. Trace each requirement to design decisions and tests so that missing coverage is visible early.</p>

<h2 id="design-boundaries">2. Design Clear Hardware and Software Boundaries</h2>
<p>Keep hardware access behind small, well-defined interfaces. A hardware abstraction layer should expose the behaviour the application needs without spreading register operations, pin configuration, or vendor-specific details throughout the codebase.</p>
<ul>
    <li>Keep drivers responsible for hardware access and device protocols.</li>
    <li>Keep application logic independent from a particular board where practical.</li>
    <li>Document ownership, units, valid ranges, and concurrency for every interface.</li>
    <li>Make initialization and shutdown order explicit.</li>
</ul>
<p>This separation makes unit testing easier and reduces the cost of moving to a new processor, board revision, or communication peripheral.</p>

<h2 id="choose-architecture">3. Choose an Appropriate Architecture</h2>
<p>A small cooperative loop can be the clearest choice for a simple device. As timing, communication, and fault-isolation needs grow, an RTOS or a more structured component model may be justified. The architecture should follow the system's timing and reliability needs rather than fashion.</p>
<p>Define tasks, interrupt responsibilities, queues, and shared state before implementation. Interrupt handlers should do the minimum work required to capture an event and defer longer processing to scheduled code. Keep blocking operations out of time-critical paths and make priority choices explainable.</p>

<h2 id="make-time-visible">4. Make Time, Memory, and Failure Behaviour Visible</h2>
<p>Embedded systems have finite resources, so resource use is part of the design contract. Measure stack high-water marks, heap usage, CPU load, interrupt latency, communication throughput, and worst-case execution time on representative hardware.</p>
<p>Prefer bounded operations and deliberate allocation strategies. Decide how the system behaves when a buffer fills, a sensor stops responding, a message is malformed, or a watchdog fires. Fault handling should move the device to a known state, preserve useful diagnostics, and avoid repeatedly restarting without recording the cause.</p>

<h2 id="test-in-layers">5. Test in Layers</h2>
<p>Testing should begin before the full device is assembled. Use several complementary layers:</p>
<ol>
    <li><strong>Static analysis:</strong> enforce language rules, style, complexity limits, and suspicious-pattern checks.</li>
    <li><strong>Unit tests:</strong> verify protocol parsing, state machines, calculations, and error paths on the host.</li>
    <li><strong>Integration tests:</strong> exercise drivers, buses, storage, and RTOS interactions on target hardware.</li>
    <li><strong>System tests:</strong> validate complete workflows, timing, power states, recovery, and upgrade behaviour.</li>
    <li><strong>Stress and fault tests:</strong> inject invalid input, communication loss, resets, brownouts, and resource pressure.</li>
</ol>
<p>Automate repeatable tests in continuous integration and keep hardware-in-the-loop tests reproducible. A test that cannot be diagnosed or repeated will gradually lose its value.</p>

<h2 id="build-security-in">6. Build Security into the Lifecycle</h2>
<p>Security is not a final scan. Establish a threat model early, identify trust boundaries, and protect interfaces that accept external data. Validate lengths and ranges before parsing, use authenticated updates, protect signing keys, and disable or restrict unused debug interfaces in production.</p>
<p>Record the software bill of materials and track third-party dependencies. Apply least privilege where the platform supports it, protect sensitive data at rest and in transit, and ensure that security events can be investigated without exposing secrets.</p>

<h2 id="release-maintain">7. Plan for Release and Maintenance</h2>
<p>A dependable product needs more than a successful firmware build. Make builds reproducible, review changes through version control, and record the compiler, SDK, configuration, and hardware revision used for every release.</p>
<p>Design field updates around interrupted power, rollback, compatibility, and authenticity. Include meaningful version information and diagnostic commands in the device. Finally, monitor field failures and feed the lessons back into requirements, tests, and architecture decisions.</p>

<h2 id="conclusion">8. Conclusion</h2>
<p>Strong embedded software development is a systems discipline. Begin with measurable requirements, isolate hardware dependencies, make resource and timing behaviour explicit, verify in layers, and treat security and maintenance as design responsibilities.</p>
<div class="mt-8 p-6 bg-blue-50 border border-blue-100 rounded-2xl">
    <p class="font-bold text-brand-navy mb-0">Key takeaway:</p>
    <p class="mb-0 mt-2">The most dependable embedded products are built by making constraints visible early and turning every important assumption into an interface, measurement, or test.</p>
</div>
""".strip()

HTML_CONTENT = """
<div class="blog-slide blog-slide-dark">
    <span class="slide-number">01</span>
    <p class="slide-kicker">Apt Computing Labs | 65-minute session</p>
    <h2>Guidance on Embedded Systems Software Development</h2>
    <p>“This session isn’t about another tutorial. It is about seeing what it really takes to ship a constrained, reliable product, from a smartwatch to an automotive ECU, and leaving with a clear skills roadmap, portfolio plan, and job map.”</p>
    <p><strong>Positioning:</strong> what to learn, why it matters, and how to become an engineer who can build real products.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">02</span>
    <p class="slide-kicker">Opening</p>
    <h2>Start with the engineer behind the product</h2>
    <p>The session opens with the speaker’s background: Harman, automotive cybersecurity, STM32, Rust, and Intrusion Detection and Prevention Systems (IDPS). The point is not a résumé tour. It is context for the promise: the skills below are the skills used when software must work inside a real product.</p>
    <div class="slide-callout"><strong>The narrative arc:</strong> Arduino trap &rarr; smartwatch reality &rarr; automotive parallels &rarr; job market &rarr; standards &rarr; next steps.</div>
</div>

<div class="blog-slide">
    <span class="slide-number">03</span>
    <p class="slide-kicker">02–07 min | The Arduino trap</p>
    <h2>Boards and tutorials are a beginning, not product engineering</h2>
    <p>Arduino projects are valuable because they make hardware approachable. But a board lighting an LED does not prove that a product can meet deadlines, survive brownouts, recover from a failed sensor, protect an update, or run for years on a battery.</p>
    <div class="slide-grid">
        <div class="slide-card"><strong>Tutorial success</strong><p>“It works once” on a development board.</p></div>
        <div class="slide-card"><strong>Product success</strong><p>It is measurable, testable, recoverable, secure, and maintainable.</p></div>
        <div class="slide-card"><strong>The bridge</strong><p>C, datasheets, drivers, RTOS concepts, protocols, toolchains, and debugging.</p></div>
    </div>
    <p><strong>Challenge:</strong> keep the board, but remove the tutorial. Write the requirements, measure the timing, inject failures, and explain every resource trade-off.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">04</span>
    <p class="slide-kicker">07–22 min | Smartwatch case study</p>
    <h2>A smartwatch is a small product with a large engineering surface</h2>
    <p>A smartwatch makes the constraints visceral. A product engineer is not only drawing pixels. They are making a complete system behave reliably inside a tiny enclosure with a tiny battery.</p>
    <div class="slide-grid">
        <div class="slide-card"><strong>Compute</strong><p>CPU budget, memory, flash, stack sizes, scheduling, and startup time.</p></div>
        <div class="slide-card"><strong>Sensing</strong><p>Accelerometer, heart-rate and other sensors, calibration, filtering, and sensor fusion.</p></div>
        <div class="slide-card"><strong>Power</strong><p>Battery limits, sleep and wake states, charging, duty cycles, and power regressions.</p></div>
        <div class="slide-card"><strong>Communication</strong><p>Bluetooth/BLE, Wi-Fi, pairing, packet loss, reconnection, and coexistence.</p></div>
        <div class="slide-card"><strong>Product features</strong><p>Display, haptics, buttons, notifications, timekeeping, and responsive user interaction.</p></div>
        <div class="slide-card"><strong>Operations</strong><p>OTA updates, rollback, reliability, manufacturing tests, telemetry, and privacy.</p></div>
    </div>
    <p>Each feature crosses hardware and software boundaries. A sensor driver can affect battery life; a communication retry can affect responsiveness; an update can fail if power disappears at the wrong moment. That is why embedded engineering is systems engineering.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">05</span>
    <p class="slide-kicker">Smartwatch debugging story</p>
    <h2>Random reset is not a diagnosis</h2>
    <p>Imagine a watch that randomly resets when the user starts a workout. The first report is “the app crashed.” Instrumentation tells a sharper story: a high-rate sensor path increases stack usage, a task misses its deadline, and the watchdog resets the device. A second device may show a similar symptom because a power dip occurs when the radio and display wake together.</p>
    <div class="slide-grid">
        <div class="slide-card"><strong>Observe</strong><p>Reset reason, watchdog status, stack watermark, voltage trace, and timestamped logs.</p></div>
        <div class="slide-card"><strong>Hypothesize</strong><p>Separate stack overflow, watchdog starvation, and power dip instead of guessing.</p></div>
        <div class="slide-card"><strong>Prove</strong><p>Reproduce with stress data, fault injection, and an instrumented build.</p></div>
        <div class="slide-card"><strong>Prevent</strong><p>Bound stack use, schedule work correctly, monitor voltage, and add a regression test.</p></div>
    </div>
    <p><strong>Debugging expectation:</strong> an engineer must turn an intermittent field symptom into evidence, a root cause, and a durable fix.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">06</span>
    <p class="slide-kicker">22–27 min | Automotive parallels</p>
    <h2>The same constraints become safety and security concerns</h2>
    <p>An automotive ECU has the same fundamental tension as a smartwatch, with higher consequences and more coordination between systems.</p>
    <div class="slide-grid">
        <div class="slide-card"><strong>ECU constraints</strong><p>Deterministic timing, limited compute, memory budgets, thermal limits, and long support lifecycles.</p></div>
        <div class="slide-card"><strong>Vehicle networks</strong><p>CAN and CAN-FD, gateways, diagnostics, message ownership, and failure handling.</p></div>
        <div class="slide-card"><strong>Security</strong><p>Threat modeling, secure boot, authenticated updates, and IDPS telemetry.</p></div>
        <div class="slide-card"><strong>Safety and reliability</strong><p>Predictable behaviour, defensive coding, diagnostics, degradation, and recovery.</p></div>
    </div>
    <p>The lesson transfers directly: a beginner who learns to reason about constraints on a wearable is already learning how to reason about a vehicle system.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">07</span>
    <p class="slide-kicker">27–32 min | What to self-learn</p>
    <h2>The checklist beyond academics</h2>
    <div class="slide-grid">
        <div class="slide-card"><strong>Embedded C</strong><p>Pointers, memory, bit operations, volatile, undefined behaviour, and data structures.</p></div>
        <div class="slide-card"><strong>Hardware literacy</strong><p>Datasheets, reference manuals, schematics, clocks, interrupts, peripherals, and drivers.</p></div>
        <div class="slide-card"><strong>RTOS and concurrency</strong><p>Tasks, queues, mutexes, priorities, timing, race conditions, and watchdogs.</p></div>
        <div class="slide-card"><strong>Protocols</strong><p>UART, SPI, I2C, CAN/CAN-FD, MQTT, BLE, Wi-Fi, and diagnostics.</p></div>
        <div class="slide-card"><strong>Toolchain</strong><p>Compiler and linker, build systems, Git, CI, flashing, tracing, and static analysis.</p></div>
        <div class="slide-card"><strong>Debugging</strong><p>GDB, JTAG/SWD, logic analyzers, oscilloscopes, logs, assertions, and fault injection.</p></div>
    </div>
</div>

<div class="blog-slide">
    <span class="slide-number">08</span>
    <p class="slide-kicker">32–47 min | Jobs, companies, expectations</p>
    <h2>What the industry expects from freshers</h2>
    <p>Entry-level hiring is not only about knowing a programming language. Teams look for evidence that a candidate can read unfamiliar hardware documentation, reason about failure, collaborate through Git, and explain a technical decision.</p>
    <div class="slide-grid">
        <div class="slide-card"><strong>Roles</strong><p>Firmware engineer, embedded software engineer, device driver engineer, BSP engineer, validation engineer, automotive software engineer, and embedded security engineer.</p></div>
        <div class="slide-card"><strong>Company types</strong><p>Product companies, semiconductor vendors, automotive OEMs, Tier-1 suppliers, robotics and IoT companies, consumer devices, and engineering services.</p></div>
        <div class="slide-card"><strong>Interview focus</strong><p>C fundamentals, debugging, operating-system concepts, protocols, data structures, electronics basics, and project trade-offs.</p></div>
        <div class="slide-card"><strong>Proof of readiness</strong><p>A working project, readable code, tests, a clear README, and the ability to explain what failed and how it was fixed.</p></div>
    </div>
</div>

<div class="blog-slide">
    <span class="slide-number">09</span>
    <p class="slide-kicker">Portfolio beat | 2–3 minutes</p>
    <h2>Build a portfolio that looks like engineering</h2>
    <p>Plan 3–5 GitHub-ready projects instead of collecting disconnected tutorials. Each project should demonstrate a different layer of the stack and include a README, architecture sketch, build instructions, test evidence, and a short demo video.</p>
    <ul>
        <li><strong>STM32:</strong> a driver and sensor project with interrupts, timing measurements, and a hardware test plan.</li>
        <li><strong>ESP32:</strong> a connected device using BLE or Wi-Fi, with reconnection and failure handling.</li>
        <li><strong>FreeRTOS:</strong> a multi-task application showing queues, priorities, synchronization, and watchdog recovery.</li>
        <li><strong>CAN:</strong> a small vehicle-network simulator with message validation, diagnostics, and fault injection.</li>
        <li><strong>MQTT:</strong> an edge-to-cloud telemetry path with offline buffering and secure reconnect.</li>
    </ul>
    <p>Show debugging artifacts too: logic-analyzer captures, GDB screenshots, reset-reason logs, test reports, power measurements, and a short “what I would improve” section.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">10</span>
    <p class="slide-kicker">47–57 min | India market snapshot</p>
    <h2>Map the market without treating salary as the whole map</h2>
    <p>India has opportunities across automotive, semiconductors, consumer electronics, industrial automation, robotics, IoT, and engineering services. Hiring titles vary, so search by the work as well as the title: firmware, BSP, device drivers, validation, AUTOSAR, embedded Linux, functional safety, and automotive cybersecurity.</p>
    <div class="slide-grid">
        <div class="slide-card"><strong>Illustrative salary bands</strong><p>Entry: ₹4–8 LPA. Early professional: ₹8–18 LPA. Experienced specialists: ₹18–35+ LPA. These are indicative ranges, not guarantees; location, domain, company, and demonstrable skill change the result.</p></div>
        <div class="slide-card"><strong>Hiring signals</strong><p>STM32/ARM, Embedded C/C++, Rust, FreeRTOS, embedded Linux, CAN/CAN-FD, AUTOSAR, BLE, MQTT, Git, CI, and debugging instruments.</p></div>
        <div class="slide-card"><strong>Company map</strong><p>Automotive OEMs and Tier-1s, chip and tool vendors, product startups, industrial and medical-device teams, consumer-device companies, and services firms.</p></div>
    </div>
    <p>Use public job descriptions to update this map. The goal is not to memorize a statistic; it is to identify a target role and build the evidence that role asks for.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">11</span>
    <p class="slide-kicker">57–67 min | Standards and quality</p>
    <h2>Professional embedded software has a quality system</h2>
    <div class="slide-grid">
        <div class="slide-card"><strong>MISRA C</strong><p>Use a documented subset of C to reduce dangerous language patterns and make reviews more consistent.</p></div>
        <div class="slide-card"><strong>CERT C</strong><p>Use secure-coding guidance for integer handling, memory, strings, input validation, and undefined behaviour.</p></div>
        <div class="slide-card"><strong>Secure embedded practice</strong><p>Threat model interfaces, protect keys, authenticate firmware, lock debug paths, and plan vulnerability response.</p></div>
        <div class="slide-card"><strong>Static analysis</strong><p>Run compiler warnings, linters, MISRA/CERT checks, complexity checks, and dependency scans in CI.</p></div>
    </div>
    <p>Standards are not paperwork added after the code is finished. They are a shared language for making code reviewable, analyzable, testable, and safer to maintain.</p>
</div>

<div class="blog-slide">
    <span class="slide-number">12</span>
    <p class="slide-kicker">90–150 day learning roadmap</p>
    <h2>Leave with a sequence, not only motivation</h2>
    <div class="slide-timeline">
        <div><strong>Days 1–30</strong><p>C fundamentals, Git, build tools, debugging basics, and one small bare-metal driver.</p></div>
        <div><strong>Days 31–60</strong><p>STM32 or ESP32 peripherals, datasheets, interrupts, UART/SPI/I2C, and a tested sensor project.</p></div>
        <div><strong>Days 61–90</strong><p>FreeRTOS, concurrency, watchdog recovery, power states, and a documented connected device.</p></div>
        <div><strong>Days 91–120</strong><p>CAN/CAN-FD or MQTT, security basics, static analysis, CI, and fault-injection tests.</p></div>
        <div><strong>Days 121–150</strong><p>Polish 3–5 portfolio projects, record demos, practice interviews, and apply to targeted roles.</p></div>
    </div>
    <p>Adjust the pace to your schedule, but keep the order: fundamentals, hardware, real-time behaviour, communication, quality, and public proof.</p>
</div>

<div class="blog-slide blog-slide-dark">
    <span class="slide-number">13</span>
    <p class="slide-kicker">67–72 min | Call to action</p>
    <h2>Build one real thing next</h2>
    <p>Choose a constrained product, write its requirements, make one failure visible, and publish the evidence. Then repeat with a new protocol or architecture.</p>
    <ul>
        <li>Pick one STM32/ESP32 project and define its success criteria.</li>
        <li>Use FreeRTOS, CAN, MQTT, or BLE where it adds a real product constraint.</li>
        <li>Capture the debugging trail, tests, measurements, and trade-offs.</li>
        <li>Publish the README and demo video so another engineer can reproduce it.</li>
    </ul>
    <p><strong>Notes are published at Apt Computing Labs.</strong> The next step is not another tutorial tab. It is a small product with a deadline, a failure mode, and a test that proves the fix.</p>
</div>

<div class="blog-slide">
    <p class="slide-kicker">Closing thought</p>
    <h2>From board projects to product engineers</h2>
    <p>Embedded systems reward engineers who can connect details to consequences: a pointer to memory safety, a task priority to timing, a retry loop to battery life, a CAN message to vehicle behaviour, and a reset log to customer trust.</p>
    <div class="slide-callout"><strong>Remember:</strong> the promise is not “learn every tool.” The promise is to learn how to reason under constraints, build evidence, and ship software that can be trusted.</div>
</div>
""".strip()

# Keep the long-form article in an editable HTML document rather than in the
# seed script so the published prose remains easy to review.
HTML_CONTENT = open(
    os.path.join(os.path.dirname(__file__), "embedded_systems_article.html"),
    encoding="utf-8",
).read().strip()


def main():
    post, created = Post.objects.get_or_create(
        slug=SLUG,
        defaults={
            "title": TITLE,
            "author": AUTHOR,
            "excerpt": EXCERPT,
            "content": HTML_CONTENT,
            "published_date": timezone.now(),
        },
    )
    if not created:
        post.title = TITLE
        post.author = AUTHOR
        post.excerpt = EXCERPT
        post.content = HTML_CONTENT
        post.save()
        print(f"Updated existing blog post: {SLUG}")
    else:
        print(f"Created new blog post: {SLUG}")


if __name__ == "__main__":
    main()
