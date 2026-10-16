from django.core.management.base import BaseCommand
from courses.models import Course


class Command(BaseCommand):
    help = "Create or update the Rust-from-C and sensor-electronics courses."

    def handle(self, *args, **options):
        courses = (
            {
                "slug": "realization-of-rust-from-c",
                "title": "Realization of Rust from C",
                "short_description": (
                    "A practical path for C and C++ developers to understand Rust's "
                    "ownership and memory-safety model, work through hands-on labs, "
                    "and explore where Rust is used in embedded, automotive, Linux "
                    "and cloud systems."
                ),
                "description": """
<div class="space-y-8">
  <div class="bg-brand-navy text-white p-6 rounded-xl shadow-lg border-l-4 border-brand-coral">
    <h3 class="text-xl font-bold mb-3">Rust concepts connected to your C/C++ experience</h3>
    <p class="text-gray-200">Start with familiar memory and systems concepts, then learn how Rust makes ownership, borrowing, errors and safe APIs explicit. The supplied course book includes the complete workshop, labs, quiz, standard-library reference, images and presentation PDFs.</p>
    <ul class="space-y-2 text-gray-200 mt-4">
      <li><strong>Session 1:</strong> Memory structure, shared data, C/C++ vulnerability patterns, tools and secure practices.</li>
      <li><strong>Session 2:</strong> Variables, types, control flow, functions, ownership, borrowing, references, mutability, errors and Cargo.</li>
      <li><strong>Session 3:</strong> Guided labs: Rust basics, ownership, buffer safety, input validation, safe files and a thread-safe counter.</li>
      <li><strong>Session 4:</strong> Rust in automotive, embedded/IoT, Linux/cloud and AI infrastructure, plus career and cybersecurity topics.</li>
      <li><strong>Practice and reference:</strong> 50-question skills test, standard-library chapters, pre-workshop setup and source references.</li>
    </ul>
  </div>
  <div>
    <h3 class="text-xl font-bold text-brand-navy mb-3">Learning outcomes</h3>
    <ul class="list-disc pl-6 text-brand-textSecondary space-y-2">
      <li>Explain how ownership, borrowing and lifetimes constrain aliases and resource access.</li>
      <li>Use Rust types, pattern matching, control flow, functions, Cargo, Option and Result.</li>
      <li>Recognise common C/C++ memory-safety risks and express safer alternatives in Rust.</li>
      <li>Build and test small programs for input handling, buffers, file operations and concurrency.</li>
      <li>Identify Rust's current applications and select suitable next steps for systems programming.</li>
    </ul>
  </div>
  <div class="bg-blue-50 p-5 rounded-xl border border-blue-100">
    <h3 class="text-lg font-bold text-brand-navy mb-2">Course materials</h3>
    <p class="text-brand-textSecondary">The protected course reader includes every page and asset supplied in the BookToStartRust docs folder.</p>
    <ul class="list-disc pl-6 text-brand-textSecondary">
      <li><a href="/courses/realization-of-rust-from-c/book/1.Introduction%20to%20Rust.pdf">Introduction to Rust presentation (PDF)</a></li>
      <li><a href="/courses/realization-of-rust-from-c/book/2.OwnershipBorrowReference_v3.pptx%20-%20Google%20Slides.pdf">Ownership, borrowing and references presentation (PDF)</a></li>
      <li>Pre-workshop guides, all session chapters and lab exercises, five-session quiz, standard-library guide, illustrations and references.</li>
    </ul>
  </div>
</div>
""",
                "skill_level": "Beginner to Intermediate",
                "technologies": "Rust, C, C++, Cargo, ownership, systems programming",
                "duration_weeks": 0,
                "duration_hours": 0,
                "prerequisites": "Basic programming experience in C or C++ is recommended; no prior Rust experience is required.",
                "materials_info": (
                    "Complete digital course book with workshop sessions, labs, quiz, "
                    "reference chapters, images and presentation PDFs."
                ),
                "training_mode": "Online",
            },
            {
                "slug": "basic-electronics-sensors-actuators",
                "title": "Basic Electronics with Sensors and Actuators",
                "short_description": (
                    "Learn how common sensor families turn physical quantities into "
                    "electrical signals, then condition those signals to operate "
                    "actuators with practical circuit diagrams—without microcontrollers."
                ),
                "description": """
<div class="space-y-8">
  <div class="bg-brand-navy text-white p-6 rounded-xl shadow-lg border-l-4 border-brand-coral">
    <h3 class="text-xl font-bold mb-3">Circuit-first electronics, no microcontrollers</h3>
    <p class="text-gray-200">Explore sensors and actuators by their electrical behaviour. Build and explain circuits using resistors, capacitors, transistors, op-amps, comparators, 555 timers and dedicated driver ICs. The handbook includes original vector illustrations and labelled circuit schematics.</p>
    <ul class="space-y-2 text-gray-200 mt-4">
      <li><strong>Sensor families:</strong> resistive, capacitive, optical, thermal, magnetic, inductive, piezoelectric, acoustic, pressure, force, flow, chemical, gas, humidity, moisture, inertial and position sensors.</li>
      <li><strong>Signal conditioning:</strong> dividers, Wheatstone bridges, amplification, filtering, threshold comparators, hysteresis and pulse shaping.</li>
      <li><strong>Actuators:</strong> LEDs and lamps, buzzers, relays, solenoids, valves, DC motors, fans, stepper motors, servos, heaters and Peltier devices.</li>
      <li><strong>Hands-on projects:</strong> light switch, thermostat/fan, door alarm, optical counter, vibration indicator and water-level alarm.</li>
    </ul>
  </div>
  <div>
    <h3 class="text-xl font-bold text-brand-navy mb-3">Students will learn to</h3>
    <ul class="list-disc pl-6 text-brand-textSecondary space-y-2">
      <li>Identify what a sensor measures and whether its output is resistance, voltage, current, capacitance, charge, pulse or contact state.</li>
      <li>Read schematics, calculate current paths and choose a suitable first signal-conditioning circuit.</li>
      <li>Measure sensor outputs with a multimeter and explain calibration, noise, hysteresis and limitations.</li>
      <li>Drive loads safely with a transistor, MOSFET, relay or dedicated driver, including inductive-load suppression.</li>
      <li>Build low-voltage demonstrations using analog and discrete circuits without an MCU or firmware.</li>
    </ul>
  </div>
  <div class="bg-blue-50 p-5 rounded-xl border border-blue-100">
    <h3 class="text-lg font-bold text-brand-navy mb-2">Safety and scope</h3>
    <p class="text-brand-textSecondary">Experiments are designed for safe extra-low-voltage supplies. Hobby sensor circuits are learning tools, not certified life-safety, gas-detection or mains-control systems.</p>
  </div>
</div>
""",
                "skill_level": "Beginner",
                "technologies": "Analog electronics, sensors, actuators, op-amps, comparators, 555 timers, transistors, MOSFETs",
                "duration_weeks": 0,
                "duration_hours": 0,
                "prerequisites": "No microcontroller or programming experience required; basic algebra is helpful.",
                "materials_info": (
                    "Illustrated digital handbook with sensor-family explanations, "
                    "labelled circuit diagrams, measurement guidance and guided projects."
                ),
                "training_mode": "Online",
            },
        )

        for defaults in courses:
            course, created = Course.objects.update_or_create(
                slug=defaults["slug"],
                defaults=defaults,
            )
            action = "Created" if created else "Updated"
            self.stdout.write(f"{action} course: {course.title}")
