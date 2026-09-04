"""
Seed script for the "Why Bluetooth Security Matters in Modern Vehicles" blog post.

Source: articl3 3 pdf.pdf (Roopashree M, AptComputingLabs)
This script is idempotent - it can be re-run safely (get_or_create + update).

Usage:
    python seed_bluetooth_security_blog.py
"""
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from django.utils import timezone
from blogs.models import Post

SLUG = "why-bluetooth-security-matters-in-modern-vehicles"

TITLE = "Why Bluetooth Security Matters in Modern Vehicles"

AUTHOR = "Roopashree M"

EXCERPT = (
    "Bluetooth has become essential to the modern connected vehicle - but every wireless "
    "channel is a potential entry point. Here's why continuous, behaviour-based Bluetooth "
    "monitoring and intrusion detection are now a core part of automotive cybersecurity."
)

IMG_BASE = "/static/blogs/bluetooth-security"
COVER_IMAGE = f"{IMG_BASE}/bluetooth-communication-connected-vehicle.jpg"

HTML_CONTENT = f"""
<p class="lead" style="font-size: 1.25rem; font-weight: 500; color: #1E3557;">Modern vehicles are no longer just mechanical machines. They have evolved into intelligent systems equipped with advanced communication technologies that improve safety, convenience, and the overall driving experience - and Bluetooth sits at the center of that shift.</p>

<div class="toc-box mt-8 mb-10 p-6 bg-brand-light border border-gray-100 rounded-2xl">
    <p class="font-bold text-brand-navy mb-3">In this article</p>
    <ol style="margin:0; padding-left: 1.25rem; line-height: 1.9;">
        <li><a href="#hidden-challenge">The Hidden Cybersecurity Challenge Behind Wireless Connectivity</a></li>
        <li><a href="#bluetooth-in-connected-vehicles">Bluetooth in Connected Vehicles</a></li>
        <li><a href="#continuous-monitoring">The Need for Continuous Monitoring</a></li>
        <li><a href="#intrusion-detection">Intrusion Detection in Automotive Cybersecurity</a></li>
        <li><a href="#bluetooth-security-sensor">Bluetooth as a Security Sensor</a></li>
        <li><a href="#rule-based-ml">Rule-Based Detection and Machine Learning</a></li>
        <li><a href="#detection-to-response">From Detection to Response</a></li>
        <li><a href="#why-this-matters">Why This Matters</a></li>
        <li><a href="#looking-ahead">Looking Ahead</a></li>
        <li><a href="#conclusion">Conclusion</a></li>
    </ol>
</div>

<h2 id="hidden-challenge">1. The Hidden Cybersecurity Challenge Behind Wireless Connectivity</h2>
<p>Modern vehicles are no longer just mechanical machines. They have evolved into intelligent systems equipped with advanced communication technologies that improve safety, convenience, and the overall driving experience. Features such as hands-free calling, wireless music streaming, smartphone integration, keyless entry, and over-the-air software updates have become standard in many vehicles. Among these technologies, Bluetooth plays an essential role by enabling seamless communication between the vehicle and external devices.</p>
<p>While Bluetooth offers convenience, it also introduces new cybersecurity challenges. Every wireless communication channel can become a potential entry point for attackers if it is not properly monitored and secured. As vehicles continue to adopt more connected technologies, protecting Bluetooth communication has become an important aspect of automotive cybersecurity.</p>

<h2 id="bluetooth-in-connected-vehicles">2. Bluetooth in Connected Vehicles</h2>
<p>Bluetooth is one of the most widely used wireless technologies in modern automobiles. Drivers use it daily to connect smartphones, answer calls, play music, synchronize contacts, and interact with infotainment systems. Some vehicles also rely on Bluetooth for communication with wearable devices, diagnostic tools, and other electronic accessories.</p>
<p>Because Bluetooth connections occur wirelessly, they are constantly exposed to nearby devices. Although Bluetooth includes several built-in security mechanisms, attackers may still attempt activities such as unauthorized pairing, repeated connection attempts, device scanning, or connection flooding. These actions may not immediately compromise a vehicle, but they can indicate suspicious behaviour that deserves attention.</p>
<p>This is why monitoring Bluetooth activity is becoming increasingly important.</p>

<figure>
    <img src="{IMG_BASE}/bluetooth-communication-connected-vehicle.jpg" alt="Bluetooth communication in a connected vehicle environment, showing a laptop, phone, and diagnostic tool connecting through Bluetooth with a monitoring workstation observing the traffic">
    <figcaption>Figure 1. Bluetooth Communication in Connected Vehicle Environment</figcaption>
</figure>

<h2 id="continuous-monitoring">3. The Need for Continuous Monitoring</h2>
<p>Traditional security mechanisms often focus on preventing attacks before they occur. However, in today's connected vehicles, prevention alone is not sufficient. Security systems must also be capable of detecting unusual behaviour while the system is operating.</p>
<p>Continuous monitoring allows security components to observe Bluetooth activities in real time and identify behaviours that deviate from normal communication patterns. Instead of relying only on known attack signatures, behavioural monitoring can recognize unexpected activity that may represent a potential threat.</p>
<p>Early detection enables security teams to investigate suspicious events before they develop into larger security incidents.</p>

<h2 id="intrusion-detection">4. Intrusion Detection in Automotive Cybersecurity</h2>
<p>Modern automotive cybersecurity frameworks increasingly rely on Intrusion Detection Systems (IDS) to monitor different parts of the vehicle. Rather than blocking every communication attempt, an IDS continuously observes system behaviour and generates alerts whenever suspicious activity is detected.</p>
<p>In an automotive environment, intrusion detection can monitor various communication channels, including:</p>
<ul>
    <li>Controller Area Network (CAN)</li>
    <li>Ethernet</li>
    <li>Wi-Fi</li>
    <li>Bluetooth</li>
    <li>Vehicle diagnostics</li>
    <li>External interfaces</li>
</ul>

<figure>
    <img src="{IMG_BASE}/automotive-ids-soc-workflow.png" alt="Automotive intrusion detection and Security Operations Center workflow: detect and record qualified onboard security events, report data of qualified security events, analyze qualified security event data for single vehicles and the whole fleet, develop threat responses, and deploy software updates to mitigate threats">
    <figcaption>Figure 2. Automotive Intrusion Detection and Security Operations Center Workflow</figcaption>
</figure>

<p>Each monitoring component acts as a specialized sensor that focuses on a particular communication technology. Together, these sensors provide a comprehensive view of the vehicle's security status.</p>

<h2 id="bluetooth-security-sensor">5. Bluetooth as a Security Sensor</h2>
<p>A dedicated Bluetooth Threat Detection Sensor focuses specifically on monitoring Bluetooth-related activities.</p>
<p>Instead of inspecting every Bluetooth packet in detail, the sensor analyzes behavioural patterns over time. Examples of monitored activities include:</p>
<ul>
    <li>Repeated device discovery</li>
    <li>Frequent connection attempts</li>
    <li>Connection cycling</li>
    <li>Unusual scanning behaviour</li>
    <li>Changes in signal characteristics</li>
    <li>Repeated observations of unknown devices</li>
</ul>
<p>By analysing these patterns, the sensor can distinguish normal Bluetooth usage from potentially suspicious behaviour.</p>

<h2 id="rule-based-ml">6. Rule-Based Detection and Machine Learning</h2>
<p>One effective approach to Bluetooth threat detection is combining rule-based detection with machine learning.</p>
<p>Rule-based detection is useful for identifying known attack patterns. For example, if a device repeatedly attempts to establish connections within a short period, predefined security rules can immediately classify the behaviour as suspicious.</p>

<figure>
    <img src="{IMG_BASE}/rule-based-vs-ml-detection.jpg" alt="Comparison diagram of a rule-based system versus a machine learning-based system: the rule-based system passes input data through a predefined rules engine to a decision output, while the ML-based system passes input data through an ML model trained on historical data to a predicted output">
    <figcaption>Figure 3. Rule-Based Detection vs Machine Learning-Based Detection</figcaption>
</figure>

<p>Machine learning provides an additional layer of intelligence by analysing behavioural features rather than relying solely on predefined rules. It can recognize unusual patterns that may not exactly match known attack signatures, improving the overall detection capability of the system.</p>
<p>The combination of these two techniques creates a more flexible and reliable intrusion detection solution.</p>

<h2 id="detection-to-response">7. From Detection to Response</h2>
<p>Detecting suspicious Bluetooth activity is only the first step. Once a potential threat has been identified, the information must be communicated to the rest of the vehicle's security infrastructure.</p>
<p>A typical workflow includes:</p>
<ol>
    <li>Monitoring Bluetooth activities.</li>
    <li>Collecting behavioural information.</li>
    <li>Detecting suspicious behaviour.</li>
    <li>Generating a security event.</li>
    <li>Sending the event to the Intrusion Detection System Manager (IdsM).</li>
    <li>Qualifying the event based on predefined criteria.</li>
    <li>Forwarding qualified events to the Intrusion Detection System Reporter (IdsR).</li>
    <li>Reporting the event to the Security Operations Center (SOC).</li>
</ol>
<p>This structured process ensures that only relevant security events are escalated while minimizing unnecessary alerts.</p>

<h2 id="why-this-matters">8. Why This Matters</h2>
<p>As vehicles become more connected, the number of wireless interfaces continues to grow. Every additional communication technology increases the importance of cybersecurity monitoring.</p>
<p>Bluetooth may appear to be a simple convenience feature, but it is also part of the vehicle's communication ecosystem. Monitoring Bluetooth behaviour helps improve overall situational awareness and provides another layer of protection against potential wireless threats.</p>
<p>By integrating Bluetooth monitoring with automotive intrusion detection frameworks, manufacturers can enhance vehicle security while maintaining the convenience that drivers expect from modern connected vehicles.</p>

<h2 id="looking-ahead">9. Looking Ahead</h2>
<p>The future of automotive cybersecurity will rely on intelligent, cooperative security components that continuously monitor different communication channels. Bluetooth monitoring is one part of this larger security architecture.</p>
<p>As connected vehicles continue to evolve, future intrusion detection systems are expected to combine behavioural analysis, artificial intelligence, and real-time event correlation to improve threat detection accuracy. Integrating specialized sensors for wireless technologies will play an important role in protecting next-generation vehicles against emerging cyber threats.</p>

<h2 id="conclusion">10. Conclusion</h2>
<p>Bluetooth technology has become an essential feature of modern vehicles, offering convenience and seamless connectivity. However, with increased connectivity comes increased responsibility to monitor and protect wireless communication. Dedicated Bluetooth Threat Detection Sensors provide valuable visibility into Bluetooth activities and complement existing automotive intrusion detection systems.</p>

<div class="mt-8 p-6 bg-blue-50 border border-blue-100 rounded-2xl">
    <p class="font-bold text-brand-navy mb-0">Conclusion:</p>
    <p class="mb-0 mt-2">By combining continuous monitoring, behavioural analysis, and intelligent event reporting, automotive security frameworks can better identify suspicious activities and strengthen the overall cybersecurity posture of connected vehicles.</p>
</div>
""".strip()


def main():
    post, created = Post.objects.get_or_create(
        slug=SLUG,
        defaults={
            "title": TITLE,
            "author": AUTHOR,
            "excerpt": EXCERPT,
            "content": HTML_CONTENT,
            "image_url": COVER_IMAGE,
            "published_date": timezone.now(),
        },
    )
    if not created:
        post.title = TITLE
        post.author = AUTHOR
        post.excerpt = EXCERPT
        post.content = HTML_CONTENT
        post.image_url = COVER_IMAGE
        post.save()
        print(f"Updated existing blog post: {SLUG}")
    else:
        print(f"Created new blog post: {SLUG}")


if __name__ == "__main__":
    main()
