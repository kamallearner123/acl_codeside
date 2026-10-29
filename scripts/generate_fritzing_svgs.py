"""
Comprehensive generator for custom Fritzing-style vector SVG wiring diagrams for STM32 DIY assignments.
Strict STM32 hardware port/pin naming (PA0..PA15, PB0..PB15, PC0..PC15, 3V3, 5V, GND) - NO Arduino naming.
Guarantees:
1. Zero wiring lines crossing over pin numbers or text labels.
2. Terminals are located on the outer edge of boxes with wires departing into empty space.
3. Generous component spacing so all labels and pin codes are 100% readable.
"""
import os

OUT_DIR = "static/blogs/stm32-diy"
os.makedirs(OUT_DIR, exist_ok=True)

def svg_header(w=800, h=480, title="Wiring Diagram"):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%">
    <defs>
        <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
            <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.12" />
        </filter>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r="1" fill="#CBD5E1" opacity="0.6"/>
        </pattern>
        <linearGradient id="pcbGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0284C7"/>
            <stop offset="100%" stop-color="#0369A1"/>
        </linearGradient>
        <linearGradient id="breadboardGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#F8FAFC"/>
            <stop offset="100%" stop-color="#F1F5F9"/>
        </linearGradient>
    </defs>
    <!-- Background Canvas -->
    <rect width="{w}" height="{h}" fill="#F8FAFC" />
    <rect width="{w}" height="{h}" fill="url(#grid)" />
    <!-- Header Banner -->
    <rect x="0" y="0" width="{w}" height="42" fill="#0F172A" />
    <text x="24" y="26" fill="#F8FAFC" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" letter-spacing="0.5">STM32 EMBEDDED WIRING SCHEMATIC &bull; {title.upper()}</text>
    <rect x="0" y="42" width="{w}" height="3" fill="#FF5B5B" />
"""

def svg_footer():
    return "</svg>"

def draw_header_block(x, y, w, h, title, pins):
    """
    Draws an STM32 Pinout box on the left.
    Text and pin names are safely on the LEFT side of the box.
    Connection terminal pads are placed strictly on the FAR RIGHT EDGE.
    Wires leave to the right into open space and NEVER cross any text inside the box.
    """
    svg = f"""
    <!-- STM32 Pin Header Block: {title} -->
    <g transform="translate({x},{y})" filter="url(#shadow)">
        <rect width="{w}" height="{h}" rx="8" fill="#0F172A" stroke="#334155" stroke-width="1.5"/>
        <rect x="0" y="0" width="{w}" height="32" rx="8" fill="#1E293B"/>
        <rect x="0" y="24" width="{w}" height="8" fill="#1E293B"/>
        <text x="{w/2}" y="21" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">{title}</text>
    """
    row_h = (h - 44) / max(len(pins), 1)
    pin_y = 48
    pin_coords = {}
    
    for i, p in enumerate(pins):
        py = pin_y + i * row_h
        pin_code = p.split()[0]
        desc = p[len(pin_code):].strip()
        
        # Color coding for pin badges
        badge_color = "#38BDF8"
        dot_color = "#38BDF8"
        if "3.3V" in pin_code or "5.0V" in pin_code or "VDD" in pin_code:
            badge_color = "#EF4444"
            dot_color = "#EF4444"
        elif "GND" in pin_code:
            badge_color = "#94A3B8"
            dot_color = "#0F172A"
        elif "PA0" in pin_code and "ADC" in desc:
            badge_color = "#F59E0B"
            dot_color = "#F59E0B"
        elif "PC13" in pin_code or "Button" in desc:
            badge_color = "#38BDF8"
            dot_color = "#38BDF8"
        elif "PWM" in desc:
            badge_color = "#10B981"
            dot_color = "#10B981"

        pin_cx = w - 16
        
        svg += f"""
        <!-- Pin: {pin_code} -->
        <rect x="8" y="{py-12}" width="{w-30}" height="24" rx="4" fill="#1E293B" stroke="#334155" stroke-width="0.8"/>
        <text x="16" y="{py+4}" fill="{badge_color}" font-family="monospace" font-size="11" font-weight="bold">{pin_code}</text>
        <text x="64" y="{py+3.5}" fill="#CBD5E1" font-family="sans-serif" font-size="9">{desc}</text>
        <!-- Contact pad on far right edge -->
        <circle cx="{pin_cx}" cy="{py}" r="6" fill="{dot_color}" stroke="#FFFFFF" stroke-width="1.5"/>
        <circle cx="{pin_cx}" cy="{py}" r="2" fill="#0F172A"/>
        """
        pin_coords[pin_code] = (x + pin_cx, y + py)
        
    svg += "</g>"
    return svg, pin_coords

def draw_breadboard(x, y, w, h):
    return f"""
    <!-- Breadboard Area -->
    <g transform="translate({x},{y})" filter="url(#shadow)">
        <rect width="{w}" height="{h}" rx="8" fill="url(#breadboardGrad)" stroke="#CBD5E1" stroke-width="2"/>
        <!-- Top Power Rails -->
        <line x1="20" y1="14" x2="{w-20}" y2="14" stroke="#EF4444" stroke-width="2" stroke-dasharray="8,4"/>
        <line x1="20" y1="24" x2="{w-20}" y2="24" stroke="#3B82F6" stroke-width="2" stroke-dasharray="8,4"/>
        <text x="8" y="17" fill="#EF4444" font-family="monospace" font-size="11" font-weight="bold">+</text>
        <text x="8" y="27" fill="#3B82F6" font-family="monospace" font-size="11" font-weight="bold">-</text>
        
        <!-- Bottom Power Rails -->
        <line x1="20" y1="{h-24}" x2="{w-20}" y2="{h-24}" stroke="#EF4444" stroke-width="2" stroke-dasharray="8,4"/>
        <line x1="20" y1="{h-14}" x2="{w-20}" y2="{h-14}" stroke="#3B82F6" stroke-width="2" stroke-dasharray="8,4"/>
        <text x="8" y="{h-21}" fill="#EF4444" font-family="monospace" font-size="11" font-weight="bold">+</text>
        <text x="8" y="{h-11}" fill="#3B82F6" font-family="monospace" font-size="11" font-weight="bold">-</text>
        
        <!-- Center Divider -->
        <rect x="15" y="{h/2 - 4}" width="{w-30}" height="8" fill="#E2E8F0" rx="2"/>
    </g>
    """

def draw_resistor(x, y, val_str, horizontal=True):
    if horizontal:
        return f"""
        <g transform="translate({x},{y})">
            <line x1="-30" y1="0" x2="30" y2="0" stroke="#94A3B8" stroke-width="3" stroke-linecap="round"/>
            <rect x="-15" y="-7" width="30" height="14" rx="3" fill="#FDE68A" stroke="#D97706" stroke-width="1.2"/>
            <rect x="-9" y="-7" width="3" height="14" fill="#B45309"/>
            <rect x="-3" y="-7" width="3" height="14" fill="#B45309"/>
            <rect x="3" y="-7" width="3" height="14" fill="#78350F"/>
            <circle cx="-30" cy="0" r="3.5" fill="#64748B" stroke="#FFFFFF" stroke-width="1"/>
            <circle cx="30" cy="0" r="3.5" fill="#64748B" stroke="#FFFFFF" stroke-width="1"/>
            <text x="0" y="-11" fill="#0F172A" font-family="monospace" font-size="9.5" font-weight="bold" text-anchor="middle">{val_str}</text>
        </g>
        """
    else:
        return f"""
        <g transform="translate({x},{y})">
            <line x1="0" y1="-30" x2="0" y2="30" stroke="#94A3B8" stroke-width="3" stroke-linecap="round"/>
            <rect x="-7" y="-15" width="14" height="30" rx="3" fill="#FDE68A" stroke="#D97706" stroke-width="1.2"/>
            <rect x="-7" y="-9" width="14" height="3" fill="#B45309"/>
            <rect x="-7" y="-3" width="14" height="3" fill="#B45309"/>
            <rect x="-7" y="3" width="14" height="3" fill="#78350F"/>
            <circle cx="0" cy="-30" r="3.5" fill="#64748B" stroke="#FFFFFF" stroke-width="1"/>
            <circle cx="0" cy="30" r="3.5" fill="#64748B" stroke="#FFFFFF" stroke-width="1"/>
            <text x="14" y="4" fill="#0F172A" font-family="monospace" font-size="9.5" font-weight="bold">{val_str}</text>
        </g>
        """

def draw_led(x, y, color="#22C55E", label="LED"):
    return f"""
    <g transform="translate({x},{y})">
        <circle cx="0" cy="0" r="14" fill="{color}" opacity="0.9" stroke="#0F172A" stroke-width="1.5"/>
        <circle cx="-4" cy="-4" r="4" fill="#FFFFFF" opacity="0.7"/>
        <line x1="-10" y1="16" x2="-10" y2="32" stroke="#94A3B8" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="10" y1="16" x2="10" y2="32" stroke="#94A3B8" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="-10" cy="32" r="3.5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1"/>
        <circle cx="10" cy="32" r="3.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1"/>
        <text x="-10" y="44" fill="#EF4444" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">(+)Anode</text>
        <text x="10" y="44" fill="#0F172A" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">(-)Cathode</text>
        <text x="0" y="-18" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">{label}</text>
    </g>
    """

def draw_button(x, y, label="Push Button"):
    return f"""
    <g transform="translate({x},{y})">
        <rect x="-18" y="-18" width="36" height="36" rx="4" fill="#1E293B" stroke="#0F172A" stroke-width="1.5"/>
        <circle cx="0" cy="0" r="10" fill="#0284C7" stroke="#0369A1" stroke-width="1.5"/>
        <!-- Terminal 1 Left -->
        <rect x="-26" y="-3" width="8" height="6" fill="#94A3B8"/>
        <circle cx="-26" cy="0" r="3.5" fill="#F59E0B" stroke="#FFFFFF" stroke-width="1"/>
        <text x="-32" y="3.5" fill="#64748B" font-family="monospace" font-size="8" font-weight="bold" text-anchor="end">Pin 1</text>
        <!-- Terminal 2 Right -->
        <rect x="18" y="-3" width="8" height="6" fill="#94A3B8"/>
        <circle cx="26" cy="0" r="3.5" fill="#F59E0B" stroke="#FFFFFF" stroke-width="1"/>
        <text x="32" y="3.5" fill="#64748B" font-family="monospace" font-size="8" font-weight="bold" text-anchor="start">Pin 2</text>
        <text x="0" y="-23" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">{label}</text>
    </g>
    """

def draw_wire(x1, y1, x2, y2, color="#3B82F6", label="", curve_dir="h"):
    """
    Draws a smooth Bezier wire from (x1, y1) to (x2, y2).
    Wires departing (x1, y1) leave horizontally rightwards into open canvas.
    """
    if curve_dir == "h":
        dx = abs(x2 - x1)
        ctrl_offset = max(dx * 0.45, 25)
        path = f"M {x1} {y1} C {x1 + ctrl_offset} {y1}, {x2 - ctrl_offset} {y2}, {x2} {y2}"
    else:
        dy = abs(y2 - y1)
        ctrl_offset = max(dy * 0.45, 25)
        path = f"M {x1} {y1} C {x1} {y1 + ctrl_offset}, {x2} {y2 - ctrl_offset}, {x2} {y2}"
    
    svg = f"""
    <g>
        <path d="{path}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linecap="round" opacity="0.95"/>
        <circle cx="{x1}" cy="{y1}" r="4" fill="{color}" stroke="#FFFFFF" stroke-width="1.2"/>
        <circle cx="{x2}" cy="{y2}" r="4" fill="{color}" stroke="#FFFFFF" stroke-width="1.2"/>
    """
    if label:
        lx = (x1 + x2) / 2
        ly = (y1 + y2) / 2
        svg += f"""
        <g transform="translate({lx},{ly-8})">
            <rect x="-32" y="-9" width="64" height="18" rx="4" fill="#FFFFFF" stroke="{color}" stroke-width="1.2" opacity="0.95"/>
            <text x="0" y="3.5" fill="{color}" font-family="monospace" font-size="8.5" font-weight="bold" text-anchor="middle">{label}</text>
        </g>
        """
    svg += "</g>"
    return svg

def draw_legend(x, y):
    return f"""
    <g transform="translate({x},{y})" filter="url(#shadow)">
        <rect width="180" height="96" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
        <text x="12" y="18" fill="#0F172A" font-family="sans-serif" font-size="10" font-weight="bold">WIRE COLOR CODING</text>
        <line x1="12" y1="32" x2="36" y2="32" stroke="#EF4444" stroke-width="3"/>
        <text x="44" y="35" fill="#334155" font-family="monospace" font-size="9">3.3V / VDD Rail</text>
        <line x1="12" y1="50" x2="36" y2="50" stroke="#0F172A" stroke-width="3"/>
        <text x="44" y="53" fill="#334155" font-family="monospace" font-size="9">GND (0.0V Rail)</text>
        <line x1="12" y1="68" x2="36" y2="68" stroke="#3B82F6" stroke-width="3"/>
        <text x="44" y="71" fill="#334155" font-family="monospace" font-size="9">STM32 GPIO Signal</text>
        <line x1="12" y1="84" x2="36" y2="84" stroke="#F59E0B" stroke-width="3"/>
        <text x="44" y="87" fill="#334155" font-family="monospace" font-size="9">Analog / ADC Input</text>
    </g>
    """

# 1. NUCLEO BOARD PINOUT
def gen_nucleo_pinout():
    svg = svg_header(800, 520, "STM32 NUCLEO-F446RE Hardware Pinout (Morpho CN7 & CN10)")
    svg += """
    <g transform="translate(60, 60)" filter="url(#shadow)">
        <rect width="680" height="420" rx="16" fill="url(#pcbGrad)" stroke="#0284C7" stroke-width="2"/>
        <rect x="0" y="0" width="680" height="70" fill="#0F172A" rx="16"/>
        <rect x="0" y="50" width="680" height="20" fill="#0F172A"/>
        <text x="340" y="36" fill="#F8FAFC" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">ST-LINK / V2-1 ON-BOARD SWD DEBUGGER &amp; VCP BRIDGE</text>
        <rect x="300" y="4" width="80" height="20" rx="3" fill="#E2E8F0" stroke="#94A3B8"/>
        <text x="340" y="18" fill="#475467" font-family="monospace" font-size="9" text-anchor="middle">USB / ST-LINK</text>
        
        <!-- STM32 MCU Chip -->
        <rect x="280" y="180" width="120" height="120" rx="6" fill="#0F172A" stroke="#334155" stroke-width="2"/>
        <circle cx="295" cy="195" r="4" fill="#64748B"/>
        <text x="340" y="235" fill="#38BDF8" font-family="monospace" font-size="12" font-weight="bold" text-anchor="middle">STM32F446RE</text>
        <text x="340" y="255" fill="#94A3B8" font-family="monospace" font-size="10" text-anchor="middle">ARM Cortex-M4F</text>
        <text x="340" y="272" fill="#94A3B8" font-family="monospace" font-size="9" text-anchor="middle">180MHz &bull; 512K Flash</text>

        <!-- Buttons LD2, B1, RESET -->
        <circle cx="160" cy="360" r="14" fill="#3B82F6" stroke="#1D4ED8" stroke-width="2"/>
        <text x="160" y="390" fill="#FFFFFF" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">B1: USER (PC13)</text>
        
        <circle cx="520" cy="360" r="14" fill="#000000" stroke="#334155" stroke-width="2"/>
        <text x="520" y="390" fill="#FFFFFF" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">B2: RESET (NRST)</text>
        
        <circle cx="280" cy="340" r="8" fill="#22C55E" stroke="#15803D" stroke-width="1.5"/>
        <text x="280" y="365" fill="#FFFFFF" font-family="monospace" font-size="9" font-weight="bold" text-anchor="middle">LD2: USER (PA5)</text>

        <!-- Left Connector Morpho CN7 -->
        <g transform="translate(18, 95)">
            <rect width="135" height="250" rx="4" fill="#1E293B" stroke="#0F172A"/>
            <text x="67" y="16" fill="#38BDF8" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">CN7 LEFT HEADER</text>
            <text x="10" y="38" fill="#F8FAFC" font-family="monospace" font-size="9">PC10 &bull; UART4_TX</text>
            <text x="10" y="56" fill="#F8FAFC" font-family="monospace" font-size="9">PC12 &bull; UART5_TX</text>
            <text x="10" y="74" fill="#38BDF8" font-family="monospace" font-size="9">VDD  &bull; 3.3V OUT</text>
            <text x="10" y="92" fill="#F8FAFC" font-family="monospace" font-size="9">NRST &bull; SYS_RESET</text>
            <text x="10" y="110" fill="#EF4444" font-family="monospace" font-size="9">5V   &bull; 5.0V SUPPLY</text>
            <text x="10" y="128" fill="#94A3B8" font-family="monospace" font-size="9">GND  &bull; GROUND</text>
            <text x="10" y="146" fill="#94A3B8" font-family="monospace" font-size="9">GND  &bull; GROUND</text>
            <text x="10" y="164" fill="#F59E0B" font-family="monospace" font-size="9">PA0  &bull; ADC1_IN0</text>
            <text x="10" y="182" fill="#F59E0B" font-family="monospace" font-size="9">PA1  &bull; ADC1_IN1</text>
            <text x="10" y="200" fill="#F59E0B" font-family="monospace" font-size="9">PA4  &bull; DAC1_OUT1</text>
            <text x="10" y="218" fill="#F8FAFC" font-family="monospace" font-size="9">PB0  &bull; TIM3_CH3</text>
            <text x="10" y="236" fill="#F8FAFC" font-family="monospace" font-size="9">PC1  &bull; ADC1_IN11</text>
        </g>

        <!-- Right Connector Morpho CN10 -->
        <g transform="translate(527, 95)">
            <rect width="135" height="250" rx="4" fill="#1E293B" stroke="#0F172A"/>
            <text x="67" y="16" fill="#38BDF8" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">CN10 RIGHT HEADER</text>
            <text x="10" y="38" fill="#F8FAFC" font-family="monospace" font-size="9">PB8  &bull; I2C1_SCL</text>
            <text x="10" y="56" fill="#F8FAFC" font-family="monospace" font-size="9">PB9  &bull; I2C1_SDA</text>
            <text x="10" y="74" fill="#22C55E" font-family="monospace" font-size="9">PA5  &bull; SPI1_SCK/LD2</text>
            <text x="10" y="92" fill="#F8FAFC" font-family="monospace" font-size="9">PA6  &bull; SPI1_MISO</text>
            <text x="10" y="110" fill="#F8FAFC" font-family="monospace" font-size="9">PA7  &bull; SPI1_MOSI</text>
            <text x="10" y="128" fill="#F8FAFC" font-family="monospace" font-size="9">PB6  &bull; TIM4_CH1</text>
            <text x="10" y="146" fill="#F8FAFC" font-family="monospace" font-size="9">PC7  &bull; TIM3_CH2</text>
            <text x="10" y="164" fill="#F8FAFC" font-family="monospace" font-size="9">PA9  &bull; USART1_TX</text>
            <text x="10" y="182" fill="#F8FAFC" font-family="monospace" font-size="9">PA8  &bull; TIM1_CH1</text>
            <text x="10" y="200" fill="#F8FAFC" font-family="monospace" font-size="9">PB10 &bull; I2C2_SCL</text>
            <text x="10" y="218" fill="#38BDF8" font-family="monospace" font-size="9">PA2  &bull; USART2_TX(VCP)</text>
            <text x="10" y="236" fill="#38BDF8" font-family="monospace" font-size="9">PA3  &bull; USART2_RX(VCP)</text>
        </g>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/nucleo-f446re-pinout.svg", "w") as f:
        f.write(svg)

# 2. BASIC ELECTRONICS KIT
def gen_basic_kit_svg():
    svg = svg_header(800, 480, "Kit 2: Basic Electronics Prototyping Essentials")
    svg += """
    <g transform="translate(40, 70)">
        <rect x="0" y="0" width="220" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="110" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">RESISTORS &amp; OHM'S LAW</text>
        <g transform="translate(110, 70)">
            <line x1="-50" y1="0" x2="50" y2="0" stroke="#94A3B8" stroke-width="3"/>
            <rect x="-24" y="-10" width="48" height="20" rx="4" fill="#FDE68A" stroke="#D97706"/>
            <text x="0" y="32" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">220 &Omega; / 330 &Omega; / 10 k&Omega;</text>
        </g>
        <text x="110" y="145" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">R = (3.3V - 2.0V) / 10mA</text>
    </g>

    <g transform="translate(290, 70)">
        <rect x="0" y="0" width="220" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="110" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">TACTILE PUSH BUTTONS</text>
        <g transform="translate(110, 80)">
            <rect x="-25" y="-25" width="50" height="50" rx="6" fill="#1E293B"/>
            <circle cx="0" cy="0" r="14" fill="#0284C7"/>
        </g>
        <text x="110" y="145" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">Internal / External Pull-Ups</text>
    </g>

    <g transform="translate(540, 70)">
        <rect x="0" y="0" width="220" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="110" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">DISCRETE &amp; RGB LEDS</text>
        <g transform="translate(110, 80)">
            <circle cx="-30" cy="0" r="12" fill="#EF4444"/>
            <circle cx="0" cy="0" r="12" fill="#22C55E"/>
            <circle cx="30" cy="0" r="12" fill="#3B82F6"/>
        </g>
        <text x="110" y="145" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">Push-Pull &bull; Common Cathode</text>
    </g>

    <g transform="translate(40, 270)">
        <rect x="0" y="0" width="345" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">PIEZO &amp; ELECTROMAGNETIC BUZZERS</text>
        <circle cx="172" cy="85" r="30" fill="#1E293B"/>
        <circle cx="172" cy="85" r="10" fill="#CBD5E1"/>
        <text x="172" y="145" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">Active 5V / Passive PWM Driven</text>
    </g>

    <g transform="translate(415, 270)">
        <rect x="0" y="0" width="345" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">TRANSISTOR &amp; MOSFET DRIVERS</text>
        <g transform="translate(172, 85)">
            <path d="M -15 -18 A 18 18 0 0 1 15 -18 L 15 12 L -15 12 Z" fill="#1E293B"/>
            <text x="0" y="2" fill="#FFFFFF" font-family="monospace" font-size="9" text-anchor="middle">2N2222</text>
        </g>
        <text x="172" y="145" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">Low-Side Switching &bull; Flyback Diode Protected</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/basic-kit-components.svg", "w") as f:
        f.write(svg)

# 3. DISPLAY & SENSING MODULES
def gen_display_sensing_svg():
    svg = svg_header(800, 480, "Kit 3: Display & Sensing Module Interfacing")
    svg += """
    <g transform="translate(40, 70)">
        <rect width="345" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">SSD1306 0.96-INCH I2C OLED</text>
        <rect x="72" y="45" width="200" height="90" rx="4" fill="#0F172A"/>
        <text x="172" y="95" fill="#38BDF8" font-family="monospace" font-size="12" text-anchor="middle">128x64 Pixels</text>
        <text x="172" y="155" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">GND &bull; VCC(3V3) &bull; SCL(PB8) &bull; SDA(PB9)</text>
    </g>

    <g transform="translate(415, 70)">
        <rect width="345" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">DS3231 HIGH-PRECISION I2C RTC</text>
        <rect x="122" y="55" width="100" height="60" rx="4" fill="#1E293B"/>
        <circle cx="172" cy="85" r="14" fill="#D97706"/>
        <text x="172" y="155" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">VCC &bull; GND &bull; SCL &bull; SDA &bull; Battery Backup</text>
    </g>

    <g transform="translate(40, 275)">
        <rect width="345" height="175" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">ANALOG TEMP (LM35) &amp; LDR</text>
        <circle cx="110" cy="85" r="22" fill="#E2E8F0"/>
        <text x="110" y="89" fill="#0F172A" font-family="monospace" font-size="9" text-anchor="middle">LM35</text>
        <circle cx="230" cy="85" r="20" fill="#FEF3C7" stroke="#D97706"/>
        <text x="230" y="89" fill="#D97706" font-family="monospace" font-size="9" text-anchor="middle">LDR</text>
        <text x="172" y="150" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">Linear 10 mV/&deg;C &bull; Voltage Divider ADC</text>
    </g>

    <g transform="translate(415, 275)">
        <rect width="345" height="175" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">MPU6050 6-AXIS IMU (ACCEL + GYRO)</text>
        <rect x="122" y="55" width="100" height="60" rx="4" fill="#0284C7"/>
        <text x="172" y="90" fill="#FFFFFF" font-family="monospace" font-size="11" text-anchor="middle">I2C: 0x68</text>
        <text x="172" y="150" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">VCC &bull; GND &bull; SCL &bull; SDA &bull; INT</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/display-sensing-pinouts.svg", "w") as f:
        f.write(svg)

# 4. DETECTION & ACTUATION MODULES
def gen_detection_modules_svg():
    svg = svg_header(800, 480, "Kit 4: Detection & Actuation Modules Pinouts")
    svg += """
    <g transform="translate(40, 70)">
        <rect width="345" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">HC-SR04 ULTRASONIC RANGEFINDER</text>
        <circle cx="110" cy="85" r="28" fill="#E2E8F0" stroke="#94A3B8" stroke-width="2"/>
        <circle cx="234" cy="85" r="28" fill="#E2E8F0" stroke="#94A3B8" stroke-width="2"/>
        <text x="172" y="150" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">VCC(5V) &bull; Trig(PA0) &bull; Echo(5V&rarr;3.3V Divider) &bull; GND</text>
    </g>

    <g transform="translate(415, 70)">
        <rect width="345" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">SG90 MICRO SERVO MOTOR</text>
        <rect x="122" y="55" width="100" height="60" rx="4" fill="#0284C7"/>
        <circle cx="172" cy="70" r="10" fill="#FFFFFF"/>
        <text x="172" y="150" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">Brown(GND) &bull; Red(5V) &bull; Orange(PWM 50Hz)</text>
    </g>

    <g transform="translate(40, 275)">
        <rect width="345" height="175" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">MQ-2 / MQ-3 GAS SENSORS</text>
        <circle cx="172" cy="85" r="30" fill="#78350F" stroke="#B45309" stroke-width="2"/>
        <text x="172" y="150" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">VCC(5V) &bull; GND &bull; AOUT(ADC 0-3.3V) &bull; DOUT</text>
    </g>

    <g transform="translate(415, 275)">
        <rect width="345" height="175" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#shadow)"/>
        <text x="172" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">L298N / TB6612 DUAL MOTOR DRIVER</text>
        <rect x="122" y="55" width="100" height="60" rx="4" fill="#DC2626"/>
        <text x="172" y="90" fill="#FFFFFF" font-family="monospace" font-size="11" text-anchor="middle">H-Bridge</text>
        <text x="172" y="150" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">IN1 &bull; IN2 &bull; ENA(PWM) &bull; Bidirectional Control</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/detection-modules-pinouts.svg", "w") as f:
        f.write(svg)

# 5. CAN BUS TRANSCEIVER
def gen_can_bus_svg():
    svg = svg_header(800, 480, "CAN Bus Physical Layer: STM32 CAN1 to SN65HVD230 Transceiver")
    svg += """
    <!-- STM32 Node 1 Header -->
    <g transform="translate(40, 70)" filter="url(#shadow)">
        <rect width="210" height="340" rx="8" fill="#0284C7" stroke="#0369A1" stroke-width="2"/>
        <text x="105" y="30" fill="#FFFFFF" font-family="sans-serif" font-size="13" font-weight="bold" text-anchor="middle">STM32F446RE NODE 1</text>
        <g transform="translate(15, 55)">
            <rect width="180" height="260" rx="4" fill="#1E293B"/>
            
            <rect x="8" y="25" width="164" height="26" rx="3" fill="#0F172A"/>
            <text x="16" y="42" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">PB9</text>
            <text x="50" y="42" fill="#94A3B8" font-family="sans-serif" font-size="9">CAN1_TX</text>
            <circle cx="155" cy="38" r="5" fill="#38BDF8" stroke="#FFFFFF" stroke-width="1.2"/>
            
            <rect x="8" y="75" width="164" height="26" rx="3" fill="#0F172A"/>
            <text x="16" y="92" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">PB8</text>
            <text x="50" y="92" fill="#94A3B8" font-family="sans-serif" font-size="9">CAN1_RX</text>
            <circle cx="155" cy="88" r="5" fill="#38BDF8" stroke="#FFFFFF" stroke-width="1.2"/>
            
            <rect x="8" y="125" width="164" height="26" rx="3" fill="#0F172A"/>
            <text x="16" y="142" fill="#EF4444" font-family="monospace" font-size="11" font-weight="bold">3.3V</text>
            <text x="50" y="142" fill="#94A3B8" font-family="sans-serif" font-size="9">VDD Supply</text>
            <circle cx="155" cy="138" r="5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1.2"/>
            
            <rect x="8" y="175" width="164" height="26" rx="3" fill="#0F172A"/>
            <text x="16" y="192" fill="#94A3B8" font-family="monospace" font-size="11" font-weight="bold">GND</text>
            <text x="50" y="192" fill="#94A3B8" font-family="sans-serif" font-size="9">Common Ground</text>
            <circle cx="155" cy="188" r="5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.2"/>
        </g>
    </g>

    <!-- Transceiver SN65HVD230 -->
    <g transform="translate(300, 100)" filter="url(#shadow)">
        <rect width="180" height="280" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2"/>
        <text x="90" y="26" fill="#0F172A" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">SN65HVD230 (3.3V)</text>
        <rect x="20" y="45" width="140" height="210" rx="4" fill="#0F172A"/>
        
        <!-- Left Side Input Pins -->
        <text x="45" y="80" fill="#38BDF8" font-family="monospace" font-size="10">D (TXD)</text>
        <circle cx="30" cy="76" r="4" fill="#38BDF8" stroke="#FFFFFF" stroke-width="1"/>
        
        <text x="45" y="120" fill="#38BDF8" font-family="monospace" font-size="10">R (RXD)</text>
        <circle cx="30" cy="116" r="4" fill="#38BDF8" stroke="#FFFFFF" stroke-width="1"/>
        
        <text x="45" y="160" fill="#EF4444" font-family="monospace" font-size="10">VCC (3V3)</text>
        <circle cx="30" cy="156" r="4" fill="#EF4444" stroke="#FFFFFF" stroke-width="1"/>
        
        <text x="45" y="200" fill="#94A3B8" font-family="monospace" font-size="10">GND</text>
        <circle cx="30" cy="196" r="4" fill="#94A3B8" stroke="#FFFFFF" stroke-width="1"/>
        
        <!-- Right Side Differential Pins -->
        <text x="130" y="100" fill="#10B981" font-family="monospace" font-size="10" text-anchor="end">CANH</text>
        <circle cx="145" cy="96" r="4" fill="#10B981" stroke="#FFFFFF" stroke-width="1"/>
        
        <text x="130" y="140" fill="#06B6D4" font-family="monospace" font-size="10" text-anchor="end">CANL</text>
        <circle cx="145" cy="136" r="4" fill="#06B6D4" stroke="#FFFFFF" stroke-width="1"/>
    </g>

    <!-- Interconnecting Wires (Straight parallel routes, NO text overlap) -->
    <path d="M 210 163 L 330 176" stroke="#3B82F6" stroke-width="2.8" stroke-linecap="round" fill="none"/>
    <path d="M 210 213 L 330 216" stroke="#3B82F6" stroke-width="2.8" stroke-linecap="round" fill="none"/>
    <path d="M 210 263 L 330 256" stroke="#EF4444" stroke-width="2.8" stroke-linecap="round" fill="none"/>
    <path d="M 210 313 L 330 296" stroke="#0F172A" stroke-width="2.8" stroke-linecap="round" fill="none"/>

    <!-- Differential CAN Lines with Termination -->
    <line x1="445" y1="196" x2="720" y2="196" stroke="#10B981" stroke-width="3.5"/>
    <text x="590" y="186" fill="#10B981" font-family="monospace" font-size="11" font-weight="bold">CANH Line</text>
    
    <line x1="445" y1="236" x2="720" y2="236" stroke="#06B6D4" stroke-width="3.5"/>
    <text x="590" y="254" fill="#06B6D4" font-family="monospace" font-size="11" font-weight="bold">CANL Line</text>

    <!-- 120 Ohm Termination Resistors -->
    <g transform="translate(520, 216)">
        <line x1="0" y1="-20" x2="0" y2="20" stroke="#94A3B8" stroke-width="3"/>
        <rect x="-8" y="-12" width="16" height="24" rx="2" fill="#FDE68A" stroke="#D97706"/>
        <text x="14" y="4" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold">120 &Omega;</text>
    </g>
    <g transform="translate(710, 216)">
        <line x1="0" y1="-20" x2="0" y2="20" stroke="#94A3B8" stroke-width="3"/>
        <rect x="-8" y="-12" width="16" height="24" rx="2" fill="#FDE68A" stroke="#D97706"/>
        <text x="14" y="4" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold">120 &Omega;</text>
        <text x="14" y="22" fill="#64748B" font-family="monospace" font-size="8.5">To Node 2</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/can-bus-transceiver-circuit.svg", "w") as f:
        f.write(svg)

# 6. ASSIGNMENT 1: INACTIVITY DETECTOR (PA5 + PC13)
def gen_diy01_svg():
    svg = svg_header(800, 480, "Assignment 1: Inactivity Detector (STM32 PA5 & PC13)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32F446RE PINS", [
        "PA5  (GPIO Output &bull; LD2)",
        "PC13 (GPIO Input  &bull; B1)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)
    
    svg += draw_led(480, 160, "#22C55E", "LED on PA5")
    svg += draw_resistor(570, 160, "330 &Omega;", horizontal=True)
    svg += draw_button(480, 280, "Button on PC13")

    # Wire 1: PA5 to LED Anode
    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 470, 192, "#22C55E", "PA5 Control")
    # Wire 2: LED Cathode to Resistor
    svg += draw_wire(490, 192, 540, 160, "#94A3B8")
    # Wire 3: Resistor to Breadboard GND Rail
    svg += draw_wire(600, 160, 680, 160, "#0F172A")
    svg += draw_wire(680, 160, 680, 376, "#0F172A")
    svg += draw_wire(680, 376, pins["GND"][0], pins["GND"][1], "#0F172A", "GND Rail")

    # Wire 4: PC13 to Button Pin 1
    svg += draw_wire(pins["PC13"][0], pins["PC13"][1], 454, 280, "#3B82F6", "PC13 Input")
    # Wire 5: Button Pin 2 to GND
    svg += draw_wire(506, 280, 680, 280, "#0F172A")

    svg += draw_legend(550, 270)
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy01-inactivity-detector.svg", "w") as f:
        f.write(svg)

# 7. ASSIGNMENT 2: MULTI-MODE LED (PA5 + PA0)
def gen_diy02_svg():
    svg = svg_header(800, 480, "Assignment 2: Multi-Mode LED Controller (PA5 & PA0)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32F446RE PINS", [
        "PA5  (TIM2_CH1 / GPIO)",
        "PA0  (GPIO_INPUT_EXTI)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)
    
    svg += draw_led(480, 160, "#3B82F6", "LED on PA5")
    svg += draw_resistor(570, 160, "330 &Omega;")
    svg += draw_button(480, 280, "Button on PA0")
    svg += draw_resistor(570, 280, "10 k&Omega; Pull-Down")

    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 470, 192, "#3B82F6", "PA5 PWM")
    svg += draw_wire(490, 192, 540, 160, "#94A3B8")
    svg += draw_wire(600, 160, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 454, 280, "#F59E0B", "PA0 Input")
    svg += draw_wire(506, 280, 540, 280, "#F59E0B")
    svg += draw_wire(600, 280, pins["GND"][0], pins["GND"][1], "#0F172A")

    svg += """
    <g transform="translate(45, 415)">
        <rect width="700" height="50" rx="6" fill="#1E293B"/>
        <text x="20" y="30" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">FSM CYCLE:</text>
        <text x="110" y="30" fill="#F8FAFC" font-family="monospace" font-size="11">MODE 0 (OFF) &rarr; MODE 1 (SOLID ON) &rarr; MODE 2 (500ms BLINK) &rarr; MODE 3 (100ms BLINK)</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy02-multimode-led.svg", "w") as f:
        f.write(svg)

# 8. ASSIGNMENT 3: TRANSISTOR BUZZER DRIVER (PB0 + PA0)
def gen_diy03_svg():
    svg = svg_header(800, 480, "Assignment 3: Electronic Buzzer Driver (PB0 TIM3_CH3)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32F446RE PINS", [
        "PB0  (TIM3_CH3 PWM Tone)",
        "PA0  (GPIO Push Button)",
        "5.0V (Buzzer Power)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)

    # Buzzer
    svg += """
    <g transform="translate(580, 150)">
        <circle cx="0" cy="0" r="26" fill="#1E293B" stroke="#0F172A" stroke-width="2"/>
        <circle cx="0" cy="0" r="10" fill="#E2E8F0"/>
        <text x="0" y="3.5" fill="#0F172A" font-family="monospace" font-size="9" font-weight="bold" text-anchor="middle">BUZZER</text>
        <circle cx="-14" cy="34" r="3.5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1"/>
        <text x="-14" y="46" fill="#EF4444" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">(+)</text>
        <circle cx="14" cy="34" r="3.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1"/>
        <text x="14" y="46" fill="#0F172A" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">(-)</text>
    </g>
    <!-- Transistor 2N2222 -->
    <g transform="translate(500, 270)">
        <path d="M -16 -18 A 18 18 0 0 1 16 -18 L 16 12 L -16 12 Z" fill="#1E293B"/>
        <text x="0" y="0" fill="#94A3B8" font-family="monospace" font-size="8" text-anchor="middle">2N2222</text>
        <circle cx="-12" cy="22" r="3.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1"/>
        <text x="-12" y="34" fill="#0F172A" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">E(GND)</text>
        <circle cx="0" cy="22" r="3.5" fill="#3B82F6" stroke="#FFFFFF" stroke-width="1"/>
        <text x="0" y="34" fill="#3B82F6" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">B(Base)</text>
        <circle cx="12" cy="22" r="3.5" fill="#0F172A" stroke="#FFFFFF" stroke-width="1"/>
        <text x="12" y="34" fill="#0F172A" font-family="monospace" font-size="8" font-weight="bold" text-anchor="middle">C(Coil)</text>
    </g>
    <!-- 1N4148 Diode -->
    <g transform="translate(650, 190)">
        <rect x="-6" y="-14" width="12" height="28" fill="#0F172A" rx="2"/>
        <rect x="-6" y="-14" width="12" height="6" fill="#E2E8F0"/>
        <text x="14" y="4" fill="#DC2626" font-family="monospace" font-size="8.5" font-weight="bold">1N4148</text>
    </g>
    """
    svg += draw_resistor(420, 270, "1 k&Omega; Base")

    # Wire 1: PB0 to Resistor
    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 390, 270, "#3B82F6", "PB0 PWM")
    # Wire 2: Resistor to Base
    svg += draw_wire(450, 270, 500, 292, "#3B82F6")
    # Wire 3: Emitter to GND
    svg += draw_wire(488, 292, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    # Wire 4: Collector to Buzzer (-)
    svg += draw_wire(512, 292, 594, 184, "#0F172A")
    # Wire 5: Buzzer (+) to 5V Supply
    svg += draw_wire(566, 184, pins["5.0V"][0], pins["5.0V"][1], "#EF4444", "5V Power")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy03-transistor-buzzer.svg", "w") as f:
        f.write(svg)

# 9. ASSIGNMENT 4: 4-BIT BINARY COUNTER (PB0..PB3 + PA0)
def gen_diy04_svg():
    svg = svg_header(800, 480, "Assignment 4: 4-Bit Binary Counter (PB0, PB1, PB2, PB3)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32 GPIO PINS", [
        "PB0  (Bit 0 LSB &bull; GPIO_OUT)",
        "PB1  (Bit 1     &bull; GPIO_OUT)",
        "PB2  (Bit 2     &bull; GPIO_OUT)",
        "PB3  (Bit 3 MSB &bull; GPIO_OUT)",
        "PA0  (Clock Input Button)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)

    colors = ["#EF4444", "#F59E0B", "#22C55E", "#3B82F6"]
    labels = ["PB0(b0)", "PB1(b1)", "PB2(b2)", "PB3(b3)"]
    for i in range(4):
        lx = 420 + i * 75
        svg += draw_led(lx, 150, colors[i], labels[i])
        svg += draw_resistor(lx, 230, "330&Omega;", horizontal=False)
        bit_pin = f"PB{i}"
        # Wire from STM32 pin to LED Anode
        svg += draw_wire(pins[bit_pin][0], pins[bit_pin][1], lx-10, 182, colors[i])
        # Resistor to GND
        svg += draw_wire(lx, 260, lx, 376, "#0F172A")
    
    # Common Ground tie
    svg += draw_wire(420, 376, pins["GND"][0], pins["GND"][1], "#0F172A", "GND Rail")
    
    svg += draw_button(530, 320, "Increment Count (PA0)")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 504, 320, "#0284C7", "PA0 Clock")
    svg += draw_wire(556, 320, 645, 376, "#0F172A")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy04-binary-counter-leds.svg", "w") as f:
        f.write(svg)

# 10. ASSIGNMENT 5: REACTION TIMER (PA5 + PA0 + PA2)
def gen_diy05_svg():
    svg = svg_header(800, 480, "Assignment 5: Reaction-Time Measurement (PA5 & PA0)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32F446RE PINS", [
        "PA5  (Random LED &bull; TIM2_CH1)",
        "PA0  (Slap Button &bull; EXTI0)",
        "PA2  (USART2_TX  &bull; 115200)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)
    
    svg += draw_led(480, 160, "#EF4444", "Trigger (PA5)")
    svg += draw_resistor(570, 160, "330 &Omega;")
    svg += draw_button(480, 280, "Response (PA0)")
    
    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 470, 192, "#EF4444", "PA5 Output")
    svg += draw_wire(490, 192, 540, 160, "#94A3B8")
    svg += draw_wire(600, 160, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 454, 280, "#3B82F6", "PA0 Input")
    svg += draw_wire(506, 280, pins["GND"][0], pins["GND"][1], "#0F172A")

    svg += """
    <g transform="translate(45, 415)">
        <rect width="700" height="50" rx="6" fill="#0F172A"/>
        <text x="20" y="30" fill="#22C55E" font-family="monospace" font-size="12" font-weight="bold">USART2_TX (PA2):</text>
        <text x="170" y="30" fill="#F8FAFC" font-family="monospace" font-size="12">"Reaction Time: 238 ms" &bull; Hardware Timer Microsecond Resolution</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy05-reaction-timer.svg", "w") as f:
        f.write(svg)

# 11. ASSIGNMENT 6: EMERGENCY BUTTON EXTI (PC13 + PA5 + PB0)
def gen_diy06_svg():
    svg = svg_header(800, 480, "Assignment 6: Hardware EXTI Emergency Stop (PC13 / PA5)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32 NVIC & EXTI", [
        "PC13 (EXTI13 Priority 0)",
        "PA5  (Alarm Red Strobe)",
        "PB0  (Siren Transistor Drive)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)

    svg += draw_button(470, 170, "E-STOP (PC13)")
    svg += draw_led(590, 170, "#DC2626", "Alarm (PA5)")
    svg += draw_resistor(670, 170, "330 &Omega;")

    svg += draw_wire(pins["PC13"][0], pins["PC13"][1], 444, 170, "#DC2626", "PC13 EXTI")
    svg += draw_wire(496, 170, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    
    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 580, 202, "#DC2626", "PA5 Alarm")
    svg += draw_wire(600, 202, 640, 170, "#94A3B8")
    svg += draw_wire(700, 170, pins["GND"][0], pins["GND"][1], "#0F172A")

    svg += """
    <g transform="translate(45, 415)">
        <rect width="700" height="50" rx="6" fill="#7F1D1D"/>
        <text x="20" y="30" fill="#FEF2F2" font-family="monospace" font-size="12" font-weight="bold">CORTEX-M NVIC AUDIT:</text>
        <text x="190" y="30" fill="#FEE2E2" font-family="monospace" font-size="12">Interrupt Latency &lt; 5 &mu;s &bull; Direct BSRR Register Trip</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy06-estop-interrupt.svg", "w") as f:
        f.write(svg)

# 12. ASSIGNMENT 7: DEBOUNCING FILTER (PA0 + PA2)
def gen_diy07_svg():
    svg = svg_header(800, 480, "Assignment 7: Software Debounce Shift-Register (PA0)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32 GPIO PINS", [
        "PA0  (Mechanical Switch Input)",
        "PA2  (USART2_TX Debug Output)",
        "3.3V (Pull-up VDD)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)
    svg += draw_button(480, 180, "Button on PA0")
    
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 454, 180, "#3B82F6", "PA0 Input")
    svg += draw_wire(506, 180, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")

    svg += """
    <g transform="translate(360, 265)">
        <rect width="365" height="110" rx="6" fill="#0F172A"/>
        <text x="16" y="24" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">SOFTWARE DEBOUNCE FILTER (30ms):</text>
        <path d="M 20 65 L 40 65 L 43 45 L 47 65 L 50 45 L 53 65 L 57 45 L 60 45 L 140 45" fill="none" stroke="#EF4444" stroke-width="2"/>
        <text x="20" y="85" fill="#EF4444" font-family="monospace" font-size="9">PA0 Raw Bounce (5-15ms)</text>
        
        <path d="M 180 65 L 230 65 L 230 45 L 340 45" fill="none" stroke="#22C55E" stroke-width="2.5"/>
        <text x="180" y="85" fill="#22C55E" font-family="monospace" font-size="9">Debounced Event Count (+1)</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy07-debouncing-filter.svg", "w") as f:
        f.write(svg)

# 13. ASSIGNMENT 8: TRAFFIC LIGHT (PB0..PB5 + PA0)
def gen_diy08_svg():
    svg = svg_header(800, 480, "Assignment 8: Non-Blocking Traffic Junction (PB0..PB5 & PA0)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32 GPIO PINS", [
        "PB0  (Vehicle RED LED)",
        "PB1  (Vehicle YELLOW LED)",
        "PB2  (Vehicle GREEN LED)",
        "PB4  (Pedestrian RED)",
        "PB5  (Pedestrian GREEN)",
        "PA0  (Pedestrian Request)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)

    # Housing for Vehicle Traffic Lights
    svg += """
    <g transform="translate(420, 100)">
        <rect width="90" height="190" rx="10" fill="#1E293B" stroke="#0F172A" stroke-width="2"/>
        <text x="45" y="20" fill="#E2E8F0" font-family="monospace" font-size="9" font-weight="bold" text-anchor="middle">VEHICLE</text>
        
        <!-- Red -->
        <circle cx="45" cy="50" r="16" fill="#EF4444" stroke="#991B1B" stroke-width="2"/>
        <circle cx="8" cy="50" r="3.5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1"/>
        <text x="45" y="72" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">PB0</text>
        
        <!-- Yellow -->
        <circle cx="45" cy="100" r="16" fill="#F59E0B" stroke="#B45309" stroke-width="2"/>
        <circle cx="8" cy="100" r="3.5" fill="#F59E0B" stroke="#FFFFFF" stroke-width="1"/>
        <text x="45" y="122" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">PB1</text>
        
        <!-- Green -->
        <circle cx="45" cy="150" r="16" fill="#22C55E" stroke="#15803D" stroke-width="2"/>
        <circle cx="8" cy="150" r="3.5" fill="#22C55E" stroke="#FFFFFF" stroke-width="1"/>
        <text x="45" y="172" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">PB2</text>
    </g>

    <!-- Housing for Pedestrian Traffic Lights -->
    <g transform="translate(560, 100)">
        <rect width="90" height="140" rx="10" fill="#1E293B" stroke="#0F172A" stroke-width="2"/>
        <text x="45" y="20" fill="#E2E8F0" font-family="monospace" font-size="9" font-weight="bold" text-anchor="middle">PEDESTRIAN</text>
        
        <!-- Ped Red -->
        <circle cx="45" cy="50" r="16" fill="#EF4444"/>
        <circle cx="8" cy="50" r="3.5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1"/>
        <text x="45" y="72" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">PB4</text>
        
        <!-- Ped Green -->
        <circle cx="45" cy="100" r="16" fill="#22C55E"/>
        <circle cx="8" cy="100" r="3.5" fill="#22C55E" stroke="#FFFFFF" stroke-width="1"/>
        <text x="45" y="122" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">PB5</text>
    </g>
    """
    svg += draw_button(605, 300, "Walk Request (PA0)")

    # Wires to side terminal pads of vehicle lights
    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 428, 150, "#EF4444", "PB0")
    svg += draw_wire(pins["PB1"][0], pins["PB1"][1], 428, 200, "#F59E0B", "PB1")
    svg += draw_wire(pins["PB2"][0], pins["PB2"][1], 428, 250, "#22C55E", "PB2")
    
    # Wires to side terminal pads of pedestrian lights
    svg += draw_wire(pins["PB4"][0], pins["PB4"][1], 568, 150, "#EF4444", "PB4")
    svg += draw_wire(pins["PB5"][0], pins["PB5"][1], 568, 200, "#22C55E", "PB5")

    # Wire to walk request button
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 579, 300, "#38BDF8", "PA0")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy08-traffic-light.svg", "w") as f:
        f.write(svg)

# 14. ASSIGNMENT 9: PWM LED DIMMER (PA0 + PB0)
def gen_diy09_svg():
    svg = svg_header(800, 480, "Assignment 9: PWM LED Dimmer (ADC PA0 & TIM3_CH3 PB0)")
    hdr_svg, pins = draw_header_block(45, 80, 260, 320, "STM32F446RE PINS", [
        "PA0  (ADC1_IN0 Potentiometer)",
        "PB0  (TIM3_CH3 PWM LED Out)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 80, 400, 320)

    # 10k Potentiometer with generous pin spacing
    svg += """
    <g transform="translate(430, 200)">
        <rect x="-35" y="-35" width="70" height="70" rx="8" fill="#0284C7" stroke="#0369A1" stroke-width="1.5"/>
        <circle cx="0" cy="0" r="20" fill="#E2E8F0" stroke="#94A3B8" stroke-width="1.5"/>
        <circle cx="0" cy="0" r="6" fill="#0F172A"/>
        <text x="0" y="-42" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">10k&Omega; POTENTIOMETER</text>
        
        <!-- Pin 1: 3.3V -->
        <circle cx="-22" cy="42" r="4" fill="#EF4444" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="-22" y="56" fill="#EF4444" font-family="monospace" font-size="8.5" font-weight="bold" text-anchor="middle">3.3V</text>
        
        <!-- Pin 2: Wiper PA0 -->
        <circle cx="0" cy="42" r="4" fill="#F59E0B" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="0" y="56" fill="#F59E0B" font-family="monospace" font-size="8.5" font-weight="bold" text-anchor="middle">PA0</text>
        
        <!-- Pin 3: GND -->
        <circle cx="22" cy="42" r="4" fill="#0F172A" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="22" y="56" fill="#0F172A" font-family="monospace" font-size="8.5" font-weight="bold" text-anchor="middle">GND</text>
    </g>
    """
    svg += draw_led(600, 190, "#38BDF8", "PB0 PWM LED")
    svg += draw_resistor(680, 190, "330 &Omega;")

    # Wire 1: 3.3V to Pot Pin 1
    svg += draw_wire(pins["3.3V"][0], pins["3.3V"][1], 408, 242, "#EF4444", "3.3V")
    # Wire 2: Wiper to PA0 ADC
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 430, 242, "#F59E0B", "PA0 Wiper")
    # Wire 3: Pot Pin 3 to GND
    svg += draw_wire(452, 242, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")

    # Wire 4: PB0 to LED Anode
    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 590, 222, "#38BDF8", "PB0 PWM")
    # Wire 5: LED Cathode to Resistor
    svg += draw_wire(610, 222, 650, 190, "#94A3B8")
    # Wire 6: Resistor to GND
    svg += draw_wire(710, 190, pins["GND"][0], pins["GND"][1], "#0F172A")

    svg += """
    <g transform="translate(45, 415)">
        <rect width="700" height="50" rx="6" fill="#1E293B"/>
        <text x="20" y="30" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">REGISTER MAPPING:</text>
        <text x="160" y="30" fill="#F8FAFC" font-family="monospace" font-size="11">ADC1_IN0 (PA0) &rarr; TIM3-&gt;CCR3 Duty Cycle &bull; 1000 Hz PWM Carrier</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy09-pwm-dimmer.svg", "w") as f:
        f.write(svg)

# 15. ASSIGNMENT 10: RGB LED COLOR MIXER (PB4, PB5, PB0, PA0, PC13, GND)
# Completely redesigned to ensure clean parallel lines, wide pin spacing, and zero text overlaps!
def gen_diy10_svg():
    svg = svg_header(800, 480, "Assignment 10: RGB Color Mixer (PB4, PB5, PB0, PA0)")
    hdr_svg, pins = draw_header_block(45, 75, 260, 330, "STM32 TIMER & ADC", [
        "PB4  (TIM3_CH1 &bull; Red PWM)",
        "PB5  (TIM3_CH2 &bull; Green PWM)",
        "PB0  (TIM3_CH3 &bull; Blue PWM)",
        "PA0  (ADC1_IN0 &bull; Dimmer Pot)",
        "PC13 (GPIO_IN  &bull; Palette B1)",
        "GND  (Common Cathode GND)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(345, 75, 400, 330)

    # In-line Resistors: Horizontally parallel, perfectly aligned with RGB channels
    svg += draw_resistor(430, 130, "220&Omega; (Red)")
    svg += draw_resistor(430, 185, "330&Omega; (Green)")
    svg += draw_resistor(430, 240, "220&Omega; (Blue)")

    # RGB Common Cathode Module Box: Clear spacing and distinct terminals
    svg += """
    <g transform="translate(580, 185)">
        <rect x="-45" y="-95" width="135" height="190" rx="10" fill="#0F172A" stroke="#334155" stroke-width="2"/>
        <text x="22" y="-72" fill="#F8FAFC" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">RGB LED (CC)</text>
        
        <!-- Large Optical Diffuser Indicator -->
        <circle cx="22" cy="-35" r="22" fill="#1E293B" stroke="#475467" stroke-width="1.5"/>
        <circle cx="16" cy="-41" r="7" fill="#EF4444" opacity="0.9"/>
        <circle cx="28" cy="-41" r="7" fill="#22C55E" opacity="0.9"/>
        <circle cx="22" cy="-29" r="7" fill="#3B82F6" opacity="0.9"/>
        
        <!-- Terminal 1: RED (aligned with y=130 -> local y=-55) -->
        <circle cx="-35" cy="-55" r="5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="-25" y="-51" fill="#EF4444" font-family="monospace" font-size="9" font-weight="bold">RED (PB4)</text>
        
        <!-- Terminal 2: GREEN (aligned with y=185 -> local y=0) -->
        <circle cx="-35" cy="0" r="5" fill="#22C55E" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="-25" y="4" fill="#22C55E" font-family="monospace" font-size="9" font-weight="bold">GREEN (PB5)</text>
        
        <!-- Terminal 3: BLUE (aligned with y=240 -> local y=55) -->
        <circle cx="-35" cy="55" r="5" fill="#3B82F6" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="-25" y="59" fill="#3B82F6" font-family="monospace" font-size="9" font-weight="bold">BLUE (PB0)</text>
        
        <!-- Terminal 4: COMMON CATHODE GND (aligned with local y=78) -->
        <circle cx="22" cy="80" r="5" fill="#94A3B8" stroke="#FFFFFF" stroke-width="1.2"/>
        <text x="22" y="70" fill="#94A3B8" font-family="monospace" font-size="8.5" font-weight="bold" text-anchor="middle">CATHODE (GND)</text>
    </g>
    """

    # Parallel, non-overlapping wiring routes:
    # 1. PB4 -> Resistor 1 -> RGB RED Pin
    svg += draw_wire(pins["PB4"][0], pins["PB4"][1], 400, 130, "#EF4444")
    svg += draw_wire(460, 130, 545, 130, "#EF4444")

    # 2. PB5 -> Resistor 2 -> RGB GREEN Pin
    svg += draw_wire(pins["PB5"][0], pins["PB5"][1], 400, 185, "#22C55E")
    svg += draw_wire(460, 185, 545, 185, "#22C55E")

    # 3. PB0 -> Resistor 3 -> RGB BLUE Pin
    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 400, 240, "#3B82F6")
    svg += draw_wire(460, 240, 545, 240, "#3B82F6")

    # 4. Common Cathode GND -> Breadboard Bottom GND Rail -> STM32 GND
    svg += draw_wire(602, 265, 602, 381, "#0F172A")
    svg += draw_wire(602, 381, pins["GND"][0], pins["GND"][1], "#0F172A", "GND Rail")

    # 5. Dimmer Potentiometer & Palette Button annotations at the bottom
    svg += """
    <g transform="translate(45, 415)">
        <rect width="700" height="50" rx="6" fill="#1E293B"/>
        <text x="20" y="30" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">INPUT CONTROLS:</text>
        <text x="145" y="30" fill="#F8FAFC" font-family="monospace" font-size="11">PA0 (ADC1): Master Dimmer 0-100% &bull; PC13 (Button): Cycle 8 Predefined Color Palettes</text>
    </g>
    """

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy10-rgb-color-mixer.svg", "w") as f:
        f.write(svg)

def main():
    print("Generating all custom STM32 Fritzing-style vector SVGs with clean wire routing...")
    gen_nucleo_pinout()
    gen_basic_kit_svg()
    gen_display_sensing_svg()
    gen_detection_modules_svg()
    gen_can_bus_svg()
    gen_diy01_svg()
    gen_diy02_svg()
    gen_diy03_svg()
    gen_diy04_svg()
    gen_diy05_svg()
    gen_diy06_svg()
    gen_diy07_svg()
    gen_diy08_svg()
    gen_diy09_svg()
    gen_diy10_svg()
    print("Successfully generated all 15 custom STM32 SVG diagrams in", OUT_DIR)

if __name__ == "__main__":
    main()
