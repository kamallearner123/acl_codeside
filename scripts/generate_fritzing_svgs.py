"""
Comprehensive generator for custom Fritzing-style vector SVG wiring diagrams for STM32 DIY assignments.
Strict STM32 hardware port/pin naming (PA0..PA15, PB0..PB15, PC0..PC15, 3V3, 5V, GND) - NO Arduino naming.
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

def draw_header_block(x, y, w, h, title, pins, align_right=False):
    svg = f"""
    <!-- Pin Header Block: {title} -->
    <g transform="translate({x},{y})" filter="url(#shadow)">
        <rect width="{w}" height="{h}" rx="6" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>
        <rect x="0" y="0" width="{w}" height="24" rx="6" fill="#0F172A"/>
        <text x="{w/2}" y="16" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">{title}</text>
    """
    row_h = (h - 32) / max(len(pins), 1)
    pin_y = 36
    pin_coords = {}
    for i, p in enumerate(pins):
        py = pin_y + i * row_h
        pin_cx = w - 12 if align_right else 12
        text_x = w - 24 if align_right else 24
        anchor = "end" if align_right else "start"
        svg += f"""
        <circle cx="{pin_cx}" cy="{py}" r="5" fill="#F59E0B" stroke="#78350F" stroke-width="1"/>
        <circle cx="{pin_cx}" cy="{py}" r="2" fill="#0F172A"/>
        <text x="{text_x}" y="{py+4}" fill="#F1F5F9" font-family="monospace" font-size="10" font-weight="600" text-anchor="{anchor}">{p}</text>
        """
        pin_coords[p.split()[0]] = (x + pin_cx, y + py)
    svg += "</g>"
    return svg, pin_coords

def draw_breadboard(x, y, w, h):
    return f"""
    <!-- Breadboard Area -->
    <g transform="translate({x},{y})" filter="url(#shadow)">
        <rect width="{w}" height="{h}" rx="8" fill="url(#breadboardGrad)" stroke="#CBD5E1" stroke-width="2"/>
        <line x1="15" y1="12" x2="{w-15}" y2="12" stroke="#EF4444" stroke-width="2" stroke-dasharray="8,4"/>
        <line x1="15" y1="20" x2="{w-15}" y2="20" stroke="#3B82F6" stroke-width="2" stroke-dasharray="8,4"/>
        <text x="6" y="15" fill="#EF4444" font-family="monospace" font-size="11" font-weight="bold">+</text>
        <text x="6" y="23" fill="#3B82F6" font-family="monospace" font-size="11" font-weight="bold">-</text>
        
        <line x1="15" y1="{h-20}" x2="{w-15}" y2="{h-20}" stroke="#EF4444" stroke-width="2" stroke-dasharray="8,4"/>
        <line x1="15" y1="{h-12}" x2="{w-15}" y2="{h-12}" stroke="#3B82F6" stroke-width="2" stroke-dasharray="8,4"/>
        <text x="6" y="{h-17}" fill="#EF4444" font-family="monospace" font-size="11" font-weight="bold">+</text>
        <text x="6" y="{h-9}" fill="#3B82F6" font-family="monospace" font-size="11" font-weight="bold">-</text>
        <rect x="15" y="{h/2 - 4}" width="{w-30}" height="8" fill="#E2E8F0" rx="2"/>
    </g>
    """

def draw_resistor(x, y, val_str, horizontal=True):
    if horizontal:
        return f"""
        <g transform="translate({x},{y})">
            <line x1="-25" y1="0" x2="25" y2="0" stroke="#94A3B8" stroke-width="3" stroke-linecap="round"/>
            <rect x="-14" y="-6" width="28" height="12" rx="3" fill="#FDE68A" stroke="#D97706" stroke-width="1.2"/>
            <rect x="-8" y="-6" width="3" height="12" fill="#B45309"/>
            <rect x="-2" y="-6" width="3" height="12" fill="#B45309"/>
            <rect x="4" y="-6" width="3" height="12" fill="#78350F"/>
            <text x="0" y="-10" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">{val_str}</text>
        </g>
        """
    else:
        return f"""
        <g transform="translate({x},{y})">
            <line x1="0" y1="-25" x2="0" y2="25" stroke="#94A3B8" stroke-width="3" stroke-linecap="round"/>
            <rect x="-6" y="-14" width="12" height="28" rx="3" fill="#FDE68A" stroke="#D97706" stroke-width="1.2"/>
            <rect x="-6" y="-8" width="12" height="3" fill="#B45309"/>
            <rect x="-6" y="-2" width="12" height="3" fill="#B45309"/>
            <rect x="-6" y="4" width="12" height="3" fill="#78350F"/>
            <text x="14" y="4" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold">{val_str}</text>
        </g>
        """

def draw_led(x, y, color="#22C55E", label="LED"):
    return f"""
    <g transform="translate({x},{y})">
        <circle cx="0" cy="0" r="12" fill="{color}" opacity="0.85" stroke="#0F172A" stroke-width="1.5"/>
        <circle cx="-3" cy="-3" r="4" fill="#FFFFFF" opacity="0.6"/>
        <line x1="11" y1="-7" x2="11" y2="7" stroke="#0F172A" stroke-width="2"/>
        <line x1="-6" y1="12" x2="-6" y2="28" stroke="#94A3B8" stroke-width="2.5"/>
        <line x1="6" y1="12" x2="6" y2="28" stroke="#94A3B8" stroke-width="2.5"/>
        <text x="-6" y="40" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">(+)Anode</text>
        <text x="14" y="40" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">(-)Cathode</text>
        <text x="0" y="-16" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">{label}</text>
    </g>
    """

def draw_button(x, y, label="Push Button"):
    return f"""
    <g transform="translate({x},{y})">
        <rect x="-18" y="-18" width="36" height="36" rx="4" fill="#334155" stroke="#0F172A" stroke-width="1.5"/>
        <circle cx="0" cy="0" r="10" fill="#0284C7" stroke="#0369A1" stroke-width="1.5"/>
        <rect x="-24" y="-12" width="6" height="4" fill="#94A3B8"/>
        <rect x="-24" y="8" width="6" height="4" fill="#94A3B8"/>
        <rect x="18" y="-12" width="6" height="4" fill="#94A3B8"/>
        <rect x="18" y="8" width="6" height="4" fill="#94A3B8"/>
        <text x="-26" y="-7" fill="#64748B" font-family="monospace" font-size="8" text-anchor="end">Pin 1</text>
        <text x="26" y="-7" fill="#64748B" font-family="monospace" font-size="8">Pin 2</text>
        <text x="0" y="-22" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">{label}</text>
    </g>
    """

def draw_wire(x1, y1, x2, y2, color="#3B82F6", label="", curve_dir="h"):
    if curve_dir == "h":
        mid_x = (x1 + x2) / 2
        path = f"M {x1} {y1} C {mid_x} {y1}, {mid_x} {y2}, {x2} {y2}"
    else:
        mid_y = (y1 + y2) / 2
        path = f"M {x1} {y1} C {x1} {mid_y}, {x2} {mid_y}, {x2} {y2}"
    
    svg = f"""
    <g>
        <path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" opacity="0.9"/>
        <circle cx="{x1}" cy="{y1}" r="3.5" fill="{color}"/>
        <circle cx="{x2}" cy="{y2}" r="3.5" fill="{color}"/>
    """
    if label:
        lx = (x1 + x2) / 2
        ly = (y1 + y2) / 2 - 6
        svg += f"""
        <rect x="{lx-28}" y="{ly-10}" width="56" height="15" rx="3" fill="#FFFFFF" stroke="{color}" stroke-width="1" opacity="0.9"/>
        <text x="{lx}" y="{ly+1}" fill="{color}" font-family="monospace" font-size="8.5" font-weight="bold" text-anchor="middle">{label}</text>
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

# 1. NUCLEO BOARD PINOUT (PURE STM32 PINS)
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

# 2. BASIC KIT COMPONENTS
def gen_basic_kit_svg():
    svg = svg_header(800, 480, "Kit 2: Fundamental Prototyping Components")
    svg += """
    <g transform="translate(60, 70)" filter="url(#shadow)">
        <rect width="320" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="26" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">1. Tactile Push Button Switch</text>
        <rect x="40" y="60" width="50" height="50" rx="6" fill="#334155"/>
        <circle cx="65" cy="85" r="16" fill="#0284C7"/>
        <text x="120" y="70" fill="#334155" font-family="monospace" font-size="10">&bull; STM32 Pin: PC13 or PA0</text>
        <text x="120" y="90" fill="#334155" font-family="monospace" font-size="10">&bull; Mode: GPIO_MODE_INPUT</text>
        <text x="120" y="110" fill="#334155" font-family="monospace" font-size="10">&bull; Internal Pull-Up/Pull-Down</text>
    </g>
    <g transform="translate(420, 70)" filter="url(#shadow)">
        <rect width="320" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="26" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">2. LED + 330&Omega; Current Resistor</text>
        <g transform="translate(60, 95)">
            <circle cx="0" cy="0" r="14" fill="#22C55E" opacity="0.9" stroke="#15803D"/>
            <line x1="0" y1="14" x2="0" y2="35" stroke="#94A3B8" stroke-width="3"/>
            <line x1="25" y1="20" x2="70" y2="20" stroke="#94A3B8" stroke-width="3"/>
            <rect x="35" y="14" width="24" height="12" rx="3" fill="#FDE68A" stroke="#D97706"/>
            <text x="47" y="38" fill="#B45309" font-family="monospace" font-size="9" text-anchor="middle">330&Omega;</text>
        </g>
        <text x="150" y="70" fill="#334155" font-family="monospace" font-size="10">&bull; STM32 Pin: PA5 (LD2)</text>
        <text x="150" y="90" fill="#334155" font-family="monospace" font-size="10">&bull; Mode: GPIO_MODE_OUTPUT_PP</text>
        <text x="150" y="110" fill="#334155" font-family="monospace" font-size="10">&bull; Max Pin Current: 20 mA</text>
    </g>
    <g transform="translate(60, 260)" filter="url(#shadow)">
        <rect width="320" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="26" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">3. 10k&Omega; Rotary Potentiometer</text>
        <g transform="translate(60, 95)">
            <rect x="-25" y="-25" width="50" height="50" rx="4" fill="#0284C7"/>
            <circle cx="0" cy="0" r="16" fill="#E2E8F0" stroke="#94A3B8"/>
            <line x1="0" y1="0" x2="10" y2="-10" stroke="#0F172A" stroke-width="2"/>
            <circle cx="-15" cy="32" r="3" fill="#94A3B8"/><text x="-15" y="44" fill="#64748B" font-family="monospace" font-size="8" text-anchor="middle">3.3V</text>
            <circle cx="0" cy="32" r="3" fill="#F59E0B"/><text x="0" y="44" fill="#F59E0B" font-family="monospace" font-size="8" text-anchor="middle">PA0</text>
            <circle cx="15" cy="32" r="3" fill="#0F172A"/><text x="15" y="44" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">GND</text>
        </g>
        <text x="120" y="70" fill="#334155" font-family="monospace" font-size="10">&bull; Terminal 1: STM32 3.3V</text>
        <text x="120" y="90" fill="#334155" font-family="monospace" font-size="10">&bull; Wiper: STM32 PA0 (ADC1_IN0)</text>
        <text x="120" y="110" fill="#334155" font-family="monospace" font-size="10">&bull; Terminal 3: STM32 GND</text>
        <text x="120" y="130" fill="#334155" font-family="monospace" font-size="10">&bull; 12-bit Value: 0 to 4095</text>
    </g>
    <g transform="translate(420, 260)" filter="url(#shadow)">
        <rect width="320" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="26" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">4. 2N2222 Transistor Driver</text>
        <g transform="translate(60, 95)">
            <path d="M -15 -18 A 18 18 0 0 1 15 -18 L 15 12 L -15 12 Z" fill="#1E293B" stroke="#0F172A"/>
            <text x="0" y="0" fill="#94A3B8" font-family="monospace" font-size="7" text-anchor="middle">2N2222</text>
            <line x1="-10" y1="12" x2="-10" y2="30" stroke="#94A3B8" stroke-width="2"/>
            <line x1="0" y1="12" x2="0" y2="30" stroke="#94A3B8" stroke-width="2"/>
            <line x1="10" y1="12" x2="10" y2="30" stroke="#94A3B8" stroke-width="2"/>
            <text x="-10" y="40" fill="#64748B" font-family="monospace" font-size="8" text-anchor="middle">E</text>
            <text x="0" y="40" fill="#64748B" font-family="monospace" font-size="8" text-anchor="middle">B</text>
            <text x="10" y="40" fill="#64748B" font-family="monospace" font-size="8" text-anchor="middle">C</text>
        </g>
        <text x="120" y="70" fill="#334155" font-family="monospace" font-size="10">&bull; Base (B): PB0 (TIM3_CH3)</text>
        <text x="120" y="90" fill="#334155" font-family="monospace" font-size="10">&bull; Emitter (E): GND</text>
        <text x="120" y="110" fill="#334155" font-family="monospace" font-size="10">&bull; Collector (C): Buzzer / Relay</text>
        <text x="120" y="130" fill="#EF4444" font-family="monospace" font-size="10">&bull; Diode: 1N4148 Flyback</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/basic-kit-components.svg", "w") as f:
        f.write(svg)

# 3. DISPLAY & SENSING (PURE STM32 PINS)
def gen_display_sensing_svg():
    svg = svg_header(800, 480, "Kit 3: Display & Sensing Module Pin Connections")
    svg += """
    <g transform="translate(50, 70)" filter="url(#shadow)">
        <rect width="210" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">SSD1306 0.96" OLED</text>
        <rect x="25" y="40" width="160" height="80" rx="4" fill="#0F172A"/>
        <text x="105" y="85" fill="#38BDF8" font-family="monospace" font-size="11" text-anchor="middle">128x64 I2C</text>
        <rect x="25" y="125" width="160" height="30" rx="3" fill="#1E293B"/>
        <text x="35" y="144" fill="#64748B" font-family="monospace" font-size="8.5">GND</text>
        <text x="75" y="144" fill="#EF4444" font-family="monospace" font-size="8.5">3.3V</text>
        <text x="115" y="144" fill="#A855F7" font-family="monospace" font-size="8.5">PB8(SCL)</text>
        <text x="155" y="144" fill="#10B981" font-family="monospace" font-size="8.5">PB9(SDA)</text>
    </g>
    <g transform="translate(295, 70)" filter="url(#shadow)">
        <rect width="210" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">DS3231 Precision RTC</text>
        <circle cx="105" cy="80" r="30" fill="#E2E8F0" stroke="#94A3B8"/>
        <text x="105" y="85" fill="#64748B" font-family="monospace" font-size="9" text-anchor="middle">CR2032</text>
        <rect x="15" y="125" width="180" height="30" rx="3" fill="#1E293B"/>
        <text x="25" y="144" fill="#EF4444" font-family="monospace" font-size="8.5">3V3</text>
        <text x="60" y="144" fill="#64748B" font-family="monospace" font-size="8.5">GND</text>
        <text x="95" y="144" fill="#10B981" font-family="monospace" font-size="8.5">PB9(SDA)</text>
        <text x="135" y="144" fill="#A855F7" font-family="monospace" font-size="8.5">PB8(SCL)</text>
        <text x="175" y="144" fill="#F59E0B" font-family="monospace" font-size="8.5">INT</text>
    </g>
    <g transform="translate(540, 70)" filter="url(#shadow)">
        <rect width="210" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">MPU6050 6-DOF IMU</text>
        <rect x="80" y="50" width="50" height="50" rx="4" fill="#0284C7"/>
        <text x="105" y="78" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">3-Axis Accel</text>
        <text x="105" y="90" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">3-Axis Gyro</text>
        <rect x="15" y="125" width="180" height="30" rx="3" fill="#1E293B"/>
        <text x="22" y="144" fill="#EF4444" font-family="monospace" font-size="8">3V3</text>
        <text x="48" y="144" fill="#64748B" font-family="monospace" font-size="8">GND</text>
        <text x="75" y="144" fill="#A855F7" font-family="monospace" font-size="8">PB8(SCL)</text>
        <text x="105" y="144" fill="#10B981" font-family="monospace" font-size="8">PB9(SDA)</text>
        <text x="140" y="144" fill="#F59E0B" font-family="monospace" font-size="8">AD0</text>
        <text x="170" y="144" fill="#F59E0B" font-family="monospace" font-size="8">INT</text>
    </g>
    <g transform="translate(50, 260)" filter="url(#shadow)">
        <rect width="210" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">LM35 Analog Temp</text>
        <path d="M 90 60 A 20 20 0 0 1 120 60 L 120 90 L 90 90 Z" fill="#1E293B"/>
        <text x="105" y="78" fill="#FFFFFF" font-family="monospace" font-size="8" text-anchor="middle">LM35</text>
        <text x="20" y="125" fill="#334155" font-family="monospace" font-size="10">&bull; Pin 1: STM32 5.0V</text>
        <text x="20" y="145" fill="#334155" font-family="monospace" font-size="10">&bull; Pin 2: STM32 PA0 (ADC)</text>
        <text x="20" y="165" fill="#334155" font-family="monospace" font-size="10">&bull; Pin 3: STM32 GND</text>
    </g>
    <g transform="translate(295, 260)" filter="url(#shadow)">
        <rect width="210" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">LDR Ambient Light</text>
        <circle cx="105" cy="70" r="16" fill="#FDE68A" stroke="#D97706"/>
        <path d="M 95 65 Q 105 55 115 65 T 105 80" fill="none" stroke="#DC2626" stroke-width="2"/>
        <text x="20" y="125" fill="#334155" font-family="monospace" font-size="10">&bull; Divider: 10k&Omega; to GND</text>
        <text x="20" y="145" fill="#334155" font-family="monospace" font-size="10">&bull; Sense: STM32 PA1 (ADC)</text>
        <text x="20" y="165" fill="#334155" font-family="monospace" font-size="10">&bull; Top: STM32 3.3V</text>
    </g>
    <g transform="translate(540, 260)" filter="url(#shadow)">
        <rect width="210" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">HC-SR501 PIR Motion</text>
        <circle cx="105" cy="70" r="24" fill="#F1F5F9" stroke="#94A3B8"/>
        <rect x="25" y="120" width="160" height="30" rx="3" fill="#1E293B"/>
        <text x="35" y="139" fill="#EF4444" font-family="monospace" font-size="8.5">5V VCC</text>
        <text x="90" y="139" fill="#F59E0B" font-family="monospace" font-size="8.5">PA2 (OUT)</text>
        <text x="155" y="139" fill="#64748B" font-family="monospace" font-size="8.5">GND</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/display-sensing-pinouts.svg", "w") as f:
        f.write(svg)

# 4. DETECTION MODULES PINOUTS
def gen_detection_modules_svg():
    svg = svg_header(800, 480, "Kit 4: Detection & Actuation Module Pin Connections")
    svg += """
    <g transform="translate(50, 70)" filter="url(#shadow)">
        <rect width="330" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">HC-SR04 Ultrasonic Distance</text>
        <rect x="25" y="45" width="130" height="70" rx="6" fill="#0284C7"/>
        <circle cx="55" cy="80" r="22" fill="#E2E8F0" stroke="#475467" stroke-width="2"/>
        <circle cx="125" cy="80" r="22" fill="#E2E8F0" stroke="#475467" stroke-width="2"/>
        <text x="175" y="65" fill="#EF4444" font-family="monospace" font-size="10">&bull; VCC: STM32 5.0V</text>
        <text x="175" y="85" fill="#3B82F6" font-family="monospace" font-size="10">&bull; Trig: STM32 PB6 (GPIO)</text>
        <text x="175" y="105" fill="#F59E0B" font-family="monospace" font-size="10">&bull; Echo: PA6 (TIM3_CH1)</text>
        <text x="175" y="125" fill="#0F172A" font-family="monospace" font-size="10">&bull; GND: STM32 GND</text>
        <text x="175" y="145" fill="#DC2626" font-family="monospace" font-size="9">&bull; Note: 1k/2k on Echo to 3.3V</text>
    </g>
    <g transform="translate(420, 70)" filter="url(#shadow)">
        <rect width="330" height="170" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">SG90 50Hz PWM Servo Motor</text>
        <rect x="35" y="50" width="70" height="70" rx="4" fill="#0284C7"/>
        <circle cx="70" cy="70" r="16" fill="#FFFFFF" stroke="#64748B"/>
        <rect x="66" y="40" width="8" height="30" fill="#FFFFFF"/>
        <text x="130" y="65" fill="#0F172A" font-family="monospace" font-size="10">&bull; Brown Pin: Ground (GND)</text>
        <text x="130" y="85" fill="#EF4444" font-family="monospace" font-size="10">&bull; Red Pin: Ext +5.0V Power</text>
        <text x="130" y="105" fill="#F59E0B" font-family="monospace" font-size="10">&bull; Orange: STM32 PB4 (TIM3)</text>
        <text x="130" y="125" fill="#334155" font-family="monospace" font-size="10">&bull; PWM: 1.0ms to 2.0ms (50Hz)</text>
    </g>
    <g transform="translate(50, 260)" filter="url(#shadow)">
        <rect width="330" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">MQ-2 Gas / Smoke Sensor</text>
        <circle cx="75" cy="95" r="30" fill="#94A3B8" stroke="#475467" stroke-width="2"/>
        <text x="130" y="70" fill="#EF4444" font-family="monospace" font-size="10">&bull; VCC: STM32 5.0V (150mA)</text>
        <text x="130" y="90" fill="#0F172A" font-family="monospace" font-size="10">&bull; GND: STM32 GND</text>
        <text x="130" y="110" fill="#3B82F6" font-family="monospace" font-size="10">&bull; DO: STM32 PC7 (Digital)</text>
        <text x="130" y="130" fill="#F59E0B" font-family="monospace" font-size="10">&bull; AO: STM32 PA1 (ADC1_IN1)</text>
    </g>
    <g transform="translate(420, 260)" filter="url(#shadow)">
        <rect width="330" height="180" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
        <text x="16" y="24" fill="#0F172A" font-family="sans-serif" font-size="12" font-weight="bold">L298N DC Motor Driver</text>
        <rect x="30" y="55" width="80" height="70" rx="4" fill="#DC2626"/>
        <rect x="50" y="45" width="40" height="20" fill="#1E293B"/>
        <text x="130" y="65" fill="#334155" font-family="monospace" font-size="10">&bull; 12V/GND: External Battery</text>
        <text x="130" y="85" fill="#3B82F6" font-family="monospace" font-size="10">&bull; ENA / ENB: PA7 / PB0 (PWM)</text>
        <text x="130" y="105" fill="#334155" font-family="monospace" font-size="10">&bull; IN1..IN4: PC0..PC3 (GPIO)</text>
        <text x="130" y="125" fill="#334155" font-family="monospace" font-size="10">&bull; GND: Common to STM32 GND</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/detection-modules-pinouts.svg", "w") as f:
        f.write(svg)

# 5. CAN BUS TRANSCEIVER
def gen_can_circuit_svg():
    svg = svg_header(800, 480, "CAN Bus Physical Layer (STM32 CAN1_TX / CAN1_RX)")
    svg += """
    <g transform="translate(40, 70)" filter="url(#shadow)">
        <rect width="210" height="340" rx="8" fill="#0284C7" stroke="#0369A1" stroke-width="2"/>
        <text x="105" y="30" fill="#FFFFFF" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">STM32F446RE NODE 1</text>
        <g transform="translate(15, 60)">
            <rect width="180" height="250" rx="4" fill="#1E293B"/>
            <text x="20" y="40" fill="#38BDF8" font-family="monospace" font-size="12">PB9: CAN1_TX</text>
            <circle cx="160" cy="35" r="5" fill="#F59E0B"/>
            <text x="20" y="80" fill="#38BDF8" font-family="monospace" font-size="12">PB8: CAN1_RX</text>
            <circle cx="160" cy="75" r="5" fill="#F59E0B"/>
            <text x="20" y="120" fill="#EF4444" font-family="monospace" font-size="12">3.3V (VDD)</text>
            <circle cx="160" cy="115" r="5" fill="#EF4444"/>
            <text x="20" y="160" fill="#94A3B8" font-family="monospace" font-size="12">GND (GROUND)</text>
            <circle cx="160" cy="155" r="5" fill="#0F172A"/>
        </g>
    </g>
    <g transform="translate(300, 100)" filter="url(#shadow)">
        <rect width="180" height="280" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2"/>
        <text x="90" y="26" fill="#0F172A" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">SN65HVD230 (3.3V)</text>
        <rect x="20" y="45" width="140" height="210" rx="4" fill="#0F172A"/>
        <text x="35" y="80" fill="#38BDF8" font-family="monospace" font-size="11">D (TXD)</text>
        <text x="35" y="115" fill="#38BDF8" font-family="monospace" font-size="11">R (RXD)</text>
        <text x="35" y="150" fill="#EF4444" font-family="monospace" font-size="11">VCC (3V3)</text>
        <text x="35" y="185" fill="#94A3B8" font-family="monospace" font-size="11">GND</text>
        
        <text x="145" y="95" fill="#10B981" font-family="monospace" font-size="11" text-anchor="end">CANH</text>
        <circle cx="155" cy="92" r="4" fill="#10B981"/>
        <text x="145" y="135" fill="#06B6D4" font-family="monospace" font-size="11" text-anchor="end">CANL</text>
        <circle cx="155" cy="132" r="4" fill="#06B6D4"/>
    </g>
    <!-- Interconnecting Wires -->
    <path d="M 190 95 L 320 178" stroke="#3B82F6" stroke-width="3" fill="none"/>
    <path d="M 190 135 L 320 213" stroke="#3B82F6" stroke-width="3" fill="none"/>
    <path d="M 190 175 L 320 248" stroke="#EF4444" stroke-width="3" fill="none"/>
    <path d="M 190 215 L 320 283" stroke="#0F172A" stroke-width="3" fill="none"/>

    <line x1="455" y1="192" x2="720" y2="192" stroke="#10B981" stroke-width="4"/>
    <text x="590" y="184" fill="#10B981" font-family="monospace" font-size="11" font-weight="bold">CANH Line</text>
    
    <line x1="455" y1="232" x2="720" y2="232" stroke="#06B6D4" stroke-width="4"/>
    <text x="590" y="250" fill="#06B6D4" font-family="monospace" font-size="11" font-weight="bold">CANL Line</text>

    <!-- 120 Ohm Termination Resistors -->
    <g transform="translate(520, 212)">
        <line x1="0" y1="-20" x2="0" y2="20" stroke="#94A3B8" stroke-width="3"/>
        <rect x="-8" y="-12" width="16" height="24" rx="2" fill="#FDE68A" stroke="#D97706"/>
        <text x="12" y="4" fill="#0F172A" font-family="monospace" font-size="11" font-weight="bold">120 &Omega;</text>
    </g>
    <g transform="translate(710, 212)">
        <line x1="0" y1="-20" x2="0" y2="20" stroke="#94A3B8" stroke-width="3"/>
        <rect x="-8" y="-12" width="16" height="24" rx="2" fill="#FDE68A" stroke="#D97706"/>
        <text x="12" y="4" fill="#0F172A" font-family="monospace" font-size="11" font-weight="bold">120 &Omega;</text>
        <text x="12" y="24" fill="#64748B" font-family="monospace" font-size="9">To STM32 Node 2</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/can-bus-transceiver-circuit.svg", "w") as f:
        f.write(svg)

# 6. ASSIGNMENT 1: INACTIVITY DETECTOR (PA5 + PC13)
def gen_diy01_svg():
    svg = svg_header(800, 480, "Assignment 1: Inactivity Detector (STM32 PA5 & PC13)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32F446RE PINS", [
        "PA5  (GPIO Output &bull; LD2)",
        "PC13 (GPIO Input  &bull; B1)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)
    
    svg += draw_led(460, 180, "#22C55E", "LED on PA5")
    svg += draw_resistor(540, 180, "330 &Omega;", horizontal=True)
    svg += draw_button(460, 290, "Button on PC13")

    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 454, 180, "#22C55E", "PA5 Control")
    svg += draw_wire(466, 180, 515, 180, "#94A3B8", "")
    svg += draw_wire(565, 180, 680, 180, "#0F172A", "GND")
    svg += draw_wire(680, 180, 680, 380, "#0F172A", "")
    svg += draw_wire(680, 380, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")

    svg += draw_wire(pins["PC13"][0], pins["PC13"][1], 436, 290, "#3B82F6", "PC13 Input")
    svg += draw_wire(484, 290, 680, 290, "#0F172A", "")

    svg += draw_legend(600, 370)
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy01-inactivity-detector.svg", "w") as f:
        f.write(svg)

# 7. ASSIGNMENT 2: MULTI-MODE LED (PA5 + PA0)
def gen_diy02_svg():
    svg = svg_header(800, 480, "Assignment 2: Multi-Mode LED Controller (PA5 & PA0)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32F446RE PINS", [
        "PA5  (TIM2_CH1 / GPIO)",
        "PA0  (GPIO_INPUT_EXTI)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)
    svg += draw_led(460, 160, "#3B82F6", "LED on PA5")
    svg += draw_resistor(540, 160, "330 &Omega;")
    svg += draw_button(460, 280, "Button on PA0")
    svg += draw_resistor(540, 280, "10 k&Omega; Pull-Down")

    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 454, 160, "#3B82F6", "PA5 PWM")
    svg += draw_wire(565, 160, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 436, 280, "#F59E0B", "PA0 Input")
    svg += draw_wire(484, 280, pins["3.3V"][0], pins["3.3V"][1], "#EF4444", "3.3V")

    svg += """
    <g transform="translate(60, 415)">
        <rect width="680" height="50" rx="6" fill="#1E293B"/>
        <text x="20" y="30" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">FSM CYCLE:</text>
        <text x="110" y="30" fill="#F8FAFC" font-family="monospace" font-size="11">MODE 0 (OFF) &rarr; MODE 1 (SOLID ON) &rarr; MODE 2 (500ms BLINK) &rarr; MODE 3 (100ms BLINK)</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy02-multimode-led.svg", "w") as f:
        f.write(svg)

# 8. ASSIGNMENT 3: TRANSISTOR BUZZER (PB0 + PA0)
def gen_diy03_svg():
    svg = svg_header(800, 480, "Assignment 3: Electronic Buzzer Driver (PB0 TIM3_CH3)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32F446RE PINS", [
        "PB0  (TIM3_CH3 PWM Tone)",
        "PA0  (GPIO Push Button)",
        "5.0V (Buzzer Power)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)

    svg += """
    <g transform="translate(560, 160)">
        <circle cx="0" cy="0" r="28" fill="#1E293B" stroke="#0F172A" stroke-width="2"/>
        <circle cx="0" cy="0" r="12" fill="#E2E8F0"/>
        <text x="0" y="4" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">BUZZER</text>
        <circle cx="-12" cy="36" r="3" fill="#EF4444"/><text x="-12" y="48" fill="#EF4444" font-family="monospace" font-size="8" text-anchor="middle">(+)</text>
        <circle cx="12" cy="36" r="3" fill="#0F172A"/><text x="12" y="48" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">(-)</text>
    </g>
    <g transform="translate(500, 280)">
        <path d="M -15 -18 A 18 18 0 0 1 15 -18 L 15 12 L -15 12 Z" fill="#1E293B"/>
        <text x="0" y="0" fill="#94A3B8" font-family="monospace" font-size="8" text-anchor="middle">2N2222</text>
        <circle cx="-10" cy="22" r="3" fill="#0F172A"/><text x="-10" y="34" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">E(GND)</text>
        <circle cx="0" cy="22" r="3" fill="#3B82F6"/><text x="0" y="34" fill="#3B82F6" font-family="monospace" font-size="8" text-anchor="middle">B(Base)</text>
        <circle cx="10" cy="22" r="3" fill="#0F172A"/><text x="10" y="34" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">C(Coil)</text>
    </g>
    <g transform="translate(630, 200)">
        <rect x="-6" y="-14" width="12" height="28" fill="#0F172A" rx="2"/>
        <rect x="-6" y="-14" width="12" height="6" fill="#E2E8F0"/>
        <text x="16" y="4" fill="#DC2626" font-family="monospace" font-size="9">1N4148 Diode</text>
    </g>
    """
    svg += draw_resistor(420, 280, "1 k&Omega; Base")

    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 395, 280, "#3B82F6", "PB0 PWM")
    svg += draw_wire(445, 280, 500, 295, "#3B82F6", "")
    svg += draw_wire(490, 295, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    svg += draw_wire(510, 295, 572, 196, "#0F172A", "")
    svg += draw_wire(548, 196, pins["5.0V"][0], pins["5.0V"][1], "#EF4444", "5.0V")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy03-transistor-buzzer.svg", "w") as f:
        f.write(svg)

# 9. ASSIGNMENT 4: 4-BIT BINARY COUNTER (PB0..PB3 + PA0)
def gen_diy04_svg():
    svg = svg_header(800, 480, "Assignment 4: 4-Bit Binary Counter (PB0, PB1, PB2, PB3)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32 GPIO PINS", [
        "PB0  (Bit 0 LSB &bull; GPIO_OUT)",
        "PB1  (Bit 1     &bull; GPIO_OUT)",
        "PB2  (Bit 2     &bull; GPIO_OUT)",
        "PB3  (Bit 3 MSB &bull; GPIO_OUT)",
        "PA0  (Clock Input Button)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)

    colors = ["#EF4444", "#F59E0B", "#22C55E", "#3B82F6"]
    labels = ["PB0(b0)", "PB1(b1)", "PB2(b2)", "PB3(b3)"]
    for i in range(4):
        lx = 420 + i * 75
        svg += draw_led(lx, 160, colors[i], labels[i])
        svg += draw_resistor(lx, 240, "330&Omega;", horizontal=False)
        bit_pin = f"PB{i}"
        svg += draw_wire(pins[bit_pin][0], pins[bit_pin][1], lx-6, 172, colors[i])
        svg += draw_wire(lx, 265, 710, 265, "#0F172A")
    
    svg += draw_wire(710, 265, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    svg += draw_button(530, 330, "Increment Count (PA0)")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 506, 330, "#0284C7", "PA0 Count")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy04-binary-counter-leds.svg", "w") as f:
        f.write(svg)

# 10. ASSIGNMENT 5: REACTION TIMER (PA5 + PA0 + PA2)
def gen_diy05_svg():
    svg = svg_header(800, 480, "Assignment 5: Reaction-Time Measurement (PA5 & PA0)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32F446RE PINS", [
        "PA5  (Random LED &bull; TIM2_CH1)",
        "PA0  (Slap Button &bull; EXTI0)",
        "PA2  (USART2_TX  &bull; 115200)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)
    svg += draw_led(460, 180, "#EF4444", "Trigger (PA5)")
    svg += draw_resistor(540, 180, "330 &Omega;")
    svg += draw_button(460, 300, "Response (PA0)")
    
    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 454, 180, "#EF4444", "PA5 Output")
    svg += draw_wire(565, 180, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 436, 300, "#3B82F6", "PA0 Input")

    svg += """
    <g transform="translate(60, 415)">
        <rect width="680" height="50" rx="6" fill="#0F172A"/>
        <text x="20" y="30" fill="#22C55E" font-family="monospace" font-size="12" font-weight="bold">USART2_TX (PA2):</text>
        <text x="170" y="30" fill="#F8FAFC" font-family="monospace" font-size="12">"Reaction Time: 238 ms" &bull; Hardware Timer Microsecond Resolution</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy05-reaction-timer.svg", "w") as f:
        f.write(svg)

# 11. ASSIGNMENT 6: EMERGENCY BUTTON EXTI (PC13 + PA5 + PB0)
def gen_diy06_svg():
    svg = svg_header(800, 480, "Assignment 6: Hardware EXTI Emergency Stop (PC13 / PA4)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32 NVIC & EXTI", [
        "PC13 (EXTI13 Priority 0)",
        "PA5  (Alarm Red Strobe)",
        "PB0  (Siren Transistor Drive)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)

    svg += draw_button(460, 180, "E-STOP (PC13)")
    svg += draw_led(580, 180, "#DC2626", "Alarm (PA5)")
    svg += draw_resistor(660, 180, "330 &Omega;")

    svg += draw_wire(pins["PC13"][0], pins["PC13"][1], 436, 180, "#DC2626", "PC13 EXTI")
    svg += draw_wire(pins["PA5"][0], pins["PA5"][1], 574, 180, "#DC2626", "PA5 Alarm")
    svg += draw_wire(685, 180, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")

    svg += """
    <g transform="translate(60, 415)">
        <rect width="680" height="50" rx="6" fill="#7F1D1D"/>
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
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32 GPIO PINS", [
        "PA0  (Mechanical Switch Input)",
        "PA2  (USART2_TX Debug Output)",
        "3.3V (Pull-up VDD)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)
    svg += draw_button(460, 200, "Button on PA0")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 436, 200, "#3B82F6", "PA0 Input")
    svg += draw_wire(484, 200, pins["GND"][0], pins["GND"][1], "#0F172A", "GND")

    svg += """
    <g transform="translate(360, 280)">
        <rect width="360" height="100" rx="6" fill="#0F172A"/>
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

# 13. ASSIGNMENT 8: TRAFFIC LIGHT (PB0, PB1, PB2, PB4, PB5, PA0)
def gen_diy08_svg():
    svg = svg_header(800, 480, "Assignment 8: Non-Blocking Traffic Junction (PB0..PB5 & PA0)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32 GPIO PINS", [
        "PB0  (Vehicle RED LED)",
        "PB1  (Vehicle YELLOW LED)",
        "PB2  (Vehicle GREEN LED)",
        "PB4  (Pedestrian RED)",
        "PB5  (Pedestrian GREEN)",
        "PA0  (Pedestrian Request)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)

    svg += """
    <g transform="translate(420, 100)">
        <rect width="90" height="200" rx="10" fill="#1E293B" stroke="#0F172A" stroke-width="2"/>
        <text x="45" y="22" fill="#E2E8F0" font-family="monospace" font-size="9" text-anchor="middle">VEHICLE</text>
        <circle cx="45" cy="55" r="16" fill="#EF4444" stroke="#991B1B" stroke-width="2"/>
        <text x="45" y="78" fill="#FFFFFF" font-family="monospace" font-size="7" text-anchor="middle">PB0</text>
        <circle cx="45" cy="105" r="16" fill="#F59E0B" stroke="#B45309" stroke-width="2"/>
        <text x="45" y="128" fill="#FFFFFF" font-family="monospace" font-size="7" text-anchor="middle">PB1</text>
        <circle cx="45" cy="155" r="16" fill="#22C55E" stroke="#15803D" stroke-width="2"/>
        <text x="45" y="178" fill="#FFFFFF" font-family="monospace" font-size="7" text-anchor="middle">PB2</text>
    </g>
    <g transform="translate(560, 100)">
        <rect width="80" height="140" rx="10" fill="#1E293B" stroke="#0F172A" stroke-width="2"/>
        <text x="40" y="22" fill="#E2E8F0" font-family="monospace" font-size="9" text-anchor="middle">PEDESTRIAN</text>
        <circle cx="40" cy="55" r="16" fill="#EF4444"/>
        <text x="40" y="78" fill="#FFFFFF" font-family="monospace" font-size="7" text-anchor="middle">PB4</text>
        <circle cx="40" cy="105" r="16" fill="#22C55E"/>
        <text x="40" y="128" fill="#FFFFFF" font-family="monospace" font-size="7" text-anchor="middle">PB5</text>
    </g>
    """
    svg += draw_button(600, 300, "Walk Request (PA0)")

    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 420, 155, "#EF4444", "PB0")
    svg += draw_wire(pins["PB1"][0], pins["PB1"][1], 420, 205, "#F59E0B", "PB1")
    svg += draw_wire(pins["PB2"][0], pins["PB2"][1], 420, 255, "#22C55E", "PB2")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 576, 300, "#38BDF8", "PA0")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy08-traffic-light.svg", "w") as f:
        f.write(svg)

# 14. ASSIGNMENT 9: PWM LED DIMMER (PA0 + PB0)
def gen_diy09_svg():
    svg = svg_header(800, 480, "Assignment 9: PWM LED Dimmer (ADC PA0 & TIM3_CH3 PB0)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32F446RE PINS", [
        "PA0  (ADC1_IN0 Potentiometer)",
        "PB0  (TIM3_CH3 PWM LED Out)",
        "3.3V (VDD Supply)",
        "GND  (Common Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)

    svg += """
    <g transform="translate(430, 200)">
        <rect x="-30" y="-30" width="60" height="60" rx="6" fill="#0284C7"/>
        <circle cx="0" cy="0" r="18" fill="#E2E8F0" stroke="#94A3B8"/>
        <circle cx="-18" cy="38" r="4" fill="#EF4444"/><text x="-18" y="52" fill="#EF4444" font-family="monospace" font-size="8" text-anchor="middle">3.3V</text>
        <circle cx="0" cy="38" r="4" fill="#F59E0B"/><text x="0" y="52" fill="#F59E0B" font-family="monospace" font-size="8" text-anchor="middle">PA0</text>
        <circle cx="18" cy="38" r="4" fill="#0F172A"/><text x="18" y="52" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">GND</text>
        <text x="0" y="-36" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">10k&Omega; POT</text>
    </g>
    """
    svg += draw_led(580, 200, "#38BDF8", "PB0 PWM LED")
    svg += draw_resistor(660, 200, "330 &Omega;")

    svg += draw_wire(pins["3.3V"][0], pins["3.3V"][1], 412, 238, "#EF4444", "3.3V")
    svg += draw_wire(pins["PA0"][0], pins["PA0"][1], 430, 238, "#F59E0B", "PA0 Wiper")
    svg += draw_wire(pins["GND"][0], pins["GND"][1], 448, 238, "#0F172A", "GND")

    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 574, 200, "#38BDF8", "PB0 PWM")
    svg += draw_wire(685, 200, pins["GND"][0], pins["GND"][1], "#0F172A", "")

    svg += """
    <g transform="translate(60, 415)">
        <rect width="680" height="50" rx="6" fill="#1E293B"/>
        <text x="20" y="30" fill="#38BDF8" font-family="monospace" font-size="11" font-weight="bold">REGISTER MAPPING:</text>
        <text x="160" y="30" fill="#F8FAFC" font-family="monospace" font-size="11">ADC1_IN0 (PA0) &rarr; TIM3-&gt;CCR3 Duty Cycle &bull; 1000 Hz PWM Carrier</text>
    </g>
    """
    svg += svg_footer()
    with open(f"{OUT_DIR}/diy09-pwm-dimmer.svg", "w") as f:
        f.write(svg)

# 15. ASSIGNMENT 10: RGB LED MIXER (PB4, PB5, PB0, PA0, PC13)
def gen_diy10_svg():
    svg = svg_header(800, 480, "Assignment 10: RGB Color Mixer (PB4, PB5, PB0, PA0)")
    hdr_svg, pins = draw_header_block(60, 80, 230, 320, "STM32 TIMER & ADC", [
        "PB4  (TIM3_CH1 &bull; Red PWM)",
        "PB5  (TIM3_CH2 &bull; Green PWM)",
        "PB0  (TIM3_CH3 &bull; Blue PWM)",
        "PA0  (ADC1_IN0 &bull; Dimmer Pot)",
        "PC13 (GPIO_IN  &bull; Palette B1)",
        "GND  (Common Cathode Ground)"
    ])
    svg += hdr_svg
    svg += draw_breadboard(340, 80, 400, 320)

    svg += """
    <g transform="translate(560, 180)">
        <circle cx="0" cy="0" r="22" fill="#E2E8F0" stroke="#0F172A" stroke-width="2"/>
        <circle cx="-6" cy="-6" r="6" fill="#EF4444" opacity="0.8"/>
        <circle cx="6" cy="-6" r="6" fill="#22C55E" opacity="0.8"/>
        <circle cx="0" cy="6" r="6" fill="#3B82F6" opacity="0.8"/>
        <text x="0" y="-28" fill="#0F172A" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">RGB LED (CC)</text>
        <circle cx="-15" cy="30" r="3" fill="#EF4444"/><text x="-15" y="42" fill="#EF4444" font-family="monospace" font-size="8" text-anchor="middle">PB4(R)</text>
        <circle cx="-5" cy="30" r="3" fill="#0F172A"/><text x="-5" y="42" fill="#0F172A" font-family="monospace" font-size="8" text-anchor="middle">GND</text>
        <circle cx="5" cy="30" r="3" fill="#22C55E"/><text x="5" y="42" fill="#22C55E" font-family="monospace" font-size="8" text-anchor="middle">PB5(G)</text>
        <circle cx="15" cy="30" r="3" fill="#3B82F6"/><text x="15" y="42" fill="#3B82F6" font-family="monospace" font-size="8" text-anchor="middle">PB0(B)</text>
    </g>
    """
    svg += draw_resistor(450, 150, "220&Omega; PB4")
    svg += draw_resistor(450, 180, "330&Omega; PB5")
    svg += draw_resistor(450, 210, "220&Omega; PB0")

    svg += draw_wire(pins["PB4"][0], pins["PB4"][1], 425, 150, "#EF4444")
    svg += draw_wire(475, 150, 545, 210, "#EF4444")
    svg += draw_wire(pins["PB5"][0], pins["PB5"][1], 425, 180, "#22C55E")
    svg += draw_wire(475, 180, 565, 210, "#22C55E")
    svg += draw_wire(pins["PB0"][0], pins["PB0"][1], 425, 210, "#3B82F6")
    svg += draw_wire(475, 210, 575, 210, "#3B82F6")

    svg += draw_wire(555, 210, pins["GND"][0], pins["GND"][1], "#0F172A", "Cathode GND")

    svg += svg_footer()
    with open(f"{OUT_DIR}/diy10-rgb-color-mixer.svg", "w") as f:
        f.write(svg)

def main():
    print("Generating all custom STM32 Fritzing-style vector SVGs...")
    gen_nucleo_pinout()
    gen_basic_kit_svg()
    gen_display_sensing_svg()
    gen_detection_modules_svg()
    gen_can_circuit_svg()
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
