"""Builds the static digital book 'Programming STM32 with C' into static/stm32_c_firmware/.

Chapters are written in a small markdown dialect (see stm32_c_book_src/*.md):
  ## 3.1 Title            -> section heading (id sec-3-1)
  ### Title               -> sub heading
  ```c Title ... ```      -> code block with title
  > [tip|warn|hw] Header | body   -> callout
  | a | b |  table rows   -> table (second row is the separator)
  ::lab Title + "- item"  -> checklist card
  ::quiz Title with Q:/-/+ lines -> quiz
"""
import html
import os
import re
import shutil

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
ZEPHYR = os.path.join(ROOT, 'static', 'zephyr_stm32')
OUT = os.path.join(ROOT, 'static', 'stm32_rust_firmware')
COURSE_URL = '/courses/embedded-rust-with-stm32/'
BOOK_TITLE = 'Embedded Rust with STM32'
BOOK_SUB = 'From Bare-Metal Firmware to Safe Embedded Products'

CHAPTERS = [
    dict(n=1, file='ch01-why-embedded-rust', md='ch01.md', badge='Foundations', short='Why Embedded Rust',
         title='Why Embedded Rust? From C Pitfalls to Rust Guarantees', time=30, tags='rust c safety pointer lifetime buffer bounds shared mutable state error paths memory safety',
         lead="See why dangling pointers, buffer overruns, data races and ignored errors in C map directly onto Rust's compile-time guarantees.",
         lab='Compare C and Rust bugs', topics='Why Embedded Rust'),
    dict(n=2, file='ch02-rust-for-c-developers', md='ch02.md', badge='Language', short='Rust for C Developers',
         title='Rust for C Developers', time=40, tags='rust variables types let mut functions integer overflow arrays slices wrapping saturating',
         lead='Variables, types, functions and control flow, translated from the C you already know, with embedded-specific integer and array behaviour.',
         lab='Port a C routine', topics='Rust for C Developers'),
    dict(n=3, file='ch03-ownership', md='ch03.md', badge='Language', short='Ownership',
         title='Ownership: Memory Safety Without a Garbage Collector', time=40, tags='ownership move copy drop scope raii',
         lead='The core idea of Rust: every value has one owner. Learn moves, copies and Drop, and why hardware peripherals are owned values.',
         lab='Move semantics', topics='Ownership'),
    dict(n=4, file='ch04-borrowing', md='ch04.md', badge='Language', short='Borrowing',
         title='Borrowing & References', time=35, tags='borrow reference mutable immutable slice aliasing borrow checker',
         lead='Shared and mutable references, the aliasing rule, and how slices replace pointer-plus-length pairs.',
         lab='Fix borrow errors', topics='Borrowing'),
    dict(n=5, file='ch05-lifetimes', md='ch05.md', badge='Language', short='Lifetimes',
         title='Lifetimes & Their Embedded Implications', time=35, tags='lifetime static annotation dma buffer struct reference elision',
         lead="Understand lifetime annotations, 'static data and why DMA buffers and long-lived peripherals care about them.",
         lab='Lifetime puzzles', topics='Lifetimes'),
    dict(n=6, file='ch06-structs-enums', md='ch06.md', badge='Language', short='Structs & Enums',
         title='Structs & Enums: Modelling Hardware', time=40, tags='struct enum match newtype state machine type-state repr',
         lead='Model registers, states and messages with structs and enums, and let match enforce handling every case.',
         lab='Model a state machine', topics='Structs & Enums'),
    dict(n=7, file='ch07-traits-generics', md='ch07.md', badge='Language', short='Traits & Generics',
         title='Traits & Generics: Driver Abstraction', time=40, tags='trait generic embedded-hal impl bound monomorphization const generic',
         lead='Write code that works with any hardware that implements a trait, at zero run-time cost.',
         lab='Generic driver', topics='Traits & Generics'),
    dict(n=8, file='ch08-option-result', md='ch08.md', badge='Language', short='Option & Result',
         title='Option & Result: Error Handling', time=35, tags='option result question mark unwrap expect error enum map_err panic',
         lead='Replace NULL and error codes with Option and Result, and use the ? operator to propagate failure.',
         lab='Error handling refactor', topics='Option & Result'),
    dict(n=9, file='ch09-no-std', md='ch09.md', badge='Embedded', short='no_std',
         title='no_std: The Embedded Environment', time=35, tags='no_std core alloc heapless panic handler entry cortex-m-rt',
         lead='Learn what disappears without the standard library, what core provides, and how to get collections without a heap.',
         lab='First no_std crate', topics='no_std'),
    dict(n=10, file='ch10-cross-compilation', md='ch10.md', badge='Embedded', short='Cross Compilation',
         title='Cross Compilation for ARM', time=30, tags='cross compile target triple thumbv7em rustup cargo config build',
         lead='Targets, triples and the Cortex-M target family, and how cargo builds for a chip you are not running on.',
         lab='Build for the target', topics='Cross Compilation'),
    dict(n=11, file='ch11-linker-memory', md='ch11.md', badge='Embedded', short='Linker & Memory',
         title='Linker Scripts & Memory Layout', time=40, tags='linker memory.x flash ram sections vector table stack heap size',
         lead='How memory.x describes flash and SRAM, what the sections contain and how to check usage.',
         lab='Read the memory map', topics='Linker & Memory'),
    dict(n=12, file='ch12-toolchain', md='ch12.md', badge='Tools', short='Toolchain',
         title='The Embedded Rust Toolchain', time=35, tags='rustup cargo probe-rs flip-link defmt clippy cargo-binutils project template',
         lead='Install and configure rustup, targets, probe-rs, flip-link and defmt, and create a project from a template.',
         lab='Hello board', topics='Toolchain'),
    dict(n=13, file='ch13-gpio', md='ch13.md', badge='Peripherals', short='GPIO',
         title='GPIO: Your First Rust Firmware', time=40, tags='gpio pin typestate output input blink button pac hal split',
         lead='Pin type-states make misuse a compile error. Blink an LED and read a button with the HAL.',
         lab='Blink and button', topics='GPIO'),
    dict(n=14, file='ch14-interrupts', md='ch14.md', badge='Peripherals', short='Interrupts',
         title='Interrupts: ISR Architecture', time=45, tags='interrupt isr exti nvic critical section atomic mutex refcell rtic',
         lead='Share data between interrupts and main safely using atomics, critical sections and RTIC.',
         lab='Interrupt button', topics='Interrupts'),
    dict(n=15, file='ch15-timers-pwm', md='ch15.md', badge='Peripherals', short='Timers & PWM',
         title='Timers & PWM: Hardware Control', time=40, tags='timer pwm psc arr duty servo percent newtype counter',
         lead='Time bases, PWM output and typed duty cycles, with servo and motor examples.',
         lab='PWM control', topics='Timers & PWM'),
    dict(n=16, file='ch16-uart', md='ch16.md', badge='Peripherals', short='UART',
         title='UART: Serial Communication', time=40, tags='uart usart serial baud nb block queue heapless parser defmt',
         lead='Blocking and interrupt-driven UART, an SPSC queue between ISR and main, and a safe command parser.',
         lab='UART console', topics='UART'),
    dict(n=17, file='ch17-i2c', md='ch17.md', badge='Buses', short='I2C',
         title='I2C: Writing a Sensor Driver', time=45, tags='i2c bme280 embedded-hal driver generic bus shared error compensation',
         lead='Use the embedded-hal I2c trait to write a BME280 driver that is portable and testable.',
         lab='BME280 driver', topics='I2C'),
    dict(n=18, file='ch18-spi', md='ch18.md', badge='Buses', short='SPI',
         title='SPI: Displays & Storage', time=40, tags='spi spidevice flash display embedded-graphics sd chip select mode',
         lead='SPI modes, safe chip-select with SpiDevice, flash and SD storage and displays via embedded-graphics.',
         lab='SPI flash and display', topics='SPI'),
    dict(n=19, file='ch19-adc', md='ch19.md', badge='Peripherals', short='ADC',
         title='ADC: Sensor Acquisition', time=35, tags='adc analog sample time newtype units filter const generic vrefint',
         lead='Read analog signals, convert units with newtypes and filter noise with fixed-size averages.',
         lab='Analog acquisition', topics='ADC'),
    dict(n=20, file='ch20-can', md='ch20.md', badge='Buses', short='CAN',
         title='CAN: Automotive Communication', time=45, tags='can bxcan bit timing filter frame transceiver termination message loopback',
         lead='CAN bit timing, filters and typed frames, with honest notes on HAL support and alternatives.',
         lab='CAN loopback', topics='CAN'),
    dict(n=21, file='ch21-dma', md='ch21.md', badge='Peripherals', short='DMA',
         title='DMA: High-Performance Peripherals', time=45, tags='dma transfer buffer static singleton circular idle line ownership alignment ccm',
         lead="Let hardware move data while Rust's ownership rules keep the buffer safe.",
         lab='DMA transfers', topics='DMA'),
    dict(n=22, file='ch22-testing', md='ch22.md', badge='Quality', short='Testing',
         title='Testing Embedded Rust', time=40, tags='test unit host mock embedded-hal-mock defmt-test ci clippy hil',
         lead='Test logic on the host, drivers with mocks and hardware with defmt-test, and run it all in CI.',
         lab='Test your firmware', topics='Testing'),
    dict(n=23, file='ch23-debugging', md='ch23.md', badge='Quality', short='Debugging',
         title='Debugging with ST-LINK & GDB', time=45, tags='debug probe-rs gdb openocd rtt defmt panic hardfault flip-link logic analyzer',
         lead='Debug with probe-rs, defmt, GDB and the logic analyzer, and understand panics and hard faults in Rust.',
         lab='Debug session', topics='Debugging'),
    dict(n=24, file='ch24-driver-design', md='ch24.md', badge='Design', short='Driver Design',
         title='Rust Driver Design', time=40, tags='driver crate generic type-state forbid unsafe async publish documentation',
         lead='Design reusable, portable, documented driver crates with typed errors and type-state configuration.',
         lab='Write a driver', topics='Driver Design'),
    dict(n=25, file='ch25-architecture', md='ch25.md', badge='Design', short='Architecture',
         title='Embedded Architecture: Modules & Interfaces', time=45, tags='architecture workspace crates rtic embassy channel state machine watchdog panic policy',
         lead='Structure firmware as a workspace with clear boundaries, message passing, state machines and reliability features.',
         lab='Restructure project', topics='Architecture'),
    dict(n=26, file='ch26-capstone', md='ch26.md', badge='Capstone', short='Capstone',
         title='Capstone: A Safe Embedded Product', time=60, tags='capstone fan controller can uart watchdog fail-safe test plan requirements',
         lead='Build a cooling-fan and environment controller node with I2C, PWM, ADC, UART, CAN, watchdog and a full test plan.',
         lab='Capstone delivery', topics='Capstone'),
]
REFERENCES = dict(file='references', title='References & Further Reading', short='References')


def esc(s):
    return html.escape(s, quote=False)


def inline(s):
    s = esc(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    return s


def code_block(title, lang, body):
    return f'''
        <div class="code-container">
          <div class="code-header">
            <div class="code-title">
              <span class="code-lang-badge">{esc(lang.upper())}</span>
              <span>{esc(title)}</span>
            </div>
            <button class="code-copy-btn">Copy</button>
          </div>
          <div class="code-body">
            <pre><code>{esc(body)}</code></pre>
          </div>
        </div>
'''


CALLOUTS = {'tip': 'callout-tip', 'warn': 'callout-warning', 'hw': 'callout-hardware'}


def callout(kind, header, body):
    return f'''
        <div class="callout {CALLOUTS[kind]}">
          <div class="callout-header">{inline(header)}</div>
          <div class="callout-body">{inline(body)}</div>
        </div>
'''


def table(lines):
    cells = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in lines]
    head, rows = cells[0], cells[2:]
    out = ['<div style="overflow-x:auto;margin:1.5rem 0;"><table style="width:100%;border-collapse:collapse;font-size:0.875rem;">',
           '<thead><tr style="border-bottom:2px solid var(--border-strong);text-align:left;">']
    out += [f'<th style="padding:0.6rem;">{inline(h)}</th>' for h in head]
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr style="border-bottom:1px solid var(--border-subtle);">')
        out += [f'<td style="padding:0.6rem;vertical-align:top;">{inline(c)}</td>' for c in r]
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def lab_block(ch, title, items):
    lis = ''.join(f'''
            <li class="checklist-item">
              <input type="checkbox" class="checklist-checkbox" id="l{ch}-step{i}">
              <label for="l{ch}-step{i}" class="checklist-text">{inline(t)}</label>
            </li>''' for i, t in enumerate(items, 1))
    return f'''
        <div class="lab-checklist-card">
          <div class="lab-title-row">
            <h3>📋 {esc(title)}</h3>
            <span class="progress-pill">Local Storage Tracked</span>
          </div>
          <ul class="checklist-items">{lis}
          </ul>
        </div>
'''


def quiz_block(sec_id, title, questions):
    qs = []
    for i, (q, opts) in enumerate(questions, 1):
        btns = []
        for j, (ok, text, expl) in enumerate(opts):
            letter = 'ABCD'[j]
            btns.append(f'''
              <button class="quiz-option-btn" data-correct="{'true' if ok else 'false'}" data-explanation="{html.escape(expl, quote=True)}">
                {letter}) {inline(text)}
              </button>''')
        qs.append(f'''
          <div class="quiz-question-box">
            <div class="quiz-question-text">{i}. {inline(q)}</div>
            <div class="quiz-options">{''.join(btns)}
            </div>
            <div class="quiz-feedback"></div>
          </div>''')
    return f'''
        <div class="quiz-container" id="{sec_id}">
          <div class="quiz-header"><span>🧠 {esc(title)}</span></div>{''.join(qs)}
        </div>
'''


def render_md(text, ch):
    """Returns (html, [(sec_id, title)])."""
    lines = text.split('\n')
    out, secs = [], []
    i = 0
    quiz_n = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith('```'):
            parts = ln[3:].strip().split(' ', 1)
            lang, title = parts[0] or 'c', (parts[1] if len(parts) > 1 else 'Example')
            i += 1
            body = []
            while not lines[i].startswith('```'):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(code_block(title, lang, '\n'.join(body)))
        elif ln.startswith('## '):
            title = ln[3:].strip()
            m = re.match(r'(\d+)\.(\d+)', title)
            sid = f'sec-{m.group(1)}-{m.group(2)}'
            secs.append((sid, title))
            out.append(f'<h2 id="{sid}">{inline(title)}</h2>')
            i += 1
        elif ln.startswith('### '):
            out.append(f'<h3>{inline(ln[4:].strip())}</h3>')
            i += 1
        elif ln.startswith('> ['):
            m = re.match(r'> \[(tip|warn|hw)\] (.*?) \| (.*)', ln)
            out.append(callout(m.group(1), m.group(2), m.group(3)))
            i += 1
        elif ln.startswith('|'):
            tl = []
            while i < len(lines) and lines[i].startswith('|'):
                tl.append(lines[i])
                i += 1
            out.append(table(tl))
        elif ln.startswith('::lab '):
            title = ln[6:].strip()
            i += 1
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(lines[i][2:])
                i += 1
            out.append(lab_block(ch, title, items))
        elif ln.startswith('::quiz '):
            title = ln[7:].strip()
            i += 1
            qs = []
            while i < len(lines) and (lines[i].startswith('Q:') or lines[i][:2] in ('- ', '+ ')):
                if lines[i].startswith('Q:'):
                    qs.append((lines[i][2:].strip(), []))
                else:
                    ok = lines[i][0] == '+'
                    body = lines[i][2:]
                    txt, _, expl = body.partition(' | ')
                    qs[-1][1].append((ok, txt, expl or ('Correct!' if ok else 'Incorrect.')))
                i += 1
            quiz_n += 1
            sid = f'sec-{ch}-quiz'
            secs.append((sid, f'{ch}.Q Knowledge Check'))
            out.append(quiz_block(sid, title, qs))
        elif ln.startswith('- '):
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(lines[i][2:])
                i += 1
            out.append('<ul>' + ''.join(f'<li>{inline(t)}</li>' for t in items) + '</ul>')
        elif re.match(r'\d+\. ', ln):
            items = []
            while i < len(lines) and re.match(r'\d+\. ', lines[i]):
                items.append(re.sub(r'^\d+\. ', '', lines[i]))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{inline(t)}</li>' for t in items) + '</ol>')
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r'(```|## |### |> \[|\||::|- |\d+\. )', lines[i]):
                para.append(lines[i])
                i += 1
            out.append(f'<p>{inline(" ".join(para))}</p>')
    return '\n'.join(out), secs


def short_sec(t):
    t = t.strip()
    return t if len(t) <= 34 else t[:32].rstrip() + '…'


def head(prefix, title, desc):
    return f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{html.escape(desc, quote=True)}">
  <title>{esc(title)} | {BOOK_TITLE}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{prefix}css/style.css?v=2">
</head>
'''


def header(prefix):
    return f'''
  <div id="reading-progress-bar"></div>
  <header class="site-header">
    <div class="header-left">
      <button class="sidebar-toggle-btn icon-btn" id="sidebar-toggle" aria-label="Toggle Sidebar">
        <svg width="20" height="20" viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
      </button>
      <a href="{prefix}index.html" class="brand-badge">
        <img src="{prefix}assets/images/acl_logo.png" alt="Apt Computing Labs" class="brand-logo-img">
        <div>
          <span class="brand-title">STM32 + Rust</span>
          <span class="brand-subtitle">Apt Computing Labs</span>
        </div>
      </a>
    </div>
    <div class="header-center">
      <div class="search-box-wrapper">
        <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input type="text" id="header-search" class="search-input" placeholder="Search chapters, ownership, GPIO, DMA, defmt..." readonly>
        <span class="search-shortcut">⌘K</span>
      </div>
    </div>
    <div class="header-right">
      <div class="board-select-container">
        <span class="board-select-label">Board:</span>
        <span class="board-select-label" style="font-weight:700;">NUCLEO-F446RE (Cortex-M4F)</span>
      </div>
      <button id="theme-toggle-btn" class="icon-btn" aria-label="Toggle Theme"></button>
      <a href="{COURSE_URL}" class="back-link" style="display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:20px;font-size:0.75rem;font-weight:600;color:var(--text-secondary);border:1px solid var(--border-subtle);text-decoration:none;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
        <span>Course Overview</span>
      </a>
    </div>
  </header>
'''


def sidebar(prefix, current, all_secs):
    cp = '' if prefix else 'chapters/'
    idx = f'{prefix}index.html'
    items = [f'''
        <div class="nav-module-group">
          <a href="{idx}" class="chapter-nav-item">
            <span class="chapter-check-icon">📖</span>
            <span class="chapter-title-text">Course Overview &amp; Syllabus</span>
          </a>
        </div>''']
    for c in CHAPTERS:
        href = f'{cp}{c["file"]}.html'
        subs = ''.join(f'<li><a href="{href}#{sid}" class="subtopic-link"><span class="subtopic-bullet">▸</span> {esc(short_sec(t))}</a></li>'
                       for sid, t in all_secs[c['n']])
        items.append(f'''
        <div class="nav-module-group">
          <button class="module-title-btn">
            <div class="module-title-left">
              <span class="module-badge">R{c['n']}</span>
              <span>{esc(c['badge'])}</span>
            </div>
            <span class="module-chevron">▼</span>
          </button>
          <ul class="chapter-sublist">
            <li>
              <a href="{href}" class="chapter-nav-item">
                <span class="chapter-check-icon">●</span>
                <span class="chapter-title-text">Ch {c['n']}: {esc(c['short'])}</span>
                <span class="chapter-read-time">{c['time']}m</span>
              </a>
              <ul class="subtopic-list">{subs}</ul>
            </li>
          </ul>
        </div>''')
    rhref = f'{cp}{REFERENCES["file"]}.html'
    items.append(f'''
        <div class="nav-module-group">
          <a href="{rhref}" class="chapter-nav-item">
            <span class="chapter-check-icon">📚</span>
            <span class="chapter-title-text">{REFERENCES['title']}</span>
          </a>
        </div>''')
    return f'''
  <div class="app-container">
    <aside class="book-sidebar" id="book-sidebar">
      <div class="sidebar-header">
        <div class="toc-heading">
          <span>Curriculum Chapters</span>
          <span class="progress-pill">{current}</span>
        </div>
      </div>
      <nav class="sidebar-nav">{''.join(items)}
      </nav>
    </aside>
'''


def footer(prefix):
    return f'''
        <footer class="site-footer">
          <div class="footer-container">
            <div class="footer-left">
              <img src="{prefix}assets/images/acl_logo.png" alt="Apt Computing Labs Logo" class="footer-logo">
              <div class="footer-brand-text">
                <span class="footer-brand-name">Apt Computing Labs</span>
                <span class="footer-tagline">Where Knowledge Meets Innovation</span>
              </div>
            </div>
            <div class="footer-right">
              <div>© 2026 <strong>Apt Computing Labs</strong>. All Rights Reserved.</div>
              <div style="font-size:0.75rem;margin-top:0.25rem;">{BOOK_TITLE} • Confidential &amp; Proprietary Educational Material</div>
            </div>
          </div>
        </footer>
'''


def tail(prefix):
    return f'''
  </div>
  <div class="modal-overlay" id="search-modal-overlay">
    <div class="search-modal">
      <div class="modal-search-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input type="text" id="modal-search-input" class="modal-search-input" placeholder="Search chapters, ownership, GPIO, DMA, defmt...">
        <button id="close-search-modal" class="icon-btn" style="border:none;">✕</button>
      </div>
      <ul id="search-results" class="search-results-list"></ul>
    </div>
  </div>
  <script src="{prefix}js/book.js?v=2"></script>
</body>
</html>
'''


def page(prefix, chap_id, title, desc, current, all_secs, topbar, article):
    return (head(prefix, title, desc) + f'<body data-chapter-id="{chap_id}">' + header(prefix)
            + sidebar(prefix, current, all_secs)
            + f'''
    <main class="main-content">
{topbar}
      <article class="prose">
{article}
{footer(prefix)}
      </article>
    </main>''' + tail(prefix))


def setup_assets():
    for d in ('css', 'js'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    shutil.copy(os.path.join(ZEPHYR, 'css', 'style.css'), os.path.join(OUT, 'css', 'style.css'))
    os.makedirs(os.path.join(OUT, 'assets', 'images'), exist_ok=True)
    shutil.copy(os.path.join(ZEPHYR, 'assets', 'images', 'acl_logo.png'), os.path.join(OUT, 'assets', 'images', 'acl_logo.png'))
    js = open(os.path.join(ZEPHYR, 'js', 'book.js'), encoding='utf-8').read()
    js = js.replace('Mastering Zephyr RTOS on STM32 - Shared Book Client Script', f'{BOOK_TITLE} - Shared Book Client Script')
    js = js.replace("'zephyr_", "'stm32rs_")
    start = js.index('const CHAPTERS_INDEX = [')
    end = js.index('];', start) + 2
    entries = ['    { title: "Course Overview & Learning Roadmap", url: "index.html", tags: "intro overview roadmap stm32 c firmware" }']
    for c in CHAPTERS:
        entries.append('    { title: "Chapter %d: %s", url: "chapters/%s.html", tags: "%s" }' % (c['n'], c['title'].replace('"', "'"), c['file'], c['tags']))
    entries.append('    { title: "References & Further Reading", url: "chapters/references.html", tags: "references datasheet reference manual rm0390 hal user manual um1725 arm" }')
    js = js[:start] + 'const CHAPTERS_INDEX = [\n' + ',\n'.join(entries) + '\n  ];' + js[end:]
    hal = ("    code = code.replace(/\\b(?:__HAL_\\w+|HAL_\\w+|LL_\\w+|NVIC_\\w+|__NVIC_\\w+|__disable_irq|__enable_irq|__WFI|__NOP|SysTick_Config)\\b/g, "
           "m => saveToken(`<span class=\"token-zephyr\">${m}</span>`));\n")
    marker = "    code = code.replace(zephyrApis, m => saveToken(`<span class=\"token-zephyr\">${m}</span>`));\n"
    assert marker in js
    js = js.replace(marker, marker + hal)
    open(os.path.join(OUT, 'js', 'book.js'), 'w', encoding='utf-8').write(js)


def build():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, 'chapters'))
    setup_assets()

    rendered, all_secs = {}, {}
    for c in CHAPTERS:
        text = open(os.path.join(SRC, c['md']), encoding='utf-8').read()
        rendered[c['n']], all_secs[c['n']] = render_md(text, c['n'])

    for k, c in enumerate(CHAPTERS):
        prev = CHAPTERS[k - 1] if k else None
        nxt = CHAPTERS[k + 1] if k + 1 < len(CHAPTERS) else None
        prev_href = f'{prev["file"]}.html' if prev else '../index.html'
        nxt_href = f'{nxt["file"]}.html' if nxt else f'{REFERENCES["file"]}.html'
        topbar = f'''      <div class="content-top-bar">
        <div class="breadcrumbs"><a href="../index.html">Home</a><span>/</span><span>Chapter {c['n']}</span><span>/</span><span>{esc(c['short'])}</span></div>
        <div class="chapter-actions">
          <a href="{prev_href}" class="btn-secondary">← {'Chapter %d' % prev['n'] if prev else 'Overview'}</a>
          <a href="{nxt_href}" class="btn-primary">{'Chapter %d' % nxt['n'] if nxt else 'References'} →</a>
        </div>
      </div>'''
        bottom = f'''
        <div class="chapter-bottom-nav">
          <a href="{prev_href}" class="nav-card"><span class="nav-card-dir">← Previous</span><span class="nav-card-title">{esc(('Chapter %d: %s' % (prev['n'], prev['title'])) if prev else 'Course Overview')}</span></a>
          <a href="{nxt_href}" class="nav-card" style="text-align:right;"><span class="nav-card-dir">Next →</span><span class="nav-card-title">{esc(('Chapter %d: %s' % (nxt['n'], nxt['title'])) if nxt else REFERENCES['title'])}</span></a>
        </div>'''
        article = f'''        <div class="chapter-hero">
          <span class="chapter-module-tag">Chapter {c['n']:02d} • {esc(c['badge'])}</span>
          <h1 class="chapter-h1">{esc(c['title'])}</h1>
          <p class="chapter-lead">{esc(c['lead'])}</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Reading Time: {c['time']} minutes</div>
            <div class="meta-item">🛠️ Lab: {esc(c['lab'])}</div>
            <div class="meta-item">🎯 Topics: {esc(c['topics'])}</div>
          </div>
        </div>
{rendered[c['n']]}
{bottom}'''
        out = page('../', c['file'], c['title'], c['lead'], f'Ch {c["n"]} of {len(CHAPTERS)}', all_secs, topbar, article)
        open(os.path.join(OUT, 'chapters', c['file'] + '.html'), 'w', encoding='utf-8').write(out)

    # references
    ref_md = open(os.path.join(SRC, 'references.md'), encoding='utf-8').read()
    ref_html, _ = render_md(ref_md, 0)
    last = CHAPTERS[-1]
    topbar = f'''      <div class="content-top-bar">
        <div class="breadcrumbs"><a href="../index.html">Home</a><span>/</span><span>References</span></div>
        <div class="chapter-actions"><a href="{last['file']}.html" class="btn-secondary">← Chapter {last['n']}</a></div>
      </div>'''
    article = f'''        <div class="chapter-hero">
          <span class="chapter-module-tag">Appendix</span>
          <h1 class="chapter-h1">{REFERENCES['title']}</h1>
          <p class="chapter-lead">The books, crates and tools you will keep open while writing embedded Rust for STM32.</p>
        </div>
{ref_html}'''
    open(os.path.join(OUT, 'chapters', 'references.html'), 'w', encoding='utf-8').write(
        page('../', 'references', REFERENCES['title'], 'References', 'Appendix', all_secs, topbar, article))

    # index
    cards = ''.join(f'''
          <a href="chapters/{c['file']}.html" class="nav-card" style="text-decoration:none;">
            <span class="nav-card-dir">Chapter {c['n']} • {c['time']} min</span>
            <span class="nav-card-title">{esc(c['title'])}</span>
            <span style="display:block;font-size:0.8rem;margin-top:0.5rem;color:var(--text-muted);">{esc(c['lead'])}</span>
          </a>''' for c in CHAPTERS)
    topbar = '''      <div class="content-top-bar">
        <div class="breadcrumbs"><a href="index.html">Home</a><span>/</span><span>Course Roadmap</span></div>
        <div class="chapter-actions"><a href="chapters/ch01-why-embedded-rust.html" class="btn-primary">Start Chapter 1 →</a></div>
      </div>'''
    covers = ['Why Rust', 'Ownership', 'Borrowing', 'Lifetimes', 'Enums & Structs', 'Traits & Generics', 'Option & Result', 'no_std', 'Cross compilation', 'Linker & memory',
              'Toolchain', 'GPIO', 'Interrupts', 'Timers/PWM', 'UART', 'I²C', 'SPI', 'ADC', 'CAN', 'DMA', 'Testing', 'Debugging', 'Driver design', 'Architecture', 'Capstone']
    pills = ''.join(f'<span class="progress-pill" style="margin:0.2rem;">{c}</span>' for c in covers)
    article = f'''        <div class="chapter-hero">
          <span class="chapter-module-tag">Digital Book &amp; Laboratory Course</span>
          <h1 class="chapter-h1">{BOOK_TITLE}: {BOOK_SUB}</h1>
          <p class="chapter-lead">Book 2 for engineers who already know STM32 and C. This is not Book 1 retyped in Rust: it teaches you to think differently - ownership, borrowing, lifetimes, Option and Result, no_std - and then applies that thinking to every peripheral, to testing and to product architecture, ending in a safe embedded capstone.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">🎯 Level: Intermediate (prerequisite: STM32 with C or equivalent)</div>
            <div class="meta-item">⏱️ {len(CHAPTERS)} chapters • lab and quiz in every chapter • capstone product</div>
            <div class="meta-item">⚡ Reference board: NUCLEO-F446RE</div>
          </div>
        </div>
        <h2 id="sec-0-1">What You Will Learn</h2>
        <div style="margin:1rem 0;">{pills}</div>
        <h2 id="sec-0-2">Hardware You Need</h2>
        <ul>
          <li>NUCLEO-F446RE board (ST-LINK/V2-1, user LED on PA5, user button on PC13).</li>
          <li>Breadboard, jumper wires, a 10 kΩ potentiometer, LEDs, resistors, a push button.</li>
          <li>BME280 sensor, SPI flash or display, SG90 servo or small fan with MOSFET driver.</li>
          <li>CAN transceiver (e.g. SN65HVD230) and 120 Ω termination - or a second board.</li>
          <li>An 8-channel USB logic analyzer (sigrok/PulseView compatible).</li>
        </ul>
        <h2 id="sec-0-3">Software You Need</h2>
        <ul>
          <li>rustup with the <code>thumbv7em-none-eabihf</code> target, cargo, clippy and rustfmt.</li>
          <li>probe-rs, flip-link and cargo-binutils; optionally OpenOCD and arm-none-eabi-gdb.</li>
          <li>VS Code with rust-analyzer and the probe-rs debugger extension.</li>
          <li>A serial terminal and PulseView.</li>
        </ul>
        <h2 id="sec-0-4">Chapter Roadmap</h2>
        <div class="chapter-bottom-nav" style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">{cards}
        </div>
        <div class="callout callout-tip" style="margin-top:2rem;">
          <div class="callout-header">💡 How to use this book</div>
          <div class="callout-body">Type the code yourself, build it, and tick the lab checklist at the end of each chapter. Every chapter ends with a quiz - if you cannot answer a question, re-read that section before moving on.</div>
        </div>'''
    all_secs[0] = []
    open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8').write(
        page('', 'index', 'Course Overview', f'{BOOK_TITLE}: {BOOK_SUB}', 'Overview', all_secs, topbar, article))
    print('Built book into', OUT)


if __name__ == '__main__':
    build()
