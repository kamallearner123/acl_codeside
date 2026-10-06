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
OUT = os.path.join(ROOT, 'static', 'cortexm_architecture')
COURSE_URL = '/courses/arm-cortex-m-architecture/'
BOOK_TITLE = 'ARM Cortex-M Architecture'
BOOK_SUB = 'From CPU Core to STM32 Firmware'

CHAPTERS = [
    dict(n=1, file='ch01-embedded-processor-fundamentals', md='ch01.md', badge='Core', short='Processor Basics',
         title='Embedded Processor Fundamentals', time=30, tags='embedded processor fundamentals cortex-m arm',
         lead='A vendor-neutral look at Processor Basics, with the STM32F446 as a worked example.',
         lab='Lab 1', topics='Embedded Processor Fundamentals'),
    dict(n=2, file='ch02-arm-architecture-overview', md='ch02.md', badge='Core', short='Arm Overview',
         title='ARM Architecture Overview', time=30, tags='arm architecture overview cortex-m arm',
         lead='A vendor-neutral look at Arm Overview, with the STM32F446 as a worked example.',
         lab='Lab 2', topics='ARM Architecture Overview'),
    dict(n=3, file='ch03-the-cortex-m-family', md='ch03.md', badge='Core', short='Cortex-M Family',
         title='The Cortex-M Family', time=30, tags='the cortex m family cortex-m arm',
         lead='A vendor-neutral look at Cortex-M Family, with the STM32F446 as a worked example.',
         lab='Lab 3', topics='The Cortex-M Family'),
    dict(n=4, file='ch04-cortex-m0-m0-m3-m4-m7-and-m33', md='ch04.md', badge='Core', short='Core Variants',
         title='Cortex-M0, M0+, M3, M4, M7 and M33', time=30, tags='cortex m0  m0  m3  m4  m7 and m33 cortex-m arm',
         lead='A vendor-neutral look at Core Variants, with the STM32F446 as a worked example.',
         lab='Lab 4', topics='Cortex-M0, M0+, M3, M4, M7 and M33'),
    dict(n=5, file='ch05-the-programmer-s-model', md='ch05.md', badge='Core', short="Programmer's Model",
         title="The Programmer's Model", time=30, tags='the programmer s model cortex-m arm',
         lead="A vendor-neutral look at Programmer's Model, with the STM32F446 as a worked example.",
         lab='Lab 5', topics="The Programmer's Model"),
    dict(n=6, file='ch06-registers', md='ch06.md', badge='Core', short='Registers',
         title='Registers', time=30, tags='registers cortex-m arm',
         lead='A vendor-neutral look at Registers, with the STM32F446 as a worked example.',
         lab='Lab 6', topics='Registers'),
    dict(n=7, file='ch07-r0-r12-sp-lr-pc-and-xpsr', md='ch07.md', badge='Core', short='R0-R12/SP/LR/PC/xPSR',
         title='R0-R12, SP, LR, PC and xPSR', time=30, tags='r0 r12  sp  lr  pc and xpsr cortex-m arm',
         lead='A vendor-neutral look at R0-R12/SP/LR/PC/xPSR, with the STM32F446 as a worked example.',
         lab='Lab 7', topics='R0-R12, SP, LR, PC and xPSR'),
    dict(n=8, file='ch08-instruction-set-basics', md='ch08.md', badge='Core', short='Instruction Set',
         title='Instruction Set Basics', time=30, tags='instruction set basics cortex-m arm',
         lead='A vendor-neutral look at Instruction Set, with the STM32F446 as a worked example.',
         lab='Lab 8', topics='Instruction Set Basics'),
    dict(n=9, file='ch09-thumb-and-thumb-2', md='ch09.md', badge='Core', short='Thumb/Thumb-2',
         title='Thumb and Thumb-2', time=30, tags='thumb and thumb 2 cortex-m arm',
         lead='A vendor-neutral look at Thumb/Thumb-2, with the STM32F446 as a worked example.',
         lab='Lab 9', topics='Thumb and Thumb-2'),
    dict(n=10, file='ch10-the-memory-map', md='ch10.md', badge='Memory', short='Memory Map',
         title='The Memory Map', time=30, tags='the memory map cortex-m arm',
         lead='A vendor-neutral look at Memory Map, with the STM32F446 as a worked example.',
         lab='Lab 10', topics='The Memory Map'),
    dict(n=11, file='ch11-flash-memory', md='ch11.md', badge='Memory', short='Flash',
         title='Flash Memory', time=30, tags='flash memory cortex-m arm',
         lead='A vendor-neutral look at Flash, with the STM32F446 as a worked example.',
         lab='Lab 11', topics='Flash Memory'),
    dict(n=12, file='ch12-sram', md='ch12.md', badge='Memory', short='SRAM',
         title='SRAM', time=30, tags='sram cortex-m arm',
         lead='A vendor-neutral look at SRAM, with the STM32F446 as a worked example.',
         lab='Lab 12', topics='SRAM'),
    dict(n=13, file='ch13-the-stack', md='ch13.md', badge='Memory', short='Stack',
         title='The Stack', time=30, tags='the stack cortex-m arm',
         lead='A vendor-neutral look at Stack, with the STM32F446 as a worked example.',
         lab='Lab 13', topics='The Stack'),
    dict(n=14, file='ch14-the-heap', md='ch14.md', badge='Memory', short='Heap',
         title='The Heap', time=30, tags='the heap cortex-m arm',
         lead='A vendor-neutral look at Heap, with the STM32F446 as a worked example.',
         lab='Lab 14', topics='The Heap'),
    dict(n=15, file='ch15-memory-mapped-i-o', md='ch15.md', badge='Memory', short='Memory-Mapped I/O',
         title='Memory-Mapped I/O', time=30, tags='memory mapped i o cortex-m arm',
         lead='A vendor-neutral look at Memory-Mapped I/O, with the STM32F446 as a worked example.',
         lab='Lab 15', topics='Memory-Mapped I/O'),
    dict(n=16, file='ch16-nvic', md='ch16.md', badge='Exceptions', short='NVIC',
         title='NVIC', time=30, tags='nvic cortex-m arm',
         lead='A vendor-neutral look at NVIC, with the STM32F446 as a worked example.',
         lab='Lab 16', topics='NVIC'),
    dict(n=17, file='ch17-interrupts', md='ch17.md', badge='Exceptions', short='Interrupts',
         title='Interrupts', time=30, tags='interrupts cortex-m arm',
         lead='A vendor-neutral look at Interrupts, with the STM32F446 as a worked example.',
         lab='Lab 17', topics='Interrupts'),
    dict(n=18, file='ch18-exceptions', md='ch18.md', badge='Exceptions', short='Exceptions',
         title='Exceptions', time=30, tags='exceptions cortex-m arm',
         lead='A vendor-neutral look at Exceptions, with the STM32F446 as a worked example.',
         lab='Lab 18', topics='Exceptions'),
    dict(n=19, file='ch19-systick', md='ch19.md', badge='Exceptions', short='SysTick',
         title='SysTick', time=30, tags='systick cortex-m arm',
         lead='A vendor-neutral look at SysTick, with the STM32F446 as a worked example.',
         lab='Lab 19', topics='SysTick'),
    dict(n=20, file='ch20-svc-and-pendsv', md='ch20.md', badge='Exceptions', short='SVC/PendSV',
         title='SVC and PendSV', time=30, tags='svc and pendsv cortex-m arm',
         lead='A vendor-neutral look at SVC/PendSV, with the STM32F446 as a worked example.',
         lab='Lab 20', topics='SVC and PendSV'),
    dict(n=21, file='ch21-fault-handling', md='ch21.md', badge='Exceptions', short='Faults',
         title='Fault Handling', time=30, tags='fault handling cortex-m arm',
         lead='A vendor-neutral look at Faults, with the STM32F446 as a worked example.',
         lab='Lab 21', topics='Fault Handling'),
    dict(n=22, file='ch22-reset-and-startup', md='ch22.md', badge='Exceptions', short='Reset & Startup',
         title='Reset and Startup', time=30, tags='reset and startup cortex-m arm',
         lead='A vendor-neutral look at Reset & Startup, with the STM32F446 as a worked example.',
         lab='Lab 22', topics='Reset and Startup'),
    dict(n=23, file='ch23-the-vector-table', md='ch23.md', badge='Exceptions', short='Vector Table',
         title='The Vector Table', time=30, tags='the vector table cortex-m arm',
         lead='A vendor-neutral look at Vector Table, with the STM32F446 as a worked example.',
         lab='Lab 23', topics='The Vector Table'),
    dict(n=24, file='ch24-the-linker-script', md='ch24.md', badge='Exceptions', short='Linker Script',
         title='The Linker Script', time=30, tags='the linker script cortex-m arm',
         lead='A vendor-neutral look at Linker Script, with the STM32F446 as a worked example.',
         lab='Lab 24', topics='The Linker Script'),
    dict(n=25, file='ch25-the-clock-system', md='ch25.md', badge='System', short='Clocks',
         title='The Clock System', time=30, tags='the clock system cortex-m arm',
         lead='A vendor-neutral look at Clocks, with the STM32F446 as a worked example.',
         lab='Lab 25', topics='The Clock System'),
    dict(n=26, file='ch26-the-mpu', md='ch26.md', badge='System', short='MPU',
         title='The MPU', time=30, tags='the mpu cortex-m arm',
         lead='A vendor-neutral look at MPU, with the STM32F446 as a worked example.',
         lab='Lab 26', topics='The MPU'),
    dict(n=27, file='ch27-the-fpu', md='ch27.md', badge='System', short='FPU',
         title='The FPU', time=30, tags='the fpu cortex-m arm',
         lead='A vendor-neutral look at FPU, with the STM32F446 as a worked example.',
         lab='Lab 27', topics='The FPU'),
    dict(n=28, file='ch28-dma', md='ch28.md', badge='System', short='DMA',
         title='DMA', time=30, tags='dma cortex-m arm',
         lead='A vendor-neutral look at DMA, with the STM32F446 as a worked example.',
         lab='Lab 28', topics='DMA'),
    dict(n=29, file='ch29-cache-and-memory-systems', md='ch29.md', badge='System', short='Cache & Memory',
         title='Cache and Memory Systems', time=30, tags='cache and memory systems cortex-m arm',
         lead='A vendor-neutral look at Cache & Memory, with the STM32F446 as a worked example.',
         lab='Lab 29', topics='Cache and Memory Systems'),
    dict(n=30, file='ch30-debug-architecture', md='ch30.md', badge='Debug & Tools', short='Debug Architecture',
         title='Debug Architecture', time=30, tags='debug architecture cortex-m arm',
         lead='A vendor-neutral look at Debug Architecture, with the STM32F446 as a worked example.',
         lab='Lab 30', topics='Debug Architecture'),
    dict(n=31, file='ch31-swd-and-jtag', md='ch31.md', badge='Debug & Tools', short='SWD/JTAG',
         title='SWD and JTAG', time=30, tags='swd and jtag cortex-m arm',
         lead='A vendor-neutral look at SWD/JTAG, with the STM32F446 as a worked example.',
         lab='Lab 31', topics='SWD and JTAG'),
    dict(n=32, file='ch32-gdb', md='ch32.md', badge='Debug & Tools', short='GDB',
         title='GDB', time=30, tags='gdb cortex-m arm',
         lead='A vendor-neutral look at GDB, with the STM32F446 as a worked example.',
         lab='Lab 32', topics='GDB'),
    dict(n=33, file='ch33-cmsis', md='ch33.md', badge='Debug & Tools', short='CMSIS',
         title='CMSIS', time=30, tags='cmsis cortex-m arm',
         lead='A vendor-neutral look at CMSIS, with the STM32F446 as a worked example.',
         lab='Lab 33', topics='CMSIS'),
    dict(n=34, file='ch34-stm32-peripheral-architecture', md='ch34.md', badge='Integration', short='STM32 Peripherals',
         title='STM32 Peripheral Architecture', time=30, tags='stm32 peripheral architecture cortex-m arm',
         lead='A vendor-neutral look at STM32 Peripherals, with the STM32F446 as a worked example.',
         lab='Lab 34', topics='STM32 Peripheral Architecture'),
    dict(n=35, file='ch35-from-reset-to-main', md='ch35.md', badge='Integration', short='Reset to main()',
         title='From Reset to main()', time=30, tags='from reset to main  cortex-m arm',
         lead='A vendor-neutral look at Reset to main(), with the STM32F446 as a worked example.',
         lab='Lab 35', topics='From Reset to main()'),
    dict(n=36, file='ch36-from-c-code-to-machine-code', md='ch36.md', badge='Integration', short='C to Machine Code',
         title='From C Code to Machine Code', time=30, tags='from c code to machine code cortex-m arm',
         lead='A vendor-neutral look at C to Machine Code, with the STM32F446 as a worked example.',
         lab='Lab 36', topics='From C Code to Machine Code'),
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
          <span class="brand-title">Cortex-M</span>
          <span class="brand-subtitle">Apt Computing Labs</span>
        </div>
      </a>
    </div>
    <div class="header-center">
      <div class="search-box-wrapper">
        <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input type="text" id="header-search" class="search-input" placeholder="Search chapters, NVIC, stack, linker, faults, CMSIS..." readonly>
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
              <span class="module-badge">C{c['n']}</span>
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
        <input type="text" id="modal-search-input" class="modal-search-input" placeholder="Search chapters, NVIC, stack, linker, faults, CMSIS...">
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
    js = js.replace("'zephyr_", "'cortexm_")
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
          <p class="chapter-lead">The Arm documents, vendor manuals, books and tools to keep open while studying Cortex-M.</p>
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
        <div class="chapter-actions"><a href="chapters/ch01-embedded-processor-fundamentals.html" class="btn-primary">Start Chapter 1 →</a></div>
      </div>'''
    covers = ['Programmer\'s model','Registers','Memory map','Stack & heap','NVIC','Exceptions','SysTick','Faults','Startup','Linker script','Clocks','MPU','FPU','DMA','Cache','SWD/JTAG','GDB','CMSIS','STM32 peripherals','C to machine code']
    pills = ''.join(f'<span class="progress-pill" style="margin:0.2rem;">{c}</span>' for c in covers)
    article = f'''        <div class="chapter-hero">
          <span class="chapter-module-tag">Digital Book &amp; Laboratory Course</span>
          <h1 class="chapter-h1">{BOOK_TITLE}: {BOOK_SUB}</h1>
          <p class="chapter-lead">Book 3, the engineering reference behind Books 1 and 2. A relatively vendor-independent tour of the Arm Cortex-M core, from processor fundamentals to the exact path from reset to main(), with the STM32 as the worked example implementation.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">🎯 Level: Intermediate (prerequisite: basic C and digital electronics)</div>
            <div class="meta-item">⏱️ {len(CHAPTERS)} chapters • lab and quiz in every chapter • reference-style chapters</div>
            <div class="meta-item">⚡ Reference board: NUCLEO-F446RE</div>
          </div>
        </div>
        <h2 id="sec-0-1">What You Will Learn</h2>
        <div style="margin:1rem 0;">{pills}</div>
        <h2 id="sec-0-2">Hardware You Need</h2>
        <ul>
          <li>Any Cortex-M board with a debug probe; the examples use NUCLEO-F446RE (Cortex-M4F).</li>
          <li>Optional: a second board with a Cortex-M0+ or M33 core to compare variants.</li>
          <li>An 8-channel USB logic analyzer (sigrok/PulseView compatible).</li>
        </ul>
        <h2 id="sec-0-3">Software You Need</h2>
        <ul>
          <li>arm-none-eabi-gcc toolchain with objdump, size, nm and readelf.</li>
          <li>OpenOCD (or ST-LINK/J-Link tools) and arm-none-eabi-gdb.</li>
          <li>Arm documentation, CMSIS headers and your vendor's reference manual.</li>
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
