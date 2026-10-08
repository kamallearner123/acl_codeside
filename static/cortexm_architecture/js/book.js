/* ==========================================================================
   ARM Cortex-M Architecture - Shared Book Client Script
   Handles: Theme toggle, Board switcher, Search, Copy code, Quizzes, Lab persistence
   ========================================================================== */

(function () {
  'use strict';

  // --- Theme Management ---
  const THEME_KEY = 'cortexm_book_theme';
  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(THEME_KEY, next);
    updateThemeIcon(next);
  }

  function updateThemeIcon(theme) {
    const btn = document.getElementById('theme-toggle-btn');
    if (!btn) return;
    btn.innerHTML = theme === 'dark' 
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
    btn.title = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
  }

  // --- Board Switcher Management ---
  const BOARD_KEY = 'cortexm_selected_board';
  function initBoard() {
    const savedBoard = localStorage.getItem(BOARD_KEY) || 'nucleo_f401re';
    const select = document.getElementById('board-selector');
    if (select) {
      select.value = savedBoard;
      select.addEventListener('change', function () {
        setBoard(this.value);
      });
    }
    applyBoardSelection(savedBoard);
  }

  function setBoard(boardId) {
    localStorage.setItem(BOARD_KEY, boardId);
    applyBoardSelection(boardId);
  }

  function applyBoardSelection(boardId) {
    // Show matching board-specific content and hide others
    document.querySelectorAll('[data-board-target]').forEach(el => {
      const targets = el.getAttribute('data-board-target').split(',');
      if (targets.includes(boardId) || targets.includes('all')) {
        el.style.display = '';
      } else {
        el.style.display = 'none';
      }
    });

    // Update dynamic command lines (e.g. west build -b <board>)
    document.querySelectorAll('.cmd-board-name').forEach(el => {
      el.textContent = boardId;
    });
  }

  // --- Reading Progress Tracker ---
  function initReadingProgress() {
    const progressBar = document.getElementById('reading-progress-bar');
    if (!progressBar) return;

    window.addEventListener('scroll', () => {
      const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = height > 0 ? (winScroll / height) * 100 : 0;
      progressBar.style.width = scrolled + '%';
    });
  }

  // --- Sidebar & Vertical Tab Toggle Management ---
  const SIDEBAR_COLLAPSED_KEY = 'cortexm_sidebar_hidden';

  function initSidebar() {
    const toggleBtn = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('book-sidebar');
    if (!sidebar) return;
    const sidebarNav = sidebar.querySelector('.sidebar-nav');
    const chapterPath = window.location.pathname.split('/chapters/')[0].replace(/\/+$/, '') || '/';
    const sidebarStateKey = `acl_book_sidebar_state:${chapterPath}`;

    function saveNavigationState(link = null) {
      if (!sidebarNav) return;
      try {
        const previousState = JSON.parse(sessionStorage.getItem(sidebarStateKey) || 'null');
        const navRect = sidebarNav.getBoundingClientRect();
        const linkOffset = link
          ? link.getBoundingClientRect().top - navRect.top
          : null;
        sessionStorage.setItem(sidebarStateKey, JSON.stringify({
          scrollTop: sidebarNav.scrollTop,
          anchorHref: link?.getAttribute('href') || previousState?.anchorHref || null,
          anchorOffset: link ? linkOffset : previousState?.anchorOffset ?? null,
          collapsedGroups: Array.from(sidebarNav.querySelectorAll('.nav-module-group'))
            .map((group, index) => group.classList.contains('collapsed') ? index : null)
            .filter(index => index !== null)
        }));
      } catch (error) {
        console.warn('Could not preserve course navigation state', error);
      }
    }

    function syncModuleButton(group) {
      const button = group.querySelector('.module-title-btn');
      if (button) button.setAttribute('aria-expanded', String(!group.classList.contains('collapsed')));
    }

    if (sidebarNav) {
      try {
        const savedState = JSON.parse(sessionStorage.getItem(sidebarStateKey) || 'null');
        if (savedState) {
          const collapsedGroups = new Set(savedState.collapsedGroups || []);
          sidebarNav.querySelectorAll('.nav-module-group').forEach((group, index) => {
            group.classList.toggle('collapsed', collapsedGroups.has(index));
            syncModuleButton(group);
          });
          const activeGroup = sidebarNav.querySelector('.chapter-nav-item.active')?.closest('.nav-module-group');
          if (activeGroup) {
            activeGroup.classList.remove('collapsed');
            syncModuleButton(activeGroup);
          }
          requestAnimationFrame(() => {
            sidebarNav.scrollTop = Number(savedState.scrollTop) || 0;
            requestAnimationFrame(() => {
              if (!savedState.anchorHref || savedState.anchorOffset === null) return;
              const anchor = Array.from(sidebarNav.querySelectorAll('a.chapter-nav-item, a.subtopic-link'))
                .find(link => link.getAttribute('href') === savedState.anchorHref);
              if (!anchor) return;
              const navTop = sidebarNav.getBoundingClientRect().top;
              const anchorOffset = anchor.getBoundingClientRect().top - navTop;
              sidebarNav.scrollTop += anchorOffset - Number(savedState.anchorOffset);
            });
          });
        }
      } catch (error) {
        console.warn('Could not restore course navigation state', error);
      }

      sidebarNav.querySelectorAll('a.chapter-nav-item, a.subtopic-link').forEach(link => {
        link.addEventListener('click', () => saveNavigationState(link));
      });
      window.addEventListener('pagehide', saveNavigationState);
    }

    // Create floating unhide tab if not present
    let floatingTab = document.getElementById('sidebar-floating-tab');
    if (!floatingTab) {
      floatingTab = document.createElement('button');
      floatingTab.id = 'sidebar-floating-tab';
      floatingTab.className = 'sidebar-floating-tab';
      floatingTab.setAttribute('aria-label', 'Show Navigation Sidebar');
      floatingTab.setAttribute('title', 'Show Navigation Sidebar (press [)');
      floatingTab.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg><span>Show Menu</span>';
      document.body.appendChild(floatingTab);
    }

    // Add collapse button to sidebar header if missing
    const tocHeading = sidebar.querySelector('.toc-heading');
    if (tocHeading && !sidebar.querySelector('.sidebar-collapse-btn')) {
      const colBtn = document.createElement('button');
      colBtn.className = 'sidebar-collapse-btn';
      colBtn.id = 'sidebar-collapse-btn';
      colBtn.title = 'Hide Navigation Tab (or press [)';
      colBtn.setAttribute('aria-label', 'Hide Navigation Tab');
      colBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>';
      colBtn.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          sidebar.classList.remove('open');
        } else {
          document.body.classList.add('sidebar-hidden');
          localStorage.setItem(SIDEBAR_COLLAPSED_KEY, 'true');
          updateToggleTooltip();
        }
      });
      tocHeading.appendChild(colBtn);
    }

    // Restore desktop collapsed state
    const wasHidden = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === 'true';
    if (window.innerWidth > 768 && wasHidden) {
      document.body.classList.add('sidebar-hidden');
    }

    function toggleSidebar() {
      if (window.innerWidth <= 768) {
        // Mobile: toggle overlay drawer
        sidebar.classList.toggle('open');
      } else {
        // Desktop: toggle hidden/visible vertical tab
        const isHidden = document.body.classList.toggle('sidebar-hidden');
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, isHidden);
      }
      updateToggleTooltip();
    }

    function updateToggleTooltip() {
      if (!toggleBtn) return;
      const isHidden = document.body.classList.contains('sidebar-hidden');
      toggleBtn.title = isHidden 
        ? 'Show Navigation Sidebar (press [)' 
        : 'Hide Navigation Sidebar (press [)';
      toggleBtn.setAttribute('aria-expanded', !isHidden);
    }

    if (toggleBtn) {
      toggleBtn.addEventListener('click', toggleSidebar);
      updateToggleTooltip();
    }

    if (floatingTab) {
      floatingTab.addEventListener('click', () => {
        document.body.classList.remove('sidebar-hidden');
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, 'false');
        updateToggleTooltip();
      });
    }

    // Close when clicking outside on mobile
    document.addEventListener('click', (e) => {
      if (window.innerWidth <= 768 && 
          sidebar.classList.contains('open') && 
          !sidebar.contains(e.target) && 
          toggleBtn && !toggleBtn.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });

    // Keyboard shortcut: '[' or Ctrl+B / Cmd+B to toggle vertical tab
    document.addEventListener('keydown', (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) {
        return;
      }
      if (e.key === '[' || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b')) {
        e.preventDefault();
        toggleSidebar();
      }
    });

    // Module collapse toggles
    document.querySelectorAll('.module-title-btn').forEach(btn => {
      const group = btn.closest('.nav-module-group');
      if (group) syncModuleButton(group);
      btn.addEventListener('click', () => {
        if (group) {
          group.classList.toggle('collapsed');
          syncModuleButton(group);
          saveNavigationState();
        }
      });
    });
  }

  function initPlatformNote() {
    const main = document.querySelector('.main-content');
    if (!main || main.querySelector('.platform-note')) return;
    const note = document.createElement('aside');
    note.className = 'platform-note';
    note.setAttribute('role', 'note');
    note.innerHTML = '<strong>Board portability:</strong> The NUCLEO-F446RE (Cortex-M4F) is a reference board for selected hands-on examples, not a requirement for every lesson. Cortex-M architecture concepts apply across Cortex-M processors; STM32 peripheral and RTOS examples transfer only where equivalent hardware and software support exist. Pins, clocks, memory maps, peripheral instances, interrupt numbering, startup files, board support/DeviceTree, and wiring vary; verify against your exact MCU reference manual, datasheet, RTOS/SDK documentation, and board schematic.';
    main.prepend(note);
  }

  // --- Code Copy Buttons ---
  function initCodeCopy() {
    document.querySelectorAll('.code-copy-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const container = btn.closest('.code-container');
        if (!container) return;
        const codeEl = container.querySelector('code') || container.querySelector('pre');
        if (!codeEl) return;

        try {
          await navigator.clipboard.writeText(codeEl.innerText);
          const originalText = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.color = '#10b981';
          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.color = '';
          }, 2000);
        } catch (err) {
          console.error('Failed to copy', err);
        }
      });
    });
  }

  // --- Lab Checklists State Persistence ---
  function initLabChecklists() {
    const pageId = document.body.getAttribute('data-chapter-id') || window.location.pathname;
    const key = 'cortexm_lab_check_' + pageId;
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(key) || '{}');
    } catch (e) {}

    document.querySelectorAll('.checklist-checkbox').forEach((checkbox, idx) => {
      if (saved[idx]) {
        checkbox.checked = true;
        const text = checkbox.closest('.checklist-item')?.querySelector('.checklist-text');
        if (text) text.classList.add('checked');
      }

      checkbox.addEventListener('change', () => {
        saved[idx] = checkbox.checked;
        localStorage.setItem(key, JSON.stringify(saved));
        const text = checkbox.closest('.checklist-item')?.querySelector('.checklist-text');
        if (text) {
          if (checkbox.checked) text.classList.add('checked');
          else text.classList.remove('checked');
        }
      });
    });
  }

  // --- Interactive Quiz Logic ---
  function initQuizzes() {
    document.querySelectorAll('.quiz-container').forEach(quiz => {
      const options = quiz.querySelectorAll('.quiz-option-btn');
      const feedback = quiz.querySelector('.quiz-feedback');

      options.forEach(btn => {
        btn.addEventListener('click', () => {
          // Reset previous states
          options.forEach(o => {
            o.classList.remove('correct', 'wrong');
            o.disabled = true;
          });

          const isCorrect = btn.getAttribute('data-correct') === 'true';
          if (isCorrect) {
            btn.classList.add('correct');
            if (feedback) {
              feedback.className = 'quiz-feedback show correct';
              feedback.innerHTML = '<strong>Correct!</strong> ' + (btn.getAttribute('data-explanation') || 'Well done!');
            }
          } else {
            btn.classList.add('wrong');
            // Highlight the correct one
            options.forEach(o => {
              if (o.getAttribute('data-correct') === 'true') o.classList.add('correct');
            });
            if (feedback) {
              feedback.className = 'quiz-feedback show wrong';
              feedback.innerHTML = '<strong>Incorrect.</strong> ' + (btn.getAttribute('data-explanation') || 'Review the chapter section above for clarification.');
            }
          }
        });
      });
    });
  }

  // --- Search System Across All Chapters ---
  const CHAPTERS_INDEX = [
    { title: "Course Overview & Learning Roadmap", url: "index.html", tags: "intro overview roadmap stm32 c firmware" },
    { title: "Chapter 1: Embedded Processor Fundamentals", url: "chapters/ch01-embedded-processor-fundamentals.html", tags: "embedded processor fundamentals cortex-m arm" },
    { title: "Chapter 2: ARM Architecture Overview", url: "chapters/ch02-arm-architecture-overview.html", tags: "arm architecture overview cortex-m arm" },
    { title: "Chapter 3: The Cortex-M Family", url: "chapters/ch03-the-cortex-m-family.html", tags: "the cortex m family cortex-m arm" },
    { title: "Chapter 4: Cortex-M0, M0+, M3, M4, M7 and M33", url: "chapters/ch04-cortex-m0-m0-m3-m4-m7-and-m33.html", tags: "cortex m0  m0  m3  m4  m7 and m33 cortex-m arm" },
    { title: "Chapter 5: The Programmer's Model", url: "chapters/ch05-the-programmer-s-model.html", tags: "the programmer s model cortex-m arm" },
    { title: "Chapter 6: Registers", url: "chapters/ch06-registers.html", tags: "registers cortex-m arm" },
    { title: "Chapter 7: R0-R12, SP, LR, PC and xPSR", url: "chapters/ch07-r0-r12-sp-lr-pc-and-xpsr.html", tags: "r0 r12  sp  lr  pc and xpsr cortex-m arm" },
    { title: "Chapter 8: Instruction Set Basics", url: "chapters/ch08-instruction-set-basics.html", tags: "instruction set basics cortex-m arm" },
    { title: "Chapter 9: Thumb and Thumb-2", url: "chapters/ch09-thumb-and-thumb-2.html", tags: "thumb and thumb 2 cortex-m arm" },
    { title: "Chapter 10: The Memory Map", url: "chapters/ch10-the-memory-map.html", tags: "the memory map cortex-m arm" },
    { title: "Chapter 11: Flash Memory", url: "chapters/ch11-flash-memory.html", tags: "flash memory cortex-m arm" },
    { title: "Chapter 12: SRAM", url: "chapters/ch12-sram.html", tags: "sram cortex-m arm" },
    { title: "Chapter 13: The Stack", url: "chapters/ch13-the-stack.html", tags: "the stack cortex-m arm" },
    { title: "Chapter 14: The Heap", url: "chapters/ch14-the-heap.html", tags: "the heap cortex-m arm" },
    { title: "Chapter 15: Memory-Mapped I/O", url: "chapters/ch15-memory-mapped-i-o.html", tags: "memory mapped i o cortex-m arm" },
    { title: "Chapter 16: NVIC", url: "chapters/ch16-nvic.html", tags: "nvic cortex-m arm" },
    { title: "Chapter 17: Interrupts", url: "chapters/ch17-interrupts.html", tags: "interrupts cortex-m arm" },
    { title: "Chapter 18: Exceptions", url: "chapters/ch18-exceptions.html", tags: "exceptions cortex-m arm" },
    { title: "Chapter 19: SysTick", url: "chapters/ch19-systick.html", tags: "systick cortex-m arm" },
    { title: "Chapter 20: SVC and PendSV", url: "chapters/ch20-svc-and-pendsv.html", tags: "svc and pendsv cortex-m arm" },
    { title: "Chapter 21: Fault Handling", url: "chapters/ch21-fault-handling.html", tags: "fault handling cortex-m arm" },
    { title: "Chapter 22: Reset and Startup", url: "chapters/ch22-reset-and-startup.html", tags: "reset and startup cortex-m arm" },
    { title: "Chapter 23: The Vector Table", url: "chapters/ch23-the-vector-table.html", tags: "the vector table cortex-m arm" },
    { title: "Chapter 24: The Linker Script", url: "chapters/ch24-the-linker-script.html", tags: "the linker script cortex-m arm" },
    { title: "Chapter 25: The Clock System", url: "chapters/ch25-the-clock-system.html", tags: "the clock system cortex-m arm" },
    { title: "Chapter 26: The MPU", url: "chapters/ch26-the-mpu.html", tags: "the mpu cortex-m arm" },
    { title: "Chapter 27: The FPU", url: "chapters/ch27-the-fpu.html", tags: "the fpu cortex-m arm" },
    { title: "Chapter 28: DMA", url: "chapters/ch28-dma.html", tags: "dma cortex-m arm" },
    { title: "Chapter 29: Cache and Memory Systems", url: "chapters/ch29-cache-and-memory-systems.html", tags: "cache and memory systems cortex-m arm" },
    { title: "Chapter 30: Debug Architecture", url: "chapters/ch30-debug-architecture.html", tags: "debug architecture cortex-m arm" },
    { title: "Chapter 31: SWD and JTAG", url: "chapters/ch31-swd-and-jtag.html", tags: "swd and jtag cortex-m arm" },
    { title: "Chapter 32: GDB", url: "chapters/ch32-gdb.html", tags: "gdb cortex-m arm" },
    { title: "Chapter 33: CMSIS", url: "chapters/ch33-cmsis.html", tags: "cmsis cortex-m arm" },
    { title: "Chapter 34: STM32 Peripheral Architecture", url: "chapters/ch34-stm32-peripheral-architecture.html", tags: "stm32 peripheral architecture cortex-m arm" },
    { title: "Chapter 35: From Reset to main()", url: "chapters/ch35-from-reset-to-main.html", tags: "from reset to main  cortex-m arm" },
    { title: "Chapter 36: From C Code to Machine Code", url: "chapters/ch36-from-c-code-to-machine-code.html", tags: "from c code to machine code cortex-m arm" },
    { title: "References & Further Reading", url: "chapters/references.html", tags: "references datasheet reference manual rm0390 hal user manual um1725 arm" }
  ];

  function initSearch() {
    const searchInputs = [document.getElementById('header-search'), document.getElementById('modal-search-input')];
    const modal = document.getElementById('search-modal-overlay');
    const resultsContainer = document.getElementById('search-results');
    const closeBtn = document.getElementById('close-search-modal');

    function openModal(query = '') {
      if (!modal) return;
      modal.classList.add('active');
      const input = document.getElementById('modal-search-input');
      if (input) {
        input.value = query;
        input.focus();
        runSearch(query);
      }
    }

    function closeModal() {
      if (modal) modal.classList.remove('active');
    }

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
      });
    }

    // Ctrl+K / Cmd+K shortcut
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        openModal();
      }
      if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
        closeModal();
      }
    });

    const headerSearch = document.getElementById('header-search');
    if (headerSearch) {
      headerSearch.addEventListener('click', () => openModal());
      headerSearch.addEventListener('focus', () => openModal());
    }

    const modalInput = document.getElementById('modal-search-input');
    if (modalInput) {
      modalInput.addEventListener('input', (e) => {
        runSearch(e.target.value);
      });
    }

    function runSearch(query) {
      if (!resultsContainer) return;
      const clean = query.trim().toLowerCase();
      if (!clean) {
        resultsContainer.innerHTML = '<li style="padding:1rem;color:var(--text-muted);text-align:center;">Type keywords to search across all book chapters...</li>';
        return;
      }

      // Determine relative path prefix depending on if we are in /chapters/ or root
      const isInChapters = window.location.pathname.includes('/chapters/');
      const prefix = isInChapters ? '../' : '';

      const matched = CHAPTERS_INDEX.filter(item => 
        item.title.toLowerCase().includes(clean) || item.tags.toLowerCase().includes(clean)
      );

      if (matched.length === 0) {
        resultsContainer.innerHTML = '<li style="padding:1rem;color:var(--text-muted);text-align:center;">No chapters found matching "' + escapeHtml(query) + '"</li>';
        return;
      }

      resultsContainer.innerHTML = matched.map(m => {
        let dest = m.url;
        if (isInChapters) {
          dest = m.url.startsWith('chapters/') ? m.url.replace('chapters/', '') : '../' + m.url;
        }
        return `
          <li class="search-result-item" onclick="window.location.href='${dest}'">
            <div class="search-result-title">${escapeHtml(m.title)}</div>
            <div class="search-result-snippet">Tags: ${escapeHtml(m.tags)}</div>
          </li>
        `;
      }).join('');
    }
  }

  // --- Subtopic Smooth Anchor Navigation & Mobile Drawer Close ---
  function initAnchorNavigation() {
    const subtopicLinks = document.querySelectorAll('.subtopic-link, a[href*="#sec-"]');

    subtopicLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        if (!href) return;

        const hashIndex = href.indexOf('#');
        if (hashIndex === -1) return;

        const targetHash = href.substring(hashIndex); // e.g. '#sec-2-1'
        const pathPart = href.substring(0, hashIndex); // e.g. 'ch02-devicetree-kconfig.html'

        // Determine if target element exists on the CURRENT page
        const currentPath = window.location.pathname.split('?')[0];
        const currentFileName = currentPath.substring(currentPath.lastIndexOf('/') + 1) || 'index.html';

        const isSamePage = !pathPart ||
          pathPart === currentFileName ||
          currentPath.endsWith(pathPart) ||
          (pathPart.startsWith('chapters/') && currentPath.endsWith(pathPart.replace('chapters/', ''))) ||
          (currentFileName === 'index.html' && (pathPart === '' || pathPart === 'index.html'));

        const targetElement = isSamePage ? document.querySelector(targetHash) : null;

        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });

          // Update active link state
          document.querySelectorAll('.subtopic-link').forEach(l => l.classList.remove('active'));
          link.classList.add('active');

          if (history.pushState) {
            history.pushState(null, '', targetHash);
          } else {
            window.location.hash = targetHash;
          }

          // Auto-close sidebar drawer on mobile devices
          const sidebar = document.getElementById('book-sidebar');
          if (sidebar && window.innerWidth <= 768) {
            sidebar.classList.remove('open');
          }
        } else {
          // If navigating across pages, also close mobile drawer so next view is uncluttered
          const sidebar = document.getElementById('book-sidebar');
          if (sidebar && window.innerWidth <= 768) {
            sidebar.classList.remove('open');
          }
        }
      });
    });

    // Check if initial page load arrived with an anchor hash
    if (window.location.hash) {
      setTimeout(() => {
        try {
          const target = document.querySelector(window.location.hash);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Highlight corresponding subtopic link if present
            document.querySelectorAll('.subtopic-link').forEach(l => {
              const h = l.getAttribute('href') || '';
              if (h.endsWith(window.location.hash)) {
                l.classList.add('active');
              }
            });
          }
        } catch (err) {
          console.debug('Invalid anchor hash', err);
        }
      }, 150);
    }
  }

  // --- Scrollspy: Highlight Active Subtopic on Scroll ---
  function initScrollspy() {
    const sections = document.querySelectorAll('[id^="sec-"]');
    if (!sections.length) return;

    const subtopicLinks = document.querySelectorAll('.subtopic-link');
    if (!subtopicLinks.length) return;

    window.addEventListener('scroll', () => {
      const scrollPos = window.scrollY + 120; // offset below sticky header
      let currentSectionId = null;

      sections.forEach(sec => {
        const top = sec.offsetTop;
        if (scrollPos >= top) {
          currentSectionId = sec.getAttribute('id');
        }
      });

      if (currentSectionId) {
        subtopicLinks.forEach(link => {
          const href = link.getAttribute('href') || '';
          if (href.endsWith('#' + currentSectionId)) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
    }, { passive: true });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ==========================================================================
  // Client-Side Syntax Highlighter for C, DTS, Bash, and Kconfig
  // ==========================================================================
  function highlightCCode(code) {
    const tokens = [];
    function saveToken(html) {
      tokens.push(html);
      return `___CTOK${tokens.length - 1}___`;
    }

    // 1. Comments
    code = code.replace(/\/\*[\s\S]*?\*\//g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));
    code = code.replace(/\/\/[^\n\r]*/g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));

    // 2. Preprocessor directives
    code = code.replace(/#\s*(?:include|define|undef|ifdef|ifndef|if|else|elif|endif|pragma|error)\b[^\n\r]*/g, m => {
      const formatted = escapeHtml(m).replace(/(&lt;[^&]+&gt;|"[^"]+")/g, '<span class="token-string">$1</span>');
      return saveToken(`<span class="token-preprocessor">${formatted}</span>`);
    });

    // 3. String literals & characters
    code = code.replace(/"(?:\\.|[^"\\])*"/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));
    code = code.replace(/'(?:\\.|[^'\\])+'/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));

    // 4. Numbers (hex and dec)
    code = code.replace(/\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b/g, m => saveToken(`<span class="token-number">${m}</span>`));

    // 5. Zephyr APIs, macros and subsystems
    const zephyrApis = /\b(?:printk|k_msleep|k_sleep|k_busy_wait|k_uptime_get_32|k_uptime_get|gpio_pin_configure_dt|gpio_pin_toggle_dt|gpio_pin_set_dt|gpio_pin_get_dt|gpio_is_ready_dt|gpio_init_callback|gpio_add_callback|gpio_pin_interrupt_configure_dt|pwm_set_pulse_dt|pwm_set_dt|sensor_sample_fetch|sensor_channel_get|k_msgq_put|k_msgq_get|k_sem_give|k_sem_take|k_mutex_lock|k_mutex_unlock|wdt_setup|wdt_install_timeout|wdt_feed|can_send|can_add_rx_filter|LOG_INF|LOG_ERR|LOG_WRN|LOG_DBG|DEVICE_DT_GET|DEVICE_DT_GET_ANY|DT_ALIAS|DT_NODELABEL|DT_CHOSEN|DT_NODE_HAS_STATUS|GPIO_DT_SPEC_GET|K_THREAD_DEFINE|K_MSGQ_DEFINE|K_SEM_DEFINE|K_MUTEX_DEFINE|SHELL_CMD_REGISTER|SHELL_STATIC_SUBCMD_SET_CREATE|ARG_UNUSED|BIT|GPIO_OUTPUT_INACTIVE|GPIO_OUTPUT_ACTIVE|GPIO_INPUT|GPIO_INT_EDGE_TO_ACTIVE)\b/g;
    code = code.replace(zephyrApis, m => saveToken(`<span class="token-zephyr">${m}</span>`));
    code = code.replace(/\b(?:__HAL_\w+|HAL_\w+|LL_\w+|NVIC_\w+|__NVIC_\w+|__disable_irq|__enable_irq|__WFI|__NOP|SysTick_Config)\b/g, m => saveToken(`<span class="token-zephyr">${m}</span>`));

    // 6. C Keywords
    const keywords = /\b(?:void|int|char|short|long|float|double|signed|unsigned|const|static|volatile|struct|enum|union|typedef|extern|inline|return|if|else|switch|case|default|while|for|do|break|continue|goto|sizeof)\b/g;
    code = code.replace(keywords, m => saveToken(`<span class="token-keyword">${m}</span>`));

    // 7. Types & Booleans
    const types = /\b(?:int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t|size_t|ssize_t|bool|true|false|NULL)\b/g;
    code = code.replace(types, m => saveToken(`<span class="token-type">${m}</span>`));

    // 8. Function calls
    code = code.replace(/\b(?!___)([a-zA-Z_]\w*)(?=\s*\()/g, m => saveToken(`<span class="token-func">${m}</span>`));

    // Restore tokens
    const re = /___CTOK(\d+)___/g;
    while (re.test(code)) { code = code.replace(re, (_, id) => tokens[id]); }
    return code;
  }


  function highlightRustCode(code) {
    const tokens = [];
    function saveToken(html) {
      tokens.push(html);
      return `___RTOK${tokens.length - 1}___`;
    }

    code = code.replace(/\/\*[\s\S]*?\*\//g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));
    code = code.replace(/\/\/[^\n\r]*/g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));
    code = code.replace(/b?r(#*)"[\s\S]*?"\1/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));
    code = code.replace(/b?"(?:\\.|[^"\\])*"/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));
    code = code.replace(/b?'(?:\\.|[^'\\])'/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));
    code = code.replace(/#!?\[[^\]\n]*\]/g, m => saveToken(`<span class="token-preprocessor">${escapeHtml(m)}</span>`));
    code = code.replace(/'(?:static|_|[a-z_]\w*)\b/g, m => saveToken(`<span class="token-prop">${m}</span>`));
    code = code.replace(/\b(?:0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[ui](?:8|16|32|64|128|size)|f32|f64)?)\b/g, m => saveToken(`<span class="token-number">${m}</span>`));
    code = code.replace(/\b[a-zA-Z_]\w*!(?=\s*[(\[{])/g, m => saveToken(`<span class="token-macro">${m}</span>`));
    code = code.replace(/\b(?:as|async|await|break|const|continue|crate|dyn|else|enum|extern|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|type|unsafe|use|where|while)\b/g, m => saveToken(`<span class="token-keyword">${m}</span>`));
    code = code.replace(/\b(?:u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|f32|f64|bool|char|str|true|false|None|Some|Ok|Err|Option|Result|Vec|String|Box|Self)\b/g, m => saveToken(`<span class="token-type">${m}</span>`));
    code = code.replace(/\b[A-Z][A-Za-z0-9]*\b/g, m => saveToken(`<span class="token-type">${m}</span>`));
    code = code.replace(/\b(?!___)([a-z_]\w*)(?=\s*(?:::<[^>\n]*>)?\()/g, m => saveToken(`<span class="token-func">${m}</span>`));

    const re = /___RTOK(\d+)___/g;
    while (re.test(code)) { code = code.replace(re, (_, id) => tokens[id]); }
    return code;
  }

  function highlightDTSCode(code) {
    const tokens = [];
    function saveToken(html) {
      tokens.push(html);
      return `___DTOK${tokens.length - 1}___`;
    }

    code = code.replace(/\/\*[\s\S]*?\*\//g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));
    code = code.replace(/\/\/[^\n\r]*/g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));
    code = code.replace(/"(?:\\.|[^"\\])*"/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));
    code = code.replace(/\b(?:0x[0-9a-fA-F]+|\d+)\b/g, m => saveToken(`<span class="token-number">${m}</span>`));
    code = code.replace(/\b(compatible|reg|status|interrupts|label|gpios|aliases|model|#address-cells|#size-cells)\b/g, m => saveToken(`<span class="token-prop">${m}</span>`));
    code = code.replace(/&[a-zA-Z0-9_]+/g, m => saveToken(`<span class="token-zephyr">${m}</span>`));
    code = code.replace(/\/dts-v1\/;/g, m => saveToken(`<span class="token-preprocessor">${m}</span>`));

    const re = /___DTOK(\d+)___/g;
    while (re.test(code)) { code = code.replace(re, (_, id) => tokens[id]); }
    return code;
  }

  function highlightBashCode(code) {
    const tokens = [];
    function saveToken(html) {
      tokens.push(html);
      return `___BTOK${tokens.length - 1}___`;
    }

    code = code.replace(/#[^\n\r]*/g, m => saveToken(`<span class="token-comment">${escapeHtml(m)}</span>`));
    code = code.replace(/"(?:\\.|[^"\\])*"/g, m => saveToken(`<span class="token-string">${escapeHtml(m)}</span>`));
    code = code.replace(/\b(west|cd|ninja|cmake|minicom|screen|pyocd|git|echo|mkdir|build|flash)\b/g, m => saveToken(`<span class="token-keyword">${m}</span>`));
    code = code.replace(/(?:^|\s)(-[a-zA-Z]|--[a-zA-Z0-9_-]+)/g, m => saveToken(`<span class="token-prop">${m}</span>`));
    code = code.replace(/\b(nucleo_f401re|nucleo_l476rg|nucleo_g071rb)\b/g, m => saveToken(`<span class="cmd-board-name token-string">${m}</span>`));

    const re = /___BTOK(\d+)___/g;
    while (re.test(code)) { code = code.replace(re, (_, id) => tokens[id]); }
    return code;
  }

  function initSyntaxHighlighting() {
    document.querySelectorAll('.code-container').forEach(container => {
      // Ensure copy button is present
      const header = container.querySelector('.code-header');
      if (header && !header.querySelector('.code-copy-btn')) {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy-btn';
        copyBtn.textContent = 'Copy';
        header.appendChild(copyBtn);
      }

      const langBadge = container.querySelector('.code-lang-badge');
      const codeEl = container.querySelector('code');
      if (!codeEl) return;

      const lang = (langBadge ? langBadge.textContent.trim().toUpperCase() : '') || 'C';
      const rawText = codeEl.innerText;

      if (lang === 'C') {
        codeEl.innerHTML = highlightCCode(rawText);
      } else if (lang === 'RUST' || lang === 'RS') {
        codeEl.innerHTML = highlightRustCode(rawText);
      } else if (lang === 'DTS' || lang === 'DEVICETREE') {
        codeEl.innerHTML = highlightDTSCode(rawText);
      } else if (lang === 'BASH' || lang === 'SHELL' || lang === 'TERMINAL') {
        codeEl.innerHTML = highlightBashCode(rawText);
      } else if (rawText.includes('#include') || rawText.includes('int main(')) {
        codeEl.innerHTML = highlightCCode(rawText);
      }
    });
  }

  // ==========================================================================
  // Interactive Hardware Output Simulators
  // ==========================================================================
  function initSimulations() {
    // --- Lab 1: Blinky & VCP Simulator ---
    const blinkyWidget = document.querySelector('[data-sim="blinky-vcp"]');
    if (blinkyWidget) {
      const ledBulb = blinkyWidget.querySelector('#sim-blinky-led');
      const logBox = blinkyWidget.querySelector('#sim-blinky-log');
      const pauseBtn = blinkyWidget.querySelector('#sim-blinky-pause-btn');
      const stepBtn = blinkyWidget.querySelector('#sim-blinky-step-btn');
      const clearBtn = blinkyWidget.querySelector('#sim-blinky-clear-btn');
      const runState = blinkyWidget.querySelector('.sim-run-state');

      let isPaused = false;
      let ledState = false;
      let cycleCount = 0;
      let blinkTimer = null;

      function appendLog(line) {
        if (!logBox) return;
        const now = new Date();
        const timeStr = String(now.getSeconds()).padStart(2, '0') + '.' + String(now.getMilliseconds()).padStart(3, '0');
        const p = document.createElement('div');
        p.innerHTML = `<span class="log-dim">[00:00:${timeStr}]</span> ${line}`;
        logBox.appendChild(p);
        logBox.scrollTop = logBox.scrollHeight;
        while (logBox.children.length > 15) {
          logBox.removeChild(logBox.firstChild);
        }
      }

      function tickBlinky() {
        if (isPaused) return;
        ledState = !ledState;
        if (ledBulb) {
          if (ledState) ledBulb.classList.add('on');
          else ledBulb.classList.remove('on');
        }

        if (ledState) {
          cycleCount++;
          appendLog(`<span class="log-info">&lt;inf&gt; main:</span> Blinky ping #${cycleCount} - LED Toggled <span class="log-hi">(HIGH / ON)</span>`);
        } else {
          appendLog(`<span class="log-dim">&lt;inf&gt; main:</span> Sleep cycle (1000 ms) - LED Toggled <span class="log-dim">(LOW / OFF)</span>`);
        }
      }

      appendLog(`<span class="log-hi">=== Mastering Zephyr RTOS on STM32 - Lab 1 ===</span>`);
      appendLog(`<span class="log-info">&lt;inf&gt; boot:</span> Target board initialized via ST-Link VCP`);
      blinkTimer = setInterval(tickBlinky, 1000);

      if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
          isPaused = !isPaused;
          pauseBtn.textContent = isPaused ? '▶ Resume Simulation' : '⏸ Pause Simulation';
          if (runState) runState.textContent = isPaused ? 'PAUSED' : 'RUNNING (1 Hz)';
        });
      }

      if (stepBtn) {
        stepBtn.addEventListener('click', () => {
          tickBlinky();
        });
      }

      if (clearBtn) {
        clearBtn.addEventListener('click', () => {
          if (logBox) logBox.innerHTML = '';
        });
      }
    }

    // --- Lab 2: External LED & Button Interrupt Simulator ---
    const gpioWidget = document.querySelector('[data-sim="gpio-interrupt"]');
    if (gpioWidget) {
      const extLedBulb = gpioWidget.querySelector('#sim-ext-led');
      const pressBtn = gpioWidget.querySelector('#sim-ext-btn');
      const extLog = gpioWidget.querySelector('#sim-ext-log');
      let extLedState = false;
      let btnPressCount = 0;

      function logExt(msg) {
        if (!extLog) return;
        const now = new Date();
        const timeStr = String(now.getSeconds()).padStart(2, '0') + '.' + String(now.getMilliseconds()).padStart(3, '0');
        const p = document.createElement('div');
        p.innerHTML = `<span class="log-dim">[00:00:${timeStr}]</span> ${msg}`;
        extLog.appendChild(p);
        extLog.scrollTop = extLog.scrollHeight;
      }

      logExt(`<span class="log-hi">=== Zephyr RTOS Lab 2: DTS Overlays &amp; GPIO Interrupts ===</span>`);
      logExt(`<span class="log-info">&lt;inf&gt; gpio:</span> Ext LED on Port GPIOB Pin 0 | Ext Button on Port GPIOB Pin 1`);
      logExt(`Click "Press Breadboard Button" to trigger STM32 EXTI interrupt...`);

      if (pressBtn) {
        pressBtn.addEventListener('click', () => {
          btnPressCount++;
          extLedState = !extLedState;
          if (extLedBulb) {
            if (extLedState) extLedBulb.classList.add('on');
            else extLedBulb.classList.remove('on');
          }
          logExt(`<span class="log-warn">[EXTI1_IRQ]</span> Hardware Button Pressed (Pulse #${btnPressCount})! Ext LED (PB0) -&gt; <span class="log-hi">${extLedState ? 'HIGH (ON)' : 'LOW (OFF)'}</span>`);
        });
      }
    }

    // --- Lab 3: Multithread Message Queue Pipeline Simulator ---
    const threadWidget = document.querySelector('[data-sim="thread-pipeline"]');
    if (threadWidget) {
      const threadLog = threadWidget.querySelector('#sim-thread-log');
      const queueBadge = threadWidget.querySelector('#sim-queue-count');
      const triggerBtn = threadWidget.querySelector('#sim-thread-trigger-btn');
      let seq = 0;
      let queueCount = 0;

      function logThread(msg) {
        if (!threadLog) return;
        const now = new Date();
        const timeStr = String(now.getSeconds()).padStart(2, '0') + '.' + String(now.getMilliseconds()).padStart(3, '0');
        const p = document.createElement('div');
        p.innerHTML = `<span class="log-dim">[00:00:${timeStr}]</span> ${msg}`;
        threadLog.appendChild(p);
        threadLog.scrollTop = threadLog.scrollHeight;
      }

      function produceSample() {
        seq++;
        queueCount++;
        if (queueBadge) queueBadge.textContent = `${queueCount}/10`;
        const temp = (23.0 + (seq % 12) * 0.2).toFixed(1);
        const hum = 45 + (seq % 8);
        logThread(`<span class="log-info">[Producer (Pri 6)]</span> Put payload #${seq} (Temp: ${temp} C, Hum: ${hum}%) into k_msgq`);

        // Consumer immediately preempts because Pri 4 > Pri 6
        setTimeout(() => {
          queueCount = Math.max(0, queueCount - 1);
          if (queueBadge) queueBadge.textContent = `${queueCount}/10`;
          logThread(`<span class="log-hi">[Consumer (Pri 4)]</span> Preempted! Fetched #${seq} -&gt; Dispatched over VCP UART`);
        }, 350);
      }

      logThread(`<span class="log-hi">=== Zephyr Kernel Lab 3: Thread Synchronization &amp; Queues ===</span>`);
      logThread(`Initialized producer_thread (Pri 6) and consumer_thread (Pri 4)`);

      const tInterval = setInterval(produceSample, 3000);
      if (triggerBtn) {
        triggerBtn.addEventListener('click', () => {
          produceSample();
        });
      }
    }

    // --- Lab 4: Sensor & Hardware PWM Breathing LED Simulator ---
    const pwmWidget = document.querySelector('[data-sim="sensor-pwm"]');
    if (pwmWidget) {
      const pwmBulb = pwmWidget.querySelector('#sim-pwm-led');
      const dutyLabel = pwmWidget.querySelector('#sim-pwm-duty');
      const tempVal = pwmWidget.querySelector('#sim-sensor-temp');
      const pressVal = pwmWidget.querySelector('#sim-sensor-press');

      let duty = 0;
      let step = 2;
      let angle = 0;

      function updatePwm() {
        angle += 0.05;
        duty = Math.round(((Math.sin(angle) + 1) / 2) * 100);
        if (pwmBulb) {
          pwmBulb.style.opacity = (0.2 + (duty / 100) * 0.8).toFixed(2);
          pwmBulb.style.boxShadow = `0 0 ${Math.round(duty / 4)}px #38bdf8, 0 0 ${Math.round(duty / 2)}px rgba(56, 189, 248, 0.8)`;
        }
        if (dutyLabel) dutyLabel.textContent = `${duty}%`;

        // Subtle sensor jitter
        if (Math.random() > 0.95 && tempVal) {
          const t = (24.5 + Math.random() * 0.4).toFixed(2);
          tempVal.textContent = `${t} °C`;
        }
        if (Math.random() > 0.95 && pressVal) {
          const p = (1013.1 + Math.random() * 0.5).toFixed(2);
          pressVal.textContent = `${p} hPa`;
        }
        requestAnimationFrame(updatePwm);
      }
      requestAnimationFrame(updatePwm);
    }

    // --- Lab 5: Interactive Zephyr Shell Console Simulator ---
    const shellWidget = document.querySelector('[data-sim="shell-console"]');
    if (shellWidget) {
      const shellLog = shellWidget.querySelector('#sim-shell-log');
      const shellInput = shellWidget.querySelector('#sim-shell-input');
      const shellLed = shellWidget.querySelector('#sim-shell-board-led');
      let shellLedState = false;

      function logShell(line) {
        if (!shellLog) return;
        const p = document.createElement('div');
        p.innerHTML = line;
        shellLog.appendChild(p);
        shellLog.scrollTop = shellLog.scrollHeight;
      }

      function runShellCmd(cmd) {
        const clean = cmd.trim();
        logShell(`<span style="color:#ffffff;">nucleo:~$ ${escapeHtml(clean)}</span>`);

        const lower = clean.toLowerCase();
        if (lower === 'help') {
          logShell(`Available commands:\n  led       : Control onboard LED state (on/off)\n  system    : System telemetry inspection\n  kernel    : Kernel stack and thread inspector\n  clear     : Clear terminal buffer`);
        } else if (lower === 'led on') {
          shellLedState = true;
          if (shellLed) shellLed.classList.add('on');
          logShell(`<span class="log-info">[SHELL]</span> User LD2 (PA5) turned <span class="log-hi">ON</span>`);
        } else if (lower === 'led off') {
          shellLedState = false;
          if (shellLed) shellLed.classList.remove('on');
          logShell(`<span class="log-dim">[SHELL]</span> User LD2 (PA5) turned <span class="log-dim">OFF</span>`);
        } else if (lower === 'system status' || lower === 'system') {
          logShell(`=== System Telemetry Status ===\nUptime: ${Math.round(performance.now())} ms\nTarget Board: nucleo_f401re\nZephyr Kernel: v3.7.0 (ACL LTS)`);
        } else if (lower === 'kernel threads' || lower === 'kernel') {
          logShell(`Threads:\n  0x20000a40  idle        (pri 15)  [READY]\n  0x20001020  shell_uart  (pri 7)   [ACTIVE]\n  0x20001500  logging     (pri 14)  [BLOCKED]`);
        } else if (lower === 'clear') {
          shellLog.innerHTML = '';
        } else {
          logShell(`<span class="log-err">shell: command not found: ${escapeHtml(clean)}</span>. Type 'help' for available commands.`);
        }
      }

      logShell(`*** Zephyr Diagnostic Shell ready. Type 'help' or click commands below ***`);

      if (shellInput) {
        shellInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            runShellCmd(shellInput.value);
            shellInput.value = '';
          }
        });
      }

      shellWidget.querySelectorAll('[data-shell-cmd]').forEach(chip => {
        chip.addEventListener('click', () => {
          const c = chip.getAttribute('data-shell-cmd');
          runShellCmd(c);
        });
      });
    }

    // --- Lab 6: Watchdog Supervisor Simulator ---
    const wdtWidget = document.querySelector('[data-sim="watchdog"]');
    if (wdtWidget) {
      const bar = wdtWidget.querySelector('#sim-wdt-bar');
      const wdtLog = wdtWidget.querySelector('#sim-wdt-log');
      const faultBtn = wdtWidget.querySelector('#sim-wdt-fault-btn');
      const wdtLed = wdtWidget.querySelector('#sim-wdt-led');

      let wdtMs = 2000;
      let faultActive = false;

      function logWdt(msg) {
        if (!wdtLog) return;
        const now = new Date();
        const timeStr = String(now.getSeconds()).padStart(2, '0') + '.' + String(now.getMilliseconds()).padStart(3, '0');
        const p = document.createElement('div');
        p.innerHTML = `<span class="log-dim">[00:00:${timeStr}]</span> ${msg}`;
        wdtLog.appendChild(p);
        wdtLog.scrollTop = wdtLog.scrollHeight;
      }

      logWdt(`<span class="log-hi">Hardware Watchdog active (Timeout: 2000 ms). System guarded.</span>`);

      setInterval(() => {
        if (!faultActive) {
          wdtMs = 2000;
          if (wdtLed) wdtLed.classList.toggle('on');
          logWdt(`<span class="log-info">&lt;inf&gt; wdt:</span> Supervisor heartbeat: Watchdog fed.`);
        }
      }, 700);

      setInterval(() => {
        if (faultActive) {
          wdtMs = Math.max(0, wdtMs - 100);
          if (bar) {
            bar.style.width = `${(wdtMs / 2000) * 100}%`;
            bar.style.background = '#ef4444';
          }
          if (wdtMs === 0) {
            logWdt(`<span class="log-err">🚨 [IWDG TIMEOUT] System lockup detected! Resetting Cortex-M core...</span>`);
            faultActive = false;
            if (faultBtn) faultBtn.textContent = '⚠️ Inject Thread Deadlock';
            setTimeout(() => {
              wdtMs = 2000;
              if (bar) {
                bar.style.width = '100%';
                bar.style.background = '#10b981';
              }
              logWdt(`<span class="log-hi">*** Booting Zephyr OS v3.7.0 (Recovered from Watchdog Reset) ***</span>`);
            }, 1000);
          }
        } else {
          if (bar) {
            bar.style.width = '100%';
            bar.style.background = '#10b981';
          }
        }
      }, 100);

      if (faultBtn) {
        faultBtn.addEventListener('click', () => {
          faultActive = !faultActive;
          faultBtn.textContent = faultActive ? 'Active Fault! (Watchdog Starving)' : '⚠️ Inject Thread Deadlock';
          if (faultActive) {
            logWdt(`<span class="log-warn">&lt;wrn&gt; wdt:</span> Fault injected! Supervisor thread frozen. Feeding stopped.`);
          }
        });
      }
    }
  }

  // --- Global Initialization ---
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initPlatformNote();
    const themeBtn = document.getElementById('theme-toggle-btn');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

    initBoard();
    initReadingProgress();
    initSidebar();
    initAnchorNavigation();
    initScrollspy();
    initCodeCopy();
    initSyntaxHighlighting();
    initSimulations();
    initLabChecklists();
    initQuizzes();
    initSearch();
  });
})();
