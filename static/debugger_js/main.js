// Main application logic

let memoryVisualizer = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize memory visualizer
    memoryVisualizer = new MemoryVisualizer('memoryViz');
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('Rust Visual Memory Debugger initialized');
});

function setupEventListeners() {
    // Run button
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
        runBtn.addEventListener('click', runCode);
    }
    
    // Clear button
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            clearEditor();
            clearOutput();
            memoryVisualizer.clear();
            memoryVisualizer.showPlaceholder();
        });
    }
    
    // Example button
    const exampleBtn = document.getElementById('exampleBtn');
    if (exampleBtn) {
        exampleBtn.addEventListener('click', showExampleMenu);
    }
    
    // Clear output button
    const clearOutputBtn = document.getElementById('clearOutputBtn');
    if (clearOutputBtn) {
        clearOutputBtn.addEventListener('click', clearOutput);
    }
    
    // Miri toggle
    const miriToggle = document.getElementById('miriToggle');
    if (miriToggle) {
        miriToggle.addEventListener('change', updateRunButtonText);
        updateRunButtonText(); // Initialize button text
    }
}

function updateRunButtonText() {
    const miriToggle = document.getElementById('miriToggle');
    const runBtn = document.getElementById('runBtn');
    
    if (runBtn && miriToggle) {
        if (miriToggle.checked) {
            runBtn.innerHTML = '<i class="fas fa-play"></i> Run with Miri';
        } else {
            runBtn.innerHTML = '<i class="fas fa-play"></i> Run';
        }
    }
}

function showExampleMenu() {
    const examples = [
        { name: '1. Basic Syntax & Variables', key: 'basic' },
        { name: '2. Ownership - Move Semantics', key: 'ownership_move' },
        { name: '3. Borrowing - Immutable', key: 'borrowing_immutable' },
        { name: '4. Borrowing - Mutable', key: 'borrowing_mutable' },
        { name: '5. Box - Heap Allocation', key: 'box_heap' },
        { name: '6. Vectors - Dynamic Arrays', key: 'vectors' },
        { name: '7. Strings - Owned vs Borrowed', key: 'strings' },
        { name: '8. Structs - Custom Types', key: 'structs' },
        { name: '9. Enums - Variants', key: 'enums' },
        { name: '10. Option - Handling None', key: 'option' },
        { name: '11. Result - Error Handling', key: 'result' },
        { name: '12. Lifetimes - References', key: 'lifetimes' },
        { name: '13. Closures - Anonymous Functions', key: 'closures' },
        { name: '14. Iterators - Lazy Processing', key: 'iterators' },
        { name: '15. Rc - Reference Counting', key: 'rc_pointer' },
        { name: '16. RefCell - Interior Mutability', key: 'refcell' },
        { name: '17. Traits - Shared Behavior', key: 'traits' },
        { name: '18. Pattern Matching', key: 'pattern_matching' },
        { name: '19. HashMap - Key-Value Store', key: 'hashmap' },
        { name: '20. Threads - Concurrency', key: 'threads' }
    ];
    
    const menu = document.createElement('div');
    menu.className = 'example-menu';
    
    const header = document.createElement('h4');
    header.textContent = 'Select an Example';
    header.style.marginBottom = '15px';
    menu.appendChild(header);
    
    const exampleGrid = document.createElement('div');
    exampleGrid.className = 'example-grid';
    exampleGrid.style.maxHeight = '500px';
    exampleGrid.style.overflowY = 'auto';
    exampleGrid.style.display = 'grid';
    exampleGrid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(280px, 1fr))';
    exampleGrid.style.gap = '8px';
    exampleGrid.style.marginBottom = '15px';
    
    examples.forEach(ex => {
        const btn = document.createElement('button');
        btn.className = 'btn btn-secondary btn-sm example-item';
        btn.textContent = ex.name;
        btn.style.textAlign = 'left';
        btn.style.justifyContent = 'flex-start';
        btn.onclick = () => {
            loadExample(ex.key);
            menu.remove();
        };
        exampleGrid.appendChild(btn);
    });
    
    menu.appendChild(exampleGrid);
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn btn-secondary btn-sm';
    closeBtn.textContent = 'Cancel';
    closeBtn.style.width = '100%';
    closeBtn.onclick = () => menu.remove();
    menu.appendChild(closeBtn);
    
    document.body.appendChild(menu);
}

async function runCode() {
    const code = getEditorCode();
    const miriToggle = document.getElementById('miriToggle');
    const useMiri = miriToggle ? miriToggle.checked : true;
    
    if (!code.trim()) {
        showOutput('Error: No code to execute', 'error');
        return;
    }
    
    // Show loading
    showLoading(true, useMiri);
    clearOutput();
    memoryVisualizer.clear();
    
    try {
        const response = await fetch('/coding-practices/execute/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                code: code,
                use_miri: useMiri
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            handleExecutionResult(result, useMiri);
        } else {
            showOutput(`Error: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showOutput(`Network Error: ${error.message}`, 'error');
        console.error('Execution error:', error);
    } finally {
        showLoading(false, useMiri);
    }
}

function handleExecutionResult(result, useMiri) {
    // Display stdout/stderr
    let output = '';
    
    // Show execution mode badge
    const modeBadge = useMiri 
        ? '<span class="mode-badge mode-badge-miri">Miri Mode</span>'
        : '<span class="mode-badge mode-badge-normal">Normal Mode</span>';
    
    output += `<div class="execution-mode-info">${modeBadge}</div>`;
    
    if (result.stdout) {
        output += `<div class="output-section output-stdout">
            <div class="output-label">Standard Output:</div>
            <pre>${escapeHtml(result.stdout)}</pre>
        </div>`;
    }
    
    if (result.stderr) {
        output += `<div class="output-section output-stderr">
            <div class="output-label">Standard Error:</div>
            <pre>${escapeHtml(result.stderr)}</pre>
        </div>`;
    }
    
    if (!result.stdout && !result.stderr) {
        output += '<div class="output-section"><em>No output</em></div>';
    }
    
    showOutput(output, result.success ? 'success' : 'error');
    
    // Visualize memory trace (only in Miri mode)
    if (useMiri && result.memory_trace) {
        memoryVisualizer.visualize(result.memory_trace);
    } else if (!useMiri) {
        // Show info message in memory panel for normal mode
        memoryVisualizer.showMessage('Memory visualization is only available in Miri mode. Toggle Miri to see memory traces.');
    } else {
        memoryVisualizer.showPlaceholder();
    }
}

function showOutput(content, type = 'info') {
    const outputDiv = document.getElementById('output');
    outputDiv.className = 'output-content';
    outputDiv.classList.add(`output-${type}`);
    outputDiv.innerHTML = content;
}

function clearOutput() {
    const outputDiv = document.getElementById('output');
    outputDiv.className = 'output-content';
    outputDiv.innerHTML = `
        <div class="output-placeholder">
            <i class="fas fa-info-circle"></i>
            <p>Output will appear here after running your code</p>
        </div>
    `;
}

function showLoading(show, useMiri = true) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        if (show) {
            const message = useMiri ? 'Executing code with Miri...' : 'Executing code...';
            overlay.innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>${message}</p>
                </div>
            `;
        }
        overlay.style.display = show ? 'flex' : 'none';
    }
    
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
        runBtn.disabled = show;
        if (show) {
            runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
        } else {
            updateRunButtonText();
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to run
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runCode();
    }
    
    // Ctrl/Cmd + K to clear
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        clearEditor();
        clearOutput();
        memoryVisualizer.clear();
        memoryVisualizer.showPlaceholder();
    }
});
