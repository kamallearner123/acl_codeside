// Python Editor initialization
console.log('🔵 python-editor.js loaded - Version 4 - ' + new Date().toISOString());

let pythonEditor = null;
let pythonEditorReady = false;

// Define functions first
function getEditorCode() {
    return pythonEditor ? pythonEditor.getValue() : '';
}

function setEditorCode(code) {
    console.log('=== setEditorCode called ===');
    console.log('Editor ready:', pythonEditorReady);
    console.log('Editor exists:', pythonEditor !== null);
    
    if (!pythonEditorReady || !pythonEditor) {
        console.warn('⏳ Python editor not ready yet. Waiting...');
        // Wait for editor to initialize
        setTimeout(() => setEditorCode(code), 100);
        return;
    }
    
    pythonEditor.setValue(code);
    console.log('✓ Code set in Python editor');
}

function clearEditor() {
    if (pythonEditor) {
        pythonEditor.setValue('# Start coding here...\n');
    }
}

// Make functions globally available immediately
window.getEditorCode = getEditorCode;
window.setEditorCode = setEditorCode;
window.clearEditor = clearEditor;

console.log('✓ Python editor functions registered globally');

// Initialize Monaco Editor
require.config({ 
    paths: { 
        'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' 
    }
});

require(['vs/editor/editor.main'], function() {
    pythonEditor = monaco.editor.create(document.getElementById('editor'), {
        value: `# Welcome to Python Programming!
# This editor supports Python 3 with popular libraries

import numpy as np
import pandas as pd

# Example: Create a simple array
data = np.array([1, 2, 3, 4, 5])
print(f"Array: {data}")
print(f"Mean: {data.mean()}")
print(f"Sum: {data.sum()}")
`,
        language: 'python',
        theme: 'vs-dark',
        fontSize: 14,
        minimap: { enabled: true },
        automaticLayout: true,
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        tabSize: 4,
        insertSpaces: true,
        lineNumbers: 'on',
        renderWhitespace: 'selection',
        folding: true,
        bracketPairColorization: {
            enabled: true
        }
    });
    
    pythonEditorReady = true;
    console.log('✓ Python Monaco Editor initialized and ready');
});
