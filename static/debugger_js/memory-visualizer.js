// Memory visualization module

class MemoryVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.memoryTrace = null;
    }
    
    visualize(memoryTrace) {
        this.memoryTrace = memoryTrace;
        this.clear();
        
        if (!memoryTrace || (!memoryTrace.stack && !memoryTrace.heap)) {
            this.showPlaceholder();
            return;
        }
        
        const wrapper = document.createElement('div');
        wrapper.className = 'memory-wrapper';
        
        // Create stack visualization
        const stackSection = this.createStackSection(memoryTrace.stack || []);
        wrapper.appendChild(stackSection);
        
        // Create heap visualization
        const heapSection = this.createHeapSection(memoryTrace.heap || []);
        wrapper.appendChild(heapSection);
        
        // Create pointers visualization
        const pointerSection = this.createPointerSection(memoryTrace.pointers || []);
        wrapper.appendChild(pointerSection);
        
        // Add note if present
        if (memoryTrace.note) {
            const note = document.createElement('div');
            note.className = 'memory-note';
            note.innerHTML = `<i class="fas fa-info-circle"></i> ${memoryTrace.note}`;
            wrapper.appendChild(note);
        }
        
        this.container.appendChild(wrapper);
    }
    
    createStackSection(stackFrames) {
        const section = document.createElement('div');
        section.className = 'memory-section stack-section';
        
        const header = document.createElement('h3');
        header.innerHTML = '<i class="fas fa-layer-group"></i> Stack';
        section.appendChild(header);
        
        if (!stackFrames || stackFrames.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'empty-section';
            empty.textContent = 'No stack frames';
            section.appendChild(empty);
            return section;
        }
        
        stackFrames.forEach((frame, index) => {
            const frameDiv = this.createStackFrame(frame, index);
            section.appendChild(frameDiv);
        });
        
        return section;
    }
    
    createStackFrame(frame, index) {
        const frameDiv = document.createElement('div');
        frameDiv.className = 'stack-frame';
        
        const frameName = document.createElement('div');
        frameName.className = 'frame-name';
        frameName.textContent = `Frame ${index}: ${frame.name || 'unknown'}`;
        frameDiv.appendChild(frameName);
        
        if (frame.locals && frame.locals.length > 0) {
            const localsDiv = document.createElement('div');
            localsDiv.className = 'locals';
            
            frame.locals.forEach(local => {
                const localDiv = this.createVariable(local);
                localsDiv.appendChild(localDiv);
            });
            
            frameDiv.appendChild(localsDiv);
        } else {
            const empty = document.createElement('div');
            empty.className = 'empty-locals';
            empty.textContent = 'No local variables';
            frameDiv.appendChild(empty);
        }
        
        return frameDiv;
    }
    
    createHeapSection(heapAllocations) {
        const section = document.createElement('div');
        section.className = 'memory-section heap-section';
        
        const header = document.createElement('h3');
        header.innerHTML = '<i class="fas fa-database"></i> Heap';
        section.appendChild(header);
        
        if (!heapAllocations || heapAllocations.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'empty-section';
            empty.textContent = 'No heap allocations';
            section.appendChild(empty);
            return section;
        }
        
        heapAllocations.forEach(allocation => {
            const allocDiv = this.createVariable(allocation);
            section.appendChild(allocDiv);
        });
        
        return section;
    }
    
    createVariable(variable) {
        const varDiv = document.createElement('div');
        varDiv.className = 'variable';
        varDiv.setAttribute('data-ptr', variable.ptr);
        
        // Get ownership status
        const ownership = this.getOwnership(variable.ptr);
        varDiv.classList.add(`ownership-${ownership}`);
        
        const varName = document.createElement('div');
        varName.className = 'variable-name';
        varName.textContent = variable.variable || 'unnamed';
        varDiv.appendChild(varName);
        
        const varInfo = document.createElement('div');
        varInfo.className = 'variable-info';
        
        const typeSpan = document.createElement('span');
        typeSpan.className = 'variable-type';
        typeSpan.textContent = variable.kind || 'unknown';
        varInfo.appendChild(typeSpan);
        
        const valueSpan = document.createElement('span');
        valueSpan.className = 'variable-value';
        valueSpan.textContent = variable.value !== null && variable.value !== undefined 
            ? `= ${variable.value}` 
            : '';
        varInfo.appendChild(valueSpan);
        
        varDiv.appendChild(varInfo);
        
        const addressSpan = document.createElement('div');
        addressSpan.className = 'variable-address';
        addressSpan.textContent = `${variable.ptr} (${variable.size} bytes)`;
        varDiv.appendChild(addressSpan);
        
        return varDiv;
    }
    
    createPointerSection(pointers) {
        const section = document.createElement('div');
        section.className = 'memory-section pointer-section';
        
        const header = document.createElement('h3');
        header.innerHTML = '<i class="fas fa-arrows-alt"></i> Pointers & References';
        section.appendChild(header);
        
        if (!pointers || pointers.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'empty-section';
            empty.textContent = 'No active pointers';
            section.appendChild(empty);
            return section;
        }
        
        pointers.forEach(pointer => {
            const pointerDiv = document.createElement('div');
            pointerDiv.className = 'pointer';
            
            const icon = pointer.kind === 'mutable' 
                ? '<i class="fas fa-arrow-right"></i>' 
                : '<i class="fas fa-arrow-right" style="opacity: 0.6;"></i>';
            
            pointerDiv.innerHTML = `
                ${icon}
                <span class="pointer-from">${pointer.from}</span>
                →
                <span class="pointer-to">${pointer.to}</span>
                <span class="pointer-kind">(${pointer.kind})</span>
            `;
            
            section.appendChild(pointerDiv);
        });
        
        return section;
    }
    
    getOwnership(ptr) {
        if (!this.memoryTrace || !this.memoryTrace.ownership) {
            return 'owned';
        }
        return this.memoryTrace.ownership[ptr] || 'owned';
    }
    
    clear() {
        this.container.innerHTML = '';
    }
    
    showPlaceholder() {
        this.container.innerHTML = `
            <div class="memory-placeholder">
                <i class="fas fa-memory"></i>
                <p>Memory visualization will appear here after running your code</p>
            </div>
        `;
    }
    
    showMessage(message) {
        this.container.innerHTML = `
            <div class="memory-placeholder">
                <i class="fas fa-info-circle"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

// Export for use in main.js
window.MemoryVisualizer = MemoryVisualizer;
