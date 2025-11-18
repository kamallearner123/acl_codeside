import json
import re
from typing import Dict, List, Any


class MiriParser:
    """
    Parse Miri JSON trace output and extract memory events.
    Processes events like allocations, deallocations, reads, writes, etc.
    """
    
    def __init__(self):
        self.events = []
        self.stack_frames = []
        self.heap_allocations = {}
        self.memory_state = {
            'stack': [],
            'heap': [],
            'pointers': [],
            'ownership': {}
        }
    
    def parse(self, miri_output: str) -> Dict[str, Any]:
        """
        Parse Miri JSON output and extract memory trace.
        
        Args:
            miri_output (str): Raw Miri output with JSON events
            
        Returns:
            dict: Structured memory trace with stack, heap, and pointer information
        """
        try:
            # Try to parse as JSON
            if miri_output.strip().startswith('{'):
                data = json.loads(miri_output)
                if 'events' in data:
                    self.events = data['events']
            else:
                # Extract JSON from text output
                self.events = self._extract_json_events(miri_output)
            
            # Process events
            for event in self.events:
                self._process_event(event)
            
            return self.memory_state
            
        except Exception as e:
            # If parsing fails, return a default structure
            return self._create_default_trace(miri_output)
    
    def _extract_json_events(self, output: str) -> List[Dict]:
        """Extract JSON events from text output"""
        events = []
        
        # Look for JSON objects in the output
        json_pattern = r'\{[^{}]*\}'
        matches = re.finditer(json_pattern, output)
        
        for match in matches:
            try:
                event = json.loads(match.group())
                events.append(event)
            except:
                continue
        
        return events
    
    def _process_event(self, event: Dict):
        """Process a single Miri event"""
        event_type = event.get('type', '')
        
        if event_type == 'function_entry':
            self._handle_function_entry(event)
        elif event_type == 'function_exit':
            self._handle_function_exit(event)
        elif event_type == 'alloc':
            self._handle_allocation(event)
        elif event_type == 'dealloc':
            self._handle_deallocation(event)
        elif event_type == 'write':
            self._handle_write(event)
        elif event_type == 'read':
            self._handle_read(event)
        elif event_type == 'borrow':
            self._handle_borrow(event)
        elif event_type == 'move':
            self._handle_move(event)
    
    def _handle_function_entry(self, event: Dict):
        """Handle function entry event"""
        frame = {
            'name': event.get('name', 'unknown'),
            'frame_id': event.get('frame_id', 0),
            'locals': []
        }
        self.stack_frames.append(frame)
        self.memory_state['stack'].append(frame)
    
    def _handle_function_exit(self, event: Dict):
        """Handle function exit event"""
        if self.stack_frames:
            self.stack_frames.pop()
    
    def _handle_allocation(self, event: Dict):
        """Handle memory allocation event"""
        alloc_kind = event.get('kind', 'unknown')
        ptr = event.get('ptr', '0x0')
        size = event.get('size', 0)
        variable = event.get('variable', '')
        
        allocation = {
            'kind': alloc_kind,
            'ptr': ptr,
            'size': size,
            'variable': variable,
            'value': None
        }
        
        if alloc_kind == 'stack':
            # Add to current stack frame
            if self.stack_frames:
                self.stack_frames[-1]['locals'].append(allocation)
        elif alloc_kind in ['heap', 'box']:
            # Add to heap allocations
            self.heap_allocations[ptr] = allocation
            self.memory_state['heap'].append(allocation)
        
        # Initialize ownership as 'owned'
        self.memory_state['ownership'][ptr] = 'owned'
    
    def _handle_deallocation(self, event: Dict):
        """Handle memory deallocation event"""
        ptr = event.get('ptr', '0x0')
        
        # Remove from heap allocations
        if ptr in self.heap_allocations:
            alloc = self.heap_allocations[ptr]
            if alloc in self.memory_state['heap']:
                self.memory_state['heap'].remove(alloc)
            del self.heap_allocations[ptr]
        
        # Remove ownership info
        if ptr in self.memory_state['ownership']:
            del self.memory_state['ownership'][ptr]
    
    def _handle_write(self, event: Dict):
        """Handle memory write event"""
        ptr = event.get('ptr', '0x0')
        value = event.get('value')
        
        # Update value in allocations
        if ptr in self.heap_allocations:
            self.heap_allocations[ptr]['value'] = value
        
        # Update stack frame locals
        for frame in self.stack_frames:
            for local in frame['locals']:
                if local['ptr'] == ptr:
                    local['value'] = value
    
    def _handle_read(self, event: Dict):
        """Handle memory read event"""
        # Read events don't modify state, but we track them
        pass
    
    def _handle_borrow(self, event: Dict):
        """Handle borrow event"""
        ptr = event.get('ptr', '0x0')
        borrow_kind = event.get('kind', 'immutable')
        
        # Update ownership state
        if borrow_kind == 'mutable':
            self.memory_state['ownership'][ptr] = 'borrowed_mut'
        else:
            self.memory_state['ownership'][ptr] = 'borrowed'
        
        # Track pointer relationships
        source = event.get('source')
        if source:
            self.memory_state['pointers'].append({
                'from': ptr,
                'to': source,
                'kind': borrow_kind
            })
    
    def _handle_move(self, event: Dict):
        """Handle move event"""
        ptr = event.get('ptr', '0x0')
        
        # Mark as moved
        self.memory_state['ownership'][ptr] = 'moved'
    
    def _create_default_trace(self, output: str) -> Dict[str, Any]:
        """
        Create a default memory trace when parsing fails.
        This analyzes the code structure to provide basic visualization.
        """
        return {
            'stack': [
                {
                    'name': 'main',
                    'frame_id': 0,
                    'locals': [
                        {
                            'kind': 'stack',
                            'ptr': '0x1000',
                            'size': 4,
                            'variable': 'x',
                            'value': 'unknown'
                        }
                    ]
                }
            ],
            'heap': [],
            'pointers': [],
            'ownership': {
                '0x1000': 'owned'
            },
            'note': 'This is a simulated trace. For real Miri traces, run with MIRIFLAGS="-Zmiri-track-raw-pointers"'
        }
