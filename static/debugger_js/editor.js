// Monaco Editor initialization and management

let editor = null;
let editorReady = false;

// Default Rust code example
const defaultCode = `fn main() {
    // Stack allocation
    let x = 42;
    let y = 10;
    
    // Heap allocation with Box
    let boxed = Box::new(100);
    
    // Vector (heap allocation)
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);
    
    // String (heap allocation)
    let s = String::from("Hello, Rust!");
    
    // Borrow
    let r = &x;
    
    println!("x = {}", x);
    println!("boxed = {}", boxed);
    println!("vec = {:?}", vec);
    println!("s = {}", s);
    println!("r = {}", r);
}`;

// Example codes organized by topic
const examples = {
    // 1. Basic Syntax & Variables
    basic: defaultCode,
    
    // 2. Ownership - Move Semantics
    ownership_move: `fn main() {
    // Ownership: Move semantics
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    
    // println!("{}", s1); // Error! s1 no longer valid
    println!("s2 = {}", s2);
    
    // Integers implement Copy, so they don't move
    let x = 5;
    let y = x;
    println!("x = {}, y = {}", x, y); // Both valid!
}`,
    
    // 3. Borrowing - Immutable References
    borrowing_immutable: `fn main() {
    // Immutable borrowing
    let s = String::from("hello");
    
    let r1 = &s; // First immutable borrow
    let r2 = &s; // Second immutable borrow
    let r3 = &s; // Multiple immutable borrows OK!
    
    println!("r1: {}, r2: {}, r3: {}", r1, r2, r3);
    println!("Original s: {}", s); // Original still valid
}`,
    
    // 4. Mutable Borrowing
    borrowing_mutable: `fn main() {
    // Mutable borrowing
    let mut s = String::from("hello");
    
    let r = &mut s; // Mutable borrow
    r.push_str(", world!");
    println!("r: {}", r);
    
    // Can use s again after r is done
    println!("s: {}", s);
}`,
    
    // 5. Box - Heap Allocation
    box_heap: `fn main() {
    // Box: Smart pointer for heap allocation
    let b = Box::new(5);
    println!("Boxed value: {}", b);
    
    // Larger data on heap
    let large_data = Box::new([0; 1000]);
    println!("Large array length: {}", large_data.len());
    
    // Box with String
    let boxed_str = Box::new(String::from("Heap String"));
    println!("Boxed string: {}", boxed_str);
}`,
    
    // 6. Vectors - Dynamic Arrays
    vectors: `fn main() {
    // Vec: Dynamic array on the heap
    let mut v = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    
    println!("Vector: {:?}", v);
    
    // Accessing elements
    let third = &v[2];
    println!("Third element: {}", third);
    
    // Iterating
    for i in &v {
        println!("{}", i);
    }
}`,
    
    // 7. Strings - Owned vs Borrowed
    strings: `fn main() {
    // String vs &str
    let s1 = String::from("Hello"); // Owned, heap
    let s2 = "World"; // &str, static/stack
    
    let combined = format!("{} {}", s1, s2);
    println!("{}", combined);
    
    // String manipulation
    let mut s = String::from("foo");
    s.push_str(" bar");
    println!("Mutated: {}", s);
}`,
    
    // 8. Structs - Custom Types
    structs: `struct Point {
    x: i32,
    y: i32,
}

fn main() {
    // Creating a struct
    let p1 = Point { x: 10, y: 20 };
    println!("Point: ({}, {})", p1.x, p1.y);
    
    // Ownership with structs
    let p2 = p1; // p1 moved to p2
    println!("p2: ({}, {})", p2.x, p2.y);
    // println!("{}", p1.x); // Error! p1 moved
}`,
    
    // 9. Enums - Variants
    enums: `enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
}

fn main() {
    let msg1 = Message::Quit;
    let msg2 = Message::Move { x: 10, y: 20 };
    let msg3 = Message::Write(String::from("Hello"));
    
    println!("Created 3 different message variants");
}`,
    
    // 10. Option - Handling None
    option: `fn main() {
    // Option: Rust's way to handle nullable values
    let some_number = Some(5);
    let no_number: Option<i32> = None;
    
    match some_number {
        Some(n) => println!("Got number: {}", n),
        None => println!("No number"),
    }
    
    match no_number {
        Some(n) => println!("Got number: {}", n),
        None => println!("No number"),
    }
}`,
    
    // 11. Result - Error Handling
    result: `fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err(String::from("Division by zero"))
    } else {
        Ok(a / b)
    }
}

fn main() {
    match divide(10, 2) {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
    
    match divide(10, 0) {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}`,
    
    // 12. Lifetimes - References
    lifetimes: `fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let string2 = String::from("short");
    
    let result = longest(&string1, &string2);
    println!("Longest: {}", result);
}`,
    
    // 13. Closures - Anonymous Functions
    closures: `fn main() {
    // Closure: Anonymous function
    let add = |a, b| a + b;
    println!("5 + 3 = {}", add(5, 3));
    
    // Closure capturing environment
    let x = 10;
    let print_x = || println!("x = {}", x);
    print_x();
    
    // Closure with move
    let s = String::from("hello");
    let take_ownership = move || {
        println!("Moved string: {}", s);
    };
    take_ownership();
}`,
    
    // 14. Iterators - Lazy Processing
    iterators: `fn main() {
    // Iterators: Lazy evaluation
    let v = vec![1, 2, 3, 4, 5];
    
    let doubled: Vec<i32> = v.iter()
        .map(|x| x * 2)
        .collect();
    
    println!("Original: {:?}", v);
    println!("Doubled: {:?}", doubled);
    
    let sum: i32 = v.iter().sum();
    println!("Sum: {}", sum);
}`,
    
    // 15. Smart Pointers - Rc (Reference Counting)
    rc_pointer: `use std::rc::Rc;

fn main() {
    // Rc: Multiple ownership
    let a = Rc::new(String::from("hello"));
    println!("Count: {}", Rc::strong_count(&a));
    
    let b = Rc::clone(&a);
    println!("Count: {}", Rc::strong_count(&a));
    
    let c = Rc::clone(&a);
    println!("Count: {}", Rc::strong_count(&a));
    
    println!("Value: {}", a);
}`,
    
    // 16. RefCell - Interior Mutability
    refcell: `use std::cell::RefCell;

fn main() {
    // RefCell: Interior mutability
    let data = RefCell::new(5);
    
    *data.borrow_mut() += 10;
    println!("Value: {}", data.borrow());
    
    *data.borrow_mut() *= 2;
    println!("Value: {}", data.borrow());
}`,
    
    // 17. Traits - Shared Behavior
    traits: `trait Summary {
    fn summarize(&self) -> String;
}

struct Article {
    title: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("Article: {}", self.title)
    }
}

fn main() {
    let article = Article {
        title: String::from("Rust Memory"),
    };
    println!("{}", article.summarize());
}`,
    
    // 18. Pattern Matching
    pattern_matching: `fn main() {
    let number = 7;
    
    match number {
        1 => println!("One"),
        2 | 3 | 5 | 7 => println!("Prime under 10"),
        4 | 6 | 8 | 9 | 10 => println!("Composite under 10"),
        _ => println!("Other number"),
    }
    
    // Destructuring
    let point = (3, 5);
    match point {
        (0, 0) => println!("Origin"),
        (x, 0) => println!("On x-axis at {}", x),
        (0, y) => println!("On y-axis at {}", y),
        (x, y) => println!("Point at ({}, {})", x, y),
    }
}`,
    
    // 19. HashMap - Key-Value Store
    hashmap: `use std::collections::HashMap;

fn main() {
    // HashMap: Key-value storage
    let mut scores = HashMap::new();
    
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Red"), 50);
    
    let team = String::from("Blue");
    let score = scores.get(&team);
    
    match score {
        Some(s) => println!("{}: {}", team, s),
        None => println!("Team not found"),
    }
    
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
}`,
    
    // 20. Threads - Concurrency
    threads: `use std::thread;
use std::time::Duration;

fn main() {
    // Spawning threads
    let handle = thread::spawn(|| {
        for i in 1..5 {
            println!("Thread: {}", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    for i in 1..3 {
        println!("Main: {}", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    handle.join().unwrap();
    println!("Done!");
}`
};

function initEditor() {
    require.config({ 
        paths: { 
            'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' 
        } 
    });
    
    require(['vs/editor/editor.main'], function() {
        editor = monaco.editor.create(document.getElementById('editor'), {
            value: defaultCode,
            language: 'rust',
            theme: 'vs-dark',
            fontSize: 14,
            minimap: { enabled: true },
            scrollBeyondLastLine: false,
            automaticLayout: true,
            lineNumbers: 'on',
            roundedSelection: false,
            readOnly: false,
            cursorStyle: 'line',
            wordWrap: 'on',
        });
        
        // Add keybindings
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, function() {
            runCode();
        });
        
        editorReady = true;
        console.log('✓ Monaco Editor initialized and ready');
    });
}

function getEditorCode() {
    return editor ? editor.getValue() : '';
}

function setEditorCode(code) {
    if (editor) {
        editor.setValue(code);
        console.log('✓ Code set in editor');
    } else {
        console.error('✗ Editor not initialized yet!');
    }
}

function clearEditor() {
    if (editor) {
        editor.setValue('');
    }
}

function loadExample(exampleName = 'basic') {
    console.log('=== loadExample called ===');
    console.log('Example key:', exampleName);
    console.log('Editor ready:', editorReady);
    console.log('Editor exists:', editor !== null);
    console.log('Example exists:', examples[exampleName] !== undefined);
    
    if (!editorReady || !editor) {
        console.warn('⏳ Editor not ready yet. Waiting...');
        // Wait for editor to initialize
        setTimeout(() => loadExample(exampleName), 100);
        return;
    }
    
    if (examples[exampleName]) {
        console.log('✓ Loading example code into editor...');
        setEditorCode(examples[exampleName]);
    } else {
        console.error('✗ Example not found:', exampleName);
        console.log('Available examples:', Object.keys(examples));
    }
}

// Make functions globally available
window.loadExample = loadExample;
window.clearEditor = clearEditor;
window.setEditorCode = setEditorCode;
window.getEditorCode = getEditorCode;

// Initialize editor when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEditor);
} else {
    initEditor();
}
