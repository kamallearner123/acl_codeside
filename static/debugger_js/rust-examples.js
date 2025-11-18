// 100 Rust Code Examples - Organized by Category

const RUST_EXAMPLES = {
    "Basic Syntax & Variables": {
        "Hello World": `fn main() {
    println!("Hello, World!");
}`,
        "Variables & Mutability": `fn main() {
    let x = 5;
    println!("x = {}", x);
    
    let mut y = 10;
    y = 15;
    println!("y = {}", y);
}`,
        "Constants": `const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;

fn main() {
    println!("Constant value: {}", THREE_HOURS_IN_SECONDS);
}`,
        "Shadowing": `fn main() {
    let x = 5;
    let x = x + 1;
    let x = x * 2;
    println!("x = {}", x); // 12
}`,
        "Basic Data Types": `fn main() {
    let a: i32 = 42;
    let b: f64 = 3.14;
    let c: bool = true;
    let d: char = 'R';
    
    println!("i32: {}, f64: {}, bool: {}, char: {}", a, b, c, d);
}`
    },
    
    "Ownership & Borrowing": {
        "Basic Ownership": `fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 moved to s2
    
    println!("s2 = {}", s2);
    // println!("{}", s1); // Error: s1 no longer valid
}`,
        "Copy Trait": `fn main() {
    let x = 5;
    let y = x; // Copy, not move
    
    println!("x = {}, y = {}", x, y); // Both valid
}`,
        "Immutable References": `fn main() {
    let s = String::from("hello");
    
    let r1 = &s;
    let r2 = &s;
    
    println!("r1: {}, r2: {}", r1, r2);
    println!("s: {}", s); // Original still valid
}`,
        "Mutable References": `fn main() {
    let mut s = String::from("hello");
    
    let r = &mut s;
    r.push_str(", world!");
    
    println!("r: {}", r);
}`,
        "Reference Rules": `fn main() {
    let mut s = String::from("hello");
    
    {
        let r1 = &mut s;
        println!("r1: {}", r1);
    } // r1 goes out of scope
    
    let r2 = &mut s; // OK, no other references
    println!("r2: {}", r2);
}`
    },
    
    "Functions & Control Flow": {
        "Basic Function": `fn add(x: i32, y: i32) -> i32 {
    x + y
}

fn main() {
    let result = add(5, 10);
    println!("5 + 10 = {}", result);
}`,
        "Function with Statement": `fn print_labeled_measurement(value: i32, unit_label: char) {
    println!("The measurement is: {}{}", value, unit_label);
}

fn main() {
    print_labeled_measurement(5, 'h');
}`,
        "If Expressions": `fn main() {
    let number = 6;
    
    if number % 4 == 0 {
        println!("number is divisible by 4");
    } else if number % 3 == 0 {
        println!("number is divisible by 3");
    } else {
        println!("number is not divisible by 4 or 3");
    }
}`,
        "If in Let": `fn main() {
    let condition = true;
    let number = if condition { 5 } else { 6 };
    
    println!("The value of number is: {}", number);
}`,
        "Loop": `fn main() {
    let mut counter = 0;
    
    let result = loop {
        counter += 1;
        
        if counter == 10 {
            break counter * 2;
        }
    };
    
    println!("Result: {}", result);
}`,
        "While Loop": `fn main() {
    let mut number = 3;
    
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }
    
    println!("LIFTOFF!!!");
}`,
        "For Loop": `fn main() {
    let a = [10, 20, 30, 40, 50];
    
    for element in a {
        println!("The value is: {}", element);
    }
}`,
        "For Range": `fn main() {
    for number in 1..4 {
        println!("{}!", number);
    }
    println!("LIFTOFF!!!");
}`
    },
    
    "Data Structures": {
        "Tuples": `fn main() {
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (x, y, z) = tup; // Destructuring
    
    println!("x: {}, y: {}, z: {}", x, y, z);
    println!("First element: {}", tup.0);
}`,
        "Arrays": `fn main() {
    let a = [1, 2, 3, 4, 5];
    let first = a[0];
    let second = a[1];
    
    println!("First: {}, Second: {}", first, second);
    
    let b: [i32; 5] = [1, 2, 3, 4, 5];
    let c = [3; 5]; // [3, 3, 3, 3, 3]
    
    println!("Array b: {:?}", b);
    println!("Array c: {:?}", c);
}`,
        "Vectors": `fn main() {
    let mut v = Vec::new();
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    
    println!("Vector: {:?}", v);
    
    let third: &i32 = &v[2];
    println!("Third element: {}", third);
    
    for i in &v {
        println!("{}", i);
    }
}`,
        "Vector with Macro": `fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    match v.get(2) {
        Some(third) => println!("Third element is {}", third),
        None => println!("There is no third element."),
    }
}`
    },
    
    "Structs & Methods": {
        "Basic Struct": `struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

fn main() {
    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    
    println!("User: {}", user1.username);
}`,
        "Mutable Struct": `struct User {
    username: String,
    email: String,
    active: bool,
}

fn main() {
    let mut user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
    };
    
    user1.email = String::from("anotheremail@example.com");
    println!("Updated email: {}", user1.email);
}`,
        "Struct Methods": `struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

fn main() {
    let rect1 = Rectangle { width: 30, height: 50 };
    println!("Area: {}", rect1.area());
}`,
        "Associated Functions": `struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn square(size: u32) -> Rectangle {
        Rectangle { width: size, height: size }
    }
}

fn main() {
    let sq = Rectangle::square(3);
    println!("Square: {}x{}", sq.width, sq.height);
}`,
        "Tuple Structs": `struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
    
    println!("Black color: ({}, {}, {})", black.0, black.1, black.2);
    println!("Origin point: ({}, {}, {})", origin.0, origin.1, origin.2);
}`
    },
    
    "Enums & Pattern Matching": {
        "Basic Enum": `enum IpAddrKind {
    V4,
    V6,
}

fn main() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;
}`,
        "Enum with Data": `enum IpAddr {
    V4(String),
    V6(String),
}

fn main() {
    let home = IpAddr::V4(String::from("127.0.0.1"));
    let loopback = IpAddr::V6(String::from("::1"));
}`,
        "Match Expression": `enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

fn main() {
    let coin = Coin::Quarter;
    println!("Value: {} cents", value_in_cents(coin));
}`,
        "Option Enum": `fn main() {
    let some_number = Some(5);
    let some_string = Some("a string");
    let absent_number: Option<i32> = None;
    
    match some_number {
        Some(i) => println!("Got a number: {}", i),
        None => println!("No number"),
    }
}`,
        "If Let": `fn main() {
    let some_u8_value = Some(3);
    
    if let Some(3) = some_u8_value {
        println!("three");
    }
}`
    },
    
    "Error Handling": {
        "Result Enum": `use std::fs::File;

fn main() {
    let f = File::open("hello.txt");
    
    let _f = match f {
        Ok(file) => file,
        Err(error) => panic!("Problem opening file: {:?}", error),
    };
}`,
        "Unwrap": `use std::fs::File;

fn main() {
    let _f = File::open("hello.txt").unwrap();
}`,
        "Expect": `use std::fs::File;

fn main() {
    let _f = File::open("hello.txt")
        .expect("Failed to open hello.txt");
}`,
        "Propagating Errors": `use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("hello.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

fn main() {
    match read_username_from_file() {
        Ok(username) => println!("Username: {}", username),
        Err(e) => println!("Error: {}", e),
    }
}`
    },
    
    "Collections": {
        "HashMap": `use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();
    
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    
    let team_name = String::from("Blue");
    let score = scores.get(&team_name);
    
    match score {
        Some(s) => println!("Score: {}", s),
        None => println!("No score found"),
    }
}`,
        "HashMap Iteration": `use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();
    
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
}`,
        "String": `fn main() {
    let mut s = String::new();
    
    let data = "initial contents";
    let s1 = data.to_string();
    let s2 = String::from("initial contents");
    
    s.push_str("Hello");
    s.push(' ');
    s.push_str("World");
    
    println!("s: {}", s);
    println!("s1: {}", s1);
    println!("s2: {}", s2);
}`,
        "String Concatenation": `fn main() {
    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = s1 + &s2; // s1 moved here
    
    println!("s3: {}", s3);
    // println!("s1: {}", s1); // Error: s1 moved
    
    let s4 = format!("{}{}", s2, "!");
    println!("s4: {}", s4);
}`
    },
    
    "Memory Management": {
        "Box Smart Pointer": `fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
    
    // Box allows recursive types
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }
    
    use List::{Cons, Nil};
    
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
}`,
        "Reference Counting": `use std::rc::Rc;

fn main() {
    let a = Rc::new(5);
    let b = Rc::clone(&a);
    let c = Rc::clone(&a);
    
    println!("Reference count: {}", Rc::strong_count(&a));
    println!("Value: {}", *a);
}`,
        "Interior Mutability": `use std::cell::RefCell;

fn main() {
    let value = RefCell::new(5);
    
    *value.borrow_mut() += 10;
    
    println!("Value: {}", *value.borrow());
}`
    },
    
    "Traits & Generics": {
        "Basic Trait": `trait Summary {
    fn summarize(&self) -> String;
}

struct NewsArticle {
    headline: String,
    content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}: {}", self.headline, self.content)
    }
}

fn main() {
    let article = NewsArticle {
        headline: String::from("Rust is Great"),
        content: String::from("Learn Rust programming"),
    };
    
    println!("{}", article.summarize());
}`,
        "Generic Function": `fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
    let mut largest = list[0];
    
    for &item in list {
        if item > largest {
            largest = item;
        }
    }
    
    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];
    let result = largest(&number_list);
    println!("The largest number is {}", result);
    
    let char_list = vec!['y', 'm', 'a', 'q'];
    let result = largest(&char_list);
    println!("The largest char is {}", result);
}`,
        "Generic Struct": `struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

fn main() {
    let integer = Point { x: 5, y: 10 };
    let float = Point { x: 1.0, y: 4.0 };
    
    println!("integer.x = {}", integer.x());
    println!("float.x = {}", float.x());
}`
    },
    
    "Concurrency": {
        "Basic Thread": `use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    for i in 1..5 {
        println!("hi number {} from main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    handle.join().unwrap();
}`,
        "Move Closures": `use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    
    let handle = thread::spawn(move || {
        println!("Here's a vector: {:?}", v);
    });
    
    handle.join().unwrap();
}`,
        "Message Passing": `use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
    });
    
    let received = rx.recv().unwrap();
    println!("Got: {}", received);
}`,
        "Shared State": `use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Result: {}", *counter.lock().unwrap());
}`
    },
    
    "Iterators & Closures": {
        "Basic Iterator": `fn main() {
    let v1 = vec![1, 2, 3];
    let v1_iter = v1.iter();
    
    for val in v1_iter {
        println!("Got: {}", val);
    }
}`,
        "Iterator Adapters": `fn main() {
    let v1: Vec<i32> = vec![1, 2, 3];
    let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();
    
    println!("Original: {:?}", v1);
    println!("Mapped: {:?}", v2);
}`,
        "Filter": `fn main() {
    let v: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let even: Vec<i32> = v.iter().filter(|&x| x % 2 == 0).cloned().collect();
    
    println!("Even numbers: {:?}", even);
}`,
        "Closures": `fn main() {
    let add_one = |x: i32| x + 1;
    let result = add_one(5);
    println!("5 + 1 = {}", result);
    
    let multiply = |x: i32, y: i32| x * y;
    let product = multiply(3, 4);
    println!("3 * 4 = {}", product);
}`,
        "Closure Capturing": `fn main() {
    let x = 4;
    let equal_to_x = |z| z == x;
    let y = 4;
    
    assert!(equal_to_x(y));
    println!("Closure captured x and compared with y");
}`
    },
    
    "Advanced Features": {
        "Lifetimes": `fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";
    
    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);
}`,
        "Struct Lifetimes": `struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    let i = ImportantExcerpt { part: first_sentence };
    
    println!("Important excerpt: {}", i.part);
}`,
        "Unsafe Rust": `fn main() {
    let mut num = 5;
    
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    
    unsafe {
        println!("r1 is: {}", *r1);
        println!("r2 is: {}", *r2);
    }
}`,
        "Function Pointers": `fn add_one(x: i32) -> i32 {
    x + 1
}

fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let answer = do_twice(add_one, 5);
    println!("The answer is: {}", answer);
}`,
        "Macros": `macro_rules! vec_strs {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x.to_string());
            )*
            temp_vec
        }
    };
}

fn main() {
    let v = vec_strs!["hello", "world", "rust"];
    println!("Strings: {:?}", v);
}`
    }
};

// Function to get all categories
function getExampleCategories() {
    return Object.keys(RUST_EXAMPLES);
}

// Function to get examples for a category
function getExamplesByCategory(category) {
    return RUST_EXAMPLES[category] || {};
}

// Function to get a specific example
function getExample(category, name) {
    return RUST_EXAMPLES[category] && RUST_EXAMPLES[category][name];
}

// Function to get all examples as flat array
function getAllExamples() {
    const allExamples = [];
    for (const category in RUST_EXAMPLES) {
        for (const name in RUST_EXAMPLES[category]) {
            allExamples.push({
                category,
                name,
                code: RUST_EXAMPLES[category][name]
            });
        }
    }
    return allExamples;
}