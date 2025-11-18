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
    },
    
    "Testing & Debugging": {
        "Unit Tests": `#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn test_panic() {
        panic!("This is a panic test");
    }
}

fn main() {
    println!("Run: cargo test");
}`,
        "Test with Should Panic": `fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("Division by zero!");
    }
    a / b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "Division by zero!")]
    fn test_divide_by_zero() {
        divide(10, 0);
    }
}

fn main() {
    println!("Result: {}", divide(10, 2));
}`,
        "Assert Macros": `fn main() {
    let x = 5;
    let y = 5;
    
    assert!(x == y);
    assert_eq!(x, y);
    assert_ne!(x, 3);
    
    let name = "Rust";
    assert!(name.contains("Ru"), "Name should contain 'Ru'");
    
    println!("All assertions passed!");
}`,
        "Debug Trait": `#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p = Point { x: 10, y: 20 };
    println!("Debug: {:?}", p);
    println!("Pretty: {:#?}", p);
    
    let vec = vec![1, 2, 3, 4, 5];
    println!("Vector debug: {:?}", vec);
}`
    },
    
    "File I/O & Modules": {
        "Reading Files": `use std::fs;

fn main() {
    let contents = fs::read_to_string("Cargo.toml")
        .expect("Something went wrong reading the file");
    
    println!("File contents:\\n{}", contents);
}`,
        "Writing Files": `use std::fs::File;
use std::io::prelude::*;

fn main() -> std::io::Result<()> {
    let mut file = File::create("hello.txt")?;
    file.write_all(b"Hello, file!")?;
    
    println!("File written successfully!");
    Ok(())
}`,
        "Module Declaration": `mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }
    
    pub fn multiply(a: i32, b: i32) -> i32 {
        a * b
    }
}

fn main() {
    let sum = math::add(5, 3);
    let product = math::multiply(4, 6);
    
    println!("Sum: {}, Product: {}", sum, product);
}`,
        "Using External Crates": `// Add to Cargo.toml: rand = "0.8"
use std::io;

fn main() {
    println!("Guess the number!");
    println!("Please input your guess:");
    
    let mut guess = String::new();
    io::stdin().read_line(&mut guess)
        .expect("Failed to read line");
        
    println!("You guessed: {}", guess);
}`,
        "Path and Use": `use std::collections::HashMap;
use std::io::Result;

mod network {
    pub mod server {
        pub fn connect() {
            println!("Connected to server");
        }
    }
}

fn main() {
    let mut map = HashMap::new();
    map.insert("key", "value");
    
    network::server::connect();
    
    println!("Map: {:?}", map);
}`
    },
    
    "Pattern Matching Advanced": {
        "Match Guards": `fn main() {
    let num = Some(4);
    
    match num {
        Some(x) if x < 5 => println!("less than five: {}", x),
        Some(x) => println!("{}", x),
        None => (),
    }
    
    let x = 4;
    let y = false;
    
    match x {
        4 | 5 | 6 if y => println!("yes"),
        _ => println!("no"),
    }
}`,
        "Destructuring": `struct Point { x: i32, y: i32 }

fn main() {
    let p = Point { x: 0, y: 7 };
    
    let Point { x: a, y: b } = p;
    println!("a: {}, b: {}", a, b);
    
    match p {
        Point { x, y: 0 } => println!("On the x axis at {}", x),
        Point { x: 0, y } => println!("On the y axis at {}", y),
        Point { x, y } => println!("On neither axis: ({}, {})", x, y),
    }
}`,
        "Ignoring Values": `fn main() {
    let numbers = (2, 4, 8, 16, 32);
    
    match numbers {
        (first, _, third, _, fifth) => {
            println!("Some numbers: {}, {}, {}", first, third, fifth)
        },
    }
    
    let mut setting_value = Some(5);
    let new_setting_value = Some(10);
    
    match (setting_value, new_setting_value) {
        (Some(_), Some(_)) => {
            println!("Can't overwrite an existing customized value");
        }
        _ => {
            setting_value = new_setting_value;
        }
    }
}`,
        "At Bindings": `enum Message {
    Hello { id: i32 },
}

fn main() {
    let msg = Message::Hello { id: 5 };
    
    match msg {
        Message::Hello { id: id_variable @ 3..=7 } => {
            println!("Found an id in range: {}", id_variable)
        },
        Message::Hello { id: 10..=12 } => {
            println!("Found an id in another range")
        },
        Message::Hello { id } => {
            println!("Found some other id: {}", id)
        },
    }
}`
    },
    
    "Custom Types & Implementations": {
        "Custom Display": `use std::fmt;

struct Point {
    x: i32,
    y: i32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let point = Point { x: 1, y: 2 };
    println!("Display: {}", point);
}`,
        "Custom Operators": `use std::ops::Add;

#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;
    
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = Point { x: 3, y: 4 };
    let p3 = p1 + p2;
    
    println!("Result: {:?}", p3);
}`,
        "Newtype Pattern": `struct Millimeters(u32);
struct Meters(u32);

impl Millimeters {
    fn to_meters(&self) -> Meters {
        Meters(self.0 / 1000)
    }
}

fn main() {
    let length = Millimeters(5000);
    let meters = length.to_meters();
    
    println!("{}mm = {}m", length.0, meters.0);
}`,
        "Type Aliases": `type Kilometers = i32;
type Result<T> = std::result::Result<T, std::io::Error>;

fn main() {
    let distance: Kilometers = 100;
    println!("Distance: {}km", distance);
    
    let _result: Result<String> = Ok("success".to_string());
}`,
        "Associated Types": `trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    current: usize,
    max: usize,
}

impl Counter {
    fn new(max: usize) -> Counter {
        Counter { current: 0, max }
    }
}

impl Iterator for Counter {
    type Item = usize;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.max {
            let current = self.current;
            self.current += 1;
            Some(current)
        } else {
            None
        }
    }
}

fn main() {
    let mut counter = Counter::new(3);
    
    while let Some(val) = counter.next() {
        println!("Counter: {}", val);
    }
}`
    },
    
    "Async Programming": {
        "Basic Async": `async fn hello_world() {
    println!("Hello, async world!");
}

fn main() {
    // Note: This is a simplified example
    // In real code, you'd use an async runtime like tokio
    println!("Sync before async");
    
    // Async functions return Future that need to be awaited
    let future = hello_world();
    
    println!("Sync after creating future");
}`,
        "Future and Await": `use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct SimpleFuture;

impl Future for SimpleFuture {
    type Output = String;
    
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready("Future completed!".to_string())
    }
}

fn main() {
    let future = SimpleFuture;
    // In real async runtime, this would be awaited
    println!("Created future");
}`
    },
    
    "Numeric & Mathematical": {
        "Number Parsing": `fn main() {
    let num_str = "42";
    let num: i32 = num_str.parse().expect("Not a number!");
    println!("Parsed: {}", num);
    
    let float_str = "3.14";
    let float: f64 = float_str.parse().unwrap_or(0.0);
    println!("Float: {}", float);
    
    // With error handling
    match "not_a_number".parse::<i32>() {
        Ok(n) => println!("Number: {}", n),
        Err(_) => println!("Failed to parse"),
    }
}`,
        "Math Operations": `fn main() {
    let x = 10;
    let y = 3;
    
    println!("Addition: {}", x + y);
    println!("Subtraction: {}", x - y);
    println!("Multiplication: {}", x * y);
    println!("Division: {}", x / y);
    println!("Remainder: {}", x % y);
    
    // Floating point
    let a = 10.0;
    let b = 3.0;
    println!("Float division: {}", a / b);
    
    // Powers and roots
    println!("Power: {}", x.pow(2));
    println!("Square root: {}", (64.0_f64).sqrt());
}`,
        "Random Numbers": `// This example shows the structure, would need rand crate
use std::collections::HashMap;

fn pseudo_random(seed: u64) -> u64 {
    // Simple pseudo-random number generator
    seed.wrapping_mul(1103515245).wrapping_add(12345)
}

fn main() {
    let mut seed = 12345;
    
    for i in 0..5 {
        seed = pseudo_random(seed);
        let random_num = seed % 100;
        println!("Random {}: {}", i, random_num);
    }
    
    // Simulating dice roll
    seed = pseudo_random(seed);
    let dice = (seed % 6) + 1;
    println!("Dice roll: {}", dice);
}`,
        "Bitwise Operations": `fn main() {
    let a = 0b1010; // 10 in binary
    let b = 0b1100; // 12 in binary
    
    println!("a = {:04b} ({})", a, a);
    println!("b = {:04b} ({})", b, b);
    println!("a & b = {:04b} ({})", a & b, a & b);
    println!("a | b = {:04b} ({})", a | b, a | b);
    println!("a ^ b = {:04b} ({})", a ^ b, a ^ b);
    println!("!a = {:04b} ({})", !a, !a);
    println!("a << 1 = {:04b} ({})", a << 1, a << 1);
    println!("a >> 1 = {:04b} ({})", a >> 1, a >> 1);
}`
    },
    
    "Date & Time": {
        "Duration": `use std::time::{Duration, Instant};
use std::thread;

fn main() {
    let start = Instant::now();
    
    // Simulate some work
    thread::sleep(Duration::from_millis(100));
    
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    
    // Working with durations
    let five_seconds = Duration::from_secs(5);
    let thirty_millis = Duration::from_millis(30);
    
    println!("5 seconds = {:?}", five_seconds);
    println!("30 millis = {:?}", thirty_millis);
    
    if duration < five_seconds {
        println!("That was quick!");
    }
}`,
        "Measuring Time": `use std::time::{SystemTime, UNIX_EPOCH, Instant};

fn main() {
    // System time
    let now = SystemTime::now();
    let since_epoch = now.duration_since(UNIX_EPOCH).expect("Time went backwards");
    println!("Seconds since Unix epoch: {}", since_epoch.as_secs());
    
    // High resolution timing
    let start = Instant::now();
    
    // Do some work
    for i in 0..1000 {
        let _ = i * i;
    }
    
    let elapsed = start.elapsed();
    println!("Operation took: {:?}", elapsed);
    println!("Nanoseconds: {}", elapsed.as_nanos());
}`
    },
    
    "Command Line": {
        "Command Line Args": `use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    println!("Program name: {}", args[0]);
    
    if args.len() > 1 {
        println!("Arguments:");
        for (i, arg) in args.iter().enumerate().skip(1) {
            println!("  {}: {}", i, arg);
        }
    } else {
        println!("No arguments provided");
    }
    
    // Environment variables
    match env::var("HOME") {
        Ok(home) => println!("Home directory: {}", home),
        Err(_) => println!("HOME not set"),
    }
}`,
        "Exit Codes": `use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <number>", args[0]);
        process::exit(1);
    }
    
    let number: i32 = match args[1].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: '{}' is not a valid number", args[1]);
            process::exit(2);
        }
    };
    
    if number < 0 {
        eprintln!("Error: Number must be positive");
        process::exit(3);
    }
    
    println!("Square of {} is {}", number, number * number);
    process::exit(0);
}`,
        "Process Management": `use std::process::{Command, Stdio};

fn main() {
    // Execute a simple command
    let output = Command::new("echo")
        .arg("Hello from Rust!")
        .output()
        .expect("Failed to execute command");
    
    println!("Output: {}", String::from_utf8_lossy(&output.stdout));
    
    // Execute with input/output handling
    let mut child = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to start cat");
    
    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        stdin.write_all(b"Hello from pipe!").expect("Failed to write");
    }
    
    let output = child.wait_with_output().expect("Failed to read output");
    println!("Pipe output: {}", String::from_utf8_lossy(&output.stdout));
}`
    },
    
    "Network & Web": {
        "TCP Client": `use std::io::prelude::*;
use std::net::TcpStream;

fn main() {
    match TcpStream::connect("127.0.0.1:7878") {
        Ok(mut stream) => {
            println!("Successfully connected to server");
            
            let msg = "Hello, server!";
            stream.write(msg.as_bytes()).unwrap();
            
            let mut data = [0 as u8; 50];
            match stream.read(&mut data) {
                Ok(_) => {
                    println!("Reply: {}", String::from_utf8_lossy(&data[..]));
                },
                Err(e) => {
                    println!("Failed to receive data: {}", e);
                }
            }
        },
        Err(e) => {
            println!("Failed to connect: {}", e);
        }
    }
}`,
        "HTTP Request": `// This example shows structure, real code would need reqwest crate
use std::collections::HashMap;

fn simulate_http_get(url: &str) -> Result<String, &'static str> {
    if url.starts_with("http://") || url.starts_with("https://") {
        Ok(format!("Response from {}", url))
    } else {
        Err("Invalid URL")
    }
}

fn main() {
    let url = "https://api.example.com/data";
    
    match simulate_http_get(url) {
        Ok(response) => println!("Response: {}", response),
        Err(e) => println!("Error: {}", e),
    }
    
    // Headers simulation
    let mut headers = HashMap::new();
    headers.insert("User-Agent", "Rust-App/1.0");
    headers.insert("Accept", "application/json");
    
    println!("Headers: {:?}", headers);
}`
    },
    
    "Performance & Optimization": {
        "Box for Heap Allocation": `fn main() {
    let x = 5;
    let y = Box::new(x);
    
    println!("x = {}", x);
    println!("y = {}", y);
    
    // Large data on heap
    let large_array = Box::new([0; 1000]);
    println!("Array length: {}", large_array.len());
    
    // Box with custom struct
    #[derive(Debug)]
    struct Point { x: f64, y: f64 }
    
    let point = Box::new(Point { x: 1.0, y: 2.0 });
    println!("Point: {:?}", point);
}`,
        "Rc Reference Counting": `use std::rc::Rc;

fn main() {
    let value = Rc::new(String::from("Hello, Rc!"));
    
    println!("Reference count: {}", Rc::strong_count(&value));
    
    {
        let value2 = Rc::clone(&value);
        let value3 = Rc::clone(&value);
        
        println!("Reference count: {}", Rc::strong_count(&value));
        println!("Value: {}", value2);
        println!("Value: {}", value3);
    }
    
    println!("Reference count after scope: {}", Rc::strong_count(&value));
    println!("Final value: {}", value);
}`,
        "Zero Copy with Cow": `use std::borrow::Cow;

fn process_text(input: &str) -> Cow<str> {
    if input.contains("bad") {
        Cow::Owned(input.replace("bad", "good"))
    } else {
        Cow::Borrowed(input)
    }
}

fn main() {
    let text1 = "This is good text";
    let text2 = "This is bad text";
    
    let result1 = process_text(text1);
    let result2 = process_text(text2);
    
    println!("Result 1: {}", result1);
    println!("Result 2: {}", result2);
    
    // Check if borrowed or owned
    match result1 {
        Cow::Borrowed(_) => println!("Text1 was borrowed"),
        Cow::Owned(_) => println!("Text1 was owned"),
    }
    
    match result2 {
        Cow::Borrowed(_) => println!("Text2 was borrowed"),
        Cow::Owned(_) => println!("Text2 was owned"),
    }
}`
    },
    
    "Unsafe Rust": {
        "Raw Pointers": `fn main() {
    let mut num = 5;
    
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    
    unsafe {
        println!("r1 is: {}", *r1);
        println!("r2 is: {}", *r2);
        
        *r2 = 10;
        println!("num is now: {}", num);
    }
    
    // Arbitrary memory address (don't do this in real code!)
    let address = 0x012345usize;
    let r = address as *const i32;
    
    // This would be unsafe and likely crash:
    // unsafe {
    //     println!("Value at arbitrary address: {}", *r);
    // }
    
    println!("Created pointer to address: {:p}", r);
}`,
        "Calling Unsafe Functions": `unsafe fn dangerous() {
    println!("This is an unsafe function!");
}

fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    
    assert!(mid <= len);
    
    unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn main() {
    unsafe {
        dangerous();
    }
    
    let mut v = vec![1, 2, 3, 4, 5, 6];
    let (left, right) = split_at_mut(&mut v, 3);
    
    println!("Left: {:?}", left);
    println!("Right: {:?}", right);
}`
    },
    
    "FFI & C Integration": {
        "External C Function": `extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Absolute value of -3 according to C: {}", abs(-3));
    }
}

// Calling Rust from C
#[no_mangle]
pub extern "C" fn call_from_c() {
    println!("Just got called from C!");
}

// Function that C can call
#[no_mangle]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}`,
        "Static Libraries": `// Creating a static library
#[no_mangle]
pub extern "C" fn rust_function(x: i32) -> i32 {
    x * 2
}

// Union for FFI
#[repr(C)]
union Value {
    integer: i32,
    float: f32,
}

fn main() {
    let val = Value { integer: 42 };
    
    unsafe {
        println!("Integer: {}", val.integer);
        
        let val2 = Value { float: 3.14 };
        println!("Float: {}", val2.float);
    }
}`
    },
    
    "Serialization": {
        "JSON Simulation": `use std::collections::HashMap;

#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
    email: String,
}

impl Person {
    fn to_json_string(&self) -> String {
        format!(
            r#"{{"name": "{}", "age": {}, "email": "{}"}}"#,
            self.name, self.age, self.email
        )
    }
    
    fn from_json_string(json: &str) -> Result<Person, &'static str> {
        // Simple parser simulation (real code would use serde)
        if json.contains("\"name\"") && json.contains("\"age\"") {
            Ok(Person {
                name: "Parsed Name".to_string(),
                age: 25,
                email: "parsed@example.com".to_string(),
            })
        } else {
            Err("Invalid JSON")
        }
    }
}

fn main() {
    let person = Person {
        name: "Alice".to_string(),
        age: 30,
        email: "alice@example.com".to_string(),
    };
    
    let json = person.to_json_string();
    println!("JSON: {}", json);
    
    let parsed = Person::from_json_string(&json).unwrap();
    println!("Parsed: {:?}", parsed);
}`,
        "Binary Serialization": `use std::mem;

#[repr(C)]
struct Data {
    id: u32,
    value: f64,
    flag: bool,
}

impl Data {
    fn to_bytes(&self) -> Vec<u8> {
        unsafe {
            let ptr = self as *const Data as *const u8;
            let slice = std::slice::from_raw_parts(ptr, mem::size_of::<Data>());
            slice.to_vec()
        }
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Data, &'static str> {
        if bytes.len() != mem::size_of::<Data>() {
            return Err("Invalid size");
        }
        
        unsafe {
            let ptr = bytes.as_ptr() as *const Data;
            Ok(ptr.read())
        }
    }
}

fn main() {
    let data = Data {
        id: 42,
        value: 3.14159,
        flag: true,
    };
    
    let bytes = data.to_bytes();
    println!("Serialized {} bytes", bytes.len());
    
    let restored = Data::from_bytes(&bytes).unwrap();
    println!("Restored: id={}, value={}, flag={}", restored.id, restored.value, restored.flag);
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