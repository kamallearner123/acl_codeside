"""
This file contains the comprehensive curriculum for the Python Tutor application.
It is organized into 16 main sections, each containing approximately 10 detailed examples/topics 
to provide in-depth understanding.
"""

CURRICULUM = [
    # -------------------------------------------------------------------------
    # 1. Introduction to Python
    # -------------------------------------------------------------------------
    {
        "id": "intro",
        "title": "1. Introduction to Python",
        "description": "Foundations: Syntax, Variables, Data Types, and Ops.",
        "topics": [
            {
                "title": "1.1 Hello World & Print",
                "content": "The print function is your first tool. It sends output to the console. You can print strings, numbers, and variables.",
                "code": "# 1. Basic string\nprint(\"Hello world!\")\n\n# 2. Printing numbers\nprint(2025)\n\n# 3. Printing multiple items\nprint(\"Python\", \"is\", \"fun\")"
            },
            {
                "title": "1.2 Separator & End Parameters",
                "content": "Customize print behavior using `sep` (separator between items) and `end` (what to print at the end, default is newline).",
                "code": "print(\"A\", \"B\", \"C\", sep=\"-\")\nprint(\"Line 1\", end=\" \")\nprint(\"Line 2 (same line)\")"
            },
            {
                "title": "1.3 Variables & Assignment",
                "content": "Variables store data. Python is dynamically typed, so you don't need to declare types.",
                "code": "x = 10         # Integer\nname = \"Alice\" # String\npi = 3.14      # Float\nprint(f\"{name} likes {pi}\")"
            },
            {
                "title": "1.4 Multiple Assignment",
                "content": "Assign values to multiple variables in one line. Useful for initializing or swapping values.",
                "code": "a, b, c = 1, 2, 3\nprint(a, b, c)\n\n# Swapping\nx, y = 10, 20\nx, y = y, x\nprint(f\"x: {x}, y: {y}\")"
            },
            {
                "title": "1.5 Data Types: Integers & Floats",
                "content": "Math operations behave differently depending on types. Floats have decimal points.",
                "code": "# Basic Math\nprint(10 + 5)   # Add\nprint(10 / 3)   # True division (float)\nprint(10 // 3)  # Floor division (int)\nprint(2 ** 3)   # Power"
            },
            {
                "title": "1.6 Data Types: Strings",
                "content": "Strings can be created with single, double, or triple quotes (multiline).",
                "code": "s1 = 'Single'\ns2 = \"Double\"\ns3 = \"\"\"Triple\nLine\nString\"\"\"\nprint(s3)"
            },
            {
                "title": "1.7 Data Types: Booleans",
                "content": "Booleans represent truth values: `True` or `False`. Results of comparisons.",
                "code": "is_active = True\nis_admin = False\nprint(10 > 5)  # True\nprint(10 == 9) # False"
            },
            {
                "title": "1.8 Type Checking",
                "content": "Use `type()` to inspect the data type of a variable.",
                "code": "x = 100\ny = \"100\"\nprint(type(x))\nprint(type(y))\nprint(isinstance(x, int))"
            },
            {
                "title": "1.9 Type Conversion (Casting)",
                "content": "Convert between types explicitly using `int()`, `str()`, `float()`.",
                "code": "s = \"50\"\nn = int(s) + 50\nprint(n)\n\nf = float(\"3.5\")\nprint(f * 2)\n\nmsg = \"Age: \" + str(25)\nprint(msg)"
            },
            {
                "title": "1.10 User Input",
                "content": "Use `input()` to get data from user. It always returns a string.",
                "code": "# Simulating input in this environment\n# name = input(\"Enter name: \")\nname = \"Guest\" # Hardcoded for demo\nprint(f\"Welcome, {name}!\")"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 2. Control Flow
    # -------------------------------------------------------------------------
    {
        "id": "control",
        "title": "2. Control Flow",
        "description": "Making decisions with conditionals and repeating tasks with loops.",
        "topics": [
            {
                "title": "2.1 The 'if' Statement",
                "content": "Execute code only if a condition is true.",
                "code": "age = 20\nif age >= 18:\n    print(\"You can vote.\")"
            },
            {
                "title": "2.2 'else' Statement",
                "content": "What to do if the condition is false.",
                "code": "temp = 15\nif temp > 20:\n    print(\"Warm\")\nelse:\n    print(\"Cold\")"
            },
            {
                "title": "2.3 'elif' Ladder",
                "content": "Check multiple conditions in sequence.",
                "code": "score = 75\nif score >= 90:\n    print(\"A\")\nelif score >= 70:\n    print(\"B\")\nelif score >= 50:\n    print(\"C\")\nelse:\n    print(\"F\")"
            },
            {
                "title": "2.4 Nested Conditionals",
                "content": "If statements inside other if statements.",
                "code": "x = 10\nif x > 0:\n    print(\"Positive\")\n    if x % 2 == 0:\n        print(\"Even\")\n    else:\n        print(\"Odd\")"
            },
            {
                "title": "2.5 Ternary Operator",
                "content": "One-line if-else for simple assignments.",
                "code": "age = 20\nstatus = \"Adult\" if age >= 18 else \"Minor\"\nprint(status)"
            },
            {
                "title": "2.6 'for' Loop with Ranges",
                "content": "Iterate a specific number of times.",
                "code": "for i in range(5):\n    print(i, end=\" \")\nprint(\"\\nSteps:\")\nfor i in range(0, 10, 2):\n    print(i, end=\" \")"
            },
            {
                "title": "2.7 'for' Loop over Iterables",
                "content": "Loop directly through lists or strings.",
                "code": "word = \"Python\"\nfor char in word:\n    print(char, end=\"-\")\n\nprint(\"\\nItems:\")\nfor item in [\"A\", \"B\", \"C\"]:\n    print(item)"
            },
            {
                "title": "2.8 'while' Loop",
                "content": "Keep running as long as condition is true.",
                "code": "count = 3\nwhile count > 0:\n    print(f\"Counting... {count}\")\n    count -= 1\nprint(\"Liftoff!\")"
            },
            {
                "title": "2.9 Break & Continue",
                "content": "`break` stops the loop. `continue` skips to next iteration.",
                "code": "print(\"Break at 3:\")\nfor i in range(10):\n    if i == 3: break\n    print(i, end=\" \")\n\nprint(\"\\nSkip 2:\")\nfor i in range(4):\n    if i == 2: continue\n    print(i, end=\" \")"
            },
            {
                "title": "2.10 Pass Statement",
                "content": "`pass` does nothing. Use it as a placeholder.",
                "code": "def future_feature():\n    pass # Todo\n\nfor i in range(5):\n    if i == 3:\n        pass # Do nothing special\n    print(i, end=\" \")"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 3. Functions
    # -------------------------------------------------------------------------
    {
        "id": "funcs",
        "title": "3. Functions",
        "description": "Reusable blocks of code, scoping, and advanced arguments.",
        "topics": [
            {
                "title": "3.1 Basic Definition",
                "content": "Define with `def`. Indent the body.",
                "code": "def say_hi():\n    print(\"Hi there!\")\n\nsay_hi()\nsay_hi()"
            },
            {
                "title": "3.2 Parameters & Arguments",
                "content": "Pass data into functions.",
                "code": "def greet(name):\n    print(f\"Hello, {name}\")\n\ngreet(\"Alice\")\ngreet(\"Bob\")"
            },
            {
                "title": "3.3 Return Values",
                "content": "Functions can return results to the caller.",
                "code": "def square(x):\n    return x * x\n\nres = square(4)\nprint(f\"Result: {res}\")"
            },
            {
                "title": "3.4 Default Arguments",
                "content": "Make arguments optional by providing defaults.",
                "code": "def power(num, exponent=2):\n    return num ** exponent\n\nprint(power(3))    # 3^2\nprint(power(3, 3)) # 3^3"
            },
            {
                "title": "3.5 Keyword Arguments",
                "content": "Pass arguments by name for clarity/order independence.",
                "code": "def describe(name, role):\n    print(f\"{name} is a {role}\")\n\ndescribe(role=\"Developer\", name=\"Kamal\")"
            },
            {
                "title": "3.6 Arbitrary Arguments (*args)",
                "content": "Accept a variable number of positional arguments.",
                "code": "def add_all(*args):\n    total = 0\n    for num in args:\n        total += num\n    return total\n\nprint(add_all(1, 2, 3, 4))"
            },
            {
                "title": "3.7 Arbitrary Keywords (**kwargs)",
                "content": "Accept dictionary of named arguments.",
                "code": "def print_data(**kwargs):\n    for k, v in kwargs.items():\n        print(f\"{k}: {v}\")\n\nprint_data(id=1, status=\"Active\")"
            },
            {
                "title": "3.8 Scope (Local vs Global)",
                "content": "Variables defined inside function are local.",
                "code": "g = \"Global\"\n\ndef test():\n    l = \"Local\"\n    print(l, g)\n\ntest()\n# print(l) # Error if uncommented"
            },
            {
                "title": "3.9 Lambda Functions",
                "content": "One-line anonymous functions.",
                "code": "double = lambda x: x * 2\nprint(double(5))\n\nadd = lambda a, b: a + b\nprint(add(3, 7))"
            },
            {
                "title": "3.10 Recursion",
                "content": "A function that calls itself.",
                "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n\nprint([fib(i) for i in range(6)])"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 4. Data Structures
    # -------------------------------------------------------------------------
    {
        "id": "datastruct",
        "title": "4. Data Structures",
        "description": "Lists, Tuples, Sets, Dictionaries, and Strings.",
        "topics": [
            {
                "title": "4.1 String Methods",
                "content": "Powerful text manipulation.",
                "code": "s = \"  Data Science  \"\nprint(s.upper())\nprint(s.lower())\nprint(s.strip())\nprint(s.replace(\"Data\", \"Computer\"))"
            },
            {
                "title": "4.2 String Slicing",
                "content": "Extract substrings `[start:end:step]`.",
                "code": "text = \"Python Programming\"\nprint(text[0:6])   # Python\nprint(text[-1])    # g\nprint(text[::-1])  # Reverse"
            },
            {
                "title": "4.3 List Creation & Access",
                "content": "Ordered mutable collection.",
                "code": "lst = [1, \"apple\", 3.5]\nprint(lst[0])\nprint(lst[1])\nlst[1] = \"banana\"\nprint(lst)"
            },
            {
                "title": "4.4 List Methods",
                "content": "Add, remove, and sort.",
                "code": "nums = [3, 1, 4, 1, 5]\nnums.append(9)\nnums.sort()\nprint(nums)\np = nums.pop()\nprint(f\"Popped: {p}, List: {nums}\")"
            },
            {
                "title": "4.5 Tuples",
                "content": "Immutable lists. Faster and safer for fixed data.",
                "code": "point = (10, 20)\nprint(point[0])\n# point[0] = 30 # Error!\n\n# Unpacking\nx, y = point\nprint(f\"x={x}, y={y}\")"
            },
            {
                "title": "4.6 Sets Basics",
                "content": "Unordered collection of unique items.",
                "code": "s = {1, 2, 2, 3}\nprint(s) # Duplicates removed\ns.add(4)\nprint(s)"
            },
            {
                "title": "4.7 Set Operations",
                "content": "Math set theory: Union, Intersection.",
                "code": "a = {1, 2, 3}\nb = {3, 4, 5}\nprint(f\"Union: {a | b}\")\nprint(f\"Intersection: {a & b}\")\nprint(f\"Diff: {a - b}\")"
            },
            {
                "title": "4.8 Dictionary Basics",
                "content": "Key-Value pairs.",
                "code": "person = {\"name\": \"John\", \"age\": 30}\nprint(person[\"name\"])\nperson[\"city\"] = \"New York\"\nprint(person)"
            },
            {
                "title": "4.9 Dict Methods",
                "content": "Keys, Values, Items, Get.",
                "code": "d = {\"a\": 1, \"b\": 2}\nprint(list(d.keys()))\nprint(list(d.values()))\n\n# Safe access\nprint(d.get(\"c\", \"Not Found\"))"
            },
            {
                "title": "4.10 Nested Structures",
                "content": "Combining lists and dicts.",
                "code": "users = [\n    {\"id\": 1, \"tags\": [\"admin\", \"staff\"]},\n    {\"id\": 2, \"tags\": [\"user\"]}\n]\nprint(users[0][\"tags\"][1])"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 5. File Handling
    # -------------------------------------------------------------------------
    {
        "id": "fileio",
        "title": "5. File Handling",
        "description": "Reading and writing to the file system.",
        "topics": [
            {
                "title": "5.1 Writing Text Files",
                "content": "Use 'w' mode. Creates or overwrites.",
                "code": "f = open(\"demo.txt\", \"w\")\nf.write(\"Hello File World!\\n\")\nf.close()\nprint(\"File written.\")"
            },
            {
                "title": "5.2 Reading Text Files",
                "content": "Use 'r' mode.",
                "code": "# Assuming demo.txt exists from previous step\nf = open(\"demo.txt\", \"r\")\ncontent = f.read()\nprint(f\"Read: {content}\")\nf.close()"
            },
            {
                "title": "5.3 Appending to Files",
                "content": "Use 'a' mode to add to end without deleting.",
                "code": "f = open(\"demo.txt\", \"a\")\nf.write(\"New Line Attached.\\n\")\nf.close()"
            },
            {
                "title": "5.4 The 'with' Statement",
                "content": "Best practice. Automatically closes file.",
                "code": "with open(\"demo.txt\", \"r\") as f:\n    print(f.read())"
            },
            {
                "title": "5.5 Reading Lines",
                "content": "`readline()` or iterating file object.",
                "code": "with open(\"demo.txt\", \"r\") as f:\n    lines = f.readlines()\n    print(lines)"
            },
            {
                "title": "5.6 Write Multiple Lines",
                "content": "`writelines()` takes a list of strings.",
                "code": "data = [\"One\\n\", \"Two\\n\", \"Three\\n\"]\nwith open(\"list.txt\", \"w\") as f:\n    f.writelines(data)"
            },
            {
                "title": "5.7 Binary Files",
                "content": "Use 'rb' or 'wb' for images/data.",
                "code": "# Writing bytes\nwith open(\"binary.dat\", \"wb\") as f:\n    f.write(b'\\xDE\\xAD\\xBE\\xEF')\nprint(\"Binary written.\")"
            },
            {
                "title": "5.8 Check if File Exists",
                "content": "Using `os.path`.",
                "code": "import os\nif os.path.exists(\"demo.txt\"):\n    print(\"Found it!\")"
            },
            {
                "title": "5.9 Deleting Files",
                "content": "`os.remove()`.",
                "code": "import os\n# os.remove(\"demo.txt\")\nprint(\"Code commented out for safety\")"
            },
            {
                "title": "5.10 Context Manager Custom",
                "content": "How `with` actually works (intro).",
                "code": "class MyOpen:\n    def __enter__(self):\n        print(\"Opening...\")\n    def __exit__(self, type, val, tb):\n        print(\"Closing...\")\n\nwith MyOpen() as x:\n    print(\"Doing work\")"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 6. Exception Handling
    # -------------------------------------------------------------------------
    {
        "id": "exceptions",
        "title": "6. Exception Handling",
        "description": "Making code robust against crash scenarios.",
        "topics": [
            {
                "title": "6.1 Basic Try-Except",
                "content": "Catch errors to prevent crash.",
                "code": "try:\n    print(1/0)\nexcept ZeroDivisionError:\n    print(\"Divided by zero!\")"
            },
            {
                "title": "6.2 Catching Any Error",
                "content": "Use `except Exception`.",
                "code": "try:\n    lst = []\n    print(lst[1])\nexcept Exception as e:\n    print(f\"Caught error: {e}\")"
            },
            {
                "title": "6.3 Multiple Except Blocks",
                "content": "Handle specific errors differently.",
                "code": "try:\n    val = int(\"abc\")\nexcept IndexError:\n    print(\"Index Error\")\nexcept ValueError:\n    print(\"Value Error\")"
            },
            {
                "title": "6.4 The 'else' Block",
                "content": "Runs only if NO exception occurred.",
                "code": "try:\n    res = 10 / 2\nexcept:\n    print(\"Error\")\nelse:\n    print(f\"Good result: {res}\")"
            },
            {
                "title": "6.5 The 'finally' Block",
                "content": "Runs no matter what.",
                "code": "try:\n    print(\"Working...\")\nfinally:\n    print(\"Always runs (cleanup)\")"
            },
            {
                "title": "6.6 Raising Exceptions",
                "content": "Trigger your own errors.",
                "code": "def check(age):\n    if age < 0:\n        raise ValueError(\"Age cannot be negative\")\n\ntry:\n    check(-5)\nexcept ValueError as e:\n    print(e)"
            },
            {
                "title": "6.7 Custom Exceptions",
                "content": "Inherit from `Exception` class.",
                "code": "class MyError(Exception): pass\n\ntry:\n    raise MyError(\"My custom message\")\nexcept MyError as e:\n    print(e)"
            },
            {
                "title": "6.8 Assertions",
                "content": "Debugging aid.",
                "code": "x = 10\ntry:\n    assert x == 5, \"x should be 5\"\nexcept AssertionError as e:\n    print(f\"Assert failed: {e}\")"
            },
            {
                "title": "6.9 Context of Exception",
                "content": "Accessing traceback (simplified).",
                "code": "import sys\ntry:\n    1/0\nexcept:\n    print(sys.exc_info()[0])"
            },
            {
                "title": "6.10 Chaining Exceptions",
                "content": "raise ... from ...",
                "code": "try:\n    try:\n        1/0\n    except ZeroDivisionError:\n        raise ValueError(\"Math fail\") from None\nexcept ValueError as e:\n    print(e)"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 7. Modules and Packages
    # -------------------------------------------------------------------------
    {
        "id": "modules",
        "title": "7. Modules and Packages",
        "description": "Code organization and libraries.",
        "topics": [
            {
                "title": "7.1 Standard Import",
                "content": "Importing a whole module.",
                "code": "import math\nprint(math.sqrt(25))"
            },
            {
                "title": "7.2 Import Specific Function",
                "content": "`from X import Y`.",
                "code": "from math import pi\nprint(pi)"
            },
            {
                "title": "7.3 Import with Alias",
                "content": "Shorten module names.",
                "code": "import datetime as dt\nprint(dt.date.today())"
            },
            {
                "title": "7.4 Creating a Module",
                "content": "Any .py file is a module.",
                "code": "# imagine my_functions.py exists\n# import my_functions\nprint(\"Import works by file name\")"
            },
            {
                "title": "7.5 Dir() Function",
                "content": "List contents of a module.",
                "code": "import math\nprint(dir(math)[:10]) # First 10 items"
            },
            {
                "title": "7.6 Packages Structure",
                "content": "Folders with `__init__.py`.",
                "code": "# from mypackage import submodule\nprint(\"A package is a folder of modules\")"
            },
            {
                "title": "7.7 Installing Packages (pip)",
                "content": "Use `pip install X` in terminal.",
                "code": "# Not python code, but terminal:\n# pip install requests"
            },
            {
                "title": "7.8 Random Module",
                "content": "Commonly used std lib.",
                "code": "import random\nprint(random.randint(1, 10))\nprint(random.choice(['A', 'B']))"
            },
            {
                "title": "7.9 Sys Path",
                "content": "Where python looks for files.",
                "code": "import sys\n# print(sys.path) # List of paths"
            },
            {
                "title": "7.10 Module Reloading",
                "content": "Advanced usage for dev.",
                "code": "import importlib\n# importlib.reload(math)"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 8. OOP
    # -------------------------------------------------------------------------
    {
        "id": "oop",
        "title": "8. Object-Oriented Programming",
        "description": "Class structure, inheritance, and magic methods.",
        "topics": [
            {
                "title": "8.1 Class Definition",
                "content": "Blueprint for objects.",
                "code": "class Cat:\n    pass\n\nc = Cat()\nprint(c)"
            },
            {
                "title": "8.2 The __init__ Method",
                "content": "Constructor to initialize attributes.",
                "code": "class Person:\n    def __init__(self, name):\n        self.name = name\n\np = Person(\"Kamal\")\nprint(p.name)"
            },
            {
                "title": "8.3 Instance Methods",
                "content": "Functions belonging to object.",
                "code": "class Dog:\n    def bark(self):\n        return \"Woof!\"\n\nd = Dog()\nprint(d.bark())"
            },
            {
                "title": "8.4 The 'self' Parameter",
                "content": "Refers to current instance. explicit in Python.",
                "code": "class Calc:\n    def set_val(self, v):\n        self.val = v\n    def get_val(self):\n        return self.val\n\nc = Calc()\nc.set_val(50)\nprint(c.get_val())"
            },
            {
                "title": "8.5 Inheritance",
                "content": "Child class inherits Parent.",
                "code": "class A:\n    def say(self): print(\"A\")\n    \nclass B(A):\n    pass\n\nobj = B()\nobj.say()"
            },
            {
                "title": "8.6 Overriding Methods",
                "content": "Child changes specific behavior.",
                "code": "class Bird:\n    def fly(self): print(\"Flying\")\n    \nclass Penguin(Bird):\n    def fly(self): print(\"Cannot fly\")\n    \np = Penguin()\np.fly()"
            },
            {
                "title": "8.7 Polymorphism",
                "content": "Different classes, same interface.",
                "code": "for animal in [Dog(), Cat()]:\n    # if both have make_sound()\n    pass"
            },
            {
                "title": "8.8 Encapsulation",
                "content": "Private variables with `__` prefix.",
                "code": "class Bank:\n    def __init__(self):\n        self.__money = 1000\n    \n    def get_bal(self):\n        return self.__money\n\nb = Bank()\nprint(b.get_bal())"
            },
            {
                "title": "8.9 Class vs Instance Var",
                "content": "Shared vs Unique.",
                "code": "class Team:\n    org = \"Google\" # Class var\n    def __init__(self, name):\n        self.name = name # Instance var\n\nprint(Team.org)"
            },
            {
                "title": "8.10 Magic Methods (__str__)",
                "content": "String representation.",
                "code": "class Box:\n    def __str__(self):\n        return \"I am a Box\"\n\nprint(Box())"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 9. Iterators
    # -------------------------------------------------------------------------
    {
        "id": "iters",
        "title": "9. Iterators and Generators",
        "description": "Lazy evaluation patterns.",
        "topics": [
            {
                "title": "9.1 What is an Iterator?",
                "content": "Object with `__next__`.",
                "code": "nums = [1, 2]\nit = iter(nums)\nprint(next(it))\nprint(next(it))"
            },
            {
                "title": "9.2 Creating Iterators",
                "content": "Class with `__iter__` and `__next__`.",
                "code": "class Counter:\n    def __init__(self, high):\n        self.current = 0\n        self.high = high\n    def __iter__(self): return self\n    def __next__(self):\n        if self.current < self.high:\n            self.current += 1\n            return self.current\n        raise StopIteration\n\nfor c in Counter(3):\n    print(c)"
            },
            {
                "title": "9.3 Simple Generator",
                "content": "Function with `yield`.",
                "code": "def my_gen():\n    yield 1\n    yield 2\n\nfor x in my_gen():\n    print(x)"
            },
            {
                "title": "9.4 Generator State",
                "content": "It remembers where it left off.",
                "code": "g = my_gen()\nprint(next(g))\nprint(next(g))"
            },
            {
                "title": "9.5 Infinite Generators",
                "content": "Careful with loops!",
                "code": "def inf():\n    i = 0\n    while True:\n        yield i\n        i += 1\n# Don't run in for-loop without break"
            },
            {
                "title": "9.6 Generator Expressions",
                "content": "One-liner generator.",
                "code": "sq = (x*x for x in range(5))\nprint(next(sq))"
            },
            {
                "title": "9.7 Yield From",
                "content": "Delegate to sub-generator.",
                "code": "def sub():\n    yield \"A\"\n    yield \"B\"\n\ndef top():\n    yield from sub()\n    yield \"C\"\n\nprint(list(top()))"
            },
            {
                "title": "9.8 Memory Efficiency",
                "content": "List vs Generator",
                "code": "import sys\nl = [i for i in range(1000)]\ng = (i for i in range(1000))\nprint(sys.getsizeof(l))\nprint(sys.getsizeof(g))"
            },
            {
                "title": "9.9 Pipelining",
                "content": "Chaining generators.",
                "code": "nums = (i for i in range(10))\nevens = (i for i in nums if i%2 == 0)\nprint(list(evens))"
            },
            {
                "title": "9.10 Send to Generator",
                "content": "Advanced: `gen.send()`.",
                "code": "def talk():\n    val = yield \"Ready\"\n    yield f\"Received {val}\"\n\ng = talk()\nprint(next(g))\nprint(g.send(\"Test\"))"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 10. Functional Programming
    # -------------------------------------------------------------------------
    {
        "id": "functional",
        "title": "10. Functional Programming",
        "description": "Pure functions, map, filter, reduce.",
        "topics": [
            {
                "title": "10.1 Lambda Introduction",
                "content": "Anonymous function.",
                "code": "f = lambda x, y: x + y\nprint(f(2, 3))"
            },
            {
                "title": "10.2 Map Function",
                "content": "Apply func to list.",
                "code": "nums = [1, 2, 3]\nsq = map(lambda x: x**2, nums)\nprint(list(sq))"
            },
            {
                "title": "10.3 Filter Function",
                "content": "Select items.",
                "code": "nums = [1, 2, 3, 4]\nevens = filter(lambda x: x%2==0, nums)\nprint(list(evens))"
            },
            {
                "title": "10.4 Reduce Function",
                "content": "Accumulate result.",
                "code": "from functools import reduce\nnums = [1, 2, 3, 4]\ntotal = reduce(lambda acc, x: acc + x, nums)\nprint(total)"
            },
            {
                "title": "10.5 List Comprehension",
                "content": "Pythonic functional style.",
                "code": "sq = [x**2 for x in range(5)]\nprint(sq)"
            },
            {
                "title": "10.6 Dict Comprehension",
                "content": "Creating maps.",
                "code": "keys = ['a', 'b']\nd = {k: k.upper() for k in keys}\nprint(d)"
            },
            {
                "title": "10.7 First Class Functions",
                "content": "Passing functions as args.",
                "code": "def apply(func, val):\n    return func(val)\n\nprint(apply(len, \"Hello\"))"
            },
            {
                "title": "10.8 Closures",
                "content": "Inner function remembers outer vars.",
                "code": "def outer(msg):\n    def inner():\n        print(msg)\n    return inner\n\nf = outer(\"Secret\")\nf()"
            },
            {
                "title": "10.9 Partial Functions",
                "content": "Pre-fill arguments.",
                "code": "from functools import partial\ndef power(base, exp):\n    return base ** exp\n\nsquare = partial(power, exp=2)\nprint(square(10))"
            },
            {
                "title": "10.10 Immutability Concept",
                "content": "Avoiding side effects.",
                "code": "t = (1, 2, 3)\n# t[0] = 5 # Error, keeps data safe\nprint(t)"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 11. Regex
    # -------------------------------------------------------------------------
    {
        "id": "regex",
        "title": "11. Regular Expressions",
        "description": "Pattern matching text. Powerful and cryptic.",
        "topics": [
            {
                "title": "11.1 Basic Match",
                "content": "Start of string.",
                "code": "import re\nif re.match(\"a\", \"apple\"):\n    print(\"Matches start\")"
            },
            {
                "title": "11.2 Search",
                "content": "Anywhere in string.",
                "code": "import re\nif re.search(\"na\", \"banana\"):\n    print(\"Found na\")"
            },
            {
                "title": "11.3 Find All",
                "content": "Return list of matches.",
                "code": "import re\ntext = \"10 apples, 20 pears\"\nnums = re.findall(r\"\\d+\", text)\nprint(nums)"
            },
            {
                "title": "11.4 Split",
                "content": "Split by pattern.",
                "code": "import re\ntext = \"a,b;c d\"\nparts = re.split(r\"[,; ]\", text)\nprint(parts)"
            },
            {
                "title": "11.5 Substitute",
                "content": "Replace pattern.",
                "code": "import re\n# Remove digits\ncln = re.sub(r\"\\d\", \"\", \"H3ll0\")\nprint(cln)"
            },
            {
                "title": "11.6 Character Classes",
                "content": "`[a-z]`, `\\d`, `\\w`.",
                "code": "import re\nprint(re.findall(r\"[A-Z]\", \"Hello World\"))"
            },
            {
                "title": "11.7 Quantifiers",
                "content": "`+`, `*`, `?`, `{n}`.",
                "code": "import re\n# a followed by one or more b\nprint(re.findall(r\"ab+\", \"abbb ab\"))"
            },
            {
                "title": "11.8 Groups",
                "content": "Extract sub-parts `()`.",
                "code": "import re\nemail = \"user@host.com\"\nm = re.search(r\"(.+)@(.+)\", email)\nprint(m.group(1), m.group(2))"
            },
            {
                "title": "11.9 Anchors",
                "content": "`^` (Start), `$` (End).",
                "code": "import re\nprint(re.search(r\"^Hi\", \"Hi there\"))"
            },
            {
                "title": "11.10 Compile",
                "content": "Pre-compile pattern for speed.",
                "code": "import re\npat = re.compile(r\"\\d{3}\")\nprint(pat.search(\"123 abc\"))"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 12. Concurrency
    # -------------------------------------------------------------------------
    {
        "id": "threading",
        "title": "12. Multithreading & Multiprocessing",
        "description": "Running tasks in parallel.",
        "topics": [
            {
                "title": "12.1 Creating a Thread",
                "content": "Basic usage.",
                "code": "import threading\ndef task(): print(\"Thread running\")\nt = threading.Thread(target=task)\nt.start()"
            },
            {
                "title": "12.2 Join Thread",
                "content": "Wait for finish.",
                "code": "t.join()\nprint(\"Thread finished\")"
            },
            {
                "title": "12.3 Thread Args",
                "content": "Passing data.",
                "code": "t = threading.Thread(target=print, args=(\"Hello\",))\nt.start()"
            },
            {
                "title": "12.4 Race Condition Info",
                "content": "When threads collision.",
                "code": "# Concept only: multiple threads changing same var needs locks"
            },
            {
                "title": "12.5 Locks",
                "content": "Synchronization.",
                "code": "lock = threading.Lock()\nwith lock:\n    print(\"Safe section\")"
            },
            {
                "title": "12.6 Daemon Threads",
                "content": "Background threads.",
                "code": "t = threading.Thread(target=task, daemon=True)\nt.start()"
            },
            {
                "title": "12.7 Multiprocessing Basic",
                "content": "Separate process.",
                "code": "import multiprocessing\n# p = multiprocessing.Process(target=task)\n# p.start()"
            },
            {
                "title": "12.8 Queue",
                "content": "Safe communication.",
                "code": "from queue import Queue\nq = Queue()\nq.put(5)\nprint(q.get())"
            },
            {
                "title": "12.9 ThreadPoolExecutor",
                "content": "Modern threading.",
                "code": "from concurrent.futures import ThreadPoolExecutor\nwith ThreadPoolExecutor(2) as exe:\n    exe.submit(print, \"Task\")"
            },
            {
                "title": "12.10 GIL",
                "content": "Global Interpreter Lock info.",
                "code": "print(\"Python threads are limited by GIL on CPU tasks\")"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 13. Sockets
    # -------------------------------------------------------------------------
    {
        "id": "sockets",
        "title": "13. Socket Programming",
        "description": "Networking basics.",
        "topics": [
            {
                "title": "13.1 Get Hostname",
                "content": "Identity.",
                "code": "import socket\nprint(socket.gethostname())"
            },
            {
                "title": "13.2 Create Socket",
                "content": "TCP object.",
                "code": "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nprint(\"Socket Created\")"
            },
            {
                "title": "13.3 Bind (Concept)",
                "content": "Server binds to port.",
                "code": "# s.bind(('localhost', 8888))"
            },
            {
                "title": "13.4 Listen",
                "content": "Wait for conn.",
                "code": "# s.listen(5)"
            },
            {
                "title": "13.5 Accept",
                "content": "Handle conn.",
                "code": "# conn, addr = s.accept()"
            },
            {
                "title": "13.6 Connect (Client)",
                "content": "Connect to server.",
                "code": "# s.connect(('google.com', 80))"
            },
            {
                "title": "13.7 Send Data",
                "content": "Bytes.",
                "code": "# s.send(b'Hello')"
            },
            {
                "title": "13.8 Receive Data",
                "content": "Buffer size.",
                "code": "# msg = s.recv(1024)"
            },
            {
                "title": "13.9 Close",
                "content": "Cleanup.",
                "code": "# s.close()"
            },
            {
                "title": "13.10 URL Lib",
                "content": "Higher level http.",
                "code": "import urllib.request\n# resp = urllib.request.urlopen('http://google.com')"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 14. Std Libs
    # -------------------------------------------------------------------------
    {
        "id": "stdlibs",
        "title": "14. Python Standard Libraries",
        "description": "Batteries included.",
        "topics": [
            {
                "title": "14.1 Datetime",
                "content": "Current time.",
                "code": "import datetime\nprint(datetime.datetime.now())"
            },
            {
                "title": "14.2 Date Arithmetic",
                "content": "Deltas.",
                "code": "from datetime import timedelta\nprint(timedelta(days=5))"
            },
            {
                "title": "14.3 Math",
                "content": "Constants & funcs.",
                "code": "import math\nprint(math.factorial(5))"
            },
            {
                "title": "14.4 OS Path",
                "content": "Paths.",
                "code": "import os\nprint(os.path.join(\"folder\", \"file.txt\"))"
            },
            {
                "title": "14.5 Sys Args",
                "content": "CLI arguments.",
                "code": "import sys\nprint(sys.argv)"
            },
            {
                "title": "14.6 JSON Dump",
                "content": "Dict to String.",
                "code": "import json\nprint(json.dumps([1, 2]))"
            },
            {
                "title": "14.7 JSON Load",
                "content": "String to Dict.",
                "code": "print(json.loads('[1, 2]'))"
            },
            {
                "title": "14.8 CSV Reading",
                "content": "Parsing CSV.",
                "code": "import csv\n# reader = csv.reader(file)"
            },
            {
                "title": "14.9 Statistics",
                "content": "Mean, median.",
                "code": "import statistics\nprint(statistics.mean([1, 2, 3]))"
            },
            {
                "title": "14.10 Time",
                "content": "Sleep.",
                "code": "import time\n# time.sleep(1)"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 15. Data Analysis
    # -------------------------------------------------------------------------
    {
        "id": "analysis",
        "title": "15. Data Analysis & Visualization",
        "description": "Data science primer.",
        "topics": [
            {
                "title": "15.1 Pandas Series",
                "content": "1D Data.",
                "code": "# import pandas as pd\n# s = pd.Series([1, 2])"
            },
            {
                "title": "15.2 DataFrame",
                "content": "2D Data.",
                "code": "# df = pd.DataFrame({'a': [1]})"
            },
            {
                "title": "15.3 Head/Tail",
                "content": "Inspect data.",
                "code": "# df.head()"
            },
            {
                "title": "15.4 Filtering",
                "content": "Select rows.",
                "code": "# df[df['a'] > 0]"
            },
            {
                "title": "15.5 GroupBy",
                "content": "Aggregation.",
                "code": "# df.groupby('col').mean()"
            },
            {
                "title": "15.6 Matplotlib Plot",
                "content": "Line chart.",
                "code": "# plt.plot(x, y)"
            },
            {
                "title": "15.7 Scatter Plot",
                "content": "Points.",
                "code": "# plt.scatter(x, y)"
            },
            {
                "title": "15.8 Histogram",
                "content": "Distribution.",
                "code": "# plt.hist(data)"
            },
            {
                "title": "15.9 Labels",
                "content": "Titles.",
                "code": "# plt.title(\"My Chart\")"
            },
            {
                "title": "15.10 Show",
                "content": "Render.",
                "code": "# plt.show()"
            }
        ]
    },

    # -------------------------------------------------------------------------
    # 16. Advanced
    # -------------------------------------------------------------------------
    {
        "id": "final",
        "title": "16. Advanced Topics & Final Project",
        "description": "Structure and Testing.",
        "topics": [
            {
                "title": "16.1 Project Layout",
                "content": "src, tests, docs.",
                "code": "# mkdir src tests"
            },
            {
                "title": "16.2 Unittest Class",
                "content": "Inherit TestCase.",
                "code": "import unittest\nclass Test(unittest.TestCase):\n    pass"
            },
            {
                "title": "16.3 Assertions",
                "content": "assertEqual.",
                "code": "# self.assertEqual(1, 1)"
            },
            {
                "title": "16.4 Pytest",
                "content": "Simpler testing.",
                "code": "# def test_func(): assert 1==1"
            },
            {
                "title": "16.5 Virtual Envs",
                "content": "Isolation.",
                "code": "# python -m venv .venv"
            },
            {
                "title": "16.6 Requirements.txt",
                "content": "Dependencies.",
                "code": "# pip freeze > requirements.txt"
            },
            {
                "title": "16.7 Logging",
                "content": "Better than print.",
                "code": "import logging\nlogging.warning(\"Watch out!\")"
            },
            {
                "title": "16.8 Type Hinting",
                "content": "Modern Python.",
                "code": "def f(x: int) -> int:\n    return x"
            },
            {
                "title": "16.9 Docstrings",
                "content": "Documentation.",
                "code": "def func():\n    \"\"\"This is docs\"\"\"\n    pass"
            },
            {
                "title": "16.10 Final Project Idea",
                "content": "Build a ToDo CLI.",
                "code": "# Combine all skills: File IO, Classes, Argparse"
            }
        ]
    }
]
