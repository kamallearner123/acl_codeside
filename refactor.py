import os
import re

files_to_process = [
    'templates/courses/detail.html',
    'templates/services/list.html',
    'templates/services/detail.html',
    'templates/blogs/list.html',
    'templates/blogs/detail.html',
    'templates/events/list.html',
    'templates/events/detail.html',
    'templates/contact/contact.html',
    'debugger/templates/debugger/home.html'
]

# Regex patterns
header_pattern = re.compile(r'^.*?<nav.*?</nav>\s*', re.IGNORECASE | re.DOTALL)
footer_pattern = re.compile(r'\s*<footer.*?>.*</footer>\s*</body>\s*</html>\s*$', re.IGNORECASE | re.DOTALL)

for filepath in files_to_process:
    full_path = os.path.join('/home/kamal/Documents/1.Github/acl_codeside', filepath)
    if not os.path.exists(full_path):
        print(f"Not found: {full_path}")
        continue
        
    with open(full_path, 'r') as f:
        content = f.read()
        
    # Check if already processed
    if "{% extends 'tailwind_base.html' %}" in content:
        print(f"Already processed: {filepath}")
        continue
        
    original = content
        
    # Replace header
    content = header_pattern.sub("{% extends 'tailwind_base.html' %}\n{% load static %}\n\n{% block content %}\n", content)
    
    # Check if it worked (if the original has {% load static %} at the top, we want to make sure it's handled properly)
    if content.startswith("{% load static %}"):
        content = content.replace("{% load static %}\n", "", 1)
        content = header_pattern.sub("{% extends 'tailwind_base.html' %}\n{% load static %}\n\n{% block content %}\n", content)
    
    # Replace footer
    content = footer_pattern.sub("\n{% endblock %}\n", content)
    
    # Clean up any leftover {% load static %} from the top if it got weird
    if original.startswith("{% load static %}") and not content.startswith("{% extends"):
        pass # Handle manually if needed, but regex should work
        
    with open(full_path, 'w') as f:
        f.write(content)
        
    print(f"Successfully processed: {filepath}")
