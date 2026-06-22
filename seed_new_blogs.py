import os
import django
import sys
import subprocess

# Set up Django environment
sys.path.append('/home/kamal/Documents/1.Github/acl_codeside')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from blogs.models import Post

def process_and_convert(md_path, static_media_path, temp_media_path, out_html_path):
    # Read Markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace media paths
    content = content.replace(temp_media_path, static_media_path)
    
    # Write back
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    # Convert to HTML using pandoc
    subprocess.run(['pandoc', md_path, '-t', 'html', '-o', out_html_path])
    
    # Read generated HTML
    with open(out_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return html_content

def main():
    apparmor_md = '/tmp/blogs_md/apparmor/post.md'
    apparmor_html = '/tmp/blogs_md/apparmor/post.html'
    apparmor_html_content = process_and_convert(
        apparmor_md, 
        '/static/blogs/apparmor/', 
        '/tmp/blogs_md/apparmor/media/', 
        apparmor_html
    )

    cve_md = '/tmp/blogs_md/cve/post.md'
    cve_html = '/tmp/blogs_md/cve/post.html'
    cve_html_content = process_and_convert(
        cve_md, 
        '/static/blogs/cve/', 
        '/tmp/blogs_md/cve/media/', 
        cve_html
    )

    # Blog 1
    post1, created1 = Post.objects.get_or_create(
        slug="linux-silent-knight-against-hackers",
        defaults={
            "title": "Linux's Silent Knight Against Hackers",
            "author": "ponnannarobin@gmail.com",
            "excerpt": "Exploring how AppArmor supplements traditional DAC models by providing Mandatory Access Control (MAC) for restricting program capabilities.",
            "content": apparmor_html_content
        }
    )
    if not created1:
        post1.content = apparmor_html_content
        post1.title = "Linux's Silent Knight Against Hackers"
        post1.author = "ponnannarobin@gmail.com"
        post1.save()
    print("AppArmor blog seeded successfully.")

    # Blog 2
    post2, created2 = Post.objects.get_or_create(
        slug="beyond-the-cve-number",
        defaults={
            "title": "Beyond the CVE Number by Roopashree",
            "author": "roopashreem935@gmail.com",
            "excerpt": "Understanding Common Patterns in Recent CVEs: Root Causes, Mitigations, and Security Tools.",
            "content": cve_html_content
        }
    )
    if not created2:
        post2.content = cve_html_content
        post2.title = "Beyond the CVE Number by Roopashree"
        post2.author = "roopashreem935@gmail.com"
        post2.save()
    print("CVE blog seeded successfully.")

if __name__ == '__main__':
    main()
