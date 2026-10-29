import os
import re

# Repo root is 2 levels up from scripts/blog_sections/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SVG_DIR = os.path.join(REPO_ROOT, "static", "blogs", "stm32-diy")

def get_inline_svg(filename: str) -> str:
    """Load and return an SVG file as an inline string wrapped in a responsive container."""
    filepath = os.path.join(SVG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        svg_content = f.read().strip()
    return f"""<div class="rounded-xl border border-gray-200 overflow-hidden bg-white shadow-sm p-4 w-full max-w-3xl mx-auto flex justify-center">{svg_content}</div>"""

def inline_all_stm32_svgs(html: str) -> str:
    """Replace all stm32-diy img references with their raw inline SVG markup."""
    pattern = re.compile(r'<img[^>]+src=[\"\']/static/blogs/stm32-diy/([^\"\']+)[\"\'][^>]*>')
    
    def replacer(m):
        filename = m.group(1)
        return get_inline_svg(filename)
        
    return pattern.sub(replacer, html)
