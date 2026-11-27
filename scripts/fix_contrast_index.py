import re

with open("private_books/c_programming_professional/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace all part card containers
# From:
# <div style="background:var(--bg-surface, #1e2433);border:1px solid var(--border-subtle, rgba(255,255,255,0.1));border-radius:12px;padding:1.5rem;box-shadow:0 4px 12px rgba(0,0,0,0.15);">
# To:
# <div class="part-card">
html = re.sub(
    r'<div style="background:var\(--bg-surface,\s*#1e2433\);border:1px solid var\(--border-subtle,\s*rgba\(255,255,255,0\.1\)\);border-radius:12px;padding:1\.5rem;box-shadow:0 4px 12px rgba\(0,0,0,0\.15\);">',
    '<div class="part-card">',
    html
)

# Replace part card headers
# From:
# <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
# To:
# <div class="part-card-header">
html = re.sub(
    r'<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">\s*<span class="module-badge"',
    '<div class="part-card-header">\n              <span class="module-badge"',
    html
)

# Replace the h3 in part-card-header
# <h3 style="margin:0;font-size:1.15rem;color:var(--text-primary);">
# to <h3 class="part-card-title">
html = re.sub(
    r'<h3 style="margin:0;font-size:1\.15rem;color:var\(--text-primary\);">',
    '<h3 class="part-card-title">',
    html
)

# Replace the ul in part-card
# <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:8px;">
# to <ul class="part-chapter-list">
html = re.sub(
    r'<ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:8px;">',
    '<ul class="part-chapter-list">',
    html
)

# Replace the chapter links in parts grid:
# <a href="..." style="display:flex;justify-content:space-between;align-items:center;text-decoration:none;color:var(--text-secondary);padding:6px 10px;border-radius:6px;transition:all 0.2s;" onmouseover="..." onmouseout="...">
# to:
# <a href="..." class="part-chapter-link">
html = re.sub(
    r'<a href="([^"]+)" style="display:flex;justify-content:space-between;align-items:center;text-decoration:none;color:var\(--text-secondary\);padding:6px 10px;border-radius:6px;transition:all 0\.2s;" onmouseover="[^"]*" onmouseout="[^"]*">',
    r'<a href="\1" class="part-chapter-link">',
    html
)

# Replace the read time badge inside part-chapter-link:
# <span style="font-size:0.75rem;opacity:0.7;">45m</span>
# to:
# <span class="part-chapter-time">45m</span>
html = re.sub(
    r'<span style="font-size:0\.75rem;opacity:0\.7;">([^<]+)</span>',
    r'<span class="part-chapter-time">\1</span>',
    html
)

# Clean up any leftover bg-surface inline styles in sections 1, 2, 3
html = re.sub(
    r'<div style="background:var\(--bg-surface,\s*#1e2433\);border:1px solid var\(--border-subtle,\s*rgba\(255,255,255,0\.1\)\);border-radius:12px;padding:1\.5rem;box-shadow:0 4px 14px rgba\(0,0,0,0\.1\);">',
    '<div class="info-card">',
    html
)

html = re.sub(
    r'<p style="font-size:0\.9rem;line-height:1\.6;color:var\(--text-secondary\);margin:0;">',
    '<p class="info-card-lead" style="margin:0;">',
    html
)

html = re.sub(
    r'<div style="display:flex;gap:1\.25rem;align-items:flex-start;background:var\(--bg-surface\);padding:1\.25rem;border-radius:10px;border:1px solid var(--border-subtle);">',
    '<div class="step-card">',
    html
)

# Write back
with open("private_books/c_programming_professional/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: index.html updated with clean, high-contrast CSS classes.")
