"""
Inline all PPT images as base64 data URIs in index.html.

Safari on macOS restricts loading of local resources from file:// URLs
in recent versions (Local Network / Cross-Origin policy). This script
embeds every <img src="images/..."> as a base64 data URI so the
HTML works when double-clicked (file://) in any browser without
needing a local HTTP server.
"""
import base64
import re
import sys
from pathlib import Path

PPT = Path('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/ppt')
HTML = PPT / 'index.html'
IMG_DIR = PPT / 'images'

def file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip('.')
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'svg': 'image/svg+xml'}.get(suffix, 'application/octet-stream')
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode('ascii')
    return f'data:{mime};base64,{encoded}'

html = HTML.read_text(encoding='utf-8')

# Find all <img src="images/..."> and replace with data URI
pattern = re.compile(r'src="(images/[^"]+)"')
matches = pattern.findall(html)
unique = sorted(set(matches))
print(f'Found {len(matches)} img refs ({len(unique)} unique)')

# Cache to avoid re-encoding same image
cache = {}
for rel in unique:
    p = PPT / rel
    if not p.exists():
        print(f'  ! MISSING: {rel}')
        continue
    uri = file_to_data_uri(p)
    cache[rel] = uri
    print(f'  ✓ {rel}  ({p.stat().st_size//1024} KB → {len(uri)//1024} KB base64)')

# Replace
def repl(m):
    rel = m.group(1)
    return f'src="{cache.get(rel, m.group(1))}"'

new_html = pattern.sub(repl, html)
HTML.write_text(new_html, encoding='utf-8')

# Report
before = len(html)
after = len(new_html)
print(f'\n✓ Inlined. HTML size: {before//1024} KB → {after//1024} KB  (+{after-before} bytes)')
