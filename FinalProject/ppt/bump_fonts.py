"""
Bump all font sizes in PPT index.html for better readability on projector.

Strategy:
  - Inline 'font-size:max(NNpx, ...)': NN -> max(NN + 2, 16) but cap at 18
  - Inline 'font-size: NNpx' (no 'max'): NN -> max(NN + 2, 16)
  - Inline 'font-size: NN%': keep as-is (relative to parent)
  - em-based sizes (0.78em, 0.85em): bump by absolute +0.08em
  - CSS .t-meta default 14px -> 15px
  - .t-helper default 12px -> 14px (if any)
"""
import re
from pathlib import Path

HTML = Path('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/ppt/index.html')
content = HTML.read_text(encoding='utf-8')

# Track changes
changes = 0

def bump_px(m):
    global changes
    px = int(m.group(1))
    new_px = max(px + 2, 15)
    if new_px != px:
        changes += 1
    return f'font-size:{new_px}px'

def bump_max(m):
    global changes
    px = int(m.group(1))
    new_px = max(px + 2, 15)
    if new_px != px:
        changes += 1
    return f'font-size:max({new_px}px'

# 1) 'font-size: NNpx' (with possible trailing space or ;)
content = re.sub(r'font-size:(\d+)px(?=[\s;}])', bump_px, content)
# 2) 'font-size:max(NNpx'
content = re.sub(r'font-size:max\((\d+)px', bump_max, content)

# 3) em-based bumps: 0.78em -> 0.88em, 0.85em -> 0.95em, 0.92em -> 1.0em
# Handle both 'font-size:.78em' (no leading 0) and 'font-size:0.78em' (with 0)
def bump_em(m):
    global changes
    raw = m.group(1)
    val = float('0' + raw) if not raw.startswith('0') else float(raw)
    new_val = min(val + 0.10, 1.0)  # cap at 1.0em
    if abs(new_val - val) > 0.001:
        changes += 1
    return f'font-size:{new_val:.2f}em'

content = re.sub(r'font-size:(\d*\.?\d+)em', bump_em, content)

# 4) CSS .t-meta: 14px -> 15px
old_tmeta = '.t-meta{\n    font-family:var(--mono);\n    font-size:14px;'
new_tmeta = '.t-meta{\n    font-family:var(--mono);\n    font-size:15px;'
if old_tmeta in content:
    content = content.replace(old_tmeta, new_tmeta)
    changes += 1

# 5) find any other CSS class with font-size:14px or 12px and bump
def bump_css(m):
    global changes
    px = int(m.group(2))
    if px < 14:
        new_px = max(px + 2, 14)
        changes += 1
        return f'{m.group(1)}{new_px}px'
    return m.group(0)

# Match patterns like "font-size:12px" or "font-size:14px" within CSS blocks
# (we already handled inline, this is for class rules)
content = re.sub(r'(font-size:)(\d+)(px)', bump_css, content)

HTML.write_text(content, encoding='utf-8')
print(f'✓ Bumped {changes} font-size declarations')
print(f'  HTML: {len(content)//1024} KB')
