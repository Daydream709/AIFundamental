"""
Re-inline images: replace base64 data URIs in <img src="data:..."> tags
based on the *original* src attribute (which was lost when previously inlined).

Strategy: Since the original src="images/p4_line1_data.png" was replaced
with src="data:image/png;base64,..." on the first inlining pass, we cannot
reliably recover the original filename from the data URI alone. So we
re-encode ALL .png files in the images/ directory to base64, then
sequentially match them to <img> tags in DOM order (which is the same
order the original scripts wrote them: P4, P7, P8, P9, P11, P12).

This is brittle if the order changes, but in our case the source
generator's run order is fixed.
"""
import re
import base64
from pathlib import Path

PPT = Path('/Users/daydream/Code/UndergraduateCourse/AIFundamental/FinalProject/ppt')
HTML = PPT / 'index.html'
IMG_DIR = PPT / 'images'

# Order matches the run order in generate_ppt_charts.py main()
EXPECTED_ORDER = [
    'p4_line1_data.png',
    'p7_kan_vs_baseline.png',
    'p8_lite_efficiency.png',
    'p9_dual_ablation.png',
    'p11_multimodal.png',
    'p12_viz_mockup.png',
]

# Pre-encode all
def to_data_uri(p: Path) -> str:
    data = p.read_bytes()
    encoded = base64.b64encode(data).decode('ascii')
    return f'data:image/png;base64,{encoded}'

uris = []
for name in EXPECTED_ORDER:
    p = IMG_DIR / name
    if not p.exists():
        print(f'  MISSING: {name}')
        continue
    uris.append(to_data_uri(p))
    print(f'  ✓ {name}  ({p.stat().st_size//1024} KB)')

html = HTML.read_text(encoding='utf-8')

# Replace each <img src="data:..."> in order
pattern = re.compile(r'<img src="data:image/png;base64,[A-Za-z0-9+/=]+"([^>]*)>')

matches = list(pattern.finditer(html))
print(f'\nFound {len(matches)} <img data: URIs to replace')

if len(matches) != len(uris):
    print(f'!  Mismatch: {len(matches)} <img> vs {len(uris)} files')
    print('   Using {min} matches')

# Build new HTML
result = []
last = 0
for i, m in enumerate(matches):
    if i >= len(uris):
        break
    result.append(html[last:m.start()])
    result.append(f'<img src="{uris[i]}"{m.group(1)}>')
    last = m.end()
result.append(html[last:])
new_html = ''.join(result)
HTML.write_text(new_html, encoding='utf-8')

before_size = len(html)
after_size = len(new_html)
print(f'\n✓ Done. HTML: {before_size//1024} KB → {after_size//1024} KB')
