"""比較画像を作る: 基準フレーム / ナチュラル / HDR / おまかせ / HDR+ の仕上がり

使い方:
    python sheet.py scenes out sheet.jpg シーン名1,シーン名2,...
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

scenes, outdir, dest = sys.argv[1], sys.argv[2], sys.argv[3]
names = sys.argv[4].split(',')
modes = [('ref', '合成前（基準フレーム）'), ('fusion', '従来: ナチュラル'), ('hdr', '従来: HDR'), ('learned', '新: おまかせ（学習）'), ('target', '参考: HDR+ の仕上がり')]
font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 20)
TW = int(os.environ.get('TILE', '360'))
rows = []
for name in names:
    meta = json.load(open(os.path.join(outdir, name + '.json')))
    w, h = meta['w'], meta['h']
    tiles = []
    for m, label in modes:
        if m == 'target':
            im = Image.open(os.path.join(scenes, name + '_target.png')).convert('RGB')
        else:
            im = Image.fromarray(np.fromfile(os.path.join(outdir, f'{name}_{m}.rgb'), np.uint8).reshape(h, w, 3))
        th = round(h * TW / w)
        tiles.append(im.resize((TW, th), Image.LANCZOS))
    th = tiles[0].height
    row = Image.new('RGB', (TW * len(tiles) + 6 * (len(tiles) - 1), th), (24, 24, 28))
    for i, t in enumerate(tiles):
        row.paste(t, (i * (TW + 6), 0))
    rows.append(row)
head = 34
W = rows[0].width
H = head + sum(r.height + 6 for r in rows)
sheet = Image.new('RGB', (W, H), (24, 24, 28))
d = ImageDraw.Draw(sheet)
for i, (_, label) in enumerate(modes):
    d.text((i * (TW + 6) + 6, 6), label, font=font, fill=(235, 235, 235))
y = head
for r in rows:
    sheet.paste(r, (0, y)); y += r.height + 6
sheet.save(dest, quality=90)
print('saved', dest, sheet.size)
