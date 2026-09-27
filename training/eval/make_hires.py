"""評価用: 長辺 2048px のシーンを作る（等倍での破綻チェック用）。
prepare_hdrplus.py が記録した向き・切り出し（prep_log.jsonl）を使って merged.dng を現像し直す。

使い方:
    python make_hires.py ../data hires シーン名1,シーン名2,...
"""
import json, os, sys, struct, tempfile, shutil
import numpy as np
from PIL import Image
data_dir, dst, names = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
sys.argv = sys.argv[:1] + [data_dir]  # prepare_hdrplus は読み込み時に sys.argv を見る
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import prepare_hdrplus as P
from hdrplus_decode import decode

os.makedirs(dst, exist_ok=True)
log = {}
for l in open(os.path.join(data_dir, 'prep_log.jsonl')):
    d = json.loads(l)
    if d['status'] == 'ok':
        log[d['burst']] = d['meta']
for b in names:
    m = log[b]
    d = tempfile.mkdtemp()
    R = f'{P.BASE}/results_20171023/{b}'; B = f'{P.BASE}/bursts/{b}'
    P.fetch(f'{R}/merged.dng', f'{d}/merged.dng'); P.fetch(f'{R}/final.jpg', f'{d}/final.jpg')
    P.fetch(f'{B}/rgb2rgb.txt', f'{d}/rgb2rgb.txt'); P.fetch(f'{B}/lens_shading_map_N{m["ref"]:03d}.tiff', f'{d}/lsm.tiff')
    rgb, _ = decode(f'{d}/merged.dng', f'{d}/lsm.tiff', [float(x) for x in open(f'{d}/rgb2rgb.txt').read().split()])
    rgb = np.ascontiguousarray(np.maximum(P.orient(rgb, m['raw_orientation']), 0))
    box = tuple(m['box'])
    bw, bh = box[2] - box[0], box[3] - box[1]
    w, h = P.fit(bw, bh, 2048)
    x = P.crop_resize(rgb, box, w, h)
    im = Image.open(f'{d}/final.jpg'); o = im.getexif().get(0x0112, 1)
    tl = P.orient(P.S2L[np.asarray(im.convert('RGB'))], o)
    t = np.round(P.l2s(P.box_resize(tl, w, h)) * 255).astype(np.uint8)
    y = x @ P.LUMA
    scale = 0.125 / max(float(y.mean()), 1e-8)
    with open(os.path.join(dst, b + '.bin'), 'wb') as f:
        f.write(struct.pack('<4I', w, h, 3, 0)); f.write(np.array([0.25, 1, 4], np.float32).tobytes())
        for e in (-2, 0, 2):
            f.write(np.round(np.clip(x * scale * 2.0 ** e, 0, 1) * 65535).astype('<u2').tobytes())
    Image.fromarray(t).save(os.path.join(dst, b + '_target.png'))
    shutil.rmtree(d, ignore_errors=True)
    print(b, w, h)
json.dump(names, open(os.path.join(dst, 'index.json'), 'w'))
