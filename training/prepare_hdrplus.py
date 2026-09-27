"""HDR+ データセットを少しずつダウンロードして、学習用の組（リニアな HDR 入力, 仕上がった写真）を作る。

使い方:
    python prepare_hdrplus.py data            # 全 3640 シーン（1 シーン約 20MB を順に処理して元データは捨てる）
    NPROC=6 python prepare_hdrplus.py data    # 並列数を指定

各バーストについて merged.dng（位置合わせ・合成済みの RAW）、final.jpg（HDR+ の完成画像）、
基準フレームのレンズシェーディング補正マップ、rgb2rgb（色変換行列）を取得し、
hdrplus_decode.py でリニア sRGB に現像する。向き（縦位置・上下逆さま）とデジタルズームの切り出しを
final.jpg との相関から自動で合わせ、画素単位で揃っていることを確かめてから縮小して保存する。
キュレーション済みの 153 シーン（20171106_subset）は評価用に data/test へ、残りを data/train へ入れる。

データセット: HDR+ Burst Photography Dataset (Hasinoff et al., SIGGRAPH Asia 2016), CC BY-SA
https://hdrplusdata.org/
"""
import json, os, sys, time, shutil, tempfile, urllib.parse, urllib.request
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import multiprocessing as mp
from PIL import Image
import tifffile

sys.path.insert(0, os.path.dirname(__file__))
from hdrplus_decode import decode

BASE = 'https://storage.googleapis.com/hdrplusdata/20171106'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'data'
TMP = os.path.join(OUT, 'tmp_download')

def list_bursts(prefix):
    """バケット内のバースト名の一覧（公開 JSON API）"""
    out, token = [], None
    while True:
        q = {'prefix': prefix, 'delimiter': '/', 'maxResults': '1000'}
        if token:
            q['pageToken'] = token
        url = 'https://storage.googleapis.com/storage/v1/b/hdrplusdata/o?' + urllib.parse.urlencode(q)
        d = json.load(urllib.request.urlopen(url, timeout=60))
        out += [p.rstrip('/').split('/')[-1] for p in d.get('prefixes', [])]
        token = d.get('nextPageToken')
        if not token:
            return out

def fetch(url, path, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return
        except Exception:
            if t == tries - 1: raise
            time.sleep(2 ** t)

S2L = np.where(np.arange(256) / 255 <= 0.04045, np.arange(256) / 255 / 12.92, ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4).astype(np.float32)

def l2s(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)

def orient(a, o):
    if o == 2: return a[:, ::-1]
    if o == 3: return a[::-1, ::-1]
    if o == 4: return a[::-1]
    if o == 5: return np.swapaxes(a, 0, 1)
    if o == 6: return np.rot90(a, -1)
    if o == 7: return np.rot90(a, 1)[::-1]  # transverse
    if o == 8: return np.rot90(a, 1)
    return a

def box_resize(a, w, h):
    """Area (box) resize of float HxWxC."""
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c], dtype=np.float32), 'F').resize((w, h), Image.BOX)) for c in range(a.shape[2])], -1)

def fit(w, h, side):
    s = side / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

def align_check(lin_in, lin_t):
    a = np.log2(lin_in @ LUMA + 1e-4); b = np.log2(lin_t @ LUMA + 1e-4)
    ga = np.diff(a, axis=1)[:-1]; gb = np.diff(b, axis=1)[:-1]
    ga2 = np.diff(a, axis=0)[:, :-1]; gb2 = np.diff(b, axis=0)[:, :-1]
    best = None; m = 3
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            sl = lambda g, oy, ox: g[m + oy: g.shape[0] - m + oy, m + ox: g.shape[1] - m + ox]
            num = (sl(ga, dy, dx) * sl(gb, 0, 0)).sum() + (sl(ga2, dy, dx) * sl(gb2, 0, 0)).sum()
            den = np.sqrt(((sl(ga, dy, dx) ** 2).sum() + (sl(ga2, dy, dx) ** 2).sum()) * ((sl(gb, 0, 0) ** 2).sum() + (sl(gb2, 0, 0) ** 2).sum()))
            c = float(num / max(den, 1e-12))
            if best is None or c > best[0]: best = (c, dx, dy)
    return best

def grad_corr0(a, b):
    """Gradient correlation of two log-luma images at zero shift."""
    ga = np.diff(a, axis=1)[:-1]; gb = np.diff(b, axis=1)[:-1]
    ga2 = np.diff(a, axis=0)[:, :-1]; gb2 = np.diff(b, axis=0)[:, :-1]
    num = (ga * gb).sum() + (ga2 * gb2).sum()
    den = np.sqrt(((ga ** 2).sum() + (ga2 ** 2).sum()) * ((gb ** 2).sum() + (gb2 ** 2).sum()))
    return float(num / max(den, 1e-12))

def crop_resize(a, box, w, h):
    """Area resize of the float sub-rectangle box=(x0, y0, x1, y1) of HxWxC to (w, h)."""
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c], dtype=np.float32), 'F').resize((w, h), Image.BOX, box=box)) for c in range(a.shape[2])], -1)

def match_geometry(rgb, lin_t):
    """Find raw orientation and (digital zoom) crop so that rgb matches the target lin_t.
    Returns (orientation, box in oriented half-res raw coords, (corr, dx, dy))."""
    th, tw = lin_t.shape[:2]
    cw, ch = fit(tw, th, 256)
    small_t = box_resize(lin_t, cw, ch)
    lt = np.log2(small_t @ LUMA + 1e-4)
    cands = [ro for ro in (1, 3, 6, 8, 2, 4, 5, 7)
             if abs(tw / th - orient(rgb, ro).shape[1] / orient(rgb, ro).shape[0]) < 0.01]
    best = None
    # まずズームなしで調べる（ほとんどのシーン）
    for ro in cands:
        rr = orient(rgb, ro)
        H, W = rr.shape[:2]
        c = align_check(box_resize(np.maximum(rr, 0), cw, ch), small_t)
        if best is None or c[0] > best[2][0]:
            best = (ro, (0.0, 0.0, float(W), float(H)), c)
    # ずれ 0 が最良で相関がある程度あれば、ズームなしで確定（細かい模様が多いと相関は低めに出る）
    if best[2][0] > 0.3 and best[2][1:] == (0, 0):
        return best
    nozoom = best
    # デジタルズーム: 中央を切り出して拡大した画像。倍率を粗く探索してから細かく詰める
    coarse = None
    for ro in cands:
        rr = np.maximum(orient(rgb, ro), 0)
        H, W = rr.shape[:2]
        mw, mh = fit(W, H, 1024)
        mid = box_resize(rr, mw, mh)
        for k in range(1, 71):
            z = 1.02 ** k
            bw, bh = mw / z, mh / z
            box = ((mw - bw) / 2, (mh - bh) / 2, (mw + bw) / 2, (mh + bh) / 2)
            c = grad_corr0(np.log2(crop_resize(mid, box, cw, ch) @ LUMA + 1e-4), lt)
            if coarse is None or c > coarse[0]:
                coarse = (c, ro, z)
    _, ro, z0 = coarse
    rr = np.maximum(orient(rgb, ro), 0)
    H, W = rr.shape[:2]
    mw, mh = fit(W, H, 1024)
    mid = box_resize(rr, mw, mh)
    fine = None
    for z in np.maximum(1.0, z0 * (1 + np.linspace(-0.025, 0.025, 26))):
        bw, bh = mw / z, mh / z
        box = ((mw - bw) / 2, (mh - bh) / 2, (mw + bw) / 2, (mh + bh) / 2)
        c = align_check(crop_resize(mid, box, cw, ch), small_t)
        if fine is None or c[0] > fine[0][0]:
            fine = (c, z, box)
    (c, dx, dy), z, box = fine
    # 256 スケールでのずれを切り出し位置に反映して、元の半分解像度の座標に直す
    sx = (box[2] - box[0]) / cw
    sy = (box[3] - box[1]) / ch
    box = (box[0] + dx * sx, box[1] + dy * sy, box[2] + dx * sx, box[3] + dy * sy)
    f = W / mw
    box = [v * f for v in box]
    # 画像の外にはみ出した分は内側へ戻す
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw > W + 1e-6 or bh > H + 1e-6:
        return nozoom
    box[0] = min(max(box[0], 0.0), W - bw); box[2] = box[0] + bw
    box[1] = min(max(box[1], 0.0), H - bh); box[3] = box[1] + bh
    box = tuple(box)
    c2 = align_check(crop_resize(rr, box, cw, ch), small_t)
    if c2[0] <= nozoom[2][0]:
        return nozoom
    return ro, box, c2

def exif_info(dng):
    info = {}
    try:
        with tifffile.TiffFile(dng) as t:
            tags = t.pages[0].tags
            ex = tags.get('ExifTag')
            if ex is not None:
                e = ex.value
                et = e.get('ExposureTime'); fn = e.get('FNumber')
                info['exposure_time'] = et[0] / et[1] if et else None
                info['fnumber'] = fn[0] / fn[1] if fn else None
                iso = e.get('ISOSpeedRatings'); info['iso'] = int(iso if not isinstance(iso, tuple) else iso[0]) if iso else None
            for k in ('Make', 'Model'):
                if k in tags: info[k.lower()] = str(tags[k].value)
            if 'BaselineExposure' in tags:
                v = tags['BaselineExposure'].value; info['baseline_exposure'] = v[0] / v[1]
    except Exception:
        pass
    return info

def process(args):
    burst, split = args
    dst = os.path.join(OUT, split, burst + '.npz')
    if os.path.exists(dst):
        return burst, 'exists', None
    d = tempfile.mkdtemp(dir=TMP)
    try:
        r = f'{BASE}/results_20171023/{burst}'
        b = f'{BASE}/bursts/{burst}'
        fetch(f'{r}/reference_frame.txt', f'{d}/ref.txt')
        ref = int(open(f'{d}/ref.txt').read().strip() or 0)
        fetch(f'{r}/merged.dng', f'{d}/merged.dng')
        fetch(f'{r}/final.jpg', f'{d}/final.jpg')
        fetch(f'{b}/rgb2rgb.txt', f'{d}/rgb2rgb.txt')
        fetch(f'{b}/lens_shading_map_N{ref:03d}.tiff', f'{d}/lsm.tiff')
        lsm = tifffile.imread(f'{d}/lsm.tiff')
        if lsm.ndim != 3 or lsm.shape[2] != 4:
            return burst, 'bad_lsm', None
        m = [float(x) for x in open(f'{d}/rgb2rgb.txt').read().split()]
        rgb, meta = decode(f'{d}/merged.dng', f'{d}/lsm.tiff', m)
        meta.update(exif_info(f'{d}/merged.dng'))
        im = Image.open(f'{d}/final.jpg')
        o = im.getexif().get(0x0112, 1)
        tgt = np.asarray(im.convert('RGB'))
        tl = orient(S2L[tgt], o)
        H2, W2 = rgb.shape[:2]
        th, tw = tl.shape[:2]
        # 目標画像は raw の半分解像度相当まで縮小してから比較する
        tw2, th2 = fit(tw, th, max(W2, H2))
        lin_t = box_resize(tl, tw2, th2)
        ro, box, best = match_geometry(rgb, lin_t)
        rgb = np.ascontiguousarray(np.maximum(orient(rgb, ro), 0))
        zoom = rgb.shape[1] / (box[2] - box[0])
        meta.update(orientation=o, raw_orientation=ro, zoom=zoom, box=list(box), corr=best[0], shift=[best[1], best[2]], ref=ref, full=[int(tw), int(th)])
        if best[0] < 0.3 or best[1:] != (0, 0):
            return burst, f'misaligned {best} zoom={zoom:.3f}', meta
        out = {}
        for side in ([512, 1024] if split == 'test' else [512]):
            w, h = fit(tw2, th2, side)
            out[f'in{side}'] = crop_resize(rgb, box, w, h).astype(np.float16)
            out[f'tgt{side}'] = np.round(l2s(box_resize(lin_t, w, h)) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez(dst + '.tmp.npz', meta=json.dumps(meta), **out)
        os.replace(dst + '.tmp.npz', dst)
        return burst, 'ok', meta
    except Exception as e:
        return burst, 'error ' + repr(e)[:300], None
    finally:
        shutil.rmtree(d, ignore_errors=True)

if __name__ == '__main__':
    os.makedirs(TMP, exist_ok=True)
    cache = os.path.join(OUT, 'bursts.json')
    if os.path.exists(cache):
        lists = json.load(open(cache))
    else:
        lists = {'full': list_bursts('20171106/results_20171023/'), 'subset': list_bursts('20171106_subset/results_20171023/')}
        json.dump(lists, open(cache, 'w'))
    test = set(lists['subset'])
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    jobs = [(b, 'test' if b in test else 'train') for b in (only or lists['full'])]
    t0 = time.time()
    log = open(os.path.join(OUT, 'prep_log.jsonl'), 'a')
    n = 0
    stat = {}
    mp.set_start_method('forkserver')
    with mp.Pool(int(os.environ.get('NPROC', '6'))) as pool:
        for burst, status, meta in pool.imap_unordered(process, jobs):
            n += 1
            key = status.split()[0]
            stat[key] = stat.get(key, 0) + 1
            log.write(json.dumps({'burst': burst, 'status': status, 'meta': meta}) + '\n')
            log.flush()
            if n % 20 == 0 or n == len(jobs):
                print(f'{n}/{len(jobs)} {time.time() - t0:.0f}s {stat}', flush=True)
