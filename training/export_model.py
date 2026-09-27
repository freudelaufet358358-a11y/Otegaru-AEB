"""学習済みのチェックポイントを src/core/learned.ts が読む形式（JSON ヘッダ + float16 の重み）に書き出す。

使い方:
    python export_model.py runs/v1/best.pt ../src/models/tone-hdrplus.bin --fixture ../test/fixtures/learned.json

--fixture を付けると、決まった合成画像に対する PyTorch の出力を書き出す。
test/learned.test.ts はこれと TypeScript 実装の出力が一致することを確かめる。
"""
import argparse, json, struct
import numpy as np
import torch
import model as M

p = argparse.ArgumentParser()
p.add_argument('ckpt')
p.add_argument('out')
p.add_argument('--fixture', default='')
p.add_argument('--info', default='{}', help='モデルに埋め込む説明 (JSON)')
args = p.parse_args()

ck = torch.load(args.ckpt, map_location='cpu')
net = M.ToneNet()
net.load_state_dict(ck['net'])
net.eval()

# float16 に丸めた重みで作り直す（TypeScript 側と同じ値で基準出力を作るため）
with torch.no_grad():
    for prm in net.parameters():
        prm.copy_(prm.half().float())
    gv = net.guide_values().half().float()

sd = net.state_dict()
order = [f'splat.{i}.{k}' for i in range(4) for k in ('weight', 'bias')] + \
        ['local1.weight', 'local1.bias', 'local2.weight', 'glob1.weight', 'glob1.bias', 'glob2.weight', 'glob2.bias',
         'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'out.weight', 'out.bias']
tensors = [(n, sd[n].numpy().astype(np.float16)) for n in order]
tensors.append(('guide', gv.numpy().astype(np.float16)))

hyper = dict(low=M.LOW, grid_xy=M.GRID_XY, grid_z=M.GRID_Z, ncoef=M.NCOEF, guide_lo=M.GUIDE_LO, guide_hi=M.GUIDE_HI,
             in_scale=M.IN_SCALE, log_eps=M.LOG_EPS, knee=M.KNEE)
meta = []
off = 0
for n, a in tensors:
    meta.append(dict(name=n, shape=list(a.shape), offset=off, length=int(a.size)))
    off += a.size
header = json.dumps(dict(version=1, hyper=hyper, info=json.loads(args.info), tensors=meta), separators=(',', ':'), ensure_ascii=False).encode()
with open(args.out, 'wb') as f:
    f.write(struct.pack('<II', 0x314D544F, len(header)))  # "OTM1"
    f.write(header)
    if (8 + len(header)) & 1:
        f.write(b'\0')
    for n, a in tensors:
        f.write(a.astype('<f2').tobytes())
print('wrote', args.out, 'params', off, 'bytes', 8 + len(header) + ((8 + len(header)) & 1) + 2 * off)

if args.fixture:
    # 合成した縮小画像（なめらかな明暗差 + 模様、約 10 段のダイナミックレンジ）
    L = M.LOW
    yy, xx = np.meshgrid((np.arange(L) + 0.5) / L, (np.arange(L) + 0.5) / L, indexing='ij')
    base = 2.0 ** (10 * xx - 8) * (1 + 0.5 * np.sin(yy * 20) * np.cos(xx * 13))
    thumb = np.stack([base * 0.9, base, base * (0.6 + 0.4 * yy)], 0).astype(np.float32)
    low = torch.from_numpy(thumb)[None]
    with torch.no_grad():
        k = M.key_of(low)
        raw = net.grid(M.lowres_input(low, k))
        grid = M.coef_grid(raw)[0].numpy()
        # 同じ模様の小さな画像 (40×24) を仕上げる
        H, W = 24, 40
        ys, xs = np.meshgrid((np.arange(H) + 0.5) / H, (np.arange(W) + 0.5) / W, indexing='ij')
        b2 = 2.0 ** (10 * xs - 8) * (1 + 0.5 * np.sin(ys * 20) * np.cos(xs * 13))
        full = np.stack([b2 * 0.9, b2, b2 * (0.6 + 0.4 * ys)], 0).astype(np.float32)
        out = M.render(net, torch.from_numpy(full)[None], low, k=k, grid=raw)[0].numpy()
    fx = dict(key=float(k[0]), grid_shape=list(grid.shape), grid=[round(float(v), 6) for v in grid.flatten()],
              full_hw=[H, W], out=[round(float(v), 6) for v in out.transpose(1, 2, 0).flatten()])
    json.dump(fx, open(args.fixture, 'w'), separators=(',', ':'))
    print('fixture', args.fixture, 'key', fx['key'])
