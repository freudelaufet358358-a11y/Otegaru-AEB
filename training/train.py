"""HDR+ の組（merged.dng から現像したリニア HDR → final.jpg）でトーンレンダラーを学習する。

使い方:
    python train.py --data data/train --out runs/v1 --epochs 40

CPU（4 コア）でも 1 エポック数分で回る大きさにしている。
各エポックの終わりに検証用シーンで PSNR を測り、最良のものを best.pt に保存する。
"""
import argparse, glob, json, math, os, random, time, zlib
import numpy as np
import torch
import torch.nn.functional as F
from model import ToneNet, render, LOW, curve_penalty, key_of, lowres_input

p = argparse.ArgumentParser()
p.add_argument('--data', default='data/train')
p.add_argument('--out', default='runs/v1')
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--bs', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--crop', type=int, nargs=2, default=[448, 336])  # 横位置の幅・高さ（縦位置は入れ替える）
p.add_argument('--val', type=int, default=120)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--reg_w', type=float, default=1.0, help='係数の正則化（トーンカーブ・彩度）の重み')
p.add_argument('--resume', default='')
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=1)

files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
if args.limit:
    files = files[:args.limit]
# 名前のハッシュで決まる検証用シーン（学習には使わない）
val_files = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % 1000 < 1000 * args.val / max(len(files), 1)][:args.val]
vs = set(val_files)
train_files = [f for f in files if f not in vs]
print(f'train {len(train_files)} val {len(val_files)}', flush=True)
json.dump([os.path.basename(f) for f in val_files], open(os.path.join(args.out, 'val_files.json'), 'w'))


def load(fs):
    xs, ts = [], []
    for f in fs:
        z = np.load(f)
        xs.append(torch.from_numpy(z['in512'].astype(np.float16)).permute(2, 0, 1).contiguous())
        ts.append(torch.from_numpy(z['tgt512']).permute(2, 0, 1).contiguous())
    return xs, ts


t0 = time.time()
tx, tt = load(train_files)
vx, vt = load(val_files)
print(f'loaded in {time.time() - t0:.0f}s', flush=True)


def area_low(x):
    return F.interpolate(x, size=(LOW, LOW), mode='area')


def make_batch(idx, cw, ch, xs, ts):
    """ランダムな位置で切り出し、半分の確率で左右反転する"""
    X, T = [], []
    for i in idx:
        x, t = xs[i].float(), ts[i].float() / 255
        _, H, W = x.shape
        if H < ch or W < cw:
            s = max(ch / H, cw / W)
            nh, nw = math.ceil(H * s), math.ceil(W * s)
            x = F.interpolate(x[None], size=(nh, nw), mode='bilinear', align_corners=False)[0]
            t = F.interpolate(t[None], size=(nh, nw), mode='bilinear', align_corners=False)[0]
            H, W = nh, nw
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)
        x = x[:, y0:y0 + ch, x0:x0 + cw]
        t = t[:, y0:y0 + ch, x0:x0 + cw]
        if random.random() < 0.5:
            x = x.flip(-1)
            t = t.flip(-1)
        X.append(x)
        T.append(t)
    return torch.stack(X), torch.stack(T)


def psnr(o, t):
    mse = ((o - t) ** 2).mean().item()
    return 10 * math.log10(1 / max(mse, 1e-10))


net = ToneNet()
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
start_epoch = 0
if args.resume:
    ck = torch.load(args.resume)
    net.load_state_dict(ck['net'])
    opt.load_state_dict(ck['opt'])
    start_epoch = ck['epoch'] + 1

# 縦横の違う画像は同じバッチに入れない
land = [i for i, x in enumerate(tx) if x.shape[2] >= x.shape[1]]
port = [i for i, x in enumerate(tx) if x.shape[2] < x.shape[1]]
steps_per_epoch = len(land) // args.bs + len(port) // args.bs
total = steps_per_epoch * args.epochs
print(f'landscape {len(land)} portrait {len(port)} steps/epoch {steps_per_epoch}', flush=True)


def lr_at(step):
    warm = 300
    if step < warm:
        return args.lr * (step + 1) / warm
    q = (step - warm) / max(1, total - warm)
    return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * min(1, q)))


def evaluate():
    net.eval()
    ps, l1s = [], []
    with torch.no_grad():
        for x, t in zip(vx, vt):
            x = x.float()[None]
            t = t.float()[None] / 255
            o = render(net, x, area_low(x))
            ps.append(psnr(o, t))
            l1s.append((o - t).abs().mean().item())
    net.train()
    return float(np.mean(ps)), float(np.mean(l1s))


log = open(os.path.join(args.out, 'log.jsonl'), 'a')
step = start_epoch * steps_per_epoch
best = -1.0
for ep in range(start_epoch, args.epochs):
    random.shuffle(land)
    random.shuffle(port)
    batches = [('L', land[i:i + args.bs]) for i in range(0, len(land) - args.bs + 1, args.bs)] + \
              [('P', port[i:i + args.bs]) for i in range(0, len(port) - args.bs + 1, args.bs)]
    random.shuffle(batches)
    te = time.time()
    run = 0.0
    run_mono = 0.0
    n = 0
    for kind, idx in batches:
        cw, ch = args.crop if kind == 'L' else args.crop[::-1]
        X, T = make_batch(idx, cw, ch, tx, tt)
        for gp in opt.param_groups:
            gp['lr'] = lr_at(step)
        low = area_low(X)
        k = key_of(low)
        grid = net.grid(lowres_input(low, k))
        out = render(net, X, low, k=k, grid=grid)
        l1 = (out - T).abs().mean()
        tone_reg, sat_reg = curve_penalty(net, grid)
        reg = tone_reg + sat_reg
        loss = l1 + args.reg_w * reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run += l1.item()
        run_mono += reg.item()
        n += 1
        step += 1
        if n % 50 == 0:
            print(f'ep {ep} it {n}/{len(batches)} l1 {run / n:.4f} reg {run_mono / n:.5f} lr {lr_at(step):.2e} {time.time() - te:.0f}s', flush=True)
    vp, vl = evaluate()
    rec = dict(epoch=ep, train_l1=run / max(n, 1), train_reg=run_mono / max(n, 1), val_psnr=vp, val_l1=vl, time=time.time() - te)
    print(rec, flush=True)
    log.write(json.dumps(rec) + '\n')
    log.flush()
    ck = dict(net=net.state_dict(), opt=opt.state_dict(), epoch=ep, args=vars(args), val_psnr=vp)
    torch.save(ck, os.path.join(args.out, 'last.pt'))
    if vp > best:
        best = vp
        torch.save(ck, os.path.join(args.out, 'best.pt'))
print('done best val psnr', best)
