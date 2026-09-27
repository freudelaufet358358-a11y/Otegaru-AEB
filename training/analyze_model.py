"""学習済みモデルの健全性の確認: セルごとのトーンカーブの単調性、係数の範囲、露出への依存。

使い方:
    python analyze_model.py runs/v1/best.pt data/test
"""
import glob, os, sys, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ck = torch.load(sys.argv[1], map_location='cpu')
net = M.ToneNet(); net.load_state_dict(ck['net']); net.eval()
files = sorted(glob.glob(sys.argv[2] + '/*.npz'))[:40]
viol = []; amin = []; amax = []; smin = []; smax = []; inv = []
with torch.no_grad():
    for f in files:
        x = torch.from_numpy(np.load(f)['in512'].astype(np.float32)).permute(2, 0, 1)[None]
        low = F.interpolate(x, size=(256, 256), mode='area')
        k = M.key_of(low)
        grid = net.grid(M.lowres_input(low, k))
        coef = M.coef_grid(grid)
        ln = torch.linspace(-12, 8, 201)
        g = net.guide(ln)
        fz = (g * M.GRID_Z - 0.5).clamp(0, M.GRID_Z - 1); z0 = fz.floor().clamp(max=M.GRID_Z - 2).long(); tz = (fz - z0)[None, :, None, None]
        a = coef[:, 0]; b = coef[:, 1]
        lout = (a[:, z0] * (1 - tz) + a[:, z0 + 1] * tz) * ln[None, :, None, None] + (b[:, z0] * (1 - tz) + b[:, z0 + 1] * tz)
        d = lout[:, 1:] - lout[:, :-1]
        viol.append(float((d < 0).float().mean()))
        amin.append(float(a.min())); amax.append(float(a.max())); smin.append(float(coef[:, 2].min())); smax.append(float(coef[:, 2].max()))
        # 露出不変性: 入力を 4 倍しても出力が同じか
        o1 = M.render(net, x, low); o2 = M.render(net, x * 4, low * 4)
        inv.append(float((o1 - o2).abs().max()))
print(json.dumps(dict(mono_violation_frac=float(np.mean(viol)), a_range=[min(amin), max(amax)], s_range=[min(smin), max(smax)], exposure_invariance_maxdiff=max(inv)), indent=1))
print('guide knots', [round(float(v), 3) for v in net.guide_values().detach()])
