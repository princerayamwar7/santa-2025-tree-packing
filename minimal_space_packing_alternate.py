# minimal_space_packing_alternate.py
# Alternating (0 / 180 deg) tree packing, tightened and rapid for n=1..200
# Usage: python .\minimal_space_packing_alternate.py --mode grid --range 1 200 --fast

import os, math, argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

# -------------------------
# 1) Tree polygon (placeholder)
# Replace get_tree_polygon() with real Kaggle polygon if available.
# -------------------------
def get_tree_polygon():
    tri = Polygon([(-0.6, -0.2), (0, 1.0), (0.6, -0.2)])
    tri2 = Polygon([(-0.45, -0.05), (0, 0.65), (0.45, -0.05)])
    trunk = Polygon([(-0.12, -0.6), (0.12, -0.6), (0.12, -0.2), (-0.12, -0.2)])
    combined = unary_union([tri, tri2, trunk])
    cx, cy = combined.centroid.x, combined.centroid.y
    return translate(combined, xoff=-cx, yoff=-cy)

def place_poly(base_poly, x, y, deg):
    p = rotate(base_poly, deg, origin=(0,0), use_radians=False)
    return translate(p, xoff=x, yoff=y)

# -------------------------
# 2) Grid helpers (positions) with alternating angles
# -------------------------
def grid_dimensions_for_n(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows

def create_grid_with_alternating_angles(n, spacing, stagger=False):
    """
    Create an array of x,y positions and alternating angles (0,180).
    If stagger=True, every other row shifts by spacing/2 (optional).
    """
    cols, rows = grid_dimensions_for_n(n)
    xs=[]; ys=[]; angles=[]
    for r in range(rows):
        row_offset = 0.0
        if stagger and (r % 2 == 1):
            row_offset = spacing/2.0
        for c in range(cols):
            if len(xs) >= n: break
            x = (c - (cols-1)/2.0) * spacing + row_offset
            y = ((rows-1)/2.0 - r) * spacing
            xs.append(x); ys.append(y)
            # alternate angle: checkerboard style -> neighbor attachments
            # Use (r+c) parity: even -> 0°, odd -> 180°
            ang = 0.0 if ((r + c) % 2 == 0) else 180.0
            angles.append(ang)
    xs = np.array(xs[:n], dtype=float)
    ys = np.array(ys[:n], dtype=float)
    angles = np.array(angles[:n], dtype=float)
    # center positions around origin
    xs -= xs.mean(); ys -= ys.mean()
    return xs, ys, angles

# -------------------------
# 3) Quick relax (fast, inexpensive)
# -------------------------
def quick_relax(base_poly, xs, ys, angles, steps=40, step_size=0.02):
    n = len(xs)
    pos = np.column_stack([xs, ys])
    for _ in range(steps):
        polys = [place_poly(base_poly, pos[i,0], pos[i,1], angles[i]) for i in range(n)]
        moved = False
        for i in range(n):
            pi = polys[i]
            ci = np.array([pi.centroid.x, pi.centroid.y])
            f = np.zeros(2)
            for j in range(n):
                if i == j: continue
                pj = polys[j]
                # only check bbox first for speed
                if (pi.bounds[0] > pj.bounds[2] or pi.bounds[2] < pj.bounds[0] or
                    pi.bounds[1] > pj.bounds[3] or pi.bounds[3] < pj.bounds[1]):
                    continue
                if pi.intersects(pj):
                    cj = np.array([pj.centroid.x, pj.centroid.y])
                    v = ci - cj
                    d = np.linalg.norm(v) + 1e-9
                    f += (v / d) * (1.0 / (d + 1e-6))
            if np.linalg.norm(f) > 1e-9:
                moved = True
                pos[i] += (f / (np.linalg.norm(f) + 1e-12)) * step_size
        if not moved:
            break
    return pos[:,0], pos[:,1], angles

# -------------------------
# 4) Tighten routine (shrink + relax binary search)
# -------------------------
def pack_polys(base_poly, xs, ys, angles):
    return [place_poly(base_poly, float(xs[i]), float(ys[i]), float(angles[i])) for i in range(len(xs))]

def has_any_overlap(polys, tol=1e-9):
    n = len(polys)
    for i in range(n):
        for j in range(i+1, n):
            # bbox check for speed
            a = polys[i].bounds; b = polys[j].bounds
            if a[0] > b[2] or a[2] < b[0] or a[1] > b[3] or a[3] < b[1]:
                continue
            if polys[i].intersects(polys[j]) and polys[i].intersection(polys[j]).area > tol:
                return True
    return False

def bounding_square_side(polys):
    xs=[]; ys=[]
    for p in polys:
        a,b,c,d = p.bounds
        xs.extend([a,c]); ys.extend([b,d])
    return max(max(xs)-min(xs), max(ys)-min(ys))

def tighten_packing(base_poly, xs_init, ys_init, angles_init, relax_fn, max_iter=22, low=0.6, high=1.0, relax_kwargs=None):
    if relax_kwargs is None:
        relax_kwargs = {}
    # center
    cx = (xs_init.max() + xs_init.min()) / 2.0
    cy = (ys_init.max() + ys_init.min()) / 2.0
    X = xs_init - cx
    Y = ys_init - cy
    # ensure initial relax
    xs_r, ys_r, ang_r = relax_fn(base_poly, X.copy(), Y.copy(), angles_init.copy(), **relax_kwargs)
    polys = pack_polys(base_poly, xs_r, ys_r, ang_r)
    if has_any_overlap(polys):
        # try a bit longer relax
        xs_r, ys_r, ang_r = relax_fn(base_poly, X.copy(), Y.copy(), angles_init.copy(), steps=120, step_size=0.02)
        polys = pack_polys(base_poly, xs_r, ys_r, ang_r)
        if has_any_overlap(polys):
            # can't fix; return current
            return xs_r, ys_r, ang_r, bounding_square_side(polys)
    best = (xs_r.copy(), ys_r.copy(), ang_r.copy())
    best_side = bounding_square_side(polys)
    lo, hi = low, high
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        xs_mid = X * mid
        ys_mid = Y * mid
        xs_try, ys_try, ang_try = relax_fn(base_poly, xs_mid.copy(), ys_mid.copy(), angles_init.copy(), **relax_kwargs)
        polys_try = pack_polys(base_poly, xs_try, ys_try, ang_try)
        if not has_any_overlap(polys_try):
            # success -> record and try smaller
            s = bounding_square_side(polys_try)
            best = (xs_try.copy(), ys_try.copy(), ang_try.copy())
            best_side = s
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-3:
            break
    # final polish
    xs_f, ys_f, ang_f = relax_fn(base_poly, best[0].copy(), best[1].copy(), best[2].copy(), steps=140, step_size=0.015)
    polys_f = pack_polys(base_poly, xs_f, ys_f, ang_f)
    return xs_f, ys_f, ang_f, bounding_square_side(polys_f)

# -------------------------
# 5) Output helpers (plot + csv)
# -------------------------
def save_plot(polys, side, title, path):
    fig, ax = plt.subplots(figsize=(6,6))
    cmap = plt.get_cmap('viridis')
    for i,p in enumerate(polys):
        x,y = p.exterior.xy
        ax.fill(x, y, alpha=0.85, edgecolor='k', linewidth=0.2, facecolor=cmap((i%20)/20))
    half = side/2.0
    sqx = [-half, half, half, -half, -half]
    sqy = [-half, -half, half, half, -half]
    ax.plot(sqx, sqy, linestyle='--', linewidth=2, color='red')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def write_submission_csv(n, xs, ys, angles, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ids=[]; xs_s=[]; ys_s=[]; degs=[]
    for i in range(n):
        ids.append(f"{n:03d}_{i}")
        xs_s.append('s' + f'{xs[i]:.6f}')
        ys_s.append('s' + f'{ys[i]:.6f}')
        degs.append('s' + f'{angles[i]:.6f}')
    df = pd.DataFrame({'id': ids, 'x': xs_s, 'y': ys_s, 'deg': degs})
    df.to_csv(path, index=False)

# -------------------------
# 6) Main fast pipeline per instance
# -------------------------
def run_instance(n, base_poly, mode='grid', fast=True, out_dir='out_alt'):
    os.makedirs(out_dir, exist_ok=True)
    minx,miny,maxx,maxy = base_poly.bounds
    poly_diam = max(maxx-minx, maxy-miny)
    # start with slightly reduced spacing to encourage attachments
    initial_spacing = poly_diam * (0.88 if fast else 0.95)

    # create grid + alternating angles
    xs, ys, angles = create_grid_with_alternating_angles(n, initial_spacing, stagger=True)

    # tiny jitter to break degeneracy
    rng = np.random.default_rng(n)  # deterministic per n
    xs += (rng.random(len(xs)) - 0.5) * 0.003
    ys += (rng.random(len(ys)) - 0.5) * 0.003

    # fast relax first pass
    relax_kwargs = {'steps': 40 if fast else 120, 'step_size': 0.02}
    xs, ys, angles = quick_relax(base_poly, xs, ys, angles, **relax_kwargs)

    # tighten with shrink+relax (fast settings)
    xs, ys, angles, side = tighten_packing(base_poly, xs, ys, angles,
                                           relax_fn=quick_relax,
                                           max_iter=20,
                                           low=0.6, high=1.0,
                                           relax_kwargs=relax_kwargs)

    polys = pack_polys(base_poly, xs, ys, angles)

    csv_path = os.path.join(out_dir, f'submission_n{n:03d}.csv')
    png_path = os.path.join(out_dir, f'plot_n{n:03d}.png')

    write_submission_csv(n, xs, ys, angles, csv_path)
    save_plot(polys, side, f'{n} Trees: {side:.6f}', png_path)

    return {'n': n, 'side': side, 'csv': csv_path, 'png': png_path}

# -------------------------
# 7) CLI: run range or single n
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['grid','hex'], default='grid')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--range', nargs=2, type=int, default=None)
    parser.add_argument('--out', type=str, default='out_alt')
    parser.add_argument('--fast', action='store_true', help='use faster (fewer relax steps)')
    args = parser.parse_args()

    base_poly = get_tree_polygon()

    if args.n:
        print('Running n=', args.n)
        r = run_instance(args.n, base_poly, mode=args.mode, fast=args.fast, out_dir=args.out)
        print('Done:', r)
    elif args.range:
        a,b = args.range
        results = []
        start = time.time()
        for n in range(a,b+1):
            print(f'Running n={n} ...', flush=True)
            r = run_instance(n, base_poly, mode=args.mode, fast=args.fast, out_dir=args.out)
            print(f' -> side={r["side"]:.6f} file={r["csv"]}')
            results.append(r)
        pd.DataFrame(results).to_csv(os.path.join(args.out, 'summary.csv'), index=False)
        print('ALL DONE in %.1f s' % (time.time() - start))
    else:
        print('Usage examples:')
        print(' python .\\minimal_space_packing_alternate.py --mode grid --range 1 200 --fast')


# ----------------------------
# MAIN ENTRYPOINT (replace existing final block with this)
# ----------------------------
if __name__ == '__main__':
    import argparse, time, os, pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['grid','hex'], default='grid')
    parser.add_argument('--n', type=int)
    parser.add_argument('--range', nargs=2, type=int)
    parser.add_argument('--out', type=str, default='out_alt')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    base_poly = get_tree_polygon()

    if args.n:
        print(f'Running n={args.n} ...', flush=True)
        r = run_instance(args.n, base_poly, mode=args.mode, fast=args.fast, out_dir=args.out)
        print('Done ->', r, flush=True)
    elif args.range:
        a, b = args.range
        results = []
        start = time.time()
        for n in range(a, b+1):
            print(f'Running n={n} ...', flush=True)
            r = run_instance(n, base_poly, mode=args.mode, fast=args.fast, out_dir=args.out)
            print(f' -> side={r["side"]:.6f} file={r["csv"]}', flush=True)
            results.append(r)
        pd.DataFrame(results).to_csv(os.path.join(args.out, 'summary.csv'), index=False)
        print('ALL DONE in %.1f s' % (time.time() - start), flush=True)
    else:
        print('Usage examples:')
        print(' python .\\minimal_space_packing_alternate.py --mode grid --n 10 --fast')
        print(' python .\\minimal_space_packing_alternate.py --mode grid --range 1 200 --fast')
