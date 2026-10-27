"""水面(池)の抽出。WingEarthが「水」とした点の特徴(色が一様、完全に水平、平ら、一定の高さ)を使う。

手がかり(WingEarthの水面 約71万点の分布から決めた初期値。しきい値は WaterParams にまとめてある)
- 色: 明るさ 0.54〜0.58、鮮やかさ 0.15〜0.18、緑らしさ 0.04〜0.09 と、とても狭い範囲に入る。
      (植物は、色のばらつきが大きい)
- 形: 面の向きが完全に水平(1.00)で、とても平ら。
- 広がり: 池なので、条件を満たす区画が大きくつながる。
- 高さ: 水面はどこも同じ高さ。つながりの中央の高さから大きくずれた点(水面の上の木の枝など)は、水面にしない。

限界
- 色の範囲は、水が多い3タイルの10〜90%点から決めた。反射や影で色が違う水面は、外れることがある。
- 水と同じ色で、平らな大きな面(灰色の舗装など)は、間違えることがある。
- 浅い水の底が見える場所は、水面にならない。
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, label

from .rules import _Voxels


@dataclass
class WaterParams:
    brightness: tuple = (0.50, 0.62)  # 明るさ(0〜1)
    saturation: tuple = (0.10, 0.22)  # 色の鮮やかさ(0〜1)
    exg: tuple = (-0.02, 0.14)  # 緑らしさ
    voxel: float = 1.0  # 平らさ・向きを調べるボクセルの大きさ(m)
    flat_max: float = 0.15  # 平らさ(小さいほど平ら)の上限
    normal_min: float = 0.97  # 面の向き(1なら水平)の下限
    cell: float = 1.0  # つながりを調べる区画の大きさ(m)
    min_fraction: float = 0.5  # 区画の中で、条件を満たす点の割合がこれ以上なら、水の区画
    close: int = 2  # 穴(船、反射、点の抜け)を埋める強さ(区画の数)
    min_area: float = 100.0  # 水面とみなす、つながりの最小の面積(m²)
    z_tol: float = 0.3  # つながりの中央の高さから、この範囲の点だけを水面にする(m)


def _stages(xyz: np.ndarray, rgb: np.ndarray, p: WaterParams) -> dict:
    """判定の途中経過(点ごとの、各条件に合うか)を、辞書で返す。診断(diagnose)にも使う。"""
    n = len(xyz)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    bright = (r + g + b) / 3
    exg = 2 * g - r - b
    sat = (rgb.max(axis=1) - rgb.min(axis=1)) / (rgb.max(axis=1) + 1e-6)
    ok_b = (bright >= p.brightness[0]) & (bright <= p.brightness[1])
    ok_s = (sat >= p.saturation[0]) & (sat <= p.saturation[1])
    ok_e = (exg >= p.exg[0]) & (exg <= p.exg[1])

    # 形: 平らで水平
    v = _Voxels(xyz.astype(np.float64), np.zeros(n), p.voxel, 3)
    ok_v = v.ok[v.inv]  # ボクセルの点数が足りて、形を調べられる
    ok_flat = (v.flat < p.flat_max)[v.inv]
    ok_normal = (v.normal_z > p.normal_min)[v.inv]
    cand = ok_b & ok_s & ok_e & ok_v & ok_flat & ok_normal

    # 広がり: 水の区画が大きくつながっているか
    ix = ((xyz[:, 0] - xyz[:, 0].min()) / p.cell).astype(np.int64)
    iy = ((xyz[:, 1] - xyz[:, 1].min()) / p.cell).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    cell = ix * ny + iy
    total = np.bincount(cell, minlength=nx * ny)
    hit = np.bincount(cell, weights=cand.astype(np.float64), minlength=nx * ny)
    grid = ((total > 0) & (hit >= p.min_fraction * np.maximum(total, 1))).reshape(nx, ny)
    pad = p.close + 1  # 端が削れないよう、余白をつけて穴埋めする
    grid = binary_closing(np.pad(grid, pad), iterations=p.close)[pad:-pad, pad:-pad]
    lab, nlab = label(grid, structure=np.ones((3, 3)))
    area = np.bincount(lab.ravel(), minlength=nlab + 1) * p.cell**2
    area[0] = 0
    in_big = (area >= p.min_area)[lab[ix, iy]] if nlab else np.zeros(n, dtype=bool)
    comp = lab[ix, iy]

    # 高さ: つながりごとに、水の候補の中央の高さに近い点だけを、水面にする
    water = np.zeros(n, dtype=bool)
    z_ok = np.zeros(n, dtype=bool)
    for k in np.flatnonzero(area >= p.min_area):
        in_comp = comp == k
        c = in_comp & cand
        if not c.any():
            continue
        z0 = np.median(xyz[c, 2])
        near = in_comp & (np.abs(xyz[:, 2] - z0) <= p.z_tol)
        z_ok |= near
        water |= near
    return dict(bright=bright, sat=sat, exg=exg, ok_b=ok_b, ok_s=ok_s, ok_e=ok_e, ok_v=ok_v, ok_flat=ok_flat,
                ok_normal=ok_normal, cand=cand, in_big=in_big, z_ok=z_ok, water=water)


def find_water(xyz: np.ndarray, rgb: np.ndarray, p: WaterParams | None = None) -> np.ndarray:
    """xyz: (点数,3)、rgb: (点数,3)の色(0〜1)。水面の点を True にした bool 配列を返す。"""
    return _stages(xyz, rgb, p or WaterParams())["water"]


def diagnose(xyz: np.ndarray, rgb: np.ndarray, truth: np.ndarray, p: WaterParams | None = None) -> str:
    """truth(本当の水面の点)のうち、どの条件で外れたかを、文章にする。しきい値の調整に使う。"""
    p = p or WaterParams()
    st = _stages(xyz, rgb, p)
    t = truth
    nt = int(t.sum())
    if nt == 0:
        return "水面の点がありません。"
    lines = [f"本当の水面: {nt:,}点 → 水と判定できた点: {int((st['water'] & t).sum()):,}点({(st['water'] & t).sum() / nt:.1%})", "",
             "条件ごとに、本当の水面の点が、条件に合う割合:"]
    for key, label_ in (("ok_b", f"明るさ {p.brightness}"), ("ok_s", f"鮮やかさ {p.saturation}"), ("ok_e", f"緑らしさ {p.exg}"),
                        ("ok_v", "形を調べられる(ボクセルの点数が足りる)"), ("ok_flat", f"平ら(平らさ < {p.flat_max})"),
                        ("ok_normal", f"水平(向き > {p.normal_min})"), ("cand", "上のすべて(水の候補)"),
                        ("in_big", f"大きな水の区画(≥ {p.min_area:g}m²)の中"), ("z_ok", f"水面の高さ(±{p.z_tol}m)")):
        lines.append(f"  {label_:<44} {st[key][t].mean():>7.1%}")
    # 最初に外れた条件の内訳(排他的)
    order = [("明るさ", st["ok_b"]), ("鮮やかさ", st["ok_s"]), ("緑らしさ", st["ok_e"]), ("点数不足", st["ok_v"]),
             ("平らさ", st["ok_flat"]), ("水平", st["ok_normal"]), ("水の区画の外・小さい区画", st["in_big"]), ("水面の高さ", st["z_ok"])]
    alive = t.copy()
    lines += ["", "取りこぼした点が、最初に外れた条件(合計は、取りこぼした点数):"]
    for name, ok in order:
        fail = alive & ~ok
        if fail.any():
            lines.append(f"  {name:<22} {int(fail.sum()):>10,}点")
        alive &= ok
    miss = t & ~st["water"]
    if miss.any():
        lines += ["", "取りこぼした点の、色の分布(5% / 中央値 / 95%)と、今の範囲:"]
        for key, name, rng in (("bright", "明るさ", p.brightness), ("sat", "鮮やかさ", p.saturation), ("exg", "緑らしさ", p.exg)):
            lo, mid, hi = np.percentile(st[key][miss], [5, 50, 95])
            lines.append(f"  {name:<10} {lo:6.2f} / {mid:6.2f} / {hi:6.2f}    (範囲 {rng[0]:.2f}〜{rng[1]:.2f})")
    return "\n".join(lines)