"""分類ごとの特徴の分布を見る。ルールのしきい値を決めるときの手がかりにする。

たとえば WingEarth が「水」とした点の、高さ・色・平らさが、どのくらいの範囲に入るかを出す。
"""
import numpy as np

from . import lasio
from .classes import class_name
from .features import compute_features, feature_names

# 表示する特徴(名前, 説明)
SHOW = [
    ("height", "地面からの高さ(m)"),
    ("above_cell_min", "区画の一番低い点からの高さ(m)"),
    ("brightness", "明るさ(0〜1)"),
    ("exg", "緑らしさ(大きいほど緑)"),
    ("saturation", "色の鮮やかさ(0〜1)"),
    ("flat_1.0", "平らさ(0に近いほど平ら)"),
    ("normal_z_1.0", "面の向き(1なら水平な面、0なら垂直な面)"),
    ("zstd_0.5", "高さ方向のばらつき(m)"),
    ("log_count_0.5", "点の密度(対数)"),
]


def class_stats(paths, max_points: int = 1_000_000, seed: int = 0) -> dict:
    """{分類コード: {"count": 点数, "q": {特徴名: (10%, 50%, 90%)}}} を返す。点が多いときは、ランダムに間引く。"""
    names = feature_names()
    cols = [names.index(n) for n, _ in SHOW]
    xs, ys = [], []
    for path in paths:
        xyz, rgb, inten, cls = lasio.read_points(path)
        xs.append(compute_features(xyz, rgb, inten)[:, cols])
        ys.append(cls)
    X, y = np.concatenate(xs), np.concatenate(ys)
    rng = np.random.default_rng(seed)
    out = {}
    for code in np.unique(y):
        idx = np.flatnonzero(y == code)
        total = len(idx)
        if total > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        q = {}
        for k, (n, _) in enumerate(SHOW):
            v = X[idx, k]
            v = v[~np.isnan(v)]
            q[n] = tuple(float(a) for a in np.percentile(v, [10, 50, 90])) if len(v) else (np.nan,) * 3
        out[int(code)] = {"count": total, "q": q}
    return out


def format_stats(stats: dict) -> str:
    lines = ["各クラスの特徴の分布(10% / 中央値 / 90%)", ""]
    for code, s in sorted(stats.items()):
        lines.append(f"■ {code} {class_name(code)}  {s['count']:,}点")
        for n, desc in SHOW:
            lo, mid, hi = s["q"][n]
            lines.append(f"    {desc:<34} {lo:>8.2f} / {mid:>8.2f} / {hi:>8.2f}")
        lines.append("")
    return "\n".join(lines)