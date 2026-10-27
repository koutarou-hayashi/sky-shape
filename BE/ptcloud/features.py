"""機械学習用の特徴量。点ごとに、高さ・色・まわりの点の並び方(3つの大きさ)を数値にする。

どの特徴も、場所の絶対的な値(座標や標高)は使わず、「地面からの高さ」や「まわりとの関係」だけを使う。
そうすると、学習した場所とは別の場所(別のタイル、会社のデータ)にも、当てはめやすくなる。
"""
import numpy as np

from .rules import Params, _ground_detail, _Voxels

SCALES = (0.5, 1.0, 2.0)  # まわりを見るボクセルの大きさ(m)
_VOXEL_FEATURES = ("linear", "planar", "flat", "mid", "normal_z", "axis_z", "hspread", "zstd", "log_count",
                   "mean_exg", "std_brightness")


def feature_names() -> list[str]:
    names = ["height", "above_cell_min", "cell_rise", "r", "g", "b", "brightness", "exg", "saturation", "intensity"]
    names += [f"{n}_{s}" for s in SCALES for n in _VOXEL_FEATURES]
    return names


def compute_features(xyz, rgb, intensity, params: Params | None = None) -> np.ndarray:
    """xyz: (点数,3)の座標(m、Zが上)、rgb: (点数,3)の色(0〜1)、intensity: (点数,)の強度(0〜1)。

    戻り値: (点数, 特徴の数) の float32。特徴が計算できない点(まわりの点が少ない)は NaN。
    メモリのため、タイル(数十万点)ごとに呼ぶ。
    """
    p = params or Params()
    _, height, above_cell, cell_rise = _ground_detail(xyz, p)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    bright = (r + g + b) / 3
    exg = 2 * g - r - b  # 緑らしさ(植物で大きい)
    sat = (rgb.max(axis=1) - rgb.min(axis=1)) / (rgb.max(axis=1) + 1e-6)

    cols = [height, above_cell, cell_rise, r, g, b, bright, exg, sat, intensity]
    pts = xyz.astype(np.float64)
    for scale in SCALES:
        v = _Voxels(pts, height, scale, p.min_points)
        nv = len(v.keys)

        def voxel_mean(x):
            return np.bincount(v.inv, weights=x.astype(np.float64), minlength=nv) / v.count

        mean_b = voxel_mean(bright)
        std_b = np.sqrt(np.maximum(voxel_mean(bright**2) - mean_b**2, 0))
        per_voxel = {
            "linear": v.linear, "planar": v.mid - v.flat, "flat": v.flat, "mid": v.mid,
            "normal_z": v.normal_z, "axis_z": v.axis_z, "hspread": v.hspread, "zstd": v.zstd,
            "log_count": np.log1p(v.count), "mean_exg": voxel_mean(exg), "std_brightness": std_b,
        }
        for name in _VOXEL_FEATURES:
            arr = per_voxel[name].astype(np.float32)
            if name not in ("log_count", "mean_exg", "std_brightness", "zstd", "hspread"):
                arr = np.where(v.ok, arr, np.nan)  # 点が少ないボクセルでは、並び方が信用できない
            cols.append(arr[v.inv])
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)