"""LASファイルの読み書き。大きなファイルでもメモリを使いすぎないよう、一定の点数ずつ処理する。"""
from collections import Counter
from pathlib import Path

import laspy
import numpy as np

CHUNK = 1_000_000  # 1回に読む点数


def _classes(points) -> np.ndarray:
    return np.asarray(points.classification, dtype=np.uint8)


def info(path: str | Path, chunk: int = CHUNK) -> dict:
    """ファイルの概要(点数、範囲、形式、クラスごとの点数)を返す。"""
    counts = np.zeros(256, dtype=np.int64)
    with laspy.open(path) as r:
        h = r.header
        for pts in r.chunk_iterator(chunk):
            counts += np.bincount(_classes(pts), minlength=256)
        return {
            "point_count": int(h.point_count),
            "version": str(h.version),
            "point_format": int(h.point_format.id),
            "has_rgb": "red" in h.point_format.dimension_names,
            "mins": [float(v) for v in h.mins],
            "maxs": [float(v) for v in h.maxs],
            "scales": [float(v) for v in h.scales],
            "offsets": [float(v) for v in h.offsets],
            "class_counts": {int(c): int(n) for c, n in enumerate(counts) if n > 0},
        }


def read_classification(path: str | Path, chunk: int = CHUNK) -> np.ndarray:
    """全点の分類コードを、ファイルの点の順番のまま配列で返す(2,274万点でも約22MB)。"""
    with laspy.open(path) as r:
        out = np.empty(r.header.point_count, dtype=np.uint8)
        pos = 0
        for pts in r.chunk_iterator(chunk):
            c = _classes(pts)
            out[pos : pos + len(c)] = c
            pos += len(c)
    return out


def read_origin(path: str | Path) -> np.ndarray:
    """ファイルの最小座標(read_xyz / read_points が引いている原点)を返す。相対座標を、ファイルの座標に戻すのに使う。"""
    with laspy.open(path) as r:
        return np.array(r.header.mins, dtype=np.float64)


def read_xyz(path: str | Path, chunk: int = CHUNK) -> np.ndarray:
    """全点の座標(点数, 3)を、float32で返す。

    実際の測量座標は値が大きく、float32だと精度が落ちるので、ファイルの最小値を原点にした相対座標にする。
    (分類には相対座標で足りる。2,274万点で約270MBのメモリを使う)
    """
    with laspy.open(path) as r:
        origin = np.array(r.header.mins, dtype=np.float64)
        out = np.empty((r.header.point_count, 3), dtype=np.float32)
        pos = 0
        for pts in r.chunk_iterator(chunk):
            n = len(pts)
            out[pos : pos + n, 0] = pts.x - origin[0]
            out[pos : pos + n, 1] = pts.y - origin[1]
            out[pos : pos + n, 2] = pts.z - origin[2]
            pos += n
    return out


def read_points(path: str | Path, chunk: int = CHUNK):
    """座標(原点を引いた相対座標)、色(0〜1)、強度(0〜1)、分類を、まとめて読む。(機械学習用)

    色は、16ビット(0〜65535)か8ビット(0〜255)かを、最大値で見分けて0〜1にそろえる。
    色がないファイルでは、色は0になる。
    """
    with laspy.open(path) as r:
        h = r.header
        n = h.point_count
        has_rgb = "red" in h.point_format.dimension_names
        origin = np.array(h.mins, dtype=np.float64)
        xyz = np.empty((n, 3), dtype=np.float32)
        rgb = np.zeros((n, 3), dtype=np.float32)
        inten = np.empty(n, dtype=np.float32)
        cls = np.empty(n, dtype=np.uint8)
        pos = 0
        for pts in r.chunk_iterator(chunk):
            m = len(pts)
            xyz[pos : pos + m, 0] = pts.x - origin[0]
            xyz[pos : pos + m, 1] = pts.y - origin[1]
            xyz[pos : pos + m, 2] = pts.z - origin[2]
            inten[pos : pos + m] = pts.intensity
            cls[pos : pos + m] = _classes(pts)
            if has_rgb:
                rgb[pos : pos + m, 0] = pts.red
                rgb[pos : pos + m, 1] = pts.green
                rgb[pos : pos + m, 2] = pts.blue
            pos += m
    if has_rgb and rgb.max() > 0:
        rgb /= 65535.0 if rgb.max() > 255 else 255.0
    inten /= 65535.0
    return xyz, rgb, inten, cls


def remove_classes(src: str | Path, dst: str | Path, codes: list[int], chunk: int = CHUNK) -> dict:
    """指定した分類コードの点を取り除いて、別のファイルに保存する。

    座標系、スケール、RGB、強度などの属性は、残した点にそのまま引き継ぐ(元のファイルは変えない)。
    戻り値: {"kept": 残した点数, "removed": {分類コード: 除去した点数}}
    """
    drop_codes = np.array(sorted(set(codes)), dtype=np.uint8)
    kept = 0
    removed: Counter = Counter()
    with laspy.open(src) as r, laspy.open(dst, mode="w", header=r.header) as w:
        for pts in r.chunk_iterator(chunk):
            cls = _classes(pts)
            drop = np.isin(cls, drop_codes)
            for c, n in zip(*np.unique(cls[drop], return_counts=True)):
                removed[int(c)] += int(n)
            w.write_points(pts[~drop])
            kept += int((~drop).sum())
    return {"kept": kept, "removed": dict(removed)}


def write_classification(src: str | Path, dst: str | Path, classes: np.ndarray, chunk: int = CHUNK) -> None:
    """src の点はそのままに、分類コードだけを classes で置き換えて dst に保存する。

    分類モデルの予測結果をLASに書き出すときに使う。classes は src と同じ点の順番・同じ点数。
    """
    with laspy.open(src) as r:
        if len(classes) != r.header.point_count:
            raise ValueError(f"点数が合いません(ファイル {r.header.point_count:,} / 分類 {len(classes):,})")
        if r.header.point_format.id <= 5 and int(classes.max(initial=0)) > 31:
            raise ValueError("このLASの点形式(0〜5)では、分類コードは0〜31までです")
        with laspy.open(dst, mode="w", header=r.header) as w:
            pos = 0
            for pts in r.chunk_iterator(chunk):
                pts.classification = classes[pos : pos + len(pts)]
                w.write_points(pts)
                pos += len(pts)