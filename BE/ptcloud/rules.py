"""ルールベースの簡易分類(比較の基準値)。

機械学習を使わず、点の高さと、周りの点の並び方(線状・面状・垂直かどうか)だけで分ける。
対象は5クラス: 地面(2)、植栽(5)、建物(6)、電線(14)、電柱(15)。それ以外は未分類(1)。

流れ
1. 地面を見つけて、各点の「地面からの高さ」を求める。
2. 地面以外の点を、一辺 voxel(m)の小さな立方体(ボクセル)に分ける。
3. ボクセルごとに、点の並びを調べる(主成分分析)。
   - 線状: 電線(ほぼ水平)や電柱(ほぼ垂直)
   - 面状: 建物の壁(垂直な面)や屋根(水平な面)
   - 散らばり: 植栽
4. 同じ種類のボクセルがつながった大きさで、ノイズや小さな物を除く。

限界(機械学習で改善する前提の基準値)
- 地面の高さは「近くの一番低い点」から推定するので、急な坂道には弱い。
- 建物より大きな範囲で一番低い点を探すので、地面が見えない大きな屋根は、地面と間違えることがある。
- 柱と標識の支柱、街灯などは区別しない(高さ5m以上の細い垂直な物は、すべて電柱になる)。
- 数千万点を一度に読むので、メモリを使う(2,274万点で数GB)。
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

UNCLASSIFIED, GROUND, VEGETATION, BUILDING, WIRE, POLE = 1, 2, 5, 6, 14, 15


@dataclass
class Params:
    voxel: float = 0.5  # ボクセルの一辺(m)。電線・電柱・植栽の判定に使う
    building_voxel: float = 1.0  # 建物の判定に使う、粗いボクセルの一辺(m)。点が少ない壁でも面と判定できる
    ground_cell: float = 1.0  # 地面を探すときの区画の大きさ(m)
    ground_window: float = 17.0  # 一番低い点を探す範囲(m)。建物の幅より大きくする
    ground_tol: float = 0.25  # 区画の一番低い点から、この高さまでを地面とする(m)
    ground_max_step: float = 0.5  # 周りより、これ以上高い区画は地面にしない(m)
    min_points: int = 3  # 特徴を調べるのに必要な、ボクセル内の最小の点数
    min_building_height: float = 2.5  # 建物とみなす、面状のつながりの一番高いところの高さ(m)
    min_building_voxels: int = 10  # 建物とみなす、面状ボクセルのつながりの最小の大きさ
    wire_min_height: float = 4.0  # 電線とみなす最小の高さ(m)
    min_wire_voxels: int = 6  # 電線とみなす、線状ボクセルのつながりの最小の長さ
    pole_min_height: float = 5.0  # 電柱とみなす、細い縦の列の最小の高さ(m)
    vegetation_min_height: float = 1.5  # 植栽とみなす最小の高さ(m)
    linear_min: float = 0.8  # 電線とみなす、線状の指標(0〜1)
    flat_max: float = 0.05  # 面とみなす、「一番薄い方向のばらつき÷一番長い方向のばらつき」の上限
    plane_mid_min: float = 0.1  # 面とみなす、「2番目÷一番長い」の下限(線を面と間違えないため)
    pole_max_width: float = 0.25  # 電柱とみなす、まわり(3x3のマス)も含めた水平方向の点のばらつきの上限(m)
    pole_min_fill: float = 0.6  # 電柱とみなす縦の列で、高さのうち電柱のボクセルが占める割合の下限


def _min_filter_1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    a = np.moveaxis(a, axis, 0)
    out = a.copy()
    for s in range(1, r + 1):
        if s >= a.shape[0]:
            break
        np.minimum(out[s:], a[:-s], out=out[s:])
        np.minimum(out[:-s], a[s:], out=out[:-s])
    return np.moveaxis(out, 0, axis)


def _ground_detail(xyz: np.ndarray, p: Params):
    """各点が地面かどうか、地面からの高さ、区画の一番低い点からの高さ、区画と周りの高さの差を返す。"""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ix = ((x - x.min()) / p.ground_cell).astype(np.int64)
    iy = ((y - y.min()) / p.ground_cell).astype(np.int64)
    cell = np.full((ix.max() + 1, iy.max() + 1), np.inf, dtype=np.float32)
    np.minimum.at(cell, (ix, iy), z)  # 区画ごとの一番低い点
    r = max(1, int(round(p.ground_window / 2 / p.ground_cell)))
    surface = _min_filter_1d(_min_filter_1d(cell, r, 0), r, 1)  # 近くの一番低い高さ
    cell_min, surf = cell[ix, iy], surface[ix, iy]
    is_ground = (z <= cell_min + p.ground_tol) & (cell_min - surf <= p.ground_max_step)
    return is_ground, (z - surf).astype(np.float32), (z - cell_min).astype(np.float32), (cell_min - surf).astype(np.float32)


def _ground(xyz: np.ndarray, p: Params):
    """各点が地面かどうかと、地面からの高さを返す。"""
    is_ground, height, _, _ = _ground_detail(xyz, p)
    return is_ground, height


_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)  # 26近傍の半分(13方向)。逆向きは重複するので使わない
]


def _large_components(ukeys, mask, dims, min_size, value=None, min_value=-np.inf):
    """mask が真のボクセルを、隣り合うもの同士でつなぎ、大きなつながりだけ残す。

    min_size 以上の大きさで、かつ(value を渡したとき)つながりの中の value の最大が min_value 以上のもの。
    """
    out = np.zeros(mask.shape, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    keys = ukeys[idx]  # 昇順
    rows, cols = [], []
    for dx, dy, dz in _OFFSETS:
        tgt = keys + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.minimum(np.searchsorted(keys, tgt), len(keys) - 1)
        hit = keys[pos] == tgt
        rows.append(np.flatnonzero(hit))
        cols.append(pos[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(keys), len(keys)))
    ncomp, lab = connected_components(g, directed=False)
    keep = np.bincount(lab, minlength=ncomp) >= min_size
    if value is not None:
        top = np.full(ncomp, -np.inf)
        np.maximum.at(top, lab, value[idx])
        keep &= top >= min_value
    out[idx[keep[lab]]] = True
    return out


class _Voxels:
    """点をボクセルに分けて、ボクセルごとの特徴(線状・面状の指標、向き、高さ)を求める。"""

    def __init__(self, pts: np.ndarray, height: np.ndarray, voxel: float, min_points: int):
        ijk = np.floor((pts - pts.min(axis=0)) / voxel).astype(np.int64) + 2  # 端に2マス余白(2つ隣まで調べるため)
        self.dims = ijk.max(axis=0) + 3
        key = (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]
        self.keys, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        self.inv = inv.ravel()
        nv = len(self.keys)

        def total(w):
            return np.bincount(self.inv, weights=w, minlength=nv)

        self._cnt = cnt
        self._sx, self._sy = total(pts[:, 0]), total(pts[:, 1])
        self._sxx, self._syy = total(pts[:, 0] ** 2), total(pts[:, 1] ** 2)
        m = np.stack([total(pts[:, k]) for k in range(3)], axis=1) / cnt[:, None]
        cov = np.empty((nv, 3, 3))
        for a in range(3):
            for b in range(a, 3):
                cov[:, a, b] = cov[:, b, a] = total(pts[:, a] * pts[:, b]) / cnt - m[:, a] * m[:, b]
        w, vec = np.linalg.eigh(cov)  # 固有値は小さい順。vec[:, :, k] が k番目の固有ベクトル
        w = np.clip(w, 0, None)
        l1 = np.where(w[:, 2] > 1e-12, w[:, 2], np.inf)
        self.linear = (w[:, 2] - w[:, 1]) / l1  # 線状の指標
        self.flat = w[:, 0] / l1  # 一番薄い方向の薄さ(0に近いほど平らな面)
        self.mid = w[:, 1] / l1  # 2番目の方向の広がり(0に近いと線)
        self.hspread = np.sqrt(cov[:, 0, 0] + cov[:, 1, 1])  # 水平方向の点のばらつき(m)
        self.zstd = np.sqrt(np.maximum(cov[:, 2, 2], 0))  # 高さ方向の点のばらつき(m)
        self.count = cnt  # ボクセルの中の点数
        self.axis_z = np.abs(vec[:, 2, 2])  # 主方向の鉛直成分(1なら垂直な線)
        self.normal_z = np.abs(vec[:, 2, 0])  # 法線の鉛直成分(0なら垂直な面=壁)
        self.height = total(height.astype(np.float64)) / cnt  # ボクセルの地面からの高さ
        self.ok = cnt >= min_points

    def block_hspread(self, sel: np.ndarray) -> np.ndarray:
        """sel のボクセルについて、同じ高さの3x3のマス(水平方向)に入る全ての点の、水平方向のばらつき(m)。

        電柱は細いのでばらつきが小さい。壁(横に続く)や木の葉(広がる)は大きくなる。
        """
        n = np.zeros(len(sel))
        sx, sy, sxx, syy = (np.zeros(len(sel)) for _ in range(4))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tgt = self.keys[sel] + (dx * self.dims[1] + dy) * self.dims[2]
                pos = np.minimum(np.searchsorted(self.keys, tgt), len(self.keys) - 1)
                hit = self.keys[pos] == tgt
                j = pos[hit]
                n[hit] += self._cnt[j]
                sx[hit] += self._sx[j]
                sy[hit] += self._sy[j]
                sxx[hit] += self._sxx[j]
                syy[hit] += self._syy[j]
        return np.sqrt(np.maximum(sxx / n - (sx / n) ** 2, 0) + np.maximum(syy / n - (sy / n) ** 2, 0))


def classify(xyz: np.ndarray, p: Params | None = None) -> np.ndarray:
    """xyz: (点数, 3)の座標(m、Zが上)。戻り値: 点ごとの分類コード(uint8)。"""
    p = p or Params()
    out = np.full(len(xyz), UNCLASSIFIED, dtype=np.uint8)
    is_ground, height = _ground(xyz, p)
    out[is_ground] = GROUND
    ng = np.flatnonzero(~is_ground)
    if ng.size == 0:
        return out
    pts = xyz[ng].astype(np.float64)
    h = height[ng]

    # --- 細かいボクセル: 電線・電柱・植栽 ---
    f = _Voxels(pts, h, p.voxel, p.min_points)
    wire_c = f.ok & (f.linear > p.linear_min) & (f.axis_z < 0.4) & (f.height >= p.wire_min_height)
    pole_c = f.ok & (f.hspread < 0.2) & (f.height >= 0.3)  # 水平に細いボクセル(絞り込み)
    wire = _large_components(f.keys, wire_c, f.dims, p.min_wire_voxels)

    # 電柱: 水平に細いボクセルが、同じ縦の列で p.pole_min_height 以上の高さに、隙間少なく並び、地面近くから始まるもの。
    # 「細い」は、まわり(3x3のマス)の点も含めて判定する。壁や木の葉は、横に広がるので細くならない。
    # 木の幹は、上に枝や葉が広がるので、細いボクセルが続かず、電柱にならない。
    nv = len(f.keys)
    vz = f.keys % f.dims[2]
    col = f.keys // f.dims[2]  # 縦の列の番号(x, y)
    thin = np.zeros(nv, dtype=bool)
    ci = np.flatnonzero(pole_c)
    if ci.size:
        thin[ci[f.block_hspread(ci) < p.pole_max_width]] = True
    pole = np.zeros(nv, dtype=bool)
    ci = np.flatnonzero(thin)
    if ci.size:
        _, cinv = np.unique(col[ci], return_inverse=True)
        cinv = cinv.ravel()
        ncol = cinv.max() + 1
        zmax = np.full(ncol, -np.inf)
        np.maximum.at(zmax, cinv, vz[ci])
        zmin = np.full(ncol, np.inf)
        np.minimum.at(zmin, cinv, vz[ci])
        hmin = np.full(ncol, np.inf)
        np.minimum.at(hmin, cinv, f.height[ci])
        span = zmax - zmin + 1
        fill = np.bincount(cinv, minlength=ncol) / span
        tall = (span * p.voxel >= p.pole_min_height) & (fill >= p.pole_min_fill) & (hmin <= 2.0)
        pole[ci[tall[cinv]]] = True
    trunk = thin & ~pole  # 孤立した短い細い柱は、幹とみなして植栽にする

    label = np.full(nv, UNCLASSIFIED, dtype=np.uint8)
    label[(f.height >= p.vegetation_min_height) | trunk] = VEGETATION
    label[wire] = WIRE
    label[pole] = POLE
    out[ng] = label[f.inv]
    del f

    # --- 粗いボクセル: 建物(壁や屋根の面) ---
    # 地面近くまで続く壁も建物にするため、低い面も候補に入れ、つながりの一番高いところで判定する。
    b = _Voxels(pts, h, p.building_voxel, p.min_points)
    plane_c = (
        b.ok & (b.flat < p.flat_max) & (b.mid > p.plane_mid_min)
        & ((b.normal_z < 0.3) | (b.normal_z > 0.9)) & (b.height >= 0.5)
    )
    building = _large_components(
        b.keys, plane_c, b.dims, p.min_building_voxels, value=b.height, min_value=p.min_building_height
    )
    is_building = building[b.inv]
    out[ng[is_building & (out[ng] != POLE) & (out[ng] != WIRE)]] = BUILDING
    return out