"""池・森・駐車場・道路がある現場向けの分類(ルールベース)。

順番に決めていく。先に決まった点は、あとのルールで上書きしない。
  1. 水面(9)        water.py: 色・平らさ・広がり・高さの一定さ
  2. 地面からの高さ   ground.py: 傾斜に対応した地面の平面(水面の点は、地面を探すときに使わない)
  3. 建物(6)        平らな面(壁・屋根)が大きくつながり、一番高いところが2.5m以上
  4. 車両(20)       地面から少し浮いた、緑でない、車の大きさのかたまり
  5. 植栽(低3・中4・高5) 緑らしい点、または高くて散らばった点。地面からの高さで、低・中・高に分ける
  6. 地面(2)        地面の高さにある、緑でない点。(split_road=True のときだけ、灰色で平らな点を路面(11)に分ける)
  7. その他(1)       どれにも当てはまらない点

電線(14)・電柱(15)は、この現場にほとんどないと見て、まだ入れていない(必要なら、rules.py の判定を持ってくる)。

しきい値は SiteParams にまとめてある。**初期値は、WingEarthの水面以外は、データで確かめていない。**
画面で結果を見て、調整する。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

from .ground import GroundParams, find_ground
from .rules import _large_components, _Voxels
from .water import WaterParams, find_water

OTHER, GROUND, VEG_LOW, VEG_MID, VEG_HIGH, BUILDING, WATER, ROAD, VEHICLE = 1, 2, 3, 4, 5, 6, 9, 11, 20


@dataclass
class SiteParams:
    water: WaterParams = field(default_factory=WaterParams)
    ground: GroundParams = field(default_factory=GroundParams)
    voxel: float = 1.0  # 形・色を調べるボクセルの大きさ(m)
    # 植栽
    green_margin: float = 0.02  # 「緑の強さ」(緑 − 赤と青の大きいほう、ボクセルの平均)がこれ以上なら、植物。ベージュや黄色は、緑にならない
    low_max: float = 0.5  # 地面からの高さがこれ未満なら、低い植栽(m)
    mid_max: float = 2.0  # これ未満なら中、以上なら高い植栽(m)
    dry_veg_min_height: float = 2.0  # 緑でなくても、この高さ以上で、散らばった点なら植物(枯れ葉・幹)(m)
    scatter_flat_min: float = 0.15  # 「散らばった」とみなす平らさ(大きいほど散らばる)
    # 建物
    min_building_height: float = 2.5  # 平らな面のつながりの、一番高いところの高さの下限(m)
    min_building_voxels: int = 10  # 平らな面のつながりの、最小の大きさ
    plane_flat_max: float = 0.05  # 面とみなす、平らさの上限
    # 車両
    vehicle_h: tuple = (0.15, 2.6)  # 候補にする、地面からの高さの範囲(m)
    vehicle_length: tuple = (2.5, 6.5)  # 車のかたまりの、長さ(m)
    vehicle_width: tuple = (1.3, 2.6)  # 幅(m)
    vehicle_height: tuple = (0.8, 2.4)  # 一番高いところ(m)
    vehicle_cell: float = 0.5  # かたまりを調べる区画の大きさ(m)
    # 地面と路面
    split_road: bool = False  # True にすると、灰色で平らな地面を、路面(11)として分ける。既定は、分けず、すべて地面(2)
    ground_with_water: bool = False  # True にすると、地面の平面を、水面の点も含めて探す。水際の岸が、水面の高さに引きずられ、岸が「地面より高い」と判定されやすいので、既定は、含めない
    road_sat_max: float = 0.12  # 路面とみなす、鮮やかさの上限(灰色)
    road_green_max: float = 0.02  # 路面とみなす、緑の強さの上限
    road_flat_max: float = 0.15  # 路面とみなす、平らさの上限


def _vehicles(xyz, height, cand, p: SiteParams) -> np.ndarray:
    """候補の点を、平面の区画に落として、つながりごとの大きさ(長さ・幅・高さ)が、車らしいかを調べる。"""
    out = np.zeros(len(xyz), dtype=bool)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return out
    c = p.vehicle_cell
    x0, y0 = xyz[idx, 0].min(), xyz[idx, 1].min()
    ix = ((xyz[idx, 0] - x0) / c).astype(np.int64)
    iy = ((xyz[idx, 1] - y0) / c).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    cell = ix * ny + iy
    cnt = np.bincount(cell, minlength=nx * ny)
    lab, n = label((cnt >= 2).reshape(nx, ny), structure=np.ones((3, 3)))
    if n == 0:
        return out
    cx, cy = np.nonzero(lab)
    cl = lab[cx, cy]
    m = np.bincount(cl, minlength=n + 1).astype(np.float64)
    mx, my = np.bincount(cl, weights=cx, minlength=n + 1) / np.maximum(m, 1), np.bincount(cl, weights=cy, minlength=n + 1) / np.maximum(m, 1)
    sxx = np.bincount(cl, weights=cx * cx, minlength=n + 1) / np.maximum(m, 1) - mx**2
    syy = np.bincount(cl, weights=cy * cy, minlength=n + 1) / np.maximum(m, 1) - my**2
    sxy = np.bincount(cl, weights=cx * cy, minlength=n + 1) / np.maximum(m, 1) - mx * my
    tr, det = sxx + syy, sxx * syy - sxy**2
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    # 一様に詰まった長方形の、辺の長さ = sqrt(12 × 分散)。区画の幅(c)を足して、端の区画を数える
    length, width = np.sqrt(12 * l1 + 1) * c, np.sqrt(12 * l2 + 1) * c
    pl = lab[ix, iy]
    top = np.full(n + 1, -np.inf)
    np.maximum.at(top, pl, height[idx])
    ok = (
        (length >= p.vehicle_length[0]) & (length <= p.vehicle_length[1])
        & (width >= p.vehicle_width[0]) & (width <= p.vehicle_width[1])
        & (top >= p.vehicle_height[0]) & (top <= p.vehicle_height[1])
    )
    ok[0] = False
    out[idx[ok[pl]]] = True
    return out


def classify_site(xyz: np.ndarray, rgb: np.ndarray, p: SiteParams | None = None) -> np.ndarray:
    """xyz: (点数,3)、rgb: (点数,3)の色(0〜1)。点ごとの分類コード(uint8)を返す。"""
    p = p or SiteParams()
    n = len(xyz)
    out = np.full(n, OTHER, dtype=np.uint8)

    # 1. 水面
    water = find_water(xyz, rgb, p.water)
    out[water] = WATER
    land = np.flatnonzero(~water)
    if land.size == 0:
        return out
    pts, col = xyz[land], rgb[land]

    # 2. 地面と、地面からの高さ
    if p.ground_with_water:
        g_all, h_all = find_ground(xyz, p.ground)
        is_ground, height = g_all[land], h_all[land]
    else:
        is_ground, height = find_ground(pts, p.ground)

    # ボクセルの形・色
    v = _Voxels(pts.astype(np.float64), height, p.voxel, 3)
    nv = len(v.keys)
    green_pt = col[:, 1] - np.maximum(col[:, 0], col[:, 2])  # 緑の強さ: 緑が、赤と青より、どれだけ強いか
    sat_pt = (col.max(axis=1) - col.min(axis=1)) / (col.max(axis=1) + 1e-6)
    green_v = np.bincount(v.inv, weights=green_pt.astype(np.float64), minlength=nv) / v.count
    sat_v = np.bincount(v.inv, weights=sat_pt.astype(np.float64), minlength=nv) / v.count
    green = (green_v >= p.green_margin)[v.inv]

    label_ = np.full(len(land), OTHER, dtype=np.uint8)

    # 6. 地面・路面(低い位置の、緑でない点)
    ground_like = is_ground & ~green
    road = ground_like & (sat_v[v.inv] <= p.road_sat_max) & (green_v[v.inv] <= p.road_green_max) & (v.flat[v.inv] <= p.road_flat_max)
    label_[ground_like] = GROUND
    if p.split_road:
        label_[road] = ROAD

    # 5. 植栽: 緑らしい点、または高くて散らばった点
    scattered = (v.flat >= p.scatter_flat_min)[v.inv] & v.ok[v.inv]
    veg = green | (~is_ground & scattered & (height >= p.dry_veg_min_height))
    veg_label = np.where(height < p.low_max, VEG_LOW, np.where(height < p.mid_max, VEG_MID, VEG_HIGH)).astype(np.uint8)
    label_[veg] = veg_label[veg]

    # 3. 建物: 緑でない平らな面が大きくつながり、高いところがある
    plane_c = (
        v.ok & (v.flat < p.plane_flat_max) & (v.mid > 0.1) & ((v.normal_z < 0.3) | (v.normal_z > 0.9))
        & (v.height >= 0.5) & (green_v < p.green_margin)
    )
    bld = _large_components(v.keys, plane_c, v.dims, p.min_building_voxels, value=v.height, min_value=p.min_building_height)
    is_bld = bld[v.inv]
    label_[is_bld] = BUILDING

    # 4. 車両: 緑でなく、建物でなく、地面から少し浮いた点のかたまり
    cand = ~green & ~is_bld & (height >= p.vehicle_h[0]) & (height <= p.vehicle_h[1])
    label_[_vehicles(pts, height, cand, p)] = VEHICLE

    out[land] = label_
    return out