"""地面の抽出(傾斜に対応)。「下から当てる平面」を、場所ごとに当てはめる方法。

1. 水平に cell(m)四方の区画に区切り、区画ごとの一番低い点の高さ(区画の最低点)を求める。
2. 各区画のまわり(window×window 区画)の最低点に、**平面**を当てはめる。ただし、平面より高い区画
   (建物・木・車など)は重みを小さくして、当てはめ直す。これを数回くり返すと、平面は地面に沿う。
   平面なので、傾斜があっても、地面の高さがずれない(法面にも対応)。
3. まず広い窓(coarse)で当てはめて、大きな建物の屋根を地面と間違えないようにしてから、
   狭い窓(window)で、細かい地形に合わせる。
4. 点は、その区画の平面から tol(m)以内なら地面。「地面からの高さ」は、平面からの高さ。

区画が空の場所や、タイルの端でも、まわりの区画から推定するので、穴にならない(三角網より端に強い)。

限界
- 地面がまったく見えない場所(密な森の下など)では、見えている一番低い面が、地面になる。
- 建物が coarse の窓より大きいと、その屋根が地面になる。
- 曲がった地形は、平面で近似するので、window より細かい凹凸は、ならされる。
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter


@dataclass
class GroundParams:
    cell: float = 1.0  # 区画の大きさ(m)
    window: int = 9  # 細かい当てはめの窓(区画の数)
    coarse: int = 41  # 大きな建物の屋根を除くための、広い窓(区画の数)
    tol: float = 0.15  # 平面から、この高さ以内の区画は、地面とみなして重みを大きくする(m)
    low: float = 0.5  # 平面より、これ以上低い区画は、ノイズとして重みを小さくする(m)
    point_tol: float = 0.3  # 点が、平面から、この高さ以内なら、地面(m)
    rise: float = 1.0  # 広い窓の平面より、これ以上高い区画は、細かい当てはめに使わない(m)
    slope_bias: float = 1.0  # 傾斜地で、区画の最低点が低めに出る分の補正の強さ(0なら補正しない)
    iterations: int = 6
    min_cells: float = 6.0  # 平面を当てはめるのに必要な、窓の中の(重みつき)区画の数


def _fit_plane(Z, w, win, minc):
    """窓ごとに、重みつきの最小二乗で平面 z = a + b·u + c·v を当てはめる。(u, v は区画の中心からの距離)

    戻り値: a(区画の中心での平面の高さ)、b、c(傾き)。当てはめられない所は nan。
    """
    nx, ny = Z.shape
    X, Y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
    n2 = float(win * win)

    def S(a):
        return uniform_filter(a, win, mode="constant") * n2

    W, Sx, Sy, Sz = S(w), S(w * X), S(w * Y), S(w * Z)
    Sxx, Sxy, Syy, Sxz, Syz = S(w * X * X), S(w * X * Y), S(w * Y * Y), S(w * X * Z), S(w * Y * Z)
    Su, Sv = Sx - X * W, Sy - Y * W  # 中心からの距離 u = X' - X, v = Y' - Y のモーメント
    Suu, Svv = Sxx - 2 * X * Sx + X * X * W, Syy - 2 * Y * Sy + Y * Y * W
    Suv = Sxy - X * Sy - Y * Sx + X * Y * W
    Suz, Svz = Sxz - X * Sz, Syz - Y * Sz
    A = np.empty((nx, ny, 3, 3))
    A[..., 0, 0], A[..., 0, 1], A[..., 0, 2] = W, Su, Sv
    A[..., 1, 0], A[..., 1, 1], A[..., 1, 2] = Su, Suu, Suv
    A[..., 2, 0], A[..., 2, 1], A[..., 2, 2] = Sv, Suv, Svv
    ridge = 1e-6 * (W + 1.0)  # 点が一直線にしか並ばないときも、解けるようにする
    A[..., 1, 1] += ridge
    A[..., 2, 2] += ridge
    bad = W < minc
    # 点がほとんどない窓(離れた場所の間など)は、式が解けないので、単位行列に置き換えて、結果を捨てる
    A[bad] = np.eye(3)
    rhs = np.stack([Sz, Suz, Svz], axis=-1)[..., None]
    rhs[bad] = 0.0
    sol = np.linalg.solve(A, rhs)[..., 0]
    return np.where(bad, np.nan, sol[..., 0]), np.where(bad, 0.0, sol[..., 1]), np.where(bad, 0.0, sol[..., 2])


def _fill(a, b, c):
    """nan の区画を、一番近い区画の平面で埋める。"""
    nan = np.isnan(a)
    if not nan.any():
        return a, b, c
    if nan.all():
        return np.zeros_like(a), b, c
    idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return a[tuple(idx)], b[tuple(idx)], c[tuple(idx)]


def find_ground(xyz: np.ndarray, p: GroundParams | None = None):
    """各点が地面かどうか(bool)と、地面からの高さ(float32、m)を返す。xyz: (点数,3)、Zが上。"""
    p = p or GroundParams()
    x, y, z = (xyz[:, i].astype(np.float64) for i in range(3))
    ix = ((x - x.min()) / p.cell).astype(np.int64)
    iy = ((y - y.min()) / p.cell).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    zmin = np.full((nx, ny), np.inf)
    np.minimum.at(zmin, (ix, iy), z)
    valid = np.isfinite(zmin)
    Z = np.where(valid, zmin, 0.0)
    ones = valid.astype(np.float64)

    def robust(win, w0):
        w = w0
        for _ in range(p.iterations):
            a, b, c = _fit_plane(Z, w, win, p.min_cells)
            r = Z - a
            ok = valid & np.isfinite(a) & (r <= p.tol) & (r >= -p.low)
            w = np.where(ok, 1.0, 1e-3) * ones
        return _fit_plane(Z, w, win, p.min_cells)

    # 1) 広い窓: 大きな建物の屋根・木の上面を、地面から外す
    a0, _, _ = _fill(*robust(p.coarse, ones))
    # 2) 狭い窓: 広い窓の平面より高すぎる区画を使わずに、細かい地形に合わせる
    w1 = np.where(valid & (Z <= a0 + p.rise), 1.0, 1e-3) * ones
    a, b, c = _fill(*robust(p.window, w1))

    # 点ごとの地面の高さ = その区画の平面(区画の中での位置による傾きも補正)。
    # 区画の一番低い点は、傾斜の下側の端にあるので、平面は、区画の中心より低くなる。その分(傾きの半分)を持ち上げる。
    u = (x - x.min()) / p.cell - ix  # 区画の中での位置(0〜1)
    v = (y - y.min()) / p.cell - iy
    bias = 0.5 * (np.abs(b) + np.abs(c))[ix, iy] * p.slope_bias
    surface = a[ix, iy] + b[ix, iy] * (u - 0.5) + c[ix, iy] * (v - 0.5) + bias
    height = z - surface
    return height <= p.point_tol, height.astype(np.float32)