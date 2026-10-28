"""水面抽出のテスト。人工の池(木・灰色の舗装・土の島つき)で、水だけが当たるかを見る。"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from ptcloud.water import find_water  # noqa: E402


def scene(seed=0):
    r = np.random.default_rng(seed)
    P, C, K = [], [], []

    def add(xyz, rgb, kind, noise=0.012):
        P.append(xyz)
        C.append(np.clip(rgb + r.normal(0, noise, (len(xyz), 3)), 0, 1))
        K.append(np.full(len(xyz), kind))

    n = 60000  # 池(40x30m)。水面の色は、WingEarthの水面の分布に近い値
    add(np.stack([r.uniform(0, 40, n), r.uniform(0, 30, n), 0.06 * r.normal(size=n)], 1), np.array([.50, .57, .61]), 1, 0.01)
    n = 40000  # 岸の地面(土)
    gx, gy = r.uniform(-20, 0, n), r.uniform(0, 30, n)
    add(np.stack([gx, gy, 0.3 - gx * 0.1 + 0.04 * r.normal(size=n)], 1), np.array([.55, .42, .30]), 0)
    n = 30000  # 灰色の舗装(平らだが、色が水と違う)
    add(np.stack([r.uniform(45, 70, n), r.uniform(0, 30, n), 0.3 + 0.03 * r.normal(size=n)], 1), np.array([.45, .46, .47]), 0)
    for cx, cy in [(5, 5), (35, 25), (-8, 15)]:  # 木(水の上にも張り出す)
        k = 3000
        a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 3 * np.cbrt(r.uniform(0, 1, k))
        add(np.stack([cx + rad * np.sin(b) * np.cos(a), cy + rad * np.sin(b) * np.sin(a), 5 + rad * np.cos(b)], 1),
            np.array([.28, .50, .25]), 2, 0.05)
    n = 800  # 池の中の小さな島
    add(np.stack([20 + r.uniform(-1, 1, n), 15 + r.uniform(-1, 1, n), 0.4 + 0.05 * r.normal(size=n)], 1), np.array([.5, .4, .3]), 0)
    return np.concatenate(P).astype(np.float32), np.concatenate(C).astype(np.float32), np.concatenate(K)


def test_water_is_found_and_others_are_not():
    xyz, rgb, kind = scene()
    w = find_water(xyz, rgb)
    assert w[kind == 1].mean() > 0.9  # 水面の大半が当たる
    assert (kind[w] == 1).mean() > 0.98  # 水と判定した点の、ほぼすべてが水
    assert w[kind == 2].mean() < 0.01  # 張り出した木を、水にしない


def test_flat_gray_pavement_is_not_water():
    xyz, rgb, kind = scene()
    only = kind == 0
    assert find_water(xyz[only], rgb[only]).mean() < 0.01  # 水がなければ、水は出ない(平らな舗装も水にならない)


def test_small_puddle_is_ignored():
    # 5m四方(25m²)だけの水たまりは、min_area(100m²)に満たないので、水面にしない
    r = np.random.default_rng(0)
    n = 5000
    xyz = np.stack([r.uniform(0, 5, n), r.uniform(0, 5, n), 0.04 * r.normal(size=n)], 1).astype(np.float32)
    rgb = np.clip(np.array([.50, .57, .61]) + r.normal(0, 0.01, (n, 3)), 0, 1).astype(np.float32)
    assert find_water(xyz, rgb).mean() < 0.01


def test_look_alike_road_at_a_different_height_is_not_water():
    # 池のすぐ横に、水と似た色(広い色の範囲には入るが、確かな水の範囲の外)で、平らな路面が、水面より0.6m高い位置にある。
    # 路面のほうが点が多くても、路面を水面にせず、池は水面のまま
    xyz, rgb, kind = scene()
    r = np.random.default_rng(3)
    n = 120000
    road = np.stack([r.uniform(0, 40, n), r.uniform(31, 61, n), 0.6 + 0.04 * r.normal(size=n)], 1).astype(np.float32)  # 池の北隣
    road_rgb = np.clip(np.array([.47, .58, .61]) + r.normal(0, 0.005, (n, 3)), 0, 1).astype(np.float32)
    w = find_water(np.concatenate([xyz, road]), np.concatenate([rgb, road_rgb]))
    assert w[len(xyz):].mean() < 0.02  # 似た色の路面を、水にしない
    assert w[:len(xyz)][kind == 1].mean() > 0.9  # 池は、変わらず水と判定される


def test_pond_with_slightly_different_color_is_still_found():
    # 水面の一部が、少し鮮やかな色(広い色の範囲には入るが、種の範囲の外)でも、同じ高さなら水面になる
    xyz, rgb, kind = scene()
    part = (kind == 1) & (xyz[:, 0] > 30)
    rgb = rgb.copy()
    rgb[part] = np.clip(np.array([.46, .56, .61]) + np.random.default_rng(1).normal(0, 0.01, (int(part.sum()), 3)), 0, 1)
    w = find_water(xyz, rgb)
    assert w[part].mean() > 0.8