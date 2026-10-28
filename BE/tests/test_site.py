"""池・森・駐車場・道路の現場向け分類のテスト。人工の現場(池・草地・舗装・車・木・低木・建物)で、おおむね当たるかを見る。"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from ptcloud.site import BUILDING, GROUND, OTHER, ROAD, VEG_HIGH, VEG_LOW, VEG_MID, VEHICLE, WATER, SiteParams, classify_site  # noqa: E402


def scene(seed=0):
    r = np.random.default_rng(seed)
    P, C, T = [], [], []
    slope = np.tan(np.radians(5))

    def ground_z(x):  # なだらかな斜面(池の水面は、別に一定の高さ)
        return x * slope

    def add(xyz, rgb, code, noise=0.02):
        P.append(xyz)
        C.append(np.clip(np.asarray(rgb) + r.normal(0, noise, (len(xyz), 3)), 0, 1))
        T.append(np.full(len(xyz), code))

    n = 50000  # 池(x 0-40, y 0-30)
    add(np.stack([r.uniform(0, 40, n), r.uniform(0, 30, n), 0.05 * r.normal(size=n) - 0.5], 1), [.50, .57, .61], WATER, 0.01)
    n = 40000  # 草地(緑の地面) x 45-75
    gx = r.uniform(45, 75, n)
    add(np.stack([gx, r.uniform(0, 30, n), ground_z(gx) + 0.05 * r.normal(size=n) + 0.05], 1), [.35, .55, .25], VEG_LOW)
    n = 30000  # 土(茶色の地面) x -25-0
    gx = r.uniform(-25, 0, n)
    add(np.stack([gx, r.uniform(0, 30, n), ground_z(gx) + 0.04 * r.normal(size=n)], 1), [.55, .42, .30], GROUND)
    n = 40000  # 舗装(灰色)の駐車場 x 80-120
    gx = r.uniform(80, 120, n)
    add(np.stack([gx, r.uniform(0, 30, n), ground_z(gx) + 0.03 * r.normal(size=n)], 1), [.45, .46, .47], ROAD, 0.01)
    colors = [(.7, .1, .1), (.9, .9, .9), (.2, .3, .7), (.1, .1, .1), (.8, .8, .2)]  # 車(緑でない色)
    for i in range(10):
        cx, cy = 85 + 3.8 * i, 8 if i % 2 else 20
        k = 1800
        add(np.stack([cx + r.uniform(-2.2, 2.2, k), cy + r.uniform(-0.9, 0.9, k), ground_z(cx) + r.uniform(0.1, 1.5, k)], 1),
            colors[i % 5], VEHICLE, 0.03)
    for cx, cy in [(50, 5), (60, 25), (70, 10), (-10, 5), (-15, 25)]:  # 木(樹冠)
        k = 3500
        a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 3 * np.cbrt(r.uniform(0, 1, k))
        add(np.stack([cx + rad * np.sin(b) * np.cos(a), cy + rad * np.sin(b) * np.sin(a), ground_z(cx) + 7 + rad * np.cos(b)], 1),
            [.28, .50, .25], VEG_HIGH, 0.05)
    for cx, cy in [(55, 15), (65, 5), (-5, 15)]:  # 低木(高さ0.7〜1.6m)
        k = 1500
        add(np.stack([cx + r.uniform(-1, 1, k), cy + r.uniform(-1, 1, k), ground_z(cx) + r.uniform(0.7, 1.6, k)], 1),
            [.30, .52, .22], VEG_MID, 0.05)
    k = 12000  # 建物(壁と平らな屋根) x 125-135, y 5-15, 高さ7m
    s, u, h = r.integers(0, 5, k), r.uniform(0, 1, k), r.uniform(0, 7, k)
    bx = np.where(s == 0, 125 + 10 * u, np.where(s == 1, 125 + 10 * u, np.where(s == 2, 125.0, np.where(s == 3, 135.0, 125 + 10 * u))))
    by = np.where(s == 0, 5.0, np.where(s == 1, 15.0, 5 + 10 * u))
    bz = np.where(s == 4, 7.0, h)
    add(np.stack([bx, by, ground_z(130) + bz], 1), [.80, .65, .42], BUILDING, 0.02)
    xyz = np.concatenate(P).astype(np.float32)
    return xyz, np.concatenate(C).astype(np.float32), np.concatenate(T)


def test_classes_are_mostly_found():
    xyz, rgb, truth = scene()
    pred = classify_site(xyz, rgb, SiteParams(split_road=True))  # 路面も分ける設定
    assert (pred == truth).mean() > 0.85
    for code, name, min_recall in ((WATER, "水面", 0.85), (VEG_HIGH, "高い植栽", 0.85), (BUILDING, "建物", 0.65),
                                   (VEHICLE, "車両", 0.7), (ROAD, "路面", 0.8), (VEG_LOW, "低い植栽", 0.8)):
        m = truth == code
        assert (pred[m] == code).mean() >= min_recall, f"{name}の再現率が低い: {(pred[m] == code).mean():.1%}"


def test_water_is_not_confused_with_ground():
    xyz, rgb, truth = scene()
    pred = classify_site(xyz, rgb)
    assert (pred[truth == WATER] == GROUND).mean() < 0.02  # 水面を、地面にしない
    assert (pred[truth == WATER] == ROAD).mean() < 0.02


def test_output_type():
    xyz, rgb, _ = scene()
    pred = classify_site(xyz, rgb)
    assert pred.shape == (len(xyz),) and pred.dtype == np.uint8
    assert set(np.unique(pred)) <= {OTHER, GROUND, VEG_LOW, VEG_MID, VEG_HIGH, BUILDING, WATER, ROAD, VEHICLE}


def test_road_is_merged_into_ground_by_default():
    # 既定では、路面と地面を分けない(舗装も、地面になる)。路面(11)は出てこない
    xyz, rgb, truth = scene()
    pred = classify_site(xyz, rgb)
    assert (pred == ROAD).sum() == 0
    assert (pred[truth == ROAD] == GROUND).mean() > 0.85
    assert (pred[truth == GROUND] == GROUND).mean() > 0.95