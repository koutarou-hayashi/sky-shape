"""池・森・駐車場・道路の現場向け分類のテスト。人工の現場(池・草地・舗装・車・木・低木・建物)で、おおむね当たるかを見る。"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from ptcloud.site import (  # noqa: E402
    BUILDING, FOREST, GROUND, OTHER, ROAD, VEG_HIGH, VEG_LOW, VEG_MID, VEHICLE, WATER, SiteParams, classify_site,
)


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
    pred = classify_site(xyz, rgb, SiteParams(split_road=True, grass_is_ground=False))  # 路面も分け、草地は低い植栽にする設定
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
    assert set(np.unique(pred)) <= {OTHER, GROUND, VEG_LOW, VEG_MID, VEG_HIGH, BUILDING, WATER, ROAD, VEHICLE, FOREST}


def test_road_is_merged_into_ground_by_default():
    # 既定では、路面と地面を分けない(舗装も、地面になる)。路面(11)は出てこない
    xyz, rgb, truth = scene()
    pred = classify_site(xyz, rgb)
    assert (pred == ROAD).sum() == 0
    assert (pred[truth == ROAD] == GROUND).mean() > 0.85
    assert (pred[truth == GROUND] == GROUND).mean() > 0.95


def test_grass_on_the_ground_is_ground_by_default():
    # 地面の高さの草(芝の斜面)は、既定では、植栽(低)でなく、地面になる。設定で、植栽(低)に戻せる
    xyz, rgb, truth = scene()
    grass = truth == VEG_LOW
    assert (classify_site(xyz, rgb)[grass] == GROUND).mean() > 0.9
    assert (classify_site(xyz, rgb, SiteParams(grass_is_ground=False))[grass] == VEG_LOW).mean() > 0.9


def forest_scene(seed=0):
    """左: 密集した林(4m間隔の木、70x60m)、右: 数本の木(15m以上離れている)。どちらの下にも、土の地面がある。"""
    r = np.random.default_rng(seed)
    P, C, T = [], [], []

    def add(xyz, rgb, code, noise=0.04):
        P.append(xyz)
        C.append(np.clip(np.asarray(rgb) + r.normal(0, noise, (len(xyz), 3)), 0, 1))
        T.append(np.full(len(xyz), code))

    n = 60000  # 地面(土)
    gx = np.concatenate([r.uniform(0, 70, n), r.uniform(100, 145, n // 2)])
    add(np.stack([gx, r.uniform(0, 60, len(gx)), 0.04 * r.normal(size=len(gx))], 1), [.55, .42, .30], GROUND, 0.02)
    k = 700  # 林: 樹冠
    for cx in np.arange(2, 70, 4.0):
        for cy in np.arange(2, 60, 4.0):
            x, y = cx + r.uniform(-0.5, 0.5), cy + r.uniform(-0.5, 0.5)
            a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 2.2 * np.cbrt(r.uniform(0, 1, k))
            add(np.stack([x + rad * np.sin(b) * np.cos(a), y + rad * np.sin(b) * np.sin(a), 7 + rad * np.cos(b)], 1),
                [.28, .50, .25], FOREST, 0.05)
    for cx, cy in [(105, 10), (125, 40), (140, 12), (110, 50)]:  # 数本の木(離れている)
        k = 3500
        a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 3 * np.cbrt(r.uniform(0, 1, k))
        add(np.stack([cx + rad * np.sin(b) * np.cos(a), cy + rad * np.sin(b) * np.sin(a), 7 + rad * np.cos(b)], 1),
            [.28, .50, .25], VEG_HIGH, 0.05)
    return np.concatenate(P).astype(np.float32), np.concatenate(C).astype(np.float32), np.concatenate(T)


def test_dense_trees_become_forest_and_a_few_trees_stay_planting():
    xyz, rgb, truth = forest_scene()
    pred = classify_site(xyz, rgb)
    assert (pred[truth == FOREST] == FOREST).mean() > 0.9  # 密集した木は、林
    assert (pred[truth == VEG_HIGH] == FOREST).mean() < 0.05  # 数本の木は、林にならない
    assert (pred[truth == VEG_HIGH] == VEG_HIGH).mean() > 0.85  # 植栽(高)のまま
    assert (pred[truth == GROUND] == GROUND).mean() > 0.9


def test_forest_can_be_turned_off():
    xyz, rgb, truth = forest_scene()
    pred = classify_site(xyz, rgb, SiteParams(forest=False))
    assert (pred == FOREST).sum() == 0


def test_grass_on_the_ground_is_ground_by_default():
    # 草地(緑の点が、地面の高さにある)は、林の外なら、既定で地面。grass_is_ground=False なら、低い植栽
    xyz, rgb, truth = scene()
    grass = truth == VEG_LOW
    assert (classify_site(xyz, rgb)[grass] == GROUND).mean() > 0.9
    assert (classify_site(xyz, rgb, SiteParams(grass_is_ground=False))[grass] == VEG_LOW).mean() > 0.9


def forest_scene(seed=0):
    """左半分(x<60)は、木が密集した広い林。右半分は、離れた数本の木と、草地。"""
    r = np.random.default_rng(seed)
    P, C, T = [], [], []

    def add(xyz, rgb, code, noise=0.04):
        P.append(xyz)
        C.append(np.clip(np.asarray(rgb) + r.normal(0, noise, (len(xyz), 3)), 0, 1))
        T.append(np.full(len(xyz), code))

    n = 90000  # 地面(土)全体
    gx, gy = r.uniform(0, 120, n), r.uniform(0, 60, n)
    add(np.stack([gx, gy, 0.04 * r.normal(size=n)], 1), [.55, .42, .30], GROUND, 0.02)
    for cx in np.arange(2.5, 60, 5.0):  # 林: 5m間隔に、樹冠
        for cy in np.arange(2.5, 60, 5.0):
            k = 600
            a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 2.8 * np.cbrt(r.uniform(0, 1, k))
            add(np.stack([cx + rad * np.sin(b) * np.cos(a), cy + rad * np.sin(b) * np.sin(a), 8 + rad * np.cos(b)], 1),
                [.28, .50, .25], FOREST)
    for cx, cy in [(75, 10), (95, 30), (110, 50)]:  # 離れた数本の木
        k = 3000
        a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 3 * np.cbrt(r.uniform(0, 1, k))
        add(np.stack([cx + rad * np.sin(b) * np.cos(a), cy + rad * np.sin(b) * np.sin(a), 7 + rad * np.cos(b)], 1),
            [.28, .50, .25], VEG_HIGH)
    xyz = np.concatenate(P).astype(np.float32)
    return xyz, np.concatenate(C).astype(np.float32), np.concatenate(T)


def test_dense_trees_become_forest_and_a_few_trees_stay_planting():
    xyz, rgb, truth = forest_scene()
    pred = classify_site(xyz, rgb)
    assert (pred[truth == FOREST] == FOREST).mean() > 0.9  # 密集した広い林は、林
    assert (pred[truth == VEG_HIGH] == FOREST).mean() < 0.02  # 離れた数本の木は、林にしない
    assert (pred[truth == VEG_HIGH] == VEG_HIGH).mean() > 0.85  # 植栽(高)のまま


def test_forest_can_be_turned_off():
    xyz, rgb, truth = forest_scene()
    pred = classify_site(xyz, rgb, SiteParams(forest=False))
    assert (pred == FOREST).sum() == 0
    assert (pred[truth == FOREST] == VEG_HIGH).mean() > 0.85  # 林にしなければ、高い植栽