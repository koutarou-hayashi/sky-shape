"""ルールベース分類のテスト。人工の小さな街(地面・壁・電柱・電線・木)で、各クラスが当たるかを見る。"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from ptcloud.rules import BUILDING, GROUND, POLE, VEGETATION, WIRE, classify  # noqa: E402


def make_scene(seed=0):
    r = np.random.default_rng(seed)
    parts = []

    def add(x, y, z, code):
        parts.append((np.stack([x, y, z], 1).astype(np.float32), np.full(len(x), code, np.uint8)))

    n = 6000  # 地面
    add(r.uniform(-20, 20, n), r.uniform(-20, 20, n), r.uniform(0, 0.05, n), GROUND)
    n = 1600  # 壁(幅10m・高さ8m)
    add(r.uniform(-5, 5, n), np.full(n, 15.0) + r.uniform(-0.02, 0.02, n), r.uniform(0, 8, n), BUILDING)
    n = 400  # 電柱(高さ9m)
    add(5 + r.uniform(-0.15, 0.15, n), -8 + r.uniform(-0.15, 0.15, n), r.uniform(0, 9, n), POLE)
    n = 600  # 電線(高さ7.5m、水平)
    add(r.uniform(-5, 10, n), np.full(n, -8.0), np.full(n, 7.5) + r.uniform(-0.03, 0.03, n), WIRE)
    # 木(幹と樹冠)
    n = 150
    add(-10 + r.uniform(-0.15, 0.15, n), -10 + r.uniform(-0.15, 0.15, n), r.uniform(0, 4, n), VEGETATION)
    n = 1500
    a, b, rad = r.uniform(0, 2 * np.pi, n), np.arccos(2 * r.uniform(0, 1, n) - 1), 2.5 * np.cbrt(r.uniform(0, 1, n))
    add(-10 + rad * np.sin(b) * np.cos(a), -10 + rad * np.sin(b) * np.sin(a), 5.5 + rad * np.cos(b), VEGETATION)
    xyz = np.concatenate([p[0] for p in parts])
    truth = np.concatenate([p[1] for p in parts])
    order = r.permutation(len(xyz))  # 点の順番を混ぜる
    return xyz[order], truth[order]


@pytest.mark.parametrize("code,name,min_recall", [
    (GROUND, "地面", 0.95),
    (BUILDING, "建物", 0.85),
    (POLE, "電柱", 0.60),
    (WIRE, "電線", 0.75),
    (VEGETATION, "植栽", 0.90),
])
def test_each_class_is_found(code, name, min_recall):
    xyz, truth = make_scene()
    pred = classify(xyz)
    m = truth == code
    recall = (pred[m] == code).mean()
    assert recall >= min_recall, f"{name}の再現率が低い: {recall:.1%}"


def test_overall_accuracy():
    xyz, truth = make_scene()
    assert (classify(xyz) == truth).mean() >= 0.93


def test_output_shape_and_type():
    xyz, _ = make_scene()
    pred = classify(xyz)
    assert pred.shape == (len(xyz),) and pred.dtype == np.uint8


def test_no_ground_only_ground():
    # 平らな地面だけの点群は、すべて地面になる
    r = np.random.default_rng(1)
    xyz = np.stack([r.uniform(0, 30, 3000), r.uniform(0, 30, 3000), r.uniform(0, 0.03, 3000)], 1).astype(np.float32)
    assert (classify(xyz) == GROUND).all()