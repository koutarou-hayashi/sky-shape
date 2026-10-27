"""地面抽出のテスト。傾斜した地面、法面の段差の上に、木と建物の屋根を置いて、地面が当たるかを見る。"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from ptcloud.ground import find_ground  # noqa: E402


def terrain(x, kind):
    if kind == "slope15":
        return x * np.tan(np.radians(15))
    if kind == "slope30":
        return x * np.tan(np.radians(30))
    if kind == "embankment":  # 平ら → 45度の法面 → 10m高い平ら
        return np.clip(x - 20, 0, 10)
    return 0 * x


def scene(kind, seed=0):
    r = np.random.default_rng(seed)
    n = 60000
    gx, gy = r.uniform(0, 60, n), r.uniform(0, 30, n)
    parts = [(np.stack([gx, gy, terrain(gx, kind) + 0.04 * r.normal(size=n)], 1), 0)]
    for cx, cy in [(8, 8), (45, 22)]:  # 木(樹冠)
        k = 2500
        a, b, rad = r.uniform(0, 6.28, k), np.arccos(2 * r.uniform(0, 1, k) - 1), 2.5 * np.cbrt(r.uniform(0, 1, k))
        parts.append((np.stack([cx + rad * np.sin(b) * np.cos(a), cy + rad * np.sin(b) * np.sin(a),
                                terrain(cx, kind) + 5 + rad * np.cos(b)], 1), 1))
    m = 4000  # 平らな屋根(地面から8m)
    bx, by = r.uniform(5, 15, m), r.uniform(18, 26, m)
    parts.append((np.stack([bx, by, terrain(10, kind) + 8 + 0.02 * r.normal(size=m)], 1), 2))
    xyz = np.concatenate([p[0] for p in parts]).astype(np.float32)
    kind_id = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
    return xyz, kind_id, xyz[:, 2] - terrain(xyz[:, 0], kind)


@pytest.mark.parametrize("kind,min_recall", [("flat", 0.98), ("slope15", 0.98), ("slope30", 0.98), ("embankment", 0.90)])
def test_ground_found_on_slopes(kind, min_recall):
    xyz, k, _ = scene(kind)
    is_ground, _ = find_ground(xyz)
    assert is_ground[k == 0].mean() >= min_recall
    assert is_ground[k == 1].mean() < 0.02  # 木の上は地面にしない
    assert is_ground[k == 2].mean() < 0.02  # 大きな屋根を、地面にしない


@pytest.mark.parametrize("kind", ["slope15", "slope30", "embankment"])
def test_height_above_ground_is_correct_on_slopes(kind):
    xyz, k, true_h = scene(kind)
    _, height = find_ground(xyz)
    tree = k == 1
    assert np.abs(height[tree] - true_h[tree]).mean() < 0.3  # 傾斜があっても、樹冠の高さが合う


def test_gap_between_clusters_does_not_crash():
    # 離れた2か所だけに点がある(間は空っぽ)。窓の中に点がない場所があっても、止まらない
    r = np.random.default_rng(0)
    a = np.stack([r.uniform(0, 30, 20000), r.uniform(0, 20, 20000), 0.04 * r.normal(size=20000)], 1)
    b = np.stack([r.uniform(200, 230, 20000), r.uniform(0, 20, 20000), 5 + 0.04 * r.normal(size=20000)], 1)
    is_ground, height = find_ground(np.concatenate([a, b]).astype(np.float32))
    assert is_ground.mean() > 0.95
    assert np.isfinite(height).all()