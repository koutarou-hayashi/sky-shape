"""機械学習の部分のテスト。人工の小さな街に、色をつけて試す。LASの読み書きは差し替えるので、laspyは要らない。"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("laspy")

from ptcloud import lasio, ml  # noqa: E402
from ptcloud.features import compute_features, feature_names  # noqa: E402
from ptcloud.rules import BUILDING, GROUND, POLE, VEGETATION, WIRE  # noqa: E402
from tests.test_rules import make_scene  # noqa: E402

COLORS = {GROUND: (.5, .5, .48), BUILDING: (.8, .65, .42), VEGETATION: (.3, .62, .3), WIRE: (.15, .15, .17), POLE: (.43, .33, .27)}


def scene(seed):
    xyz, truth = make_scene(seed)
    noise = np.random.default_rng(seed + 100).normal(0, 0.04, (len(truth), 3))
    rgb = np.clip(np.array([COLORS[int(c)] for c in truth]) + noise, 0, 1).astype(np.float32)
    return xyz, rgb, np.zeros(len(truth), dtype=np.float32), truth


def test_features_shape():
    xyz, rgb, inten, _ = scene(0)
    X = compute_features(xyz, rgb, inten)
    assert X.shape == (len(xyz), len(feature_names()))
    assert X.dtype == np.float32


def test_validation_blocks_are_spread_out():
    xyz, _ = make_scene(0)
    train = ml._train_mask(xyz, 0.3, block=4.0)
    assert 0.15 < (~train).mean() < 0.45
    # 検証用の帯が、1か所に固まっていない(検証用の点の間に、学習用の点が入っている)
    axis = 0 if np.ptp(xyz[:, 0]) >= np.ptp(xyz[:, 1]) else 1
    v = xyz[~train, axis]
    between = train & (xyz[:, axis] > v.min()) & (xyz[:, axis] < v.max())
    assert between.any()


def test_train_and_predict_roundtrip(tmp_path, monkeypatch):
    store = {}
    monkeypatch.setattr(lasio, "read_points", lambda p: scene(int(str(p)[0])))
    monkeypatch.setattr(lasio, "write_classification", lambda s, d, c: store.update(pred=c))
    model = tmp_path / "m.joblib"
    result = ml.train(["0.las"], model, valid_frac=0.3, block=4.0, log=lambda m: None)
    assert result.accuracy > 0.95  # 人工の街は、色と形で分けやすい
    counts = ml.predict_file(ml.load_model(model), "1.las", "out.las")  # 別の街に当てはめる
    _, truth = make_scene(1)
    assert (store["pred"] == truth).mean() > 0.95
    assert sum(counts.values()) == len(truth)