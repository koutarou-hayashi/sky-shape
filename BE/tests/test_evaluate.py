import numpy as np
import pytest

from ptcloud.evaluate import evaluate, remap


def test_perfect_match():
    t = np.array([2, 2, 5, 6, 6, 6], dtype=np.uint8)
    r = evaluate(t, t.copy())
    assert r.corrections == 0
    assert r.accuracy == 1.0
    assert r.miou == 1.0


def test_counts_and_metrics():
    #            正解: 2 2 2 5 5 6
    truth = np.array([2, 2, 2, 5, 5, 6], dtype=np.uint8)
    pred = np.array([2, 2, 5, 5, 6, 6], dtype=np.uint8)
    r = evaluate(truth, pred)
    assert r.labels == [2, 5, 6]
    assert r.corrections == 2  # 3番目(2→5)と5番目(5→6)
    assert r.correct == 4
    # クラス2: 正解3点のうち2点を当て、予測2点はすべて正解
    assert r.recall[0] == pytest.approx(2 / 3)
    assert r.precision[0] == 1.0
    assert r.iou[0] == pytest.approx(2 / 3)
    assert r.confusion.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_ignore_unlabeled_truth():
    truth = np.array([1, 1, 2, 2], dtype=np.uint8)  # 1は正解データを作っていない点
    pred = np.array([5, 6, 2, 5], dtype=np.uint8)
    r = evaluate(truth, pred, ignore=[1])
    assert r.total == 2
    assert r.corrections == 1


def test_mapping_pred():
    truth = np.array([5, 5, 5], dtype=np.uint8)
    pred = np.array([3, 4, 5], dtype=np.uint8)  # 他のソフトの低い植生・中位の植生・高い植生
    assert evaluate(truth, pred).corrections == 2
    assert evaluate(truth, pred, mapping={3: 5, 4: 5}).corrections == 0
    assert remap(pred, {3: 5}).tolist() == [5, 4, 5]


def test_size_mismatch_raises():
    with pytest.raises(ValueError, match="点数が違います"):
        evaluate(np.zeros(3, np.uint8), np.zeros(4, np.uint8))


def test_all_ignored_raises():
    with pytest.raises(ValueError):
        evaluate(np.ones(3, np.uint8), np.ones(3, np.uint8), ignore=[1])


def test_mapping_truth():
    truth = np.array([3, 4, 5, 2], dtype=np.uint8)  # 正解データを作ったソフトでは、植栽が3・4・5に分かれている
    pred = np.array([5, 5, 5, 2], dtype=np.uint8)
    assert evaluate(truth, pred).corrections == 2
    assert evaluate(truth, pred, truth_mapping={3: 5, 4: 5}).corrections == 0