"""LASの読み書きのテスト。laspyが入っていない環境では自動的に飛ばす。"""
import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from ptcloud import lasio  # noqa: E402


def make_las(path, classes, fmt=3):
    header = laspy.LasHeader(point_format=fmt, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [35000.0, -82000.0, 10.0]
    las = laspy.LasData(header)
    n = len(classes)
    las.x = 35000.0 + np.arange(n) * 0.5
    las.y = -82000.0 + np.arange(n) * 0.25
    las.z = 10.0 + np.arange(n) * 0.1
    las.classification = np.asarray(classes, dtype=np.uint8)
    las.red = (np.arange(n) * 100).astype(np.uint16)
    las.write(path)


def test_info_and_read(tmp_path):
    p = tmp_path / "a.las"
    make_las(p, [2, 2, 5, 14, 14, 14])
    i = lasio.info(p, chunk=4)  # チャンクをまたいでも数えられること
    assert i["point_count"] == 6
    assert i["class_counts"] == {2: 2, 5: 1, 14: 3}
    assert lasio.read_classification(p, chunk=4).tolist() == [2, 2, 5, 14, 14, 14]


def test_remove_classes_keeps_attributes(tmp_path):
    src, dst = tmp_path / "a.las", tmp_path / "b.las"
    make_las(src, [2, 14, 5, 14, 2, 15])
    res = lasio.remove_classes(src, dst, [14, 15], chunk=4)
    assert res == {"kept": 3, "removed": {14: 2, 15: 1}}
    out = laspy.read(dst)
    assert np.asarray(out.classification).tolist() == [2, 5, 2]
    assert out.header.point_count == 3
    assert list(out.header.offsets) == [35000.0, -82000.0, 10.0]  # 座標系(ずらし量)を維持
    assert out.red.tolist() == [0, 200, 400]  # RGBも残った点の分だけ引き継ぐ
    assert laspy.read(src).header.point_count == 6  # 元のファイルは変わらない


def test_write_classification(tmp_path):
    src, dst = tmp_path / "a.las", tmp_path / "b.las"
    make_las(src, [1] * 6)
    lasio.write_classification(src, dst, np.array([2, 2, 5, 6, 14, 15], dtype=np.uint8), chunk=4)
    assert lasio.read_classification(dst).tolist() == [2, 2, 5, 6, 14, 15]
    with pytest.raises(ValueError):
        lasio.write_classification(src, dst, np.zeros(5, dtype=np.uint8))


def test_crop_keeps_points_in_range_and_attributes(tmp_path):
    src, dst = tmp_path / "a.las", tmp_path / "b.las"
    make_las(src, [2, 14, 5, 14, 2, 15])  # x = 35000 + 0.5*i, y = -82000 + 0.25*i
    kept = lasio.crop(src, dst, xmin=35000.4, xmax=35001.6, ymin=-82001.0, ymax=-81999.0, chunk=4)
    out = laspy.read(dst)
    assert kept == 3  # i = 1, 2, 3 (x = 35000.5, 35001.0, 35001.5)
    assert np.asarray(out.classification).tolist() == [14, 5, 14]
    assert out.red.tolist() == [100, 200, 300]  # RGBも引き継ぐ
    assert laspy.read(src).header.point_count == 6  # 元のファイルは変わらない