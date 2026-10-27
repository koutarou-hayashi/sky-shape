"""機械学習による分類(第1段階: 特徴量 + 勾配ブースティング)。

正解つきのLAS(WingEarthなどで手動分類したもの)で学習して、別のLASの分類コードを予測する。
学習したモデルは1つのファイル(.joblib)に保存して、別のPCにも持ち運べる。
"""
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import evaluate as ev
from . import lasio
from .features import compute_features, feature_names

MODEL_VERSION = 1


def _load(path):
    xyz, rgb, inten, cls = lasio.read_points(path)
    return compute_features(xyz, rgb, inten), cls, xyz


def _train_mask(xyz: np.ndarray, valid_frac: float, block: float) -> np.ndarray:
    """学習用の点を True、検証用の点を False にする。長い方の水平軸を block(m)ごとの帯に区切って、
    約 valid_frac の帯を、間をあけて検証用にする。

    ランダムに点を分けると、隣り合う点が学習と検証の両方に入り、精度が実際より高く出る。
    かといって、端をまとめて検証用にすると、そこにしかないクラスが学習から消える。そのため、帯で分ける。
    """
    axis = 0 if np.ptp(xyz[:, 0]) >= np.ptp(xyz[:, 1]) else 1
    ids = np.floor((xyz[:, axis] - xyz[:, axis].min()) / block).astype(np.int64)
    return ~(np.floor((ids + 1) * valid_frac) > np.floor(ids * valid_frac))


def train(truth_paths, model_path, valid_paths=(), valid_frac=0.3, block=25.0, balanced=False, max_iter=200, seed=0,
          log=print):
    """正解つきLASで学習して、モデルを保存する。検証の結果(ev.Result)を返す(検証用がなければ None)。

    valid_paths があれば、それを検証用に使う。なければ、各ファイルを block(m)の帯に区切り、約 valid_frac を検証用にする。
    """
    xs, ys, xv, yv = [], [], [], []
    for path in truth_paths:
        log(f"読み込み・特徴量の計算: {path}")
        X, y, xyz = _load(path)
        if valid_paths or valid_frac <= 0:
            xs.append(X), ys.append(y)
        else:
            tr = _train_mask(xyz, valid_frac, block)
            xs.append(X[tr]), ys.append(y[tr])
            xv.append(X[~tr]), yv.append(y[~tr])
    for path in valid_paths:
        log(f"読み込み・特徴量の計算(検証用): {path}")
        X, y, _ = _load(path)
        xv.append(X), yv.append(y)
    X, y = np.concatenate(xs), np.concatenate(ys)
    if xv:
        lost = sorted(set(np.concatenate(yv).tolist()) - set(y.tolist()))
        if lost:
            log(f"注意: 検証用にしかないクラスがあります: {lost}。学習に含まれないので、当たりません。--block を小さくするか、別のファイルを検証にしてください。")
    log(f"学習: {len(y):,}点  " + "  ".join(f"{c}:{n:,}" for c, n in zip(*np.unique(y, return_counts=True))))
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, class_weight="balanced" if balanced else None, random_state=seed
    ).fit(X, y)
    joblib.dump({"version": MODEL_VERSION, "features": feature_names(), "model": clf}, model_path)
    log(f"モデルを保存しました: {model_path}")
    if not xv:
        return None
    Xv, yv = np.concatenate(xv), np.concatenate(yv)
    log(f"検証: {len(yv):,}点")
    return ev.evaluate(yv, clf.predict(Xv).astype(np.uint8))


def load_model(path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("features") != feature_names():
        raise ValueError("このモデルは、今の特徴量と合いません。もう一度、学習してください。")
    return bundle


def predict_file(bundle: dict, src, dst) -> dict:
    """1ファイルを分類して保存し、分類コードごとの点数を返す。"""
    xyz, rgb, inten, _ = lasio.read_points(src)
    pred = bundle["model"].predict(compute_features(xyz, rgb, inten)).astype(np.uint8)
    lasio.write_classification(src, dst, pred)
    return {int(c): int(n) for c, n in zip(*np.unique(pred, return_counts=True))}