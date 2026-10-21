"""分類結果の評価。正解の分類と、予測(ツールの出力)の分類を点ごとに比べる。

比較の中心は「手動で修正が必要な点数」(= 正解と違う点の数)。
そのほか、クラスごとの適合率・再現率・IoU、全体の正解率、mIoU を出す。
"""
import csv
from dataclasses import dataclass

import numpy as np

from .classes import class_name


@dataclass
class Result:
    labels: list[int]  # 評価したクラスの分類コード(小さい順)
    confusion: np.ndarray  # 混同行列。行=正解、列=予測

    @property
    def total(self) -> int:
        return int(self.confusion.sum())

    @property
    def correct(self) -> int:
        return int(np.trace(self.confusion))

    @property
    def corrections(self) -> int:
        """手動で修正が必要な点数(正解と違う点の数)"""
        return self.total - self.correct

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    def _ratio(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros(len(num)), where=den > 0)

    @property
    def precision(self) -> np.ndarray:
        return self._ratio(np.diag(self.confusion), self.confusion.sum(axis=0))

    @property
    def recall(self) -> np.ndarray:
        return self._ratio(np.diag(self.confusion), self.confusion.sum(axis=1))

    @property
    def iou(self) -> np.ndarray:
        tp = np.diag(self.confusion)
        return self._ratio(tp, self.confusion.sum(axis=0) + self.confusion.sum(axis=1) - tp)

    @property
    def miou(self) -> float:
        return float(self.iou.mean())


def remap(classes: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """分類コードを読み替える(例: 他のソフトの「3,4,5」を、植栽の「5」にそろえる)。"""
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut[classes]


def evaluate(truth: np.ndarray, pred: np.ndarray, ignore=(), mapping: dict[int, int] | None = None) -> Result:
    """truth: 正解の分類、pred: 予測の分類(同じ点の順番・同じ点数)。

    ignore: 評価から外す正解の分類コード(例: 正解データを作っていない未分類の点)
    mapping: 予測の分類コードの読み替え
    """
    if truth.shape != pred.shape:
        raise ValueError(
            f"点数が違います(正解 {truth.size:,} / 予測 {pred.size:,})。"
            "同じ点を同じ順番で保存したファイル同士でないと比較できません。"
        )
    if mapping:
        pred = remap(pred, mapping)
    if len(ignore) > 0:
        keep = ~np.isin(truth, list(ignore))
        truth, pred = truth[keep], pred[keep]
    if truth.size == 0:
        raise ValueError("評価する点がありません(すべて除外されました)")

    labels = sorted(set(np.unique(truth).tolist()) | set(np.unique(pred).tolist()))
    n = len(labels)
    lut = np.full(256, -1, dtype=np.int64)
    lut[labels] = np.arange(n)
    cm = np.bincount(lut[truth] * n + lut[pred], minlength=n * n).reshape(n, n)
    return Result(labels=labels, confusion=cm)


def format_report(r: Result) -> str:
    lines = [
        f"評価した点数: {r.total:,}",
        f"手動で修正が必要な点数: {r.corrections:,}({r.corrections / r.total:.2%})",
        f"全体の正解率: {r.accuracy:.2%}   mIoU: {r.miou:.3f}",
        "",
        "コード  クラス          正解の点数    予測の点数    適合率   再現率    IoU",
    ]
    truth_n, pred_n = r.confusion.sum(axis=1), r.confusion.sum(axis=0)
    for i, code in enumerate(r.labels):
        lines.append(
            f"{code:>4}   {class_name(code):<10} {truth_n[i]:>12,} {pred_n[i]:>12,}"
            f"   {r.precision[i]:>6.1%}   {r.recall[i]:>6.1%}   {r.iou[i]:.3f}"
        )
    return "\n".join(lines)


def save_csv(r: Result, path: str) -> None:
    """クラスごとの結果と混同行列を、1つのCSVに保存する(論文の表づくり用)。"""
    truth_n, pred_n = r.confusion.sum(axis=1), r.confusion.sum(axis=0)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["コード", "クラス", "正解の点数", "予測の点数", "適合率", "再現率", "IoU"])
        for i, code in enumerate(r.labels):
            w.writerow([code, class_name(code), truth_n[i], pred_n[i],
                        round(float(r.precision[i]), 4), round(float(r.recall[i]), 4), round(float(r.iou[i]), 4)])
        w.writerow([])
        w.writerow(["手動修正が必要な点数", r.corrections, "全体の正解率", round(r.accuracy, 4), "mIoU", round(r.miou, 4)])
        w.writerow([])
        w.writerow(["混同行列(行=正解、列=予測)"] + [class_name(c) for c in r.labels])
        for i, code in enumerate(r.labels):
            w.writerow([class_name(code)] + [int(v) for v in r.confusion[i]])