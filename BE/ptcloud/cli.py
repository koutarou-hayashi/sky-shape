"""コマンドの入口。 python -m ptcloud <コマンド> で使う。

  info      ファイルの概要を表示する
  evaluate  正解のLASと予測のLASを比べて、精度を出す
  classify      ルールベースの簡易分類をして、分類つきのLASを保存する
  classify-dir  フォルダ内のLASを、まとめて簡易分類する
  remove    指定したクラスの点を取り除いて、別のファイルに保存する
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

from . import evaluate as ev
from . import lasio
from .classes import class_name


def _parse_map(items: list[str]) -> dict[int, int]:
    out = {}
    for s in items:
        a, _, b = s.partition("=")
        out[int(a)] = int(b)
    return out


def cmd_info(args) -> int:
    i = lasio.info(args.file)
    print(f"点数: {i['point_count']:,}   LAS {i['version']}   点形式: {i['point_format']}   RGB: {'あり' if i['has_rgb'] else 'なし'}")
    print(f"範囲 最小: {i['mins']}")
    print(f"範囲 最大: {i['maxs']}")
    print("クラスごとの点数:")
    for code, n in i["class_counts"].items():
        print(f"  {code:>3} {class_name(code):<10} {n:>12,}({n / i['point_count']:.1%})")
    return 0


def cmd_evaluate(args) -> int:
    truth = lasio.read_classification(args.truth)
    pred = lasio.read_classification(args.pred)
    try:
        r = ev.evaluate(truth, pred, ignore=args.ignore, mapping=_parse_map(args.map_pred),
                        truth_mapping=_parse_map(args.map_truth))
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(ev.format_report(r))
    if args.csv:
        ev.save_csv(r, args.csv)
        print(f"\nCSVに保存しました: {args.csv}")
    return 0


# classify / classify-dir で共通のしきい値の指定(rules.py の Params と同じ名前)
_PARAM_ARGS = (
    ("voxel", float), ("building_voxel", float), ("ground_cell", float), ("ground_window", float),
    ("ground_tol", float), ("ground_max_step", float), ("min_points", int), ("min_building_height", float),
    ("min_building_voxels", int), ("wire_min_height", float), ("min_wire_voxels", int), ("pole_min_height", float),
    ("vegetation_min_height", float), ("linear_min", float), ("flat_max", float), ("plane_mid_min", float),
    ("pole_max_width", float), ("pole_min_fill", float),
)


def _add_param_args(parser) -> None:
    for name, typ in _PARAM_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None,
                            help="しきい値。既定値は rules.py の Params を参照")


def _params(args):
    from . import rules

    return rules.Params(**{n: getattr(args, n) for n, _ in _PARAM_ARGS if getattr(args, n) is not None})


def _classify_file(src: Path, dst: Path, params) -> dict:
    """1ファイルを分類して保存し、分類コードごとの点数を返す。"""
    from . import rules  # scipyが必要なので、分類するときだけ読み込む

    xyz = lasio.read_xyz(src)
    classes = rules.classify(xyz, params)
    del xyz
    lasio.write_classification(src, dst, classes)
    return {int(c): int(n) for c, n in zip(*np.unique(classes, return_counts=True))}


def _summary(counts: dict) -> str:
    total = sum(counts.values())
    return "  ".join(f"{class_name(c)} {n / total:.1%}" for c, n in sorted(counts.items()))


def cmd_classify(args) -> int:
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    print("分類中...")
    counts = _classify_file(src, dst, _params(args))
    for code, n in counts.items():
        print(f"  {code:>3} {class_name(code):<10} {n:>12,}({n / sum(counts.values()):.1%})")
    print(f"保存しました: {dst}")
    return 0


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def cmd_classify_dir(args) -> int:
    import time

    src_dir, dst_dir = Path(args.src_dir), Path(args.dst_dir)
    if not src_dir.is_dir():
        print(f"エラー: フォルダが見つかりません: {src_dir}", file=sys.stderr)
        return 1
    if src_dir.resolve() == dst_dir.resolve():
        print("エラー: 保存先は、元のフォルダとは別にしてください。", file=sys.stderr)
        return 1
    files = sorted(src_dir.glob(args.pattern), key=_natural_key)
    if not files:
        print(f"エラー: {src_dir} に {args.pattern} に合うファイルがありません。", file=sys.stderr)
        return 1
    dst_dir.mkdir(parents=True, exist_ok=True)
    params = _params(args)
    total: dict = {}
    failed, skipped = [], 0
    t0 = time.time()
    for i, src in enumerate(files, 1):
        dst = dst_dir / src.name
        if dst.exists() and not args.force:
            skipped += 1
            print(f"[{i}/{len(files)}] {src.name}: すでにあるのでスキップ(上書きは --force)")
            continue
        t1 = time.time()
        try:
            counts = _classify_file(src, dst, params)
        except Exception as e:  # 1ファイルの失敗で、全体を止めない
            failed.append(src.name)
            print(f"[{i}/{len(files)}] {src.name}: 失敗 ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        for c, n in counts.items():
            total[c] = total.get(c, 0) + n
        print(f"[{i}/{len(files)}] {src.name}: {sum(counts.values()):,}点 {time.time() - t1:.1f}秒  {_summary(counts)}")
    print(f"\n完了: {len(files) - len(failed) - skipped}件を分類、{skipped}件をスキップ、{len(failed)}件が失敗({time.time() - t0:.0f}秒)")
    if total:
        print("全体(今回分類した分):  " + _summary(total))
    if failed:
        print("失敗したファイル: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


def cmd_remove(args) -> int:
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    i = lasio.info(src)
    targets = {c: n for c, n in i["class_counts"].items() if c in set(args.classes)}
    missing = sorted(set(args.classes) - set(targets))
    if missing:
        print(f"注意: 次の分類コードの点は、ファイルにありません: {missing}")
    total = sum(targets.values())
    print(f"全体: {i['point_count']:,}点")
    for c, n in targets.items():
        print(f"  除去: {c:>3} {class_name(c):<10} {n:>12,}点")
    print(f"除去する点数の合計: {total:,}({total / i['point_count']:.1%})  → 残り {i['point_count'] - total:,}点")
    if args.dry_run:
        print("(--dry-run のため、ファイルは作りません)")
        return 0
    if not args.yes and input("この内容で保存しますか?[y/N] ").strip().lower() != "y":
        print("中止しました。")
        return 0
    res = lasio.remove_classes(src, dst, args.classes)
    print(f"保存しました: {dst}(残した点 {res['kept']:,}点)")
    return 0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ptcloud", description="点群の分類・評価・除去")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("info", help="ファイルの概要を表示する")
    a.add_argument("file")
    a.set_defaults(func=cmd_info)

    a = sub.add_parser("evaluate", help="正解と予測のLASを比べて精度を出す")
    a.add_argument("--truth", required=True, help="正解の分類が入ったLAS")
    a.add_argument("--pred", required=True, help="予測(ツールの出力)のLAS")
    a.add_argument("--ignore", type=int, nargs="*", default=[], help="評価から外す正解の分類コード")
    a.add_argument("--map-pred", nargs="*", default=[], metavar="元=先", help="予測の分類コードの読み替え(例: 3=5 4=5)")
    a.add_argument("--map-truth", nargs="*", default=[], metavar="元=先", help="正解の分類コードの読み替え(例: 3=5 4=5)")
    a.add_argument("--csv", help="結果をCSVに保存する")
    a.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("classify", help="ルールベースの簡易分類をして保存する(1ファイル)")
    a.add_argument("src", help="元のLAS(変更しない)")
    a.add_argument("dst", help="保存先のLAS(分類コードが書き込まれる)")
    a.add_argument("--force", action="store_true", help="保存先がすでにあっても上書きする")
    _add_param_args(a)
    a.set_defaults(func=cmd_classify)

    a = sub.add_parser("classify-dir", help="フォルダ内のLASを、まとめて簡易分類して保存する")
    a.add_argument("src_dir", help="元のLASが入ったフォルダ(変更しない)")
    a.add_argument("dst_dir", help="保存先のフォルダ(同じファイル名で保存する)")
    a.add_argument("--pattern", default="*.las", help="対象のファイル名(既定: *.las)")
    a.add_argument("--force", action="store_true", help="保存先にすでにあるファイルも、上書きする")
    _add_param_args(a)
    a.set_defaults(func=cmd_classify_dir)

    a = sub.add_parser("remove", help="指定したクラスの点を取り除いて保存する")
    a.add_argument("src", help="元のLAS(変更しない)")
    a.add_argument("dst", help="保存先のLAS")
    a.add_argument("--classes", type=int, nargs="+", required=True, help="除去する分類コード")
    a.add_argument("--dry-run", action="store_true", help="点数だけ表示して、保存しない")
    a.add_argument("--yes", action="store_true", help="確認を省く")
    a.add_argument("--force", action="store_true", help="保存先がすでにあっても上書きする")
    a.set_defaults(func=cmd_remove)

    args = p.parse_args(argv)
    sys.exit(args.func(args))