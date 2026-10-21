"""コマンドの入口。 python -m ptcloud <コマンド> で使う。

  info      ファイルの概要を表示する
  evaluate  正解のLASと予測のLASを比べて、精度を出す
  classify  ルールベースの簡易分類をして、分類つきのLASを保存する
  remove    指定したクラスの点を取り除いて、別のファイルに保存する
"""
import argparse
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
        r = ev.evaluate(truth, pred, ignore=args.ignore, mapping=_parse_map(args.map_pred))
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(ev.format_report(r))
    if args.csv:
        ev.save_csv(r, args.csv)
        print(f"\nCSVに保存しました: {args.csv}")
    return 0


def cmd_classify(args) -> int:
    from dataclasses import fields

    from . import rules  # scipyが必要なので、このコマンドを使うときだけ読み込む

    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    # --voxel など、Paramsの項目を指定した分だけ上書きする
    params = rules.Params(**{f.name: getattr(args, f.name) for f in fields(rules.Params) if getattr(args, f.name) is not None})
    print("座標を読み込み中...")
    xyz = lasio.read_xyz(src)
    print(f"{len(xyz):,}点を分類中...")
    classes = rules.classify(xyz, params)
    del xyz
    for code, n in zip(*np.unique(classes, return_counts=True)):
        print(f"  {int(code):>3} {class_name(int(code)):<10} {int(n):>12,}({n / len(classes):.1%})")
    lasio.write_classification(src, dst, classes)
    print(f"保存しました: {dst}")
    return 0


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
    a.add_argument("--csv", help="結果をCSVに保存する")
    a.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("classify", help="ルールベースの簡易分類をして保存する")
    a.add_argument("src", help="元のLAS(変更しない)")
    a.add_argument("dst", help="保存先のLAS(分類コードが書き込まれる)")
    a.add_argument("--force", action="store_true", help="保存先がすでにあっても上書きする")
    for name, typ, default in (("voxel", float, 0.5), ("building_voxel", float, 1.0), ("ground_window", float, 17.0),
                               ("pole_min_height", float, 5.0), ("min_building_voxels", int, 10)):
        a.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None, help=f"しきい値(既定 {default})。rules.py の Params を参照")
    for name in ("ground_cell", "ground_tol", "ground_max_step", "min_points", "min_building_height",
                 "wire_min_height", "min_wire_voxels", "vegetation_min_height", "linear_min", "flat_max",
                 "plane_mid_min", "pole_max_width", "pole_min_fill"):
        a.set_defaults(**{name: None})
    a.set_defaults(func=cmd_classify)

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