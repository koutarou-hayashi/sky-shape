"""コマンドの入口。 python -m ptcloud <コマンド> で使う。

  info      ファイルの概要を表示する
  evaluate  正解のLASと予測のLASを比べて、精度を出す
  classify      ルールベースの簡易分類をして、分類つきのLASを保存する
  classify-dir  フォルダ内のLASを、まとめて簡易分類する
  classify-site 池・森・駐車場・道路向けのルールで分類する(classify-site-dir はフォルダまとめて)
  ground        地面だけを抽出する(傾斜に対応)
  crop          水平の範囲を切り出す(画面で見る、WingEarthで手動分類する、ための小さなLASを作る)
  water-check   水面の取りこぼしの原因を調べる(water-tune はしきい値をまとめて試す)
  water         水面だけを抽出する(色・平らさ・広がり。water-dir はフォルダまとめて)
  stats         分類ごとの特徴の分布を見る(ルールのしきい値を決める手がかり)
  train         正解つきのLASで学習して、モデルを保存する
  predict       学習したモデルで分類する(predict-dir はフォルダまとめて)
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


def _run_dir(args, work) -> int:
    """src_dir のLASを番号順に、work(src, dst) -> {分類コード: 点数} で処理して、dst_dir に同じ名前で保存する。"""
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
            counts = work(src, dst)
        except Exception as e:  # 1ファイルの失敗で、全体を止めない
            failed.append(src.name)
            print(f"[{i}/{len(files)}] {src.name}: 失敗 ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        for code, n in counts.items():
            total[code] = total.get(code, 0) + n
        print(f"[{i}/{len(files)}] {src.name}: {sum(counts.values()):,}点 {time.time() - t1:.1f}秒  {_summary(counts)}")
    print(f"\n完了: {len(files) - len(failed) - skipped}件を分類、{skipped}件をスキップ、{len(failed)}件が失敗({time.time() - t0:.0f}秒)")
    if total:
        print("全体(今回分類した分):  " + _summary(total))
    if failed:
        print("失敗したファイル: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


def cmd_classify_dir(args) -> int:
    params = _params(args)
    return _run_dir(args, lambda src, dst: _classify_file(src, dst, params))


def _water_params(args):
    from . import water

    return water.WaterParams(**({"min_area": args.min_area} if getattr(args, "min_area", None) else {}))


def _water_file(src: Path, dst: Path, params=None) -> dict:
    """1ファイルの水面を抽出して保存する(水面=9、それ以外=1)。分類コードごとの点数を返す。"""
    from . import water

    xyz, rgb, _, _ = lasio.read_points(src)
    classes = np.where(water.find_water(xyz, rgb, params), 9, 1).astype(np.uint8)
    lasio.write_classification(src, dst, classes)
    return {int(c): int(n) for c, n in zip(*np.unique(classes, return_counts=True))}


def cmd_water(args) -> int:
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    print(_summary(_water_file(src, dst, _water_params(args))))
    print(f"保存しました: {dst}")
    return 0


def cmd_water_check(args) -> int:
    from . import water

    xyz, rgb, _, cls = lasio.read_points(args.file)
    print(water.diagnose(xyz, rgb, cls == args.truth_class))
    return 0


def cmd_water_tune(args) -> int:
    from . import water

    xyz, rgb, _, cls = lasio.read_points(args.file)
    print(water.tune(xyz, rgb, cls == args.truth_class))
    return 0


def cmd_water_diff(args) -> int:
    """水面の判定を、WingEarthの水(正解)と見比べるためのLASを作る。画面で、取りこぼしや追加の場所を見る。"""
    from . import water

    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    xyz, rgb, _, cls = lasio.read_points(src)
    truth, ours = cls == args.truth_class, water.find_water(xyz, rgb, _water_params(args))
    out = np.ones(len(cls), dtype=np.uint8)  # 1: どちらも水でない
    out[truth & ours] = 9  # 9: 両方とも水
    out[ours & ~truth] = 30  # 30: こちらだけが水(WingEarthにない)
    out[truth & ~ours] = 31  # 31: WingEarthだけが水(こちらの取りこぼし)
    lasio.write_classification(src, dst, out)
    print()
    for code, label in ((9, "両方とも水"), (30, "こちらだけが水"), (31, "WingEarthだけが水(取りこぼし)"), (1, "どちらも水でない")):
        print(f"  {code:>3} {label:<24} {int((out == code).sum()):>12,}点")
    print()
    print(water.describe_diff(xyz, truth, ours, lasio.read_origin(src)))
    print(f"\n保存しました: {dst}")
    return 0


def _site_params(args):
    """classify-site のオプションから、SiteParams を作る。指定しなかった項目は、site.py の既定値のまま。"""
    from . import site

    kw = {}
    for name in ("forest_cover_min", "forest_min_area", "forest_tall_min", "low_max", "mid_max"):
        if getattr(args, name, None) is not None:
            kw[name] = getattr(args, name)
    if getattr(args, "no_forest", False):
        kw["forest"] = False
    if getattr(args, "no_grass_ground", False):
        kw["grass_is_ground"] = False
    if getattr(args, "split_road", False):
        kw["split_road"] = True
    return site.SiteParams(**kw)


def _add_site_args(parser) -> None:
    parser.add_argument("--forest-cover-min", type=float, help="林とみなす、樹冠が占める割合の下限(既定 0.6)")
    parser.add_argument("--forest-min-area", type=float, help="林とみなす、つながりの最小の面積(m²、既定 2000)")
    parser.add_argument("--forest-tall-min", type=float, help="背の高い植生とみなす高さの下限(m、既定 2.0)")
    parser.add_argument("--low-max", type=float, help="植栽(低)の高さの上限(m、既定 0.5)")
    parser.add_argument("--mid-max", type=float, help="植栽(中)の高さの上限(m、既定 2.0)")
    parser.add_argument("--no-forest", action="store_true", help="林(24)を作らず、植栽(低・中・高)だけにする")
    parser.add_argument("--no-grass-ground", action="store_true", help="地面の高さの草を、地面でなく、植栽(低)にする")
    parser.add_argument("--split-road", action="store_true", help="灰色で平らな地面を、路面(11)に分ける")


def _site_file(src: Path, dst: Path, params=None) -> dict:
    """1ファイルを、池・森・駐車場・道路向けのルールで分類して保存する。分類コードごとの点数を返す。"""
    from . import site

    xyz, rgb, _, _ = lasio.read_points(src)
    classes = site.classify_site(xyz, rgb, params)
    lasio.write_classification(src, dst, classes)
    return {int(c): int(n) for c, n in zip(*np.unique(classes, return_counts=True))}


def cmd_classify_site(args) -> int:
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    print("分類中...")
    counts = _site_file(src, dst, _site_params(args))
    for code, n in sorted(counts.items()):
        print(f"  {code:>3} {class_name(code):<10} {n:>12,}({n / sum(counts.values()):.1%})")
    print(f"保存しました: {dst}")
    return 0


def cmd_classify_site_dir(args) -> int:
    params = _site_params(args)
    return _run_dir(args, lambda src, dst: _site_file(src, dst, params))


def cmd_crop(args) -> int:
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    if args.center:
        cx, cy = args.center
        r = args.radius
        xmin, xmax, ymin, ymax = cx - r, cx + r, cy - r, cy + r
    elif None not in (args.xmin, args.xmax, args.ymin, args.ymax):
        xmin, xmax, ymin, ymax = args.xmin, args.xmax, args.ymin, args.ymax
    else:
        print("エラー: --center X Y と --radius、または --xmin --xmax --ymin --ymax を指定してください。", file=sys.stderr)
        return 1
    kept = lasio.crop(src, dst, xmin, xmax, ymin, ymax)
    print(f"範囲 X {xmin:.1f}〜{xmax:.1f}、Y {ymin:.1f}〜{ymax:.1f} の {kept:,}点を保存しました: {dst}")
    return 0


def cmd_water_dir(args) -> int:
    params = _water_params(args)
    return _run_dir(args, lambda src, dst: _water_file(src, dst, params))


def cmd_ground(args) -> int:
    from . import ground

    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    p = ground.GroundParams(**{n: getattr(args, n) for n in ("cell", "window", "coarse", "tol", "point_tol")
                               if getattr(args, n) is not None})
    xyz = lasio.read_xyz(src)
    is_ground, height = ground.find_ground(xyz, p)
    classes = np.where(is_ground, 2, 1).astype(np.uint8)  # 地面=2、それ以外=1
    lasio.write_classification(src, dst, classes)
    print(f"{len(xyz):,}点のうち、地面 {int(is_ground.sum()):,}点({is_ground.mean():.1%})")
    print(f"地面からの高さ(地面以外): 中央値 {np.median(height[~is_ground]):.1f}m、90% {np.percentile(height[~is_ground], 90):.1f}m"
          if (~is_ground).any() else "地面以外の点はありません")
    print(f"保存しました: {dst}")
    return 0


def cmd_stats(args) -> int:
    from . import stats

    print(stats.format_stats(stats.class_stats(args.files, max_points=args.max_points)))
    return 0


def cmd_train(args) -> int:
    from . import ml  # scikit-learnが必要なので、このコマンドを使うときだけ読み込む

    result = ml.train(args.truth, args.model, valid_paths=args.valid, valid_frac=args.valid_frac, block=args.block,
                      balanced=args.balanced, max_iter=args.max_iter)
    if result is None:
        print("検証用のデータがないので、精度は測っていません。")
        return 0
    print(ev.format_report(result))
    if args.csv:
        ev.save_csv(result, args.csv)
        print(f"\nCSVに保存しました: {args.csv}")
    return 0


def cmd_predict(args) -> int:
    from . import ml

    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        print("エラー: 元のファイルは上書きできません。別の名前を指定してください。", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(f"エラー: {dst} がすでにあります。上書きする場合は --force を付けてください。", file=sys.stderr)
        return 1
    counts = ml.predict_file(ml.load_model(args.model), src, dst)
    print(_summary(counts))
    print(f"保存しました: {dst}")
    return 0


def cmd_predict_dir(args) -> int:
    from . import ml

    bundle = ml.load_model(args.model)
    return _run_dir(args, lambda src, dst: ml.predict_file(bundle, src, dst))


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

    a = sub.add_parser("water", help="水面だけを抽出して保存する(水面=9、それ以外=1)")
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--force", action="store_true")
    a.add_argument("--min-area", type=float, default=None, help="水面とみなす、つながりの最小の面積(m²、既定 100)")
    a.set_defaults(func=cmd_water)

    a = sub.add_parser("water-check", help="水面の判定で、本当の水面(WingEarthの水など)を、どの条件で取りこぼすかを調べる")
    a.add_argument("file", help="水面の分類が入ったLAS")
    a.add_argument("--truth-class", type=int, default=9, help="水面の分類コード(既定 9)")
    a.set_defaults(func=cmd_water_check)

    a = sub.add_parser("water-tune", help="水面のしきい値の組み合わせを、まとめて試して、表にする")
    a.add_argument("file", help="水面の分類が入ったLAS")
    a.add_argument("--truth-class", type=int, default=9, help="水面の分類コード(既定 9)")
    a.set_defaults(func=cmd_water_tune)

    a = sub.add_parser("water-diff", help="水面の判定を、WingEarthの水と見比べる色分けLASを作る(9=両方、30=こちらだけ、31=取りこぼし)")
    a.add_argument("src", help="水面の分類(WingEarthなど)が入った元のLAS")
    a.add_argument("dst")
    a.add_argument("--truth-class", type=int, default=9, help="水面の分類コード(既定 9)")
    a.add_argument("--force", action="store_true")
    a.add_argument("--min-area", type=float, default=None, help="水面とみなす、つながりの最小の面積(m²、既定 100)")
    a.set_defaults(func=cmd_water_diff)

    a = sub.add_parser("classify-site", help="池・森・駐車場・道路向けのルールで分類して保存する(水面・地面・路面・植栽・建物・車両)")
    a.add_argument("src", help="元のLAS(RGBつき。変更しない)")
    a.add_argument("dst")
    a.add_argument("--force", action="store_true")
    _add_site_args(a)
    a.set_defaults(func=cmd_classify_site)

    a = sub.add_parser("classify-site-dir", help="フォルダ内のLASを、まとめて classify-site で分類する")
    a.add_argument("src_dir")
    a.add_argument("dst_dir")
    a.add_argument("--pattern", default="*.las")
    a.add_argument("--force", action="store_true")
    _add_site_args(a)
    a.set_defaults(func=cmd_classify_site_dir)

    a = sub.add_parser("crop", help="水平の範囲(ファイルの座標 X, Y)の点だけを切り出して保存する")
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"), help="切り出す範囲の中心(ファイルの座標)")
    a.add_argument("--radius", type=float, default=30.0, help="--center のとき、中心から四方に何mか(既定 30)")
    a.add_argument("--xmin", type=float)
    a.add_argument("--xmax", type=float)
    a.add_argument("--ymin", type=float)
    a.add_argument("--ymax", type=float)
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_crop)

    a = sub.add_parser("water-dir", help="フォルダ内のLASから、水面をまとめて抽出する")
    a.add_argument("src_dir")
    a.add_argument("dst_dir")
    a.add_argument("--pattern", default="*.las")
    a.add_argument("--force", action="store_true")
    a.add_argument("--min-area", type=float, default=None, help="水面とみなす、つながりの最小の面積(m²、既定 100)")
    a.set_defaults(func=cmd_water_dir)

    a = sub.add_parser("ground", help="地面だけを抽出して保存する(傾斜に対応。地面=2、それ以外=1)")
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--force", action="store_true")
    for name, typ in (("cell", float), ("window", int), ("coarse", int), ("tol", float), ("point_tol", float)):
        a.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None, help="しきい値。既定値は ground.py の GroundParams を参照")
    a.set_defaults(func=cmd_ground)

    a = sub.add_parser("stats", help="分類ごとの特徴(高さ・色・平らさなど)の分布を見る")
    a.add_argument("files", nargs="+", help="分類が入ったLAS(WingEarthの結果など)")
    a.add_argument("--max-points", type=int, default=1_000_000, help="1クラスで使う点数の上限(既定 100万)")
    a.set_defaults(func=cmd_stats)

    a = sub.add_parser("train", help="正解つきのLASで学習して、モデルを保存する")
    a.add_argument("--truth", nargs="+", required=True, help="正解の分類が入ったLAS(学習用)")
    a.add_argument("--model", required=True, help="モデルの保存先(.joblib)")
    a.add_argument("--valid", nargs="*", default=[], help="検証用のLAS(指定しなければ、学習用を帯に区切って一部を検証に使う)")
    a.add_argument("--valid-frac", type=float, default=0.3, help="--valid がないとき、何割を検証にするか(既定 0.3)")
    a.add_argument("--block", type=float, default=25.0, help="--valid-frac で分けるときの、帯の幅(m、既定 25)")
    a.add_argument("--balanced", action="store_true", help="点数の少ないクラスを重く扱う")
    a.add_argument("--max-iter", type=int, default=200, help="学習の繰り返しの上限(既定 200)")
    a.add_argument("--csv", help="検証の結果をCSVに保存する")
    a.set_defaults(func=cmd_train)

    a = sub.add_parser("predict", help="学習したモデルで分類して保存する(1ファイル)")
    a.add_argument("--model", required=True)
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_predict)

    a = sub.add_parser("predict-dir", help="学習したモデルで、フォルダ内のLASをまとめて分類する")
    a.add_argument("--model", required=True)
    a.add_argument("src_dir")
    a.add_argument("dst_dir")
    a.add_argument("--pattern", default="*.las")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_predict_dir)

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