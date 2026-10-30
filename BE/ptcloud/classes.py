"""分類コード・名前・色・定義の表。**分類名は、ここだけで決める**(単一の情報源)。

- バックエンド(classify-site、評価、レポート)は、ここから名前を読む。
- 画面(フロント)は、`python -m ptcloud classes --ts ../frontend/src/classTable.ts` で、同じ内容のファイルを作って読む。
- 定義書は、`python -m ptcloud classes --md ../outputs/classes.md` で作る(論文・要件定義書に貼れる)。

LASの classification(分類)は、点形式0〜5(LAS 1.2など)では 0〜31 まで。独自のコードは 20番台を使っている。
クラスを変えるときは、この表だけを直す。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassInfo:
    code: int
    name: str
    color: str  # 画面の色(#rrggbb)
    group: str  # 「現場」= 今回の現場データで分類する / 「現場(未対応)」= 対象だが、まだ判定を作っていない / 「将来」= 街並みのデータ用
    definition: str  # 何を、このクラスにするか
    rule: str  # 今のルールベースの判定(簡単に)


CLASS_TABLE: list[ClassInfo] = [
    ClassInfo(1, "その他(未分類)", "#8c9298", "現場",
              "どのクラスにも当てはまらないもの。柵・擁壁・岩・人・小さな構造物など。", "ほかのどれにも、当てはまらなかった点"),
    ClassInfo(2, "地面", "#a58a6a", "現場",
              "地表面。土・砂利・裸地・草や芝に覆われた地面・斜面(法面)・舗装(路面)を含む。路面は、いまは分けない。",
              "傾斜対応の地面の平面から 0.3m 以内の点(林の中を除く)"),
    ClassInfo(3, "植栽(低)", "#9ad07a", "現場",
              "地面から 0.5m 未満の低木・草。地面にぴったりの草(芝)は、地面にする。", "緑の点で、地面からの高さ 0.5m 未満(地面の高さの草を除く)"),
    ClassInfo(4, "植栽(中)", "#5fae5f", "現場", "地面から 0.5〜2m の低木・若い木。", "緑の点で、地面からの高さ 0.5〜2m"),
    ClassInfo(5, "植栽(高)", "#3f8a48", "現場",
              "地面から 2m 以上の木。密集して広がる林にならない、数本の木・並木・庭木。", "緑の点、または高く散らばった点で、地面からの高さ 2m 以上(林を除く)"),
    ClassInfo(6, "建物", "#c9a36b", "現場", "建物の壁・屋根。", "緑でない平らな面が、大きくつながり、一番高いところが 2.5m 以上"),
    ClassInfo(9, "水面", "#3d8fd1", "現場", "池・水路などの水面。", "色が一様で、平らで水平な面が、広くつながり、水面の高さに近い点"),
    ClassInfo(11, "路面", "#6e7479", "現場",
              "舗装された道路・駐車場の路面。(いまは、地面にまとめている。路面のデータで、分け方を決める)", "split_road を有効にしたときだけ: 灰色で平らな地面の点"),
    ClassInfo(14, "電線", "#e0568a", "現場(未対応)", "架空の電線。", "(rules.py に、街並み向けの判定がある。この現場向けは、未実装)"),
    ClassInfo(15, "電柱", "#8a6bbf", "現場(未対応)", "電柱・送電塔の柱。", "(rules.py に、街並み向けの判定がある。この現場向けは、未実装)"),
    ClassInfo(20, "車両", "#e07a3d", "現場", "自動車・バイクなどの車。", "緑でなく、地面から少し浮いた、車の大きさ(長さ・幅・高さ)のかたまり"),
    ClassInfo(24, "林", "#2f6b3a", "現場",
              "背の高い木が密集して、広く連なる山林・林。高さは分けない。", "背の高い植生が、窓の中の6割以上を覆い、2,000m² 以上つながる範囲の植物"),
    ClassInfo(21, "標識", "#e8b923", "将来", "案内標識・看板など。", "(未実装)"),
    ClassInfo(22, "人物", "#c04a7a", "将来", "人。", "(未実装)"),
    ClassInfo(23, "道路標識", "#d1a02a", "将来", "道路標識(規制・警戒・案内)。", "(未実装)"),
]

# 以前からの辞書(ほかのファイルが使う)。CLASS_TABLE から作る。
PROJECT_CLASSES: dict[int, str] = {c.code: c.name for c in CLASS_TABLE if c.group.startswith("現場")}
FUTURE_CLASSES: dict[int, str] = {c.code: c.name for c in CLASS_TABLE if c.group == "将来"}

# ASPRS標準の分類コード(他のソフトが書き出したLASを読むとき用)
ASPRS_NAMES: dict[int, str] = {
    0: "未分類(分類なし)", 1: "未分類", 2: "地面", 3: "低い植生", 4: "中位の植生", 5: "高い植生",
    6: "建物", 7: "ノイズ(低)", 9: "水", 10: "鉄道", 11: "道路面",
    13: "電線(ガード)", 14: "電線(導体)", 15: "送電塔", 16: "電線接続部", 17: "橋", 18: "ノイズ(高)",
}


def class_name(code: int) -> str:
    return PROJECT_CLASSES.get(code) or FUTURE_CLASSES.get(code) or ASPRS_NAMES.get(code) or f"クラス{code}"


def to_markdown() -> str:
    """分類の定義書(Markdownの表)。"""
    lines = ["# 分類クラスの定義", "",
             "| コード | 名前 | 区分 | 定義 | 今のルールベースの判定 |", "|---|---|---|---|---|"]
    for c in CLASS_TABLE:
        lines.append(f"| {c.code} | {c.name} | {c.group} | {c.definition} | {c.rule} |")
    return "\n".join(lines) + "\n"


def to_csv() -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["コード", "名前", "色", "区分", "定義", "今のルールベースの判定"])
    for c in CLASS_TABLE:
        w.writerow([c.code, c.name, c.color, c.group, c.definition, c.rule])
    return buf.getvalue()


def to_typescript() -> str:
    """画面(フロント)で読む表。classes.py から自動で作る。手で直さない。"""
    rows = ",\n".join(f'  {c.code}: {{ code: {c.code}, name: "{c.name}", color: "{c.color}" }}' for c in CLASS_TABLE)
    return (
        "// 自動生成: backend で `python -m ptcloud classes --ts <このファイル>` を実行して作る。手で直さない(classes.py を直す)。\n"
        "export type ClassInfo = { code: number; name: string; color: string };\n\n"
        f"export const CLASS_TABLE: Record<number, ClassInfo> = {{\n{rows},\n}};\n"
    )