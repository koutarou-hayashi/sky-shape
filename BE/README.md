# ptcloud(点群の自動分類ツール:バックエンド)

卒業研究「三次元点群データの自動分類ツールの作成、既存ツールとの比較」のPython側。
画面(フロント)とは別のリポジトリで、**LASファイルを介して**つなぐ。

## 準備
```
py -3.12 -m venv .venv          # Macは python3 -m venv .venv
.venv\Scripts\activate          # Macは source .venv/bin/activate
pip install -r requirements.txt
```

## コマンド
```
python -m ptcloud info data/sample.las
python -m ptcloud evaluate --truth data/truth.las --pred data/pred.las --ignore 1 --csv outputs/result.csv
python -m ptcloud remove data/in.las outputs/out.las --classes 14 15 --dry-run
```
- `info`: 点数、範囲、点形式、クラスごとの点数。
- `evaluate`: 正解のLASと予測のLASを、点ごとに比べる(同じ点を同じ順番で保存したファイル同士)。
  **手動で修正が必要な点数**、クラスごとの適合率・再現率・IoU、混同行列を出す。
- `remove`: 指定したクラスの点を除いて、別のファイルに保存する。元のファイルは変えない。
  実行前に除去する点数を表示して確認する。`--dry-run` なら点数の表示だけ。

## 構成
- `ptcloud/classes.py` 分類コードと名前の表(**仮の割り当て**。クラスを変えるときはここを直す)
- `ptcloud/lasio.py` LASの読み書き(一定の点数ずつ処理する)
- `ptcloud/evaluate.py` 精度の評価
- `ptcloud/cli.py` コマンドの入口
- `tests/` テスト(`pip install pytest` のあと、`pytest`)

## 今後の予定
1. ルールベースの簡易分類(比較の基準値)
2. 特徴量+従来の機械学習による分類(第1段階)
3. 深層学習による分類(第2段階、余裕があれば)