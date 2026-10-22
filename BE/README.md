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
python -m ptcloud classify data/in.las outputs/classified.las
python -m ptcloud classify-dir data/tiles outputs/tiles
python -m ptcloud evaluate --truth data/truth.las --pred data/pred.las --ignore 1 --csv outputs/result.csv
python -m ptcloud remove data/in.las outputs/out.las --classes 14 15 --dry-run
```
- `info`: 点数、範囲、点形式、クラスごとの点数。
- `classify`: ルールベースの簡易分類(地面・植栽・建物・電線・電柱の5クラス)をして、分類コードを書き込んだLASを保存する。
  それ以外の点は未分類(1)。しきい値は `--voxel 0.4` のように指定できる(一覧は `ptcloud/rules.py` の `Params`)。
  座標を全部メモリに読むので、2,274万点で数GBを使う。
- `classify-dir`: フォルダ内のLASを、ファイル名の番号順にまとめて分類して、別のフォルダに同じ名前で保存する。
  すでにあるファイルはスキップ(`--force` で上書き)、1ファイルが失敗しても止まらず、最後に一覧を表示する。
  タイルごとに処理するので、タイルの境目では、まわりの点が足りず、精度が少し落ちることがある。
- `evaluate`: 正解のLASと予測のLASを、点ごとに比べる(同じ点を同じ順番で保存したファイル同士)。
  **手動で修正が必要な点数**、クラスごとの適合率・再現率・IoU、混同行列を出す。
- `remove`: 指定したクラスの点を除いて、別のファイルに保存する。元のファイルは変えない。
  実行前に除去する点数を表示して確認する。`--dry-run` なら点数の表示だけ。

## 構成
- `ptcloud/classes.py` 分類コードと名前の表(**仮の割り当て**。クラスを変えるときはここを直す)
- `ptcloud/lasio.py` LASの読み書き(一定の点数ずつ処理する)
- `ptcloud/rules.py` ルールベースの簡易分類(比較の基準値)
- `ptcloud/evaluate.py` 精度の評価
- `ptcloud/cli.py` コマンドの入口
- `tests/` テスト(`pip install pytest` のあと、`pytest`)

## 簡易分類(`rules.py`)について
- 人工の街並みで試した結果は、地面・建物・植栽が9割台、電線・電柱が5〜9割でした。**人工データでの確認なので、実データでは、しきい値の調整が必要です**。
  調整するときは、正解つきの小さなデータで `evaluate` を回しながら、`Params` の値を変えます。
- 弱いところ: 急な坂道、地面が見えない大きな屋根、木のそばの電柱、柱と標識の支柱・街灯の区別。

## 今後の予定
1. 実データ(正解つきの小さな範囲)での、簡易分類の調整と評価
2. 特徴量+従来の機械学習による分類(第1段階)
3. 深層学習による分類(第2段階、余裕があれば)