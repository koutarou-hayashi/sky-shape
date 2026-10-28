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
python -m ptcloud train --truth data/truth.las --model outputs/model.joblib --csv outputs/valid.csv
python -m ptcloud predict-dir --model outputs/model.joblib data/tiles outputs/pred
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
- `ground`: 地面だけを抽出して保存する(地面=2、それ以外=1)。**傾斜(法面)に対応**。WingEarthの地面(2)と見比べるのに使う。
- `water` / `water-dir`: 水面だけを抽出して保存する(水面=9、それ以外=1)。色・平らさ・広がり・高さの一定さで判定する。
  WingEarthの水(9)と見比べるのは、`evaluate --truth 元のLAS --pred 出力 --ignore 2` でできる。
- `water-check` / `water-tune` / `water-diff`: 水面の判定を、WingEarthの水と見比べて調整するための道具。
  `water-check`は取りこぼしの原因、`water-tune`はしきい値の組み合わせの比較、`water-diff`は色分けLAS(9=両方、30=こちらだけ、31=取りこぼし)。
- `stats`: 分類ごとの、高さ・色・平らさなどの分布を表示する(ルールのしきい値を決める手がかり)。
- `train`: 正解の分類が入ったLAS(WingEarthなどで手動分類したもの)で学習して、モデル(.joblib)を保存する。
  `--valid` で検証用のLASを別に指定できる。指定しなければ、学習用を25mの帯に区切り、間をあけて約3割を検証にする。
  検証の結果(手動で修正が必要な点数、クラスごとの精度)を表示する。**検証に正解がないクラスは、評価できない**。
  点数の少ないクラス(車両など)が当たらないときは `--balanced`。
- `predict` / `predict-dir`: 学習したモデルで、LASを分類して保存する(フォルダまとめては predict-dir)。
  特徴量は、座標や標高そのものを使わず、地面からの高さ・色・まわりの点の並び方だけなので、別のタイルや別の現場にも当てはめられる。
  メモリのため、1ファイルは数十万点(今回のタイル)ほどにしておく。
- `classify-site` / `classify-site-dir`: **池・森・駐車場・道路の現場向け**のルールで分類する(水面9、地面2、植栽 低3・中4・高5、建物6、車両20、その他1)。路面と地面は、既定では分けない(`split_road=True` で路面11を分ける)。
  水面 → 地面からの高さ → 建物 → 車両 → 植栽 → 路面・地面、の順に決める。しきい値は `ptcloud/site.py` の `SiteParams`。
  **初期値は、水面以外はデータで確かめていない。画面で結果を見て調整する。** 電線・電柱はまだ入っていない。
  メモリのため、元の2,274万点を一度にかけるより、タイル(`classify-site-dir`)を勧める。
- `crop`: 水平の範囲(ファイルの座標 X, Y)の点だけを切り出す。`--center X Y --radius 30` など。画面で見る、WingEarthで手動分類する、ための小さなLASを作る。
- `evaluate`: 正解のLASと予測のLASを、点ごとに比べる(同じ点を同じ順番で保存したファイル同士)。
  **手動で修正が必要な点数**、クラスごとの適合率・再現率・IoU、混同行列を出す。
- `remove`: 指定したクラスの点を除いて、別のファイルに保存する。元のファイルは変えない。
  実行前に除去する点数を表示して確認する。`--dry-run` なら点数の表示だけ。

## 構成
- `ptcloud/classes.py` 分類コードと名前の表(**仮の割り当て**。クラスを変えるときはここを直す)
- `ptcloud/lasio.py` LASの読み書き(一定の点数ずつ処理する)
- `ptcloud/ground.py` 地面の抽出(傾斜対応)。ルールベースへの組み込みは、これから
- `ptcloud/site.py` 池・森・駐車場・道路向けの分類(水面→地面→建物→車両→植栽→路面)
- `ptcloud/water.py` 水面の抽出(色・平らさ・広がり)
- `ptcloud/stats.py` 分類ごとの特徴の分布
- `ptcloud/rules.py` ルールベースの簡易分類(比較の基準値)
- `ptcloud/features.py` 機械学習用の特徴量(高さ、色、まわりの点の並び方)
- `ptcloud/ml.py` 学習と予測(HistGradientBoosting)
- `ptcloud/evaluate.py` 精度の評価
- `ptcloud/cli.py` コマンドの入口
- `tests/` テスト(`pip install pytest` のあと、`pytest`)

## 簡易分類(`rules.py`)について
- 人工の街並みで試した結果は、地面・建物・植栽が9割台、電線・電柱が5〜9割でした。**人工データでの確認なので、実データでは、しきい値の調整が必要です**。
  調整するときは、正解つきの小さなデータで `evaluate` を回しながら、`Params` の値を変えます。
- 弱いところ: 急な坂道、地面が見えない大きな屋根、木のそばの電柱、柱と標識の支柱・街灯の区別。

## 今後の予定
1. 実データ(正解つきの小さな範囲)での、簡易分類の調整と評価
2. 特徴量+従来の機械学習による分類(第1段階)← `train` / `predict` まで作成済み。正解データでの学習・評価はこれから
3. 深層学習による分類(第2段階、余裕があれば)