# sky-shape
【卒業研究】自動分類ツール

## ディレクトリ構成
```
/
├─ BE           // バックエンド
├─ FE           // フロントエンド
├─ outputs      // 出力データ置き場
└─ sample_data  // テスト用データ置き場
```

## 環境構築
- GitHubからリポジトリを取得してくる
```sh
git clone https://github.com/koutarou-hayashi/sky-shape.git
```
- ルートディレクトリに以下のフォルダを作成
`outputs`、`sample_data`
- サンプルデータの作成
[Googgle Colab](https://colab.research.google.com/drive/1DOMgNSBOwJfoyTQVbQHtazPAHlgNVPUX?usp=sharing)のスクリプトを実行して作成し、`/sample_data`の配下に配置
- ドキュメントを参考にフロントエンド、バックエンドの環境構築を行う

## ドキュメント
- [フロントエンド](FE/README.md)
- [バックエンド](BE/README.md)