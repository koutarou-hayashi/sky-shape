/**
 * 分類結果のデータの形(フロントとバックエンドの約束)
 *
 * - positions: 点ごとの座標 [x0, y0, z0, x1, y1, z1, ...]
 * - classes:   点ごとのクラス番号。positionsと同じ順番で、点の数だけ並ぶ
 * - classDefs: クラス番号と、名前・表示色の対応表
 * - colors:    (任意)点ごとのRGB(0-255)。LASにRGBがあるときだけ入る
 * - meta:      (任意)読み込んだファイルの情報
 *
 * バックエンドが分類した結果も、この形に変換して渡す。
 * クラスを追加・変更するときは classDefs だけ直せばよい。
 */
export type ClassDef = {
  id: number;
  name: string;
  color: string; // "#rrggbb"
};

export type PointCloudMeta = {
  fileName: string;
  totalPoints: number; // ファイル全体の点数
  stride: number; // 何点おきに読み込んだか(1なら全点)
  classCounts: Record<number, number>; // クラスごとの、ファイル全体の点数
  origin: [number, number, number]; // 座標から引いた原点(元の座標 = 画面の座標 + origin)
};

export type PointCloudData = {
  positions: Float32Array;
  classes: Uint8Array;
  classDefs: ClassDef[];
  colors?: Uint8Array;
  meta?: PointCloudMeta;
};