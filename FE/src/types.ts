/**
 * 分類結果のデータの形(フロントとバックエンドの約束)
 *
 * - positions: 点ごとの座標 [x0, y0, z0, x1, y1, z1, ...]
 * - classes:   点ごとのクラス番号。positionsと同じ順番で、点の数だけ並ぶ
 * - classDefs: クラス番号と、名前・表示色の対応表
 *
 * バックエンドが分類した結果も、この形に変換して渡す。
 * クラスを追加・変更するときは classDefs だけ直せばよい。
 */
export type ClassDef = {
  id: number;
  name: string;
  color: string; // "#rrggbb"
};

export type PointCloudData = {
  positions: Float32Array;
  classes: Uint8Array;
  classDefs: ClassDef[];
};