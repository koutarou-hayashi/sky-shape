import { CLASS_TABLE } from "./classTable";
import type { ClassDef, PointCloudData } from "./types";

// ASPRS標準の分類コード(LAS 1.4)。classTable.ts(プロジェクトの分類名・色)にないコードの名前に使う
const CLASS_NAMES: Record<number, string> = {
  0: "未分類(分類なし)", 1: "未分類", 2: "地面", 3: "低い植生", 4: "中位の植生", 5: "高い植生",
  6: "建物", 7: "ノイズ(低)", 8: "予約", 9: "水", 10: "鉄道", 11: "道路面", 12: "予約",
  13: "電線(ガード)", 14: "電線(導体)", 15: "送電塔", 16: "電線接続部", 17: "橋", 18: "ノイズ(高)",
};
const PALETTE = [
  "#8c9298", "#9aa5ad", "#a58a6a", "#7cbf6a", "#58a65a", "#3f8a48", "#c9a36b", "#555b61",
  "#aaaaaa", "#3d8fd1", "#7a6a5a", "#6e7479", "#bbbbbb", "#c04a7a", "#e0568a", "#8a6bbf",
  "#d98a3d", "#b08c5a", "#333333",
];

// 点形式ごとの最小のレコード長と、RGBの位置(ない形式は未定義)
const MIN_RECORD = [20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67];
const RGB_OFFSET: Record<number, number> = { 2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30 };

export type ReadOptions = {
  maxPoints?: number; // 画面に読み込む最大点数。超える分は等間隔で間引く
  onProgress?: (done: number, total: number) => void;
};

export async function readLas(file: File, opts: ReadOptions = {}): Promise<PointCloudData> {
  const maxPoints = opts.maxPoints ?? 3_000_000;
  if (file.size < 227) throw new Error("ファイルが小さすぎます。LASファイルではない可能性があります。");

  const h = new DataView(await file.slice(0, Math.min(375, file.size)).arrayBuffer());
  if (String.fromCharCode(h.getUint8(0), h.getUint8(1), h.getUint8(2), h.getUint8(3)) !== "LASF") {
    throw new Error("LASファイルではありません(先頭が「LASF」ではありません)。");
  }
  const minor = h.getUint8(25);
  const dataOffset = h.getUint32(96, true);
  const rawFormat = h.getUint8(104);
  if (rawFormat & 0xc0) throw new Error("圧縮されたLAZ形式には未対応です。LASで保存し直してください。");
  const format = rawFormat & 0x3f;
  if (format > 10) throw new Error(`未対応の点形式です(形式${format})。`);
  const recLen = h.getUint16(105, true);
  if (recLen < MIN_RECORD[format]) throw new Error("点のレコード長が不正です。");

  let total = h.getUint32(107, true);
  if (total === 0 && minor >= 4 && h.byteLength >= 255) total = Number(h.getBigUint64(247, true));
  if (total === 0) throw new Error("点が含まれていません。");
  if (dataOffset + total * recLen > file.size) throw new Error("ファイルが途中で切れています。");

  const sx = h.getFloat64(131, true), sy = h.getFloat64(139, true), sz = h.getFloat64(147, true);
  const ox = h.getFloat64(155, true), oy = h.getFloat64(163, true), oz = h.getFloat64(171, true);
  // 大きな座標値をそのままFloat32にすると精度が落ちるため、最小値を原点にして小さな値にする
  const originX = h.getFloat64(187, true), originY = h.getFloat64(203, true), originZ = h.getFloat64(219, true);

  const classOffset = format <= 5 ? 15 : 16;
  const classMask = format <= 5 ? 0x1f : 0xff;
  const rgbOffset = RGB_OFFSET[format];

  const stride = Math.max(1, Math.ceil(total / maxPoints));
  const capacity = Math.ceil(total / stride);
  const positions = new Float32Array(capacity * 3);
  const classes = new Uint8Array(capacity);
  const rgb = rgbOffset !== undefined ? new Uint16Array(capacity * 3) : null;

  const tally = new Uint32Array(256); // クラスごとの全点数(間引き前)
  const chunkPoints = 250_000;
  let n = 0;
  for (let start = 0; start < total; start += chunkPoints) {
    const count = Math.min(chunkPoints, total - start);
    const from = dataOffset + start * recLen;
    const buf = await file.slice(from, from + count * recLen).arrayBuffer();
    if (buf.byteLength < count * recLen) throw new Error("ファイルの読み込み中にデータが足りなくなりました。");
    const v = new DataView(buf);
    for (let i = 0; i < count; i++) tally[v.getUint8(i * recLen + classOffset) & classMask]++;
    for (let i = (stride - (start % stride)) % stride; i < count; i += stride) {
      const p = i * recLen;
      positions[n * 3] = v.getInt32(p, true) * sx + ox - originX;
      positions[n * 3 + 1] = v.getInt32(p + 4, true) * sy + oy - originY;
      positions[n * 3 + 2] = v.getInt32(p + 8, true) * sz + oz - originZ;
      classes[n] = v.getUint8(p + classOffset) & classMask;
      if (rgb) {
        rgb[n * 3] = v.getUint16(p + rgbOffset, true);
        rgb[n * 3 + 1] = v.getUint16(p + rgbOffset + 2, true);
        rgb[n * 3 + 2] = v.getUint16(p + rgbOffset + 4, true);
      }
      n++;
    }
    opts.onProgress?.(start + count, total);
  }

  // RGBが16bitの範囲か、8bitの値のままかを判定して、0-255にそろえる
  let colors: Uint8Array | undefined;
  if (rgb) {
    let max = 0;
    for (let i = 0; i < n * 3; i++) if (rgb[i] > max) max = rgb[i];
    const shift = max > 255 ? 8 : 0;
    colors = new Uint8Array(n * 3);
    for (let i = 0; i < n * 3; i++) colors[i] = rgb[i] >> shift;
  }

  const classCounts: Record<number, number> = {};
  const ids: number[] = [];
  for (let id = 0; id < 256; id++) {
    if (tally[id] > 0) {
      ids.push(id);
      classCounts[id] = tally[id];
    }
  }
  const classDefs: ClassDef[] = ids.map((id) => ({
    id,
    name: CLASS_TABLE[id]?.name ?? CLASS_NAMES[id] ?? `クラス${id}`,
    color: CLASS_TABLE[id]?.color ?? PALETTE[id % PALETTE.length],
  }));

  return {
    positions: positions.slice(0, n * 3),
    classes: classes.slice(0, n),
    classDefs,
    colors,
    meta: { fileName: file.name, totalPoints: total, stride, classCounts, origin: [originX, originY, originZ] },
  };
}