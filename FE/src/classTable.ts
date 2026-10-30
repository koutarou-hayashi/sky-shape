// 自動生成: backend で `python -m ptcloud classes --ts <このファイル>` を実行して作る。手で直さない(classes.py を直す)。
export type ClassInfo = { code: number; name: string; color: string };

export const CLASS_TABLE: Record<number, ClassInfo> = {
  1: { code: 1, name: "その他(未分類)", color: "#8c9298" },
  2: { code: 2, name: "地面", color: "#a58a6a" },
  3: { code: 3, name: "植栽(低)", color: "#9ad07a" },
  4: { code: 4, name: "植栽(中)", color: "#5fae5f" },
  5: { code: 5, name: "植栽(高)", color: "#3f8a48" },
  6: { code: 6, name: "建物", color: "#c9a36b" },
  9: { code: 9, name: "水面", color: "#3d8fd1" },
  11: { code: 11, name: "路面", color: "#6e7479" },
  14: { code: 14, name: "電線", color: "#e0568a" },
  15: { code: 15, name: "電柱", color: "#8a6bbf" },
  20: { code: 20, name: "車両", color: "#e07a3d" },
  24: { code: 24, name: "林", color: "#2f6b3a" },
  21: { code: 21, name: "標識", color: "#e8b923" },
  22: { code: 22, name: "人物", color: "#c04a7a" },
  23: { code: 23, name: "道路標識", color: "#d1a02a" },
};
