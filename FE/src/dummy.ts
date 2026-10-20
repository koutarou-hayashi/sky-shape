import type { ClassDef, PointCloudData } from "./types";

export const DUMMY_CLASSES: ClassDef[] = [
  { id: 0, name: "地面・路面", color: "#8c9298" },
  { id: 1, name: "建物", color: "#c9a36b" },
  { id: 2, name: "植栽", color: "#4f9d4f" },
  { id: 3, name: "電柱", color: "#8a6bbf" },
  { id: 4, name: "電線", color: "#e0568a" },
  { id: 5, name: "車両", color: "#3d8fd1" },
  { id: 6, name: "標識", color: "#e8b923" },
];

// 毎回同じ点群になるように、種つきの乱数を使う
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeDummyData(): PointCloudData {
  const rand = mulberry32(42);
  const pos: number[] = [];
  const cls: number[] = [];
  const add = (x: number, y: number, z: number, c: number) => {
    pos.push(x, y, z);
    cls.push(c);
  };

  // 地面
  for (let i = 0; i < 60000; i++) {
    add((rand() - 0.5) * 80, (rand() - 0.5) * 80, rand() * 0.05, 0);
  }

  // 建物(壁面のみ): [中心x, 中心y, 幅, 奥行, 高さ]
  const boxes: [number, number, number, number, number][] = [
    [-20, 15, 14, 10, 12],
    [18, 18, 10, 12, 9],
    [-18, -22, 12, 8, 7],
  ];
  for (const [cx, cy, w, d, h] of boxes) {
    for (let i = 0; i < 9000; i++) {
      const side = Math.floor(rand() * 4);
      const u = rand() - 0.5;
      const z = rand() * h;
      if (side < 2) add(cx + u * w, cy + (side ? d / 2 : -d / 2), z, 1);
      else add(cx + (side === 2 ? w / 2 : -w / 2), cy + u * d, z, 1);
    }
  }

  // 植栽(幹と樹冠)
  for (let t = 0; t < 14; t++) {
    const tx = (rand() - 0.5) * 70;
    const ty = -14 + rand() * 10;
    for (let i = 0; i < 1500; i++) {
      const a = rand() * Math.PI * 2;
      const b = Math.acos(2 * rand() - 1);
      const r = 2.5 * Math.cbrt(rand());
      add(tx + r * Math.sin(b) * Math.cos(a), ty + r * Math.sin(b) * Math.sin(a), 5 + r * Math.cos(b), 2);
    }
    for (let i = 0; i < 150; i++) add(tx + (rand() - 0.5) * 0.3, ty + (rand() - 0.5) * 0.3, rand() * 4, 2);
  }

  // 電柱と電線
  const xs = [-30, -15, 0, 15, 30];
  for (const px of xs) {
    for (let i = 0; i < 400; i++) add(px + (rand() - 0.5) * 0.3, (rand() - 0.5) * 0.3, rand() * 9, 3);
  }
  for (let k = 0; k < xs.length - 1; k++) {
    for (const [h, dy] of [[8.5, -0.4], [7.8, 0.4]]) {
      for (let i = 0; i < 300; i++) {
        const s = rand();
        add(xs[k] + (xs[k + 1] - xs[k]) * s, dy, h - 4.8 * s * (1 - s), 4);
      }
    }
  }

  // 車両
  for (const [vx, vy] of [[-8, 6], [6, 7], [20, -3]]) {
    for (let i = 0; i < 1500; i++) add(vx + (rand() - 0.5) * 4.2, vy + (rand() - 0.5) * 1.8, rand() * 1.5, 5);
  }

  // 標識(支柱と板)
  for (const [sx, sy] of [[-25, 4], [10, -4], [26, 3]]) {
    for (let i = 0; i < 120; i++) add(sx + (rand() - 0.5) * 0.1, sy, rand() * 2.5, 6);
    for (let i = 0; i < 250; i++) add(sx + (rand() - 0.5) * 0.8, sy, 2.5 + (rand() - 0.5) * 0.8, 6);
  }

  return {
    positions: new Float32Array(pos),
    classes: new Uint8Array(cls),
    classDefs: DUMMY_CLASSES,
  };
}