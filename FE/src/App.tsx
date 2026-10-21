import { useMemo, useRef, useState } from "react";
import { LayerPanel } from "./Layerpanel";
import { PointCloudView, type ColorMode } from "./Pointcloudview";
// import { makeDummyData } from "./dummy";
// ↑プロトタイプを見せたい時などはコメントアウトを外す
import { readLas } from "./lasReader";
import type { PointCloudData } from "./types";

// 画面に表示する最大の点数(これを超える分は等間隔で間引いて読み込む)
const MAX_DISPLAY_POINTS = 2_000_000;
const EMPTY_DATA: PointCloudData = {
  positions: new Float32Array(0),
  classes: new Uint8Array(0),
  classDefs: [],
};

type Status =
  | { kind: "idle" }
  | { kind: "loading"; ratio: number }
  | { kind: "error"; message: string };

const toggle = (set: Set<number>, id: number) => {
  const next = new Set(set);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
};

export default function App() {
  // const [data, setData] = useState<PointCloudData>(() => makeDummyData());
  // プロトタイプを見せたい時は上の行のコメントアウトをはずし、下の行をコメントアウトする
  const [data, setData] = useState<PointCloudData>(EMPTY_DATA);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [marked, setMarked] = useState<Set<number>>(new Set());
  // 除去の履歴。1回の除去ごとに、除去したクラス番号の配列を積む
  const [history, setHistory] = useState<number[][]>([]);
  const [colorMode, setColorMode] = useState<ColorMode>("class");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const fileRef = useRef<HTMLInputElement>(null);
  const loading = status.kind === "loading";

  const openFile = async (file: File) => {
    if (loading) return;
    setStatus({ kind: "loading", ratio: 0 });
    try {
      const loaded = await readLas(file, {
        maxPoints: MAX_DISPLAY_POINTS,
        onProgress: (done, total) => setStatus({ kind: "loading", ratio: done / total }),
      });
      setData(loaded);
      setHidden(new Set());
      setMarked(new Set());
      setHistory([]);
      setColorMode(loaded.colors ? "rgb" : "class");
      setStatus({ kind: "idle" });
    } catch (e) {
      setStatus({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  // クラスごとの点数(ファイル読み込み時は、間引く前の全点数)
  const counts = useMemo(() => {
    const m = new Map<number, number>();
    if (data.meta) {
      for (const [id, n] of Object.entries(data.meta.classCounts)) m.set(Number(id), n);
    } else {
      for (const c of data.classes) m.set(c, (m.get(c) ?? 0) + 1);
    }
    return m;
  }, [data]);
  const totalPoints = data.meta?.totalPoints ?? data.classes.length;

  const removed = useMemo(() => new Set(history.flat()), [history]);
  const hiddenAll = useMemo(() => new Set([...hidden, ...removed]), [hidden, removed]);

  const sum = (ids: Iterable<number>) => {
    let n = 0;
    for (const id of ids) n += counts.get(id) ?? 0;
    return n;
  };
  const remainingPoints = totalPoints - sum(removed);
  const markedPoints = sum(marked);

  const handleRemove = () => {
    if (marked.size === 0) return;
    const names = data.classDefs.filter((d) => marked.has(d.id)).map((d) => d.name).join("、");
    if (!window.confirm(`次のレイヤを除去します。\n${names}(合計${markedPoints.toLocaleString("ja-JP")}点)`)) return;
    setHistory((h) => [...h, [...marked]]);
    setMarked(new Set());
  };

  const meta = data.meta;
  const fmt = (n: number) => n.toLocaleString("ja-JP");
  const note =
  data.classes.length === 0
    ? "LASファイルを開くと、点群が表示されます。"
    : !meta
      ? "表示はダミーデータです。除去は画面上だけで、元のデータは変わりません。"
      : meta.stride > 1
        ? `表示を軽くするため、...(元のまま)`
        : "除去は画面上だけで、元のファイルは変わりません。";

  return (
    <div
      className="app"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) void openFile(file);
      }}
    >
      <PointCloudView data={data} hidden={hiddenAll} colorMode={colorMode} />
      <LayerPanel
        classDefs={data.classDefs}
        counts={counts}
        removed={removed}
        hidden={hidden}
        marked={marked}
        remainingPoints={remainingPoints}
        markedPoints={markedPoints}
        canUndo={history.length > 0}
        note={note}
        onToggleHidden={(id) => setHidden((s) => toggle(s, id))}
        onToggleMarked={(id) => setMarked((s) => toggle(s, id))}
        onRemove={handleRemove}
        onUndo={() => setHistory((h) => h.slice(0, -1))}
      >
        <section className="file-tools" aria-label="ファイル">
          <input
            ref={fileRef}
            type="file"
            accept=".las"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = ""; // 同じファイルをもう一度選べるようにする
              if (file) void openFile(file);
            }}
          />
          <button onClick={() => fileRef.current?.click()} disabled={loading}>
            LASファイルを開く
          </button>
          <p className="hint">または、このウィンドウにLASファイルをドロップ</p>
          {meta && <p className="file-name">{meta.fileName}</p>}
          <label className="field">
            表示色
            <select value={colorMode} onChange={(e) => setColorMode(e.target.value as ColorMode)}>
              <option value="class">レイヤの色</option>
              <option value="rgb" disabled={!data.colors}>
                元のRGB
              </option>
            </select>
          </label>
          {status.kind === "loading" && <p role="status">読み込み中 {Math.round(status.ratio * 100)}%</p>}
          {status.kind === "error" && (
            <p role="alert" className="error">
              {status.message}
            </p>
          )}
        </section>
      </LayerPanel>
    </div>
  );
}