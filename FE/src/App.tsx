import { useMemo, useState } from "react";
import { LayerPanel } from "./LayerPanel";
import { PointCloudView } from "./PointCloudView";
import { makeDummyData } from "./dummy";

const toggle = (set: Set<number>, id: number) => {
  const next = new Set(set);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
};

export default function App() {
  const data = useMemo(() => makeDummyData(), []);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [marked, setMarked] = useState<Set<number>>(new Set());
  // 除去の履歴。1回の除去ごとに、除去したクラス番号の配列を積む
  const [history, setHistory] = useState<number[][]>([]);

  const counts = useMemo(() => {
    const m = new Map<number, number>();
    for (const c of data.classes) m.set(c, (m.get(c) ?? 0) + 1);
    return m;
  }, [data]);

  const removed = useMemo(() => new Set(history.flat()), [history]);
  const hiddenAll = useMemo(() => new Set([...hidden, ...removed]), [hidden, removed]);

  const sum = (ids: Iterable<number>) => {
    let n = 0;
    for (const id of ids) n += counts.get(id) ?? 0;
    return n;
  };
  const remainingPoints = data.classes.length - sum(removed);
  const markedPoints = sum(marked);

  const handleRemove = () => {
    if (marked.size === 0) return;
    const names = data.classDefs.filter((d) => marked.has(d.id)).map((d) => d.name).join("、");
    if (!window.confirm(`次のレイヤを除去します。\n${names}(合計${markedPoints.toLocaleString("ja-JP")}点)`)) return;
    setHistory((h) => [...h, [...marked]]);
    setMarked(new Set());
  };

  return (
    <div className="app">
      <PointCloudView data={data} hidden={hiddenAll} />
      <LayerPanel
        classDefs={data.classDefs}
        counts={counts}
        removed={removed}
        hidden={hidden}
        marked={marked}
        remainingPoints={remainingPoints}
        markedPoints={markedPoints}
        canUndo={history.length > 0}
        onToggleHidden={(id) => setHidden((s) => toggle(s, id))}
        onToggleMarked={(id) => setMarked((s) => toggle(s, id))}
        onRemove={handleRemove}
        onUndo={() => setHistory((h) => h.slice(0, -1))}
      />
    </div>
  );
}