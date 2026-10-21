import type { ReactNode } from "react";
import type { ClassDef } from "./types";

type Props = {
  classDefs: ClassDef[];
  counts: Map<number, number>;
  removed: Set<number>; // 除去済みのクラス
  hidden: Set<number>; // 表示をオフにしているクラス
  marked: Set<number>; // 除去対象に選んだクラス
  remainingPoints: number;
  markedPoints: number;
  canUndo: boolean;
  note: string;
  children?: ReactNode; // 先頭に表示するファイル操作など
  onToggleHidden: (id: number) => void;
  onToggleMarked: (id: number) => void;
  onRemove: () => void;
  onUndo: () => void;
};

const fmt = (n: number) => n.toLocaleString("ja-JP");

export function LayerPanel(p: Props) {
  return (
    <aside className="panel">
      <header>
        <h1>レイヤ</h1>
        <p className="total">残りの点数 {fmt(p.remainingPoints)}</p>
      </header>

      {p.children}

      <table className="layers">
        <thead>
          <tr>
            <th scope="col">表示</th>
            <th scope="col">レイヤ</th>
            <th scope="col" className="num">点数</th>
            <th scope="col">除去対象</th>
          </tr>
        </thead>
        <tbody>
          {p.classDefs.map((c) => {
            const isRemoved = p.removed.has(c.id);
            return (
              <tr key={c.id} className={isRemoved ? "removed" : p.marked.has(c.id) ? "marked" : ""}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`${c.name}を表示`}
                    checked={!p.hidden.has(c.id) && !isRemoved}
                    disabled={isRemoved}
                    onChange={() => p.onToggleHidden(c.id)}
                  />
                </td>
                <td>
                  <span className="swatch" style={{ background: c.color }} />
                  {c.name}
                  {isRemoved && <span className="tag">除去済み</span>}
                </td>
                <td className="num">{fmt(p.counts.get(c.id) ?? 0)}</td>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`${c.name}を除去対象にする`}
                    checked={p.marked.has(c.id)}
                    disabled={isRemoved}
                    onChange={() => p.onToggleMarked(c.id)}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="actions">
        <button className="danger" onClick={p.onRemove} disabled={p.marked.size === 0}>
          選んだレイヤを除去({fmt(p.markedPoints)}点)
        </button>
        <button onClick={p.onUndo} disabled={!p.canUndo}>
          元に戻す
        </button>
      </div>
      <p className="note">{p.note}</p>
    </aside>
  );
}