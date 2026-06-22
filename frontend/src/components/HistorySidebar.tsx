import { History, Trash2 } from "lucide-react";
import type { HistoryItem } from "../lib/api";

interface Props {
  items: HistoryItem[];
  onOpen: (id: number) => void;
  onDelete: (id: number) => void;
  onClear: () => void;
}

export function HistorySidebar({ items, onOpen, onDelete, onClear }: Props) {
  return (
    <aside className="card h-fit p-5 lg:sticky lg:top-6">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History size={17} className="text-accent" />
          <h3 className="font-display text-lg font-semibold text-ink">Historial</h3>
        </div>
        {items.length > 0 && (
          <button onClick={onClear} className="text-xs text-muted transition hover:text-ai">
            Vaciar
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <p className="text-sm text-muted">Tus análisis aparecerán aquí.</p>
      ) : (
        <ul className="max-h-[70vh] space-y-2 overflow-auto pr-1">
          {items.map((it) => (
            <li key={it.id}>
              <button
                onClick={() => onOpen(it.id)}
                className="group w-full rounded-xl border border-line bg-surface-2/40 p-3 text-left transition hover:border-accent hover:bg-surface-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="line-clamp-2 text-sm font-medium text-ink">{it.title}</span>
                  <Trash2
                    size={14}
                    className="mt-0.5 shrink-0 text-muted opacity-0 transition hover:text-ai group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(it.id);
                    }}
                  />
                </div>
                <div className="mt-2 flex gap-1.5 font-mono text-[10px]">
                  <span className="rounded px-1.5 py-0.5"
                    style={{ background: "rgb(var(--good) / 0.15)", color: "rgb(var(--good))" }}>
                    O {it.originality}
                  </span>
                  <span className="rounded px-1.5 py-0.5"
                    style={{ background: "rgb(var(--source) / 0.15)", color: "rgb(var(--source))" }}>
                    P {it.plagiarism}
                  </span>
                  <span className="rounded px-1.5 py-0.5"
                    style={{ background: "rgb(var(--ai) / 0.15)", color: "rgb(var(--ai))" }}>
                    IA {it.ai_probability}
                  </span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
