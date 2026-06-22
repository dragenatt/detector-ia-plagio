import { Lightbulb } from "lucide-react";
import type { Recommendation } from "../lib/api";

const PRIORITY: Record<string, { label: string; token: string }> = {
  alta: { label: "Prioridad alta", token: "--ai" },
  media: { label: "Prioridad media", token: "--source" },
  baja: { label: "Sugerencia", token: "--good" },
};

/** Recomendaciones para mejorar el texto sin volverlo más artificial. */
export function Recommendations({ recs }: { recs: Recommendation[] }) {
  return (
    <div className="card p-6">
      <div className="mb-4 flex items-center gap-2">
        <Lightbulb size={18} className="text-accent" />
        <h3 className="font-display text-lg font-semibold text-ink">
          Recomendaciones para mejorar
        </h3>
      </div>
      <div className="space-y-3">
        {recs.map((r, i) => {
          const p = PRIORITY[r.priority] ?? PRIORITY.baja;
          return (
            <div key={i} className="rounded-xl border border-line bg-surface-2/50 p-4">
              <div className="flex items-center justify-between gap-3">
                <h4 className="font-semibold text-ink">{r.title}</h4>
                <span className="chip whitespace-nowrap"
                  style={{ color: `rgb(var(${p.token}))` }}>{p.label}</span>
              </div>
              <p className="mt-1.5 text-sm text-muted">{r.detail}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
