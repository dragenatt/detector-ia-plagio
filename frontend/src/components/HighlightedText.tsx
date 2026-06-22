import { useMemo } from "react";
import type { Segment } from "../lib/api";

const LEGEND = [
  { cat: "original", label: "Original / humano", swatch: "--good" },
  { cat: "ia", label: "Posible IA", swatch: "--ai" },
  { cat: "plagio", label: "Coincide con fuente", swatch: "--source" },
  { cat: "mixto", label: "Mixto", swatch: "--ink" },
];

interface Part {
  text: string;
  cat: string;
  reason?: string;
}

/** Reconstruye el texto original a partir de los segmentos (usando offsets) y
 *  resalta cada tramo según su categoría. El hover muestra el motivo. */
export function HighlightedText({ text, segments }: { text: string; segments: Segment[] }) {
  const parts = useMemo<Part[]>(() => {
    const sorted = [...segments].sort((a, b) => a.start - b.start);
    const out: Part[] = [];
    let cursor = 0;
    for (const s of sorted) {
      if (s.start > cursor) out.push({ text: text.slice(cursor, s.start), cat: "gap" });
      out.push({ text: text.slice(s.start, s.end), cat: s.category, reason: s.reason });
      cursor = Math.max(cursor, s.end);
    }
    if (cursor < text.length) out.push({ text: text.slice(cursor), cat: "gap" });
    return out;
  }, [text, segments]);

  return (
    <div className="card p-6">
      <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        <h3 className="mr-2 font-display text-lg font-semibold text-ink">Texto analizado</h3>
        {LEGEND.map((l) => (
          <span key={l.cat} className="inline-flex items-center gap-1.5 text-xs text-muted">
            <span className="h-3 w-3 rounded-sm"
              style={{ background: `rgb(var(${l.swatch}) / 0.5)` }} />
            {l.label}
          </span>
        ))}
      </div>
      <p className="whitespace-pre-wrap text-[15px] leading-[1.85] text-ink">
        {parts.map((p, i) =>
          p.cat === "gap" ? (
            <span key={i}>{p.text}</span>
          ) : (
            <span key={i} className={`hl-${p.cat} cursor-help`} title={p.reason}>
              {p.text}
            </span>
          )
        )}
      </p>
    </div>
  );
}
