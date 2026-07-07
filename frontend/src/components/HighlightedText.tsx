import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { X } from "lucide-react";
import type { Evidence, Segment } from "../lib/api";

const LEGEND = [
  { cat: "original", label: "Original / humano", swatch: "--good" },
  { cat: "ia", label: "Posible IA (intensidad = nivel)", swatch: "--ai" },
  { cat: "plagio", label: "Coincide con fuente", swatch: "--source" },
  { cat: "mixto", label: "Mixto", swatch: "--ink" },
];

const EVIDENCE_META: Record<string, { label: string; token: string }> = {
  copia: { label: "Copia de fuente", token: "--source" },
  generica: { label: "Frase genérica", token: "--ai" },
  conector: { label: "Conector", token: "--ai" },
  atenuador: { label: "Atenuador", token: "--ai" },
  tipografia: { label: "Tipografía", token: "--ai" },
};

interface Part {
  text: string;
  cat: string;
  seg?: Segment;
}

/** Intensidad del ámbar proporcional al puntaje de IA de la oración (0-100):
 *  cuanto más "parece IA", más marcado el resaltado. */
function aiStyle(score: number): CSSProperties {
  const alpha = 0.12 + (0.4 * Math.min(Math.max(score, 0), 100)) / 100;
  return {
    background: `rgb(var(--ai) / ${alpha.toFixed(2)})`,
    boxShadow: "inset 0 -2px 0 rgb(var(--ai) / 0.55)",
    borderRadius: "3px",
  };
}

/** Trocea el texto de un segmento en tramos normales y tramos de evidencia
 *  (usando offsets absolutos), para subrayar las palabras exactas. */
function splitByEvidence(segText: string, segStart: number, evidence: Evidence[]) {
  const pieces: { text: string; ev?: Evidence }[] = [];
  let cursor = 0;
  const sorted = [...evidence].sort((a, b) => a.start - b.start);
  for (const ev of sorted) {
    const s = ev.start - segStart;
    const e = ev.end - segStart;
    if (s < cursor || s < 0 || e > segText.length) continue; // offset fuera del tramo
    if (s > cursor) pieces.push({ text: segText.slice(cursor, s) });
    pieces.push({ text: segText.slice(s, e), ev });
    cursor = e;
  }
  if (cursor < segText.length) pieces.push({ text: segText.slice(cursor) });
  return pieces;
}

function evidenceUnderline(kind: string): CSSProperties {
  const token = EVIDENCE_META[kind]?.token ?? "--ai";
  return {
    borderBottom: `2px dotted rgb(var(${token}))`,
    fontWeight: 600,
  };
}

/** Reconstruye el texto original a partir de los segmentos (usando offsets) y
 *  resalta cada tramo según su categoría. Dentro de una oración marcada, las
 *  palabras que disparan señales van subrayadas; clic = panel de desglose. */
export function HighlightedText({ text, segments }: { text: string; segments: Segment[] }) {
  const [selected, setSelected] = useState<Segment | null>(null);

  const parts = useMemo<Part[]>(() => {
    const sorted = [...segments].sort((a, b) => a.start - b.start);
    const out: Part[] = [];
    let cursor = 0;
    for (const s of sorted) {
      if (s.start > cursor) out.push({ text: text.slice(cursor, s.start), cat: "gap" });
      out.push({ text: text.slice(s.start, s.end), cat: s.category, seg: s });
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
        <span className="text-xs text-muted">· El punteado marca las palabras exactas; clic para ver el porqué</span>
      </div>
      <p className="whitespace-pre-wrap text-[15px] leading-[1.85] text-ink">
        {parts.map((p, i) => {
          if (p.cat === "gap" || !p.seg) return <span key={i}>{p.text}</span>;
          const seg = p.seg;
          const graded = (p.cat === "ia" || p.cat === "mixto");
          const clickable = seg.category !== "original";
          const inner =
            seg.evidence && seg.evidence.length > 0 ? (
              splitByEvidence(p.text, seg.start, seg.evidence).map((piece, j) =>
                piece.ev ? (
                  <span key={j} style={evidenceUnderline(piece.ev.kind)}
                    title={`${EVIDENCE_META[piece.ev.kind]?.label ?? piece.ev.kind}: «${piece.ev.text}»`}>
                    {piece.text}
                  </span>
                ) : (
                  <span key={j}>{piece.text}</span>
                )
              )
            ) : (
              p.text
            );
          return (
            <span
              key={i}
              className={`${graded ? "" : `hl-${p.cat}`} ${clickable ? "cursor-pointer" : ""}`}
              style={graded ? aiStyle(seg.ai_score) : undefined}
              title={seg.reason}
              onClick={clickable ? () => setSelected(selected?.index === seg.index ? null : seg) : undefined}
            >
              {inner}
            </span>
          );
        })}
      </p>

      {selected && (
        <div className="mt-5 rounded-xl border border-line bg-paper p-5">
          <div className="mb-2 flex items-start justify-between gap-3">
            <h4 className="font-display text-base font-semibold text-ink">
              Desglose de la oración
            </h4>
            <button onClick={() => setSelected(null)}
              className="rounded p-1 text-muted hover:text-ink" aria-label="Cerrar">
              <X size={16} />
            </button>
          </div>
          <p className="mb-3 text-sm italic text-muted">«{selected.text}»</p>
          <div className="mb-3 flex flex-wrap gap-2 text-xs">
            <span className="chip">IA por oración: {selected.ai_score}%</span>
            {selected.plagiarism_overlap > 0 && (
              <span className="chip">Coincidencia: {selected.plagiarism_overlap}%</span>
            )}
          </div>

          {(selected.evidence ?? []).length === 0 && (
            <p className="text-sm text-muted">
              Sin evidencias léxicas puntuales: el marcado viene del estilo global de la
              oración (ritmo, longitud, ausencia de voz personal) o del modelo entrenado.
            </p>
          )}

          {(selected.evidence ?? []).map((ev, i) =>
            ev.kind === "copia" ? (
              <div key={i} className="mb-3">
                <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
                  Coincidencia con «{selected.plagiarism_source}» ({ev.matched_words} palabras seguidas)
                </p>
                <div className="grid gap-2 sm:grid-cols-2">
                  <div className="rounded-lg p-3 text-sm"
                    style={{ background: "rgb(var(--source) / 0.10)", border: "1px solid rgb(var(--source) / 0.35)" }}>
                    <p className="mb-1 text-xs font-semibold text-muted">Tu texto</p>
                    «{ev.text}»
                  </div>
                  <div className="rounded-lg p-3 text-sm"
                    style={{ background: "rgb(var(--source) / 0.10)", border: "1px solid rgb(var(--source) / 0.35)" }}>
                    <p className="mb-1 text-xs font-semibold text-muted">La fuente</p>
                    «{ev.source_fragment}»
                  </div>
                </div>
              </div>
            ) : (
              <p key={i} className="mb-1 text-sm text-ink">
                <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                  style={{ background: `rgb(var(${EVIDENCE_META[ev.kind]?.token ?? "--ai"}) / 0.7)` }} />
                <b>{EVIDENCE_META[ev.kind]?.label ?? ev.kind}:</b> «{ev.text}»
              </p>
            )
          )}
          <p className="mt-3 text-xs text-muted">{selected.reason}</p>
        </div>
      )}
    </div>
  );
}
