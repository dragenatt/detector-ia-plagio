import type { AnalysisResult } from "../lib/api";

/** Mapa de calor lineal del documento: una barra que representa el texto de
 *  principio a fin, con las ZONAS de IA marcadas en ámbar (según las regiones
 *  que detecta el motor). De un vistazo se ve DÓNDE se concentra la IA. */
export function DocumentHeatmap({ r }: { r: AnalysisResult }) {
  const regions = r.ai_detection?.regions ?? [];
  const total = r.meta?.char_count || 1;
  const coverage = r.ai_detection?.regions_summary?.coverage ?? 0;

  if (!regions.length) return null;

  return (
    <div className="mt-6">
      <div className="mb-1.5 flex items-center justify-between text-xs text-muted">
        <span>Dónde se concentra la IA en el documento</span>
        <span>{Math.round(coverage * 100)}% del texto · {regions.length} zona(s)</span>
      </div>
      <div className="relative h-4 w-full overflow-hidden rounded-full"
        style={{ background: "rgb(var(--good) / 0.18)" }}
        title="Verde = redacción propia · Ámbar = zona con estilo de IA">
        {regions.map((reg, i) => {
          const left = (reg.start / total) * 100;
          const width = Math.max(((reg.end - reg.start) / total) * 100, 1.5);
          const alpha = 0.35 + 0.5 * Math.min(Math.max(reg.score, 0), 1);
          return (
            <div key={i} className="absolute top-0 h-full"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: `rgb(var(--ai) / ${alpha.toFixed(2)})`,
              }}
              title={`Zona de IA (${Math.round(reg.score * 100)}%)`} />
          );
        })}
      </div>
      <div className="mt-1.5 flex gap-4 text-xs text-muted">
        <span className="inline-flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: "rgb(var(--good) / 0.5)" }} />
          Propio
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: "rgb(var(--ai) / 0.6)" }} />
          Zona con estilo de IA
        </span>
      </div>
    </div>
  );
}
