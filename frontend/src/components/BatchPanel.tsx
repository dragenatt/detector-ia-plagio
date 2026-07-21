import { Download, FileWarning, Users } from "lucide-react";
import type { BatchResult } from "../lib/api";

/** Color de la celda de IA según el nivel de riesgo. */
function aiCellStyle(v: number) {
  const alpha = 0.1 + (0.4 * Math.min(Math.max(v, 0), 100)) / 100;
  return { background: `rgb(var(--ai) / ${alpha.toFixed(2)})` };
}

function toCsv(res: BatchResult): string {
  const head = ["archivo", "palabras", "ia_%", "plagio_%", "originalidad_%",
    "confianza", "coincide_con", "coincidencia_%", "error"];
  const rows = res.rows.map((r) => [
    r.name, r.words ?? "", r.ai_probability ?? "", r.plagiarism ?? "",
    r.originality ?? "", r.confidence ?? "",
    r.cross_match?.source ?? "", r.cross_match?.overlap ?? "",
    r.error ?? "",
  ]);
  return [head, ...rows]
    .map((cols) => cols.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","))
    .join("\n");
}

export function BatchPanel({ res }: { res: BatchResult }) {
  const download = () => {
    const blob = new Blob([toCsv(res)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "veraz-lote.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const ok = res.rows.filter((r) => r.ok);
  const sorted = [...ok].sort((a, b) => (b.ai_probability ?? 0) - (a.ai_probability ?? 0));
  const failed = res.rows.filter((r) => !r.ok);

  return (
    <section className="card p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-ink">
          <Users size={18} /> Análisis por lotes · {res.count} archivo(s)
        </h3>
        <button className="btn-ghost" onClick={download}>
          <Download size={15} /> Exportar CSV
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
              <th className="py-2 pr-3">Archivo</th>
              <th className="px-3 text-right">IA</th>
              <th className="px-3 text-right">Plagio</th>
              <th className="px-3 text-right">Originalidad</th>
              <th className="px-3">Confianza</th>
              <th className="pl-3">¿Se copió con?</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.name} className="border-b border-line/60">
                <td className="py-2 pr-3 font-medium text-ink">{r.name}</td>
                <td className="px-3 text-right font-mono" style={aiCellStyle(r.ai_probability ?? 0)}>
                  {r.ai_probability}%
                </td>
                <td className="px-3 text-right font-mono">{r.plagiarism}%</td>
                <td className="px-3 text-right font-mono">{r.originality}%</td>
                <td className="px-3 text-muted">{r.confidence}</td>
                <td className="pl-3">
                  {r.cross_match ? (
                    <span className="text-source">
                      {r.cross_match.source} ({r.cross_match.overlap}%)
                    </span>
                  ) : (
                    <span className="text-muted">—</span>
                  )}
                  {r.warning && (
                    <span className="ml-1 text-ai" title={r.warning}>⚠</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {failed.length > 0 && (
        <div className="mt-4 rounded-lg border border-ai/40 bg-ai/10 p-3 text-sm">
          <p className="mb-1 flex items-center gap-1.5 font-medium text-ink">
            <FileWarning size={15} className="text-ai" /> {failed.length} archivo(s) no se pudieron analizar:
          </p>
          <ul className="list-inside list-disc text-muted">
            {failed.map((r) => (
              <li key={r.name}>{r.name}: {r.error}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-3 text-xs text-muted">
        "¿Se copió con?" compara cada trabajo contra los demás del lote para
        detectar copia entre ellos. Los porcentajes son estimaciones.
      </p>
    </section>
  );
}
