import { useCallback, useEffect, useState } from "react";
import { GraduationCap, Loader2, Plus, Undo2, RefreshCw, AlertTriangle } from "lucide-react";
import { api } from "../lib/api";
import type { CorpusCounts, TrainReport } from "../lib/api";

const LABELS: { id: "humano" | "ia" | "mixto"; label: string; hint: string }[] = [
  { id: "humano", label: "Humano", hint: "escrito por una persona" },
  { id: "ia", label: "IA", hint: "generado por inteligencia artificial" },
  { id: "mixto", label: "Mixto", hint: "en parte humano, en parte IA" },
];

/** Vista "Entrenar": el usuario le enseña al detector qué es humano y qué es
 *  IA añadiendo ejemplos etiquetados, y reentrena con un clic. */
export function TrainPanel({ onModelChange }: { onModelChange: () => void }) {
  const [counts, setCounts] = useState<CorpusCounts | null>(null);
  const [text, setText] = useState("");
  const [label, setLabel] = useState<"humano" | "ia" | "mixto">("humano");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [training, setTraining] = useState(false);
  const [report, setReport] = useState<TrainReport | null>(null);

  const refresh = useCallback(async () => {
    try { setCounts(await api.corpus()); } catch { /* backend caído */ }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  const add = async () => {
    setBusy(true); setMsg(null);
    try {
      const r = await api.addExample(text, label);
      setCounts(r.counts);
      setText("");
      setMsg(r.note ?? `Ejemplo añadido como "${label}". ¡Gracias por enseñarle!`);
    } catch (e) { setMsg((e as Error).message); }
    finally { setBusy(false); }
  };

  const undo = async () => {
    setBusy(true); setMsg(null);
    try {
      const r = await api.undoLastExample();
      setCounts(r.counts);
      setMsg(`Se deshizo el último ejemplo (${r.label}).`);
    } catch (e) { setMsg((e as Error).message); }
    finally { setBusy(false); }
  };

  const retrain = async () => {
    setTraining(true); setMsg(null); setReport(null);
    try {
      const r = await api.train();
      setReport(r);
      onModelChange();
      await refresh();
    } catch (e) { setMsg((e as Error).message); }
    finally { setTraining(false); }
  };

  const balancePct = counts ? Math.round(counts.balanced * 100) : 0;
  const imbalanced = counts != null && counts.balanced < 0.5;

  return (
    <section className="space-y-6">
      <div className="card p-6 md:p-7">
        <h2 className="mb-1 flex items-center gap-2 font-display text-xl font-semibold text-ink">
          <GraduationCap size={22} /> Entrenar el detector
        </h2>
        <p className="mb-5 text-sm text-muted">
          Enséñale qué es humano y qué es IA. Cuantos más ejemplos variados le
          des, mejor aprende. Se necesitan al menos <b>3 de cada clase</b> para
          entrenar; para notar mejora, apunta a <b>10 o más</b>.
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={7}
          placeholder="Pega aquí un texto de ejemplo…"
          className="w-full resize-y rounded-xl border border-line bg-paper p-4 text-[15px] leading-relaxed text-ink outline-none transition focus:border-accent"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <div className="inline-flex rounded-full border border-line bg-surface-2 p-1 text-sm">
            {LABELS.map((l) => (
              <button key={l.id} onClick={() => setLabel(l.id)} title={l.hint}
                className={`rounded-full px-4 py-1.5 font-medium transition ${
                  label === l.id ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink"}`}>
                {l.label}
              </button>
            ))}
          </div>
          <span className="text-xs text-muted">{words} palabras</span>
          <div className="ml-auto flex gap-2">
            <button className="btn-ghost" onClick={undo} disabled={busy}>
              <Undo2 size={15} /> Deshacer último
            </button>
            <button className="btn-primary" onClick={add} disabled={busy || words < 1}>
              {busy ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
              Añadir al corpus
            </button>
          </div>
        </div>
        {msg && <p className="mt-3 text-sm text-accent-ink">{msg}</p>}
      </div>

      {/* Balance del corpus */}
      <div className="card p-6">
        <h3 className="mb-3 font-display text-lg font-semibold text-ink">Balance del corpus</h3>
        {counts ? (
          <>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              {Object.entries(counts.by_label).map(([lab, c]) => (
                <div key={lab} className="rounded-xl border border-line bg-paper p-3">
                  <div className="text-2xl font-semibold text-ink">{c.total}</div>
                  <div className="text-xs capitalize text-muted">{lab}</div>
                  {c.user > 0 && <div className="text-xs text-accent-ink">{c.user} tuyos</div>}
                </div>
              ))}
            </div>
            <div className="mt-4">
              <div className="mb-1 flex justify-between text-xs text-muted">
                <span>Humano: {counts.human_axis}</span>
                <span>IA: {counts.ai_axis}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                <div className="h-full bg-good"
                  style={{ width: `${(counts.human_axis / Math.max(counts.human_axis + counts.ai_axis, 1)) * 100}%` }} />
              </div>
              {imbalanced && (
                <p className="mt-2 flex items-center gap-1.5 text-xs text-ai">
                  <AlertTriangle size={13} /> Las clases están desbalanceadas
                  ({balancePct}%). Añade más ejemplos de la clase con menos.
                </p>
              )}
            </div>
          </>
        ) : (
          <p className="text-sm text-muted">Cargando…</p>
        )}
        <div className="mt-5 flex items-center gap-3">
          <button className="btn-primary" onClick={retrain} disabled={training}>
            {training ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
            {training ? "Entrenando…" : "Reentrenar"}
          </button>
          <span className="text-xs text-muted">Aprende de todos los ejemplos actuales.</span>
        </div>

        {report && (
          <div className={`mt-4 rounded-xl border p-4 text-sm ${
            report.trained ? "border-good/40 bg-good/10" : "border-ai/40 bg-ai/10"}`}>
            {report.trained ? (
              <>
                <p className="font-medium text-ink">✓ Modelo reentrenado.</p>
                {report.cross_validation && (
                  <p className="mt-1 text-muted">
                    Precisión (validación cruzada F1): <b className="text-ink">
                      {report.cross_validation.f1_mean} ± {report.cross_validation.f1_std}</b>
                    {report.calibration_after?.ece != null &&
                      ` · calibración (ECE): ${report.calibration_after.ece}`}
                  </p>
                )}
              </>
            ) : (
              <p className="text-ink">{report.message}</p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
