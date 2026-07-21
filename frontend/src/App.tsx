import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, AlertTriangle } from "lucide-react";
import { api } from "./lib/api";
import type { AnalysisResult, BatchResult, HistoryItem, ModelStatus } from "./lib/api";
import { useTheme } from "./lib/theme";
import { Header } from "./components/Header";
import { InputPanel } from "./components/InputPanel";
import { ScorePanel } from "./components/ScorePanel";
import { SignalBars } from "./components/SignalBars";
import { HighlightedText } from "./components/HighlightedText";
import { ExplanationPanel } from "./components/ExplanationPanel";
import { Recommendations } from "./components/Recommendations";
import { HistorySidebar } from "./components/HistorySidebar";
import { Disclaimer } from "./components/Disclaimer";
import { BatchPanel } from "./components/BatchPanel";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { TrainPanel } from "./components/TrainPanel";
import { ThumbsUp, Bot } from "lucide-react";

export default function App() {
  const { theme, toggle } = useTheme();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [batch, setBatch] = useState<BatchResult | null>(null);
  const [analyzedText, setAnalyzedText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [training, setTraining] = useState(false);
  const [view, setView] = useState<"analyze" | "train">("analyze");
  const [feedback, setFeedback] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try { setHistory(await api.history()); } catch { /* backend apagado */ }
    try { setModel(await api.modelStatus()); } catch { /* backend apagado */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleText = async (text: string) => {
    setLoading(true); setError(null); setFeedback(null);
    try {
      const r = await api.analyzeText(text);
      setBatch(null); setResult(r); setAnalyzedText(r.analyzed_text ?? text); refresh();
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const handleFile = async (file: File) => {
    setLoading(true); setError(null); setFeedback(null);
    try {
      const r = await api.analyzeFile(file);
      setBatch(null); setResult(r);
      setAnalyzedText(r.analyzed_text ?? r.extracted_text ?? ""); refresh();
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const handleBatch = async (files: File[]) => {
    setLoading(true); setError(null); setFeedback(null);
    try {
      const r = await api.analyzeBatch(files);
      setResult(null); setAnalyzedText(""); setBatch(r);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const openHistory = async (id: number) => {
    setError(null);
    try {
      const d = await api.historyDetail(id);
      setResult({ ...d.payload, id });
      setAnalyzedText(d.text ?? "");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) { setError((e as Error).message); }
  };

  const train = async () => {
    setTraining(true); setError(null);
    try { await api.train(); await refresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setTraining(false); }
  };

  const sendFeedback = async (label: "humano" | "ia") => {
    if (!analyzedText) return;
    setFeedback(null);
    try {
      await api.addExample(analyzedText, label);
      setFeedback(label === "humano"
        ? "Guardado como HUMANO. Reentrena en la pestaña «Entrenar» para que aprenda."
        : "Guardado como IA. Reentrena en la pestaña «Entrenar» para que aprenda.");
    } catch (e) { setFeedback("No se pudo guardar: " + (e as Error).message); }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <Header theme={theme} onToggleTheme={toggle} model={model}
        onTrain={train} training={training} />

      <div className="mb-5 inline-flex rounded-full border border-line bg-surface-2 p-1 text-sm">
        <button onClick={() => setView("analyze")}
          className={`rounded-full px-5 py-1.5 font-medium transition ${
            view === "analyze" ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink"}`}>
          Analizar
        </button>
        <button onClick={() => setView("train")}
          className={`rounded-full px-5 py-1.5 font-medium transition ${
            view === "train" ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink"}`}>
          Entrenar
        </button>
      </div>

      {view === "train" ? (
        <ErrorBoundary onReset={() => setView("analyze")}>
          <TrainPanel onModelChange={refresh} />
        </ErrorBoundary>
      ) : (
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
        <main className="space-y-6">
          <InputPanel onAnalyzeText={handleText} onAnalyzeFile={handleFile}
            onAnalyzeBatch={handleBatch} loading={loading} />

          {error && (
            <div className="flex items-center gap-2 rounded-2xl border border-ai/40 bg-ai/10 p-4 text-sm text-ink">
              <AlertTriangle size={16} className="text-ai" />
              {error}
            </div>
          )}

          {batch && (
            <ErrorBoundary onReset={() => setBatch(null)}>
              <BatchPanel res={batch} />
            </ErrorBoundary>
          )}

          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                key={result.id ?? "result"}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                <ErrorBoundary onReset={() => setResult(null)}>
                  {result.extraction?.pages_total != null && (
                    <div className={`rounded-xl border p-3 text-sm ${
                      result.extraction.partial || result.extraction.scanned
                        ? "border-ai/40 bg-ai/10 text-ink"
                        : "border-good/40 bg-good/10 text-ink"}`}>
                      📄 {result.extraction.note ??
                        `Se leyeron ${result.extraction.pages_with_text} de ${result.extraction.pages_total} páginas · ${result.extraction.words} palabras.`}
                    </div>
                  )}
                  <ScorePanel r={result} />

                  {/* Enseñarle al detector: ¿acertó? corrígelo. */}
                  {analyzedText && (
                    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-paper p-3 text-sm">
                      <span className="text-muted">¿Sabes qué era en realidad? Enséñale:</span>
                      <button className="btn-ghost" onClick={() => sendFeedback("humano")}>
                        <ThumbsUp size={15} /> Era humano
                      </button>
                      <button className="btn-ghost" onClick={() => sendFeedback("ia")}>
                        <Bot size={15} /> Era IA
                      </button>
                      {feedback && <span className="text-accent-ink">{feedback}</span>}
                    </div>
                  )}

                  {result.id != null && (
                    <div className="flex justify-end">
                      <a href={api.reportUrl(result.id)} target="_blank" rel="noreferrer"
                        className="btn-ghost">
                        <Download size={15} /> Exportar reporte PDF
                      </a>
                    </div>
                  )}

                  {analyzedText && (
                    <HighlightedText text={analyzedText} segments={result.segments ?? []} />
                  )}

                  <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                    <SignalBars signals={result.ai_detection?.signals ?? []} />
                    <ExplanationPanel e={result.explanation} />
                  </div>

                  <Recommendations recs={result.recommendations ?? []} />
                  <Disclaimer text={result.explanation?.disclaimer ?? ""} />
                </ErrorBoundary>
              </motion.div>
            )}
          </AnimatePresence>

          {!result && !batch && (
            <Disclaimer text="Veraz ofrece estimaciones probabilísticas, no veredictos. Úsalo para revisar y mejorar tu texto, nunca como una acusación." />
          )}
        </main>

        <HistorySidebar
          items={history}
          onOpen={openHistory}
          onDelete={async (id) => {
            await api.deleteHistory(id);
            refresh();
            if (result?.id === id) setResult(null);
          }}
          onClear={async () => { await api.clearHistory(); refresh(); }}
        />
      </div>
      )}

      <footer className="py-10 text-center text-xs text-muted">
        Veraz · proyecto académico de detección estimada · ningún detector es 100 % confiable.
      </footer>
    </div>
  );
}
