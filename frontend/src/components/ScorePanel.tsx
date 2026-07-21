import { motion } from "framer-motion";
import { ShieldCheck, Sparkles, GitCompare } from "lucide-react";
import type { AnalysisResult } from "../lib/api";
import { ScoreRing } from "./ScoreRing";
import { DocumentHeatmap } from "./DocumentHeatmap";

const level = (v: number) => (v >= 66 ? "alto" : v >= 33 ? "medio" : "bajo");

/** Tarjeta principal: los tres puntajes + veredicto en lenguaje simple. */
export function ScorePanel({ r }: { r: AnalysisResult }) {
  const { originality, plagiarism, ai_probability } = r.scores;
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="card p-6 md:p-8"
    >
      {/* Veredicto arriba, en grande. */}
      <p className="mb-6 font-display text-2xl leading-snug text-ink">{r.explanation.summary}</p>

      <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
        <ScoreRing value={ai_probability} token="ai" label="Probabilidad de IA"
          level={level(ai_probability)}
          sub={r.ai_detection.used_model ? "heurística + modelo" : "heurística"} />
        <ScoreRing value={plagiarism} token="source" label="Similitud / plagio"
          level={r.plagiarism.has_corpus ? level(plagiarism) : undefined}
          sub={r.plagiarism.has_corpus ? "vs. corpus" : "sin corpus"} />
        <ScoreRing value={originality} token="good" label="Originalidad"
          level={level(originality)} sub="estimada" />
      </div>

      <DocumentHeatmap r={r} />

      <div className="mt-7 flex flex-wrap items-center gap-2">
        <span className="chip"><ShieldCheck size={14} /> Confianza: {r.confidence.level}</span>
        {r.plagiarism.has_corpus && (
          <span className="chip"><GitCompare size={14} /> Similitud temática: {r.plagiarism.topical_similarity}%</span>
        )}
        <span className="chip"><Sparkles size={14} /> {r.meta.word_count} palabras</span>
      </div>

      <p className="mt-4 text-sm text-muted">{r.confidence.reason}</p>
    </motion.section>
  );
}
