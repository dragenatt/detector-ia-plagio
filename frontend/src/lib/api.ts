// Cliente de la API de Veraz + tipos del resultado de análisis.

const BASE = import.meta.env.VITE_API_URL ?? "";

// --- Tipos que reflejan la salida del motor (backend/app/analysis/engine.py) ---
export type Category = "original" | "ia" | "plagio" | "mixto";
export type Severity = "baja" | "media" | "alta";

export interface Scores {
  originality: number;
  plagiarism: number;
  ai_probability: number;
}
export interface Confidence {
  level: string;
  reason: string;
}
export interface Signal {
  key: string;
  label: string;
  value: number;
  score: number;
  weight: number;
  severity: Severity;
  description: string;
}
export interface AiRegion {
  first_sentence: number;
  last_sentence: number;
  sentences: number[];
  start: number;
  end: number;
  score: number;
  words: number;
}
export interface AiDetection {
  probability: number;
  heuristic_probability: number;
  model_probability: number | null;
  used_model: boolean;
  signals: Signal[];
  regions?: AiRegion[];
  regions_summary?: { count: number; coverage: number };
}
export interface PlagMatch {
  source: string;
  overlap: number;
}
export interface FlaggedSentence {
  index: number;
  text: string;
  start: number;
  end: number;
  source: string;
  overlap: number;
}
export interface Plagiarism {
  similarity: number;
  topical_similarity: number;
  matches: PlagMatch[];
  flagged_sentences: FlaggedSentence[];
  has_corpus: boolean;
  note: string | null;
}
export interface Evidence {
  kind: "copia" | "generica" | "conector" | "atenuador" | "tipografia";
  label: string;
  text: string;
  start: number;
  end: number;
  source_fragment?: string;
  matched_words?: number;
}
export interface Segment {
  index: number;
  text: string;
  start: number;
  end: number;
  category: Category;
  ai_score: number;
  plagiarism_overlap: number;
  plagiarism_source?: string | null;
  reason: string;
  evidence?: Evidence[];
}
export interface Explanation {
  summary: string;
  human_signals: string[];
  ai_signals: string[];
  plagiarism_signals: string[];
  disclaimer: string;
}
export interface Recommendation {
  priority: Severity;
  title: string;
  detail: string;
}
export interface Meta {
  word_count: number;
  sentence_count: number;
  char_count: number;
  model_used: boolean;
  analyzed_at: string;
}
export interface Extraction {
  words: number;
  pages_total: number | null;
  pages_with_text: number | null;
  scanned: boolean | null;
  partial: boolean;
  note: string | null;
}
export interface AnalysisResult {
  id: number | null;
  analyzed_text?: string;
  extraction?: Extraction;
  scores: Scores;
  confidence: Confidence;
  ai_detection: AiDetection;
  plagiarism: Plagiarism;
  segments: Segment[];
  explanation: Explanation;
  recommendations: Recommendation[];
  metrics: Record<string, number>;
  meta: Meta;
  extracted_text?: string;
}
export interface HistoryItem {
  id: number;
  created_at: string;
  title: string;
  word_count: number;
  originality: number;
  plagiarism: number;
  ai_probability: number;
  summary: string;
}
export interface ModelStatus {
  loaded: boolean;
  meta: Record<string, unknown> | null;
}
export interface CorpusCounts {
  by_label: Record<string, { total: number; user: number }>;
  human_axis: number;
  ai_axis: number;
  balanced: number;
}
export interface TrainReport {
  trained: boolean;
  message: string;
  n_human?: number;
  n_ai?: number;
  cross_validation?: { f1_mean: number; f1_std: number; f1_folds: number[]; accuracy: number };
  calibration_after?: { ece: number | null; brier: number | null };
}
export interface BatchRow {
  name: string;
  ok: boolean;
  error?: string | null;
  words?: number;
  ai_probability?: number;
  plagiarism?: number;
  originality?: number;
  confidence?: string;
  cross_match?: { source: string; overlap: number } | null;
  warning?: string | null;
}
export interface BatchResult {
  count: number;
  failed: number;
  rows: BatchRow[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* respuesta sin JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  analyzeText: (text: string, opts?: { title?: string; useModel?: boolean }) =>
    req<AnalysisResult>("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        title: opts?.title,
        use_model: opts?.useModel ?? true,
      }),
    }),

  analyzeFile: (file: File, useModel = true) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<AnalysisResult>(
      `/api/analyze/file?use_model=${useModel}`,
      { method: "POST", body: fd }
    );
  },

  analyzeBatch: (files: File[], useModel = true) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    return req<BatchResult>(
      `/api/analyze/batch?use_model=${useModel}`,
      { method: "POST", body: fd }
    );
  },

  history: () => req<HistoryItem[]>("/api/history"),
  historyDetail: (id: number) =>
    req<{ payload: AnalysisResult; text: string; title: string }>(
      `/api/history/${id}`
    ),
  deleteHistory: (id: number) =>
    req(`/api/history/${id}`, { method: "DELETE" }),
  clearHistory: () => req("/api/history", { method: "DELETE" }),

  reportUrl: (id: number) => `${BASE}/api/report/${id}`,

  modelStatus: () => req<ModelStatus>("/api/model"),
  train: () => req<TrainReport>("/api/train", { method: "POST" }),

  corpus: () => req<CorpusCounts>("/api/corpus"),
  addExample: (text: string, label: "humano" | "ia" | "mixto") =>
    req<{ saved: string; words: number; note: string | null; counts: CorpusCounts }>(
      "/api/corpus/example",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, label }),
      }
    ),
  undoLastExample: () =>
    req<{ deleted: string; label: string; counts: CorpusCounts }>(
      "/api/corpus/last",
      { method: "DELETE" }
    ),

  references: () =>
    req<{ uploaded: { id: number; name: string }[]; from_folders: string[] }>(
      "/api/references"
    ),
  addReference: (name: string, text: string) =>
    req("/api/references", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, text }),
    }),
  addReferenceFile: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req("/api/references/file", { method: "POST", body: fd });
  },
  deleteReference: (id: number) =>
    req(`/api/references/${id}`, { method: "DELETE" }),
};
