import { useRef, useState } from "react";
import { FileText, Upload, Loader2, ScanLine } from "lucide-react";

interface Props {
  onAnalyzeText: (text: string) => void;
  onAnalyzeFile: (file: File) => void;
  onAnalyzeBatch: (files: File[]) => void;
  loading: boolean;
}

type Tab = "paste" | "upload" | "batch";

export function InputPanel({ onAnalyzeText, onAnalyzeFile, onAnalyzeBatch, loading }: Props) {
  const [tab, setTab] = useState<Tab>("paste");
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const batchRef = useRef<HTMLInputElement>(null);
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  const pickFile = (f: File | null) => {
    if (!f) return;
    fileRef.current = f;
    setFileName(f.name);
  };

  const tabBtn = (id: Tab, label: string) => (
    <button
      onClick={() => setTab(id)}
      className={`rounded-full px-4 py-1.5 font-medium transition ${
        tab === id ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink"
      }`}
    >
      {label}
    </button>
  );

  return (
    <section className="card p-6 md:p-7">
      <div className="mb-4 inline-flex rounded-full border border-line bg-surface-2 p-1 text-sm">
        {tabBtn("paste", "Pegar texto")}
        {tabBtn("upload", "Subir archivo")}
        {tabBtn("batch", "Varios archivos")}
      </div>

      {tab === "batch" ? (
        <>
          <div
            onClick={() => batchRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              setBatchFiles(Array.from(e.dataTransfer.files ?? []));
            }}
            className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-line bg-paper px-6 py-10 text-center transition hover:border-accent"
          >
            <Upload size={26} className="text-accent" />
            <p className="text-sm font-medium text-ink">
              {batchFiles.length
                ? `${batchFiles.length} archivo(s) seleccionados`
                : "Arrastra varios archivos o haz clic para elegir"}
            </p>
            <p className="text-xs text-muted">
              PDF, DOCX, TXT · para comparar toda una clase (máx. 60)
            </p>
            <input ref={batchRef} type="file" accept=".pdf,.docx,.txt,.md" multiple hidden
              onChange={(e) => setBatchFiles(Array.from(e.target.files ?? []))} />
          </div>
          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-xs text-muted">
              {batchFiles.length} archivo(s) · se detecta también copia entre ellos
            </span>
            <button className="btn-primary" disabled={loading || batchFiles.length < 2}
              onClick={() => onAnalyzeBatch(batchFiles)}>
              {loading ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} />}
              {loading ? "Analizando…" : "Analizar lote"}
            </button>
          </div>
        </>
      ) : tab === "paste" ? (
        <>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={9}
            placeholder="Pega aquí el texto académico que quieres analizar…"
            className="w-full resize-y rounded-xl border border-line bg-paper p-4 text-[15px] leading-relaxed text-ink outline-none transition focus:border-accent"
          />
          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-xs text-muted">
              {words} palabras · {text.length} caracteres
            </span>
            <button className="btn-primary" disabled={loading || words < 5}
              onClick={() => onAnalyzeText(text)}>
              {loading ? <Loader2 size={16} className="animate-spin" /> : <ScanLine size={16} />}
              {loading ? "Analizando…" : "Analizar"}
            </button>
          </div>
        </>
      ) : (
        <>
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              pickFile(e.dataTransfer.files?.[0] ?? null);
            }}
            className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-line bg-paper px-6 py-10 text-center transition hover:border-accent"
          >
            <Upload size={26} className="text-accent" />
            <p className="text-sm font-medium text-ink">
              {fileName ?? "Arrastra un archivo o haz clic para elegir"}
            </p>
            <p className="text-xs text-muted">PDF, DOCX, TXT (máx. 5 MB)</p>
            <input ref={inputRef} type="file" accept=".pdf,.docx,.txt,.md" hidden
              onChange={(e) => pickFile(e.target.files?.[0] ?? null)} />
          </div>
          <div className="mt-3 flex justify-end">
            <button className="btn-primary" disabled={loading || !fileRef.current}
              onClick={() => fileRef.current && onAnalyzeFile(fileRef.current)}>
              {loading ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} />}
              {loading ? "Analizando…" : "Analizar archivo"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
