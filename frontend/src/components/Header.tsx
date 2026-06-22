import { Moon, Sun, RefreshCw, Sparkles } from "lucide-react";
import type { ModelStatus } from "../lib/api";

interface Props {
  theme: string;
  onToggleTheme: () => void;
  model: ModelStatus | null;
  onTrain: () => void;
  training: boolean;
}

export function Header({ theme, onToggleTheme, model, onTrain, training }: Props) {
  const meta = (model?.meta ?? {}) as Record<string, unknown>;
  const acc = typeof meta.holdout_accuracy === "number"
    ? (meta.holdout_accuracy as number)
    : typeof meta.train_accuracy === "number"
    ? (meta.train_accuracy as number)
    : null;

  return (
    <header className="flex flex-wrap items-center justify-between gap-4 py-6">
      <div className="flex items-center gap-3">
        <div className="grid h-11 w-11 place-items-center rounded-2xl bg-accent text-accent-ink shadow-soft">
          <svg width="22" height="22" viewBox="0 0 100 100">
            <path d="M26 52l16 16 32-36" fill="none" stroke="currentColor"
              strokeWidth="11" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div>
          <h1 className="font-display text-2xl font-semibold leading-none">Veraz</h1>
          <p className="mt-1 text-xs text-muted">Estimación de originalidad · plagio · IA</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="chip">
          <Sparkles size={14} />
          {model?.loaded
            ? `Modelo activo${acc != null ? ` · ${Math.round(acc * 100)}%` : ""}`
            : "Solo heurísticas"}
        </span>
        <button className="btn-ghost" onClick={onTrain} disabled={training}>
          <RefreshCw size={15} className={training ? "animate-spin" : ""} />
          {training ? "Entrenando…" : "Reentrenar"}
        </button>
        <button className="btn-ghost !px-3" onClick={onToggleTheme} aria-label="Cambiar tema">
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
}
