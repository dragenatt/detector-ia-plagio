import type { Signal } from "../lib/api";

const SEV_TOKEN: Record<string, string> = {
  alta: "--ai",
  media: "--ai",
  baja: "--good",
};

/** Desglose de las señales de IA como barras. Cada barra es una pista; el
 *  texto deja claro que ninguna por sí sola es una prueba. */
export function SignalBars({ signals }: { signals: Signal[] }) {
  return (
    <div className="card p-6">
      <h3 className="font-display text-lg font-semibold text-ink">Señales analizadas (IA)</h3>
      <p className="mb-4 mt-1 text-sm text-muted">
        Cada barra es una pista independiente. Ninguna, por sí sola, es una prueba.
      </p>
      <div className="space-y-3.5">
        {signals.map((s) => (
          <div key={s.key}>
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium text-ink">{s.label}</span>
              <span className="font-mono text-xs text-muted">{s.score}%</span>
            </div>
            <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{ width: `${s.score}%`, background: `rgb(var(${SEV_TOKEN[s.severity]}))` }}
              />
            </div>
            <p className="mt-1 text-xs text-muted">{s.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
