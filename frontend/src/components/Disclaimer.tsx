import { Info } from "lucide-react";

/** Aviso permanente: los resultados son estimaciones, no acusaciones. */
export function Disclaimer({ text }: { text: string }) {
  return (
    <div className="flex gap-3 rounded-2xl border border-line bg-surface-2/60 p-4 text-sm text-muted">
      <Info size={18} className="mt-0.5 shrink-0 text-accent" />
      <p>{text}</p>
    </div>
  );
}
