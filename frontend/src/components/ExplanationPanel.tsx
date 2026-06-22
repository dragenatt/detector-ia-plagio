import { User, Cpu, Link2 } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { Explanation } from "../lib/api";

function Group({
  icon: Icon,
  title,
  items,
  token,
}: {
  icon: LucideIcon;
  title: string;
  items: string[];
  token: string;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="grid h-7 w-7 place-items-center rounded-lg"
          style={{ background: `rgb(var(${token}) / 0.15)`, color: `rgb(var(${token}))` }}>
          <Icon size={16} />
        </span>
        <h4 className="font-semibold text-ink">{title}</h4>
      </div>
      <ul className="space-y-1.5 pl-1 text-sm text-muted">
        {items.map((t, i) => (
          <li key={i} className="flex gap-2">
            <span aria-hidden>·</span>
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Explica el resultado en lenguaje simple, separado en tres tipos de señal. */
export function ExplanationPanel({ e }: { e: Explanation }) {
  return (
    <div className="card space-y-5 p-6">
      <h3 className="font-display text-lg font-semibold text-ink">¿Por qué este resultado?</h3>
      <Group icon={User} title="Señales humanas" items={e.human_signals} token="--good" />
      <Group icon={Cpu} title="Señales de IA" items={e.ai_signals} token="--ai" />
      <Group icon={Link2} title="Señales de plagio" items={e.plagiarism_signals} token="--source" />
    </div>
  );
}
