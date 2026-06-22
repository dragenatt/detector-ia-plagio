import { useEffect, useState } from "react";

const TOKEN: Record<string, string> = {
  good: "--good",
  source: "--source",
  ai: "--ai",
  accent: "--accent",
};

interface Props {
  value: number;
  label: string;
  sub?: string;
  token?: "good" | "source" | "ai" | "accent";
  size?: number;
}

/** Anillo de puntaje con conteo animado y arco que se dibuja al aparecer. */
export function ScoreRing({ value, label, sub, token = "accent", size = 138 }: Props) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const dur = 900;
    const tick = (t: number) => {
      const p = Math.min((t - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(Math.round(value * eased));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  const stroke = 12;
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - value / 100);
  const color = `rgb(var(${TOKEN[token]}))`;

  return (
    <div className="flex flex-col items-center gap-2.5">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none"
            stroke="rgb(var(--line))" strokeWidth={stroke} />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none"
            stroke={color} strokeWidth={stroke} strokeLinecap="round"
            strokeDasharray={circ} strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.22,1,0.36,1)" }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-display text-4xl font-semibold tabular-nums" style={{ color }}>
            {display}
            <span className="text-xl">%</span>
          </span>
        </div>
      </div>
      <div className="text-center">
        <div className="text-sm font-semibold text-ink">{label}</div>
        {sub && <div className="text-xs text-muted">{sub}</div>}
      </div>
    </div>
  );
}
