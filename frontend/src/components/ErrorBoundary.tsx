import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: ReactNode;
  /** Texto para el botón de reinicio; si se pasa onReset, se usa esa acción. */
  onReset?: () => void;
}
interface State {
  error: Error | null;
}

/** Captura cualquier error de render de sus hijos y muestra un mensaje en vez
 *  de dejar la pantalla en blanco. Evita que un análisis con datos raros
 *  "rompa" toda la aplicación. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Queda en la consola del navegador para diagnóstico.
    console.error("Veraz UI error:", error, info.componentStack);
  }

  reset = () => {
    this.setState({ error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.error) {
      return (
        <div className="card border-ai/40 bg-ai/10 p-6">
          <div className="mb-2 flex items-center gap-2 font-display text-lg font-semibold text-ink">
            <AlertTriangle size={18} className="text-ai" />
            Algo salió mal al mostrar esto
          </div>
          <p className="mb-4 text-sm text-muted">
            Ocurrió un error al dibujar esta parte de la interfaz, pero la
            aplicación sigue funcionando. Puedes reintentar o analizar otro
            texto. Detalle técnico:{" "}
            <code className="text-xs">{this.state.error.message}</code>
          </p>
          <button className="btn-primary" onClick={this.reset}>
            Reintentar
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
