import type { PlayerSnap } from "../api/types";
import { fallbackCausesText, modelFallbackText } from "../lib/copy";

/** How many of the LLM agent's moves fell back to the GA heuristic, shown on its board. */
export function ModelFallbacks({ player, className }: { player: PlayerSnap; className: string }) {
  const text = modelFallbackText(player);
  if (!text) return null;
  const causes = fallbackCausesText(player);
  return (
    <p className={className} data-testid="model-fallbacks" title={causes || undefined}>
      {text}
      {causes ? <span className="visually-hidden"> ({causes})</span> : null}
    </p>
  );
}
