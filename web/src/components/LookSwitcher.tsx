import type { LookId } from "../api/types";
import type { LookOption } from "../screens/models";

interface Props {
  looks: LookOption[];
  look: LookId;
  setLook: (id: LookId) => void;
  /** Render labels like "[A] TETROMINO" (Terminal). */
  format?: (o: LookOption, i: number) => string;
}

export function LookSwitcher({ looks, look, setLook, format }: Props) {
  return (
    <div className="look-switcher" role="group" aria-label="Look">
      <span className="look-switcher__label" aria-hidden="true">
        Look
      </span>
      {looks.map((o, i) => (
        <button key={o.id} type="button" className="look-switcher__opt" aria-pressed={o.id === look} onClick={() => setLook(o.id)}>
          {format ? format(o, i) : o.label}
        </button>
      ))}
    </div>
  );
}
