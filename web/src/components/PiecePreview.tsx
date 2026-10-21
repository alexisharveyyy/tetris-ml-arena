// A queue or hold piece (board colors), drawn with DOM cells.
import { useFitCell } from "../lib/fit";
import { PREVIEW_CELLS } from "../lib/pieces";
import { useApp } from "../state/app";

interface Props {
  letter: string | null;
  cell?: number;
  color?: string;
  dim?: boolean;
}

export function PiecePreview({ letter, cell: baseCell = 14, color, dim }: Props) {
  const { theme, look } = useApp();
  const cell = useFitCell(baseCell, 3);
  if (!letter) return <span className="piece-preview piece-preview--empty" aria-hidden="true" />;
  const cells = PREVIEW_CELLS[letter];
  if (look === "terminal") {
    const rows = [0, 1].map((r) =>
      [0, 1, 2, 3].map((c) => (cells.some(([rr, cc]) => rr === r && cc === c) ? "[]" : "  ")).join("").trimEnd(),
    );
    return (
      <span className="piece-preview piece-preview--text" aria-hidden="true">
        {rows.filter((r) => r).join("\n")}
      </span>
    );
  }
  const cols = Math.max(...cells.map(([, c]) => c)) + 1;
  const rows = Math.max(...cells.map(([r]) => r)) + 1;
  const fill = color ?? theme.pieceColors[letter];
  return (
    <span
      className={`piece-preview${dim ? " piece-preview--dim" : ""}`}
      aria-hidden="true"
      style={{ display: "inline-grid", gridTemplateColumns: `repeat(${cols}, ${cell}px)`, gridTemplateRows: `repeat(${rows}, ${cell}px)`, gap: 1 }}
    >
      {cells.map(([r, c]) => (
        <span key={`${r}-${c}`} className="piece-preview__cell" style={{ gridRow: r + 1, gridColumn: c + 1, background: fill }} />
      ))}
    </span>
  );
}
