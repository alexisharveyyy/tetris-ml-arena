// Broadcast's skewed plate: the box is skewed, its contents are counter-skewed so text stays upright.
import type { CSSProperties, ElementType, ReactNode } from "react";

interface PlateProps {
  as?: ElementType;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
  [key: string]: unknown;
}

export function Plate({ as: Tag = "div", className, children, ...rest }: PlateProps) {
  return (
    <Tag className={`plate${className ? ` ${className}` : ""}`} {...rest}>
      <span className="plate__in">{children}</span>
    </Tag>
  );
}

export function Ticker({ label, lines }: { label: string; lines: string[] }) {
  return (
    <section className="ticker" aria-label={label}>
      <span className="ticker__label">{label}</span>
      <ul className="ticker__lines" aria-live="polite">
        {lines.map((l, i) => (
          <li key={`${i}-${l}`}>{l}</li>
        ))}
      </ul>
    </section>
  );
}
