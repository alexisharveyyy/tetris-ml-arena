import type { ReactNode } from "react";
import { KeyCap } from "./KeyCap";

interface Props {
  children: ReactNode;
  variant?: "primary" | "secondary" | "link" | "ghost";
  /** Keyboard shortcut shown on a key cap inside the button. */
  shortcut?: string;
  href?: string;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  pressed?: boolean;
  label?: string;
  type?: "button" | "submit";
}

/** A real <button>, or an <a> when it navigates. */
export function Button({ children, variant = "primary", shortcut, href, onClick, disabled, className, pressed, label, type = "button" }: Props) {
  const cls = `btn btn--${variant}${className ? ` ${className}` : ""}`;
  const inner = (
    <span className="btn__inner">
      <span className="btn__label">{children}</span>
      {shortcut ? <KeyCap name={shortcut} /> : null}
    </span>
  );
  if (href && !disabled) {
    return (
      <a className={cls} href={href} aria-label={label} onClick={onClick}>
        {inner}
      </a>
    );
  }
  return (
    <button type={type} className={cls} onClick={onClick} disabled={disabled} aria-pressed={pressed} aria-label={label}>
      {inner}
    </button>
  );
}
