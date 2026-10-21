import type { KeyboardEvent } from "react";

interface Props<T extends string> {
  label: string;
  tabs: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  idPrefix: string;
  className?: string;
}

/** ARIA tabs: role=tablist, aria-selected, arrow keys move between tabs. */
export function Tabs<T extends string>({ label, tabs, value, onChange, idPrefix, className }: Props<T>) {
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = tabs.findIndex((t) => t.id === value);
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const next = tabs[(i + step + tabs.length) % tabs.length];
    onChange(next.id);
    document.getElementById(`${idPrefix}-tab-${next.id}`)?.focus();
  };
  return (
    <div role="tablist" aria-label={label} className={`tabs${className ? ` ${className}` : ""}`} onKeyDown={onKey}>
      {tabs.map((t) => (
        <button
          key={t.id}
          id={`${idPrefix}-tab-${t.id}`}
          role="tab"
          type="button"
          className="tabs__tab"
          aria-selected={t.id === value}
          aria-controls={`${idPrefix}-panel`}
          tabIndex={t.id === value ? 0 : -1}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
