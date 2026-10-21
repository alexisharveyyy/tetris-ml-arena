// A group of toggle buttons where exactly one is on (aria-pressed).
interface Option<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

interface Props<T extends string> {
  label: string;
  options: Option<T>[];
  value: T;
  onChange: (v: T) => void;
  className?: string;
  /** Show the selected option in brackets (Terminal look). */
  bracket?: boolean;
}

export function Segmented<T extends string>({ label, options, value, onChange, className, bracket }: Props<T>) {
  return (
    <div role="group" aria-label={label} className={`seg${className ? ` ${className}` : ""}`}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className="seg__opt"
          aria-pressed={o.value === value}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
        >
          {bracket && o.value === value ? `[${o.label}]` : o.label}
        </button>
      ))}
    </div>
  );
}
