import type { ReactNode } from "react";

/** Back link, title and optional controls, shared by the screens without mockups. */
export function PageHead({ title, back = "#/", backLabel = "Home", children }: { title: string; back?: string; backLabel?: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <a className="link link--strong" href={back}>
        {backLabel}
      </a>
      <h1 className="page-title">{title}</h1>
      {children ? <div className="page-head__extra">{children}</div> : null}
    </header>
  );
}
