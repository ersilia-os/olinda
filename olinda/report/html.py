"""A single-page report tying the figures and numbers together.

Deliberately plain and publication-oriented — a white page, the system font stack, a sticky
section nav and a responsive grid of figure cards — matching the reports zairachem writes. The
data colours are not restated here: the swatches in the colour key come from
:func:`olinda.style.hexcol`, so a hue cannot mean one thing in a figure and another on the page.

Light only, on purpose. stylia renders every figure on an opaque white canvas, so a dark page
would be a row of glaring white rectangles.

Figures are referenced from the sibling ``png/`` folder rather than base64-inlined, which keeps
``report.html`` a few kilobytes instead of duplicating every image into it. The page therefore only
works next to its folders — that is the intended unit, and moving the whole directory is fine.
"""

from __future__ import annotations

import html
from pathlib import Path

from olinda.style import hexcol

# What the colours mean, stated once for the whole page so no figure needs its own legend.
_KEY = (
    ("model", "olinda prediction"),
    ("teacher", "Teacher value"),
    ("active", "Active"),
    ("inactive", "Inactive"),
)

_CSS = """
:root {
  color-scheme: light;
  --fg: #1f2328; --muted: #6e7781; --line: #e6e8eb; --bg: #ffffff; --soft: #f6f8fa;
  --link: #0969da; --nav: 210px;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.shell { display: flex; align-items: flex-start; gap: 2rem; max-width: 78rem; margin: 0 auto; }
nav { position: sticky; top: 0; flex: 0 0 var(--nav); width: var(--nav); height: 100vh;
      padding: 2.5rem 0 2rem 1.5rem; overflow-y: auto; }
nav .brand { font-weight: 600; font-size: 1rem; margin-bottom: .15rem; }
nav .stamp { color: var(--muted); font-size: .74rem; margin-bottom: 1.4rem; }
nav a { display: block; padding: .3rem .55rem; margin: .1rem 0 .1rem -.55rem; border-radius: 7px;
        color: var(--fg); text-decoration: none; font-size: .82rem; }
nav a:hover { background: var(--soft); }
main { flex: 1 1 auto; min-width: 0; max-width: 57.5rem; padding: 2.5rem 1.5rem 5rem; }
h1 { font-size: 1.5rem; margin: 0 0 .2rem; letter-spacing: -.01em; }
h2 { font-size: 1.06rem; margin: 2.6rem 0 .8rem; padding-bottom: .35rem;
     border-bottom: 1px solid var(--line); scroll-margin-top: 1rem; }
.sub { color: var(--muted); margin: 0 0 1.6rem; }
.stats { display: flex; flex-wrap: wrap; gap: .6rem; margin: 0 0 1.6rem; padding: 0; list-style: none; }
.stats li { flex: 1 1 8rem; border: 1px solid var(--line); border-radius: 10px; padding: .6rem .8rem; }
.stats .k { color: var(--muted); font-size: .72rem; }
.stats .v { font-size: 1.15rem; font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-variant-numeric: tabular-nums; }
.key { display: flex; flex-wrap: wrap; gap: .9rem; margin: 0 0 1.4rem; color: var(--muted);
       font-size: .78rem; }
.key span { display: inline-flex; align-items: center; gap: .35rem; }
.key i { width: .7rem; height: .7rem; border-radius: 3px; border: 1px solid rgba(0,0,0,.08); }
/* Cards are sized so a figure renders near its true size: a (2,2) panel is 2.36 in at 200 dpi,
   and stylia's print format sets 5-6 pt type, which stops being legible much below this. */
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(24rem, 1fr)); }
/* align-self:start stops a short card being stretched to its neighbour's height, which would
   leave a block of dead space under its caption. */
figure { margin: 0; align-self: start; display: flex; flex-direction: column;
         border: 1px solid var(--line); border-radius: 12px; padding: .9rem;
         transition: box-shadow .18s ease, transform .18s ease; }
figure:hover { box-shadow: 0 8px 24px rgba(27,31,36,.09); transform: translateY(-2px); }
/* Each image box takes its figure's own aspect ratio, so a wide panel is not letterboxed inside a
   square card and a square one is not cropped. Cards in a row then differ in height, which the
   grid handles. */
figure img { width: 100%; height: auto; aspect-ratio: var(--ar, 1); object-fit: contain; display: block; }
figure.wide { grid-column: 1 / -1; }
figcaption { margin-top: .6rem; }
figcaption .t { display: flex; justify-content: space-between; gap: .5rem; font-weight: 600;
                font-size: .84rem; }
figcaption .c { color: var(--muted); font-size: .76rem; margin-top: .15rem; }
figcaption a { color: var(--link); text-decoration: none; font-weight: 400; font-size: .78rem; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; }
th, td { text-align: right; padding: .4rem .6rem; border-bottom: 1px solid var(--line);
         font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--muted); font-weight: 600; }
.wrap { overflow-x: auto; }
.note { background: var(--soft); border: 1px solid var(--line); border-radius: 8px;
        padding: .7rem 1rem; margin: 1rem 0; font-size: .84rem; }
details { margin-top: 1rem; font-size: .84rem; color: var(--muted); }
summary { cursor: pointer; color: var(--link); }
@media (max-width: 900px) {
  .shell { display: block; }
  nav { position: static; width: auto; height: auto; padding: 1.5rem 1.5rem 0; }
  nav a { display: inline-block; margin-right: .4rem; }
}
@media print {
  nav { display: none; }
  figure { break-inside: avoid; box-shadow: none; }
  main { max-width: none; }
}
"""


def _table(rows, headers) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _figures(figs) -> str:
    if not figs:
        return ""
    cards = []
    for f in figs:
        # Titles and captions come from the plots' own registry, so a figure is described the same way
        # wherever it appears; older reports without them fall back to the file name.
        title = html.escape(f.get("title") or f["name"].replace("_", " "))
        text = html.escape(f.get("caption", ""))
        aspect = float(f.get("aspect", 1) or 1)
        # A panel twice as wide as it is tall gets its own row; squeezed into a half-width card it
        # would render at half the size of its neighbours.
        cls = ' class="wide"' if aspect >= 2 else ""
        cards.append(
            # Not lazy-loaded: a report is a dozen small PNGs, and deferring them means they are simply
            # absent when the page is printed, exported to PDF or captured full-height.
            f'<figure{cls}><img src="{f["png"]}" alt="{title}" style="--ar:{aspect}">'
            f'<figcaption><span class="t">{title}<a href="{f["pdf"]}">PDF</a></span>'
            f'<span class="c">{text}</span></figcaption></figure>'
        )
    return f'<div class="grid">{"".join(cards)}</div>'


def _stats(items) -> str:
    """The headline numbers, so the answer is visible before any table is read."""
    items = [(k, v) for k, v in items if v is not None]
    if not items:
        return ""
    cells = "".join(
        f'<li><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></li>'
        for k, v in items
    )
    return f'<ul class="stats">{cells}</ul>'


def _colour_key() -> str:
    swatches = "".join(
        f'<span><i style="background:{hexcol(role)}"></i>{html.escape(label)}</span>'
        for role, label in _KEY
    )
    return f'<div class="key">{swatches}</div>'


def _section(slug: str, title: str, *blocks: str) -> str:
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    return f'<h2 id="{slug}">{html.escape(title)}</h2>' + "".join(blocks)


def write_report(out_dir: str | Path, report: dict) -> Path:
    """Render ``report.html`` from the structure :func:`olinda.report.validate_model` assembles."""
    out_dir = Path(out_dir)
    meta = report["model"]
    title = f"olinda validation · {Path(meta['path']).name}"

    config = _table(
        [
            ("Model", meta["path"]),
            ("Tasks", ", ".join(meta["columns"])),
            ("Trained", meta.get("trained_at") or "—"),
            ("olinda", meta.get("olinda_version") or "—"),
            ("RDKit", meta.get("rdkit_version") or "—"),
            ("Backend", meta.get("backend") or "—"),
            ("Hard-label head", "yes" if meta.get("has_hard") else "no"),
        ],
        ("", "Value"),
    )

    dataset_rows = []
    for key, label in (("soft", "Soft labels"), ("hard", "Hard labels")):
        d = report.get(key)
        if d:
            dataset_rows.append(
                (label, d["file"], f"{d['n']:,}", f"{d['n_unparseable']:,}")
            )
    dataset = _table(dataset_rows, ("", "File", "Compounds", "Unparseable"))

    notes = "".join(
        f'<div class="note">{html.escape(n)}</div>' for n in report.get("notes", [])
    )

    perf_rows = []
    for task, m in (report.get("soft") or {}).get("metrics", {}).items():
        perf_rows.append(
            (
                task,
                f"{m['n']:,}",
                f"{m['r2']:+.4f}",
                f"{m['pearson']:+.4f}",
                f"{m['spearman']:+.4f}",
                f"{m['rmse']:.5f}",
                f"{m['mae']:.5f}",
            )
        )
    perf = _table(perf_rows, ("Task", "n", "R²", "Pearson", "Spearman", "RMSE", "MAE"))

    rank_rows = []
    for task, m in (report.get("hard") or {}).get("metrics", {}).items():
        enr = m.get("enrichment", {})
        rank_rows.append(
            (
                task,
                f"{m['n']:,}",
                f"{m['n_positive']:,}",
                f"{m['hit_rate']:.3f}",
                _fmt(m["auroc"]),
                _fmt(m["average_precision"]),
                *[
                    _fmt(enr.get(k), "{:.1f}x")
                    for k in ("top_0.01", "top_0.05", "top_0.1")
                ],
            )
        )
    rank = _table(
        rank_rows,
        ("Task", "n", "Actives", "Hit rate", "AUROC", "AP", "EF 1%", "EF 5%", "EF 10%"),
    )

    figs = report.get("figures", {})
    sections = [
        ("config", "Configuration", (config,)),
        ("dataset", "Dataset", (dataset,)),
        (
            "teacher",
            "Agreement with the teacher",
            (perf, _colour_key(), _figures(figs.get("soft", []))),
        ),
        (
            "ranking",
            "Ranking against measured labels",
            (rank, _figures(figs.get("hard", []))),
        ),
        ("internals", "Model internals", (_figures(figs.get("internals", [])),)),
    ]
    body = "".join(_section(slug, name, *blocks) for slug, name, blocks in sections)
    nav_links = "".join(
        f'<a href="#{slug}">{html.escape(name)}</a>'
        for slug, name, blocks in sections
        if any(b for b in blocks)
    )

    page = f"""<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_CSS}</style>
<div class="shell">
<nav>
  <div class="brand">olinda</div>
  <div class="stamp">{html.escape(report.get("generated", ""))}</div>
  {nav_links}
</nav>
<main>
<h1>{html.escape(title)}</h1>
<p class="sub">Scored against the labels below — every number and figure on this page comes from that
comparison, not from the run that trained the model.</p>
{_stats(_headline(report))}
{notes}
{body}
<details><summary>How to read these figures</summary>
Metrics are computed once, in <code>olinda.metrics</code>, and reused by both the figures and
<code>metrics.json</code>, so a number on a plot and the same number in the file cannot disagree.
Density panels are binned and shaded on a log scale, so a single crowded bin does not flatten the
rest. Each figure links its vector PDF for publication use.</details>
</main>
</div>
"""
    from olinda.report import REPORT_NAME

    path = out_dir / REPORT_NAME
    path.write_text(page, encoding="utf-8")
    return path


def _headline(report: dict) -> list[tuple[str, str | None]]:
    """The two or three numbers worth reading before anything else."""
    items: list[tuple[str, str | None]] = []
    soft = (report.get("soft") or {}).get("metrics", {})
    hard = (report.get("hard") or {}).get("metrics", {})
    if soft:
        first = next(iter(soft.values()))
        items.append(("Compounds scored", f"{first['n']:,}"))
        items.append(("R²", _fmt(first["r2"], "{:+.3f}")))
        items.append(("Spearman ρ", _fmt(first["spearman"], "{:+.3f}")))
    if hard:
        first = next(iter(hard.values()))
        if not soft:
            items.append(("Compounds scored", f"{first['n']:,}"))
        items.append(("AUROC", _fmt(first["auroc"])))
        items.append(("Actives", f"{first['n_positive']:,}"))
    return items


def _fmt(value, spec: str = "{:.3f}") -> str:
    """Format a metric, showing an undefined one as an em dash rather than ``nan``."""
    if value is None:
        return "—"
    try:
        return "—" if value != value else spec.format(value)  # NaN != NaN
    except (TypeError, ValueError):
        return str(value)
