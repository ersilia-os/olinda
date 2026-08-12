"""A single-page report tying the figures and numbers together.

Figures are referenced from the sibling ``png/`` folder rather than base64-inlined, which keeps
``report.html`` a few kilobytes instead of duplicating every image into it. The page therefore only
works next to its folders — that is the intended unit, and moving the whole directory is fine.
"""

from __future__ import annotations

import html
from pathlib import Path

_CSS = """
:root {
  --ink: #201a26; --muted: #6b6478; --rule: #e6e2ea; --bg: #ffffff; --panel: #faf8fb;
  --accent: #50285a;
}
@media (prefers-color-scheme: dark) {
  :root { --ink: #ece8f0; --muted: #a49dae; --rule: #322b3a; --bg: #17141b; --panel: #1e1a24;
          --accent: #aa96fa; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 1.5rem 4rem; background: var(--bg); color: var(--ink);
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; padding-bottom: .35rem;
     border-bottom: 1px solid var(--rule); color: var(--accent); }
.sub { color: var(--muted); margin: 0 0 2rem; }
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); }
figure { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
         padding: .75rem; }
figure img { width: 100%; height: auto; display: block; border-radius: 6px; }
figcaption { color: var(--muted); font-size: .8rem; margin-top: .5rem; display: flex;
             justify-content: space-between; gap: .5rem; }
figcaption a { color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: .87rem; }
th, td { text-align: right; padding: .4rem .6rem; border-bottom: 1px solid var(--rule);
         font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--muted); font-weight: 600; }
.wrap { overflow-x: auto; }
.note { background: var(--panel); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
        padding: .7rem 1rem; margin: 1rem 0; color: var(--muted); font-size: .88rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }
"""


def _table(rows, headers) -> str:
  if not rows:
    return ""
  head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
  body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
  return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _figures(figs) -> str:
  if not figs:
    return ""
  cards = []
  for f in figs:
    label = html.escape(f["name"].replace("_", " "))
    cards.append(
      f'<figure><img src="{f["png"]}" alt="{label}">'
      f'<figcaption><span>{label}</span><a href="{f["pdf"]}">pdf</a></figcaption></figure>'
    )
  return f'<div class="grid">{"".join(cards)}</div>'


def _section(title: str, *blocks: str) -> str:
  blocks = [b for b in blocks if b]
  if not blocks:
    return ""
  return f"<h2>{html.escape(title)}</h2>" + "".join(blocks)


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
      ("Ground-truth head", "yes" if meta.get("has_ground_truth") else "no"),
    ],
    ("", "Value"),
  )

  dataset_rows = []
  for key, label in (("soft", "Soft labels"), ("hard", "Hard labels")):
    d = report.get(key)
    if d:
      dataset_rows.append((label, d["file"], f"{d['n']:,}", f"{d['n_unparseable']:,}"))
  dataset = _table(dataset_rows, ("", "File", "Compounds", "Unparseable"))

  notes = "".join(f'<div class="note">{html.escape(n)}</div>' for n in report.get("notes", []))

  perf_rows = []
  for task, m in (report.get("soft") or {}).get("metrics", {}).items():
    perf_rows.append((
      task,
      f"{m['n']:,}",
      f"{m['r2']:+.4f}",
      f"{m['pearson']:+.4f}",
      f"{m['spearman']:+.4f}",
      f"{m['rmse']:.5f}",
      f"{m['mae']:.5f}",
    ))
  perf = _table(perf_rows, ("Task", "n", "R²", "Pearson", "Spearman", "RMSE", "MAE"))

  rank_rows = []
  for task, m in (report.get("hard") or {}).get("metrics", {}).items():
    enr = m.get("enrichment", {})
    rank_rows.append((
      task,
      f"{m['n']:,}",
      f"{m['n_positive']:,}",
      f"{m['hit_rate']:.3f}",
      _fmt(m["auroc"]),
      _fmt(m["average_precision"]),
      *[_fmt(enr.get(k), "{:.1f}x") for k in ("top_0.01", "top_0.05", "top_0.1")],
    ))
  rank = _table(rank_rows, ("Task", "n", "Actives", "Hit rate", "AUROC", "AP", "EF 1%", "EF 5%", "EF 10%"))

  figs = report.get("figures", {})
  page = f"""<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_CSS}</style>
<main>
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(report.get("generated", ""))}</p>
{notes}
{_section("Configuration", config)}
{_section("Dataset", dataset)}
{_section("Agreement with the teacher", perf, _figures(figs.get("soft", [])))}
{_section("Ranking against measured labels", rank, _figures(figs.get("hard", [])))}
{_section("Model internals", _figures(figs.get("internals", [])))}
</main>
"""
  path = out_dir / "report.html"
  path.write_text(page, encoding="utf-8")
  return path


def _fmt(value, spec: str = "{:.3f}") -> str:
  """Format a metric, showing an undefined one as an em dash rather than ``nan``."""
  if value is None:
    return "—"
  try:
    return "—" if value != value else spec.format(value)  # NaN != NaN
  except (TypeError, ValueError):
    return str(value)
