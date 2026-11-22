"""Write the CPU demo's browser evidence as a portable HTML report."""

from html import escape


_STYLE = """
:root {
  color-scheme: light;
  color: #182e28;
  background: #eef4ee;
  font-family: system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1056px; margin: auto; padding: clamp(20px, 5vw, 64px); }
header { padding-bottom: 24px; }
header p, .note { color: #465f56; max-width: 65ch; line-height: 1.55; }
h1 { font-size: clamp(32px, 6vw, 58px); line-height: 1.06; margin: 12px 0; letter-spacing: -.035em; }
h2 { font-size: 23px; }
a { color: #146b55; text-underline-offset: .2em; }
a:focus-visible, summary:focus-visible { outline: 2px solid #146b55; outline-offset: 4px; }
.metrics, .pair { display: grid; gap: 14px; }
.metrics { grid-template-columns: repeat(auto-fit, minmax(min(150px, 100%), 1fr)); margin: 0 0 32px; }
.metrics div, details { background: #f9fcf9; border: 1px solid #cfdfd3; border-radius: 14px; }
.metrics div { display: flex; flex-direction: column; padding: 16px; }
.metrics dt { color: #465f56; font-size: 14px; line-height: 1.3; }
.metrics dd { margin: auto 0 0; padding-top: 4px; font-size: 26px; font-weight: 700; color: #146b55; }
details { margin-bottom: 10px; padding: 14px 16px; }
summary { cursor: pointer; font-weight: 650; }
.pair { grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); margin-top: 14px; }
figure { margin: 0; }
img { display: block; width: 100%; aspect-ratio: 5 / 2; object-fit: cover; object-position: left top;
  background: #eceef3; border-radius: 8px; margin: 8px 0; }
figcaption { font-size: 14px; margin-bottom: 8px; }
.success { color: #146b55; }
.failure { color: #92382e; }
"""


def _case_panel(row):
    name = escape(row["id"], quote=True)
    frames = {key: f"{name}/{escape(record['file'], quote=True)}" for key, record in row["frames"].items()}
    label = "Injected trailing space" if row["trailing_space"] else "Clean input"
    verdict = "Exact submission" if row["success"] else "Incorrect submission"
    style = "success" if row["success"] else "failure"
    action = "Rewrote input" if row["repaired"] else "Kept input"
    return f"""
<figure>
  <strong>{label}</strong>
  <a href="{frames['selected']}"><img src="{frames['selected']}" loading="lazy"
    alt="Browser screenshot of the selected tail in the {label.lower()} case"></a>
  <figcaption><strong class="{style}">{verdict}</strong> · {action}</figcaption>
  <a href="{frames['before']}">Before probe</a> · <a href="{frames['submitted']}">After submission</a>
</figure>"""


def write_tail_demo_report(out, result, pairs):
    count = result["layout_pairs"]
    metrics = (
        ("Clean exact submissions", result["clean_successes"], count),
        ("Trailing-space exact submissions", result["trailing_space_successes"], count),
        ("Matching starting screenshots", result["identical_unselected_pairs"], count),
        ("Abstentions on ambiguous controls", result["negative_control_abstentions"], result["negative_controls"]),
    )
    metric_html = "".join(f"<div><dt>{label}</dt><dd>{passed}/{total}</dd></div>"
                          for label, passed, total in metrics)
    pair_html = []
    for index, (clean, faulted) in enumerate(pairs):
        title = escape(clean["profile"].replace("-", " ").capitalize())
        expanded = " open" if index == 0 else ""
        pair_html.append(f"""
<details{expanded}>
  <summary>{title} · Sample {clean['layout'] + 1}</summary>
  <div class="pair">{_case_panel(clean)}{_case_panel(faulted)}</div>
</details>""")
    (out / "report.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trailing-space recovery · GUI-Recover</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
  <header>
    <strong>GUI-Recover</strong>
    <h1>Trailing-space recovery</h1>
    <p>In this CPU browser demo, an extra space can be invisible in the starting screenshot.
    Selecting the last character reveals whether it contains ink or whitespace. The policy uses that
    evidence to decide whether to replace the input before submission.</p>
    <a href="summary.json">Run summary (JSON)</a>
  </header>
  <dl class="metrics">{metric_html}</dl>
  <h2>Compare matched cases</h2>
  <p class="note">Open a pair to compare the two selections and inspect the submitted frames.</p>
  {"".join(pair_html)}
</main>
</body>
</html>
""", encoding="utf-8")
