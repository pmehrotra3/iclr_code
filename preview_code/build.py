"""build.py — generate the preview_code/*.html pages from the repository's source files.

    python preview_code/build.py

Every page shows its source files IN FULL, cut into pieces at the functions / sections that
pages.py explains, with the explanation beside each piece. The code is read from disk at build
time, so re-run this after changing the code. If a piece that pages.py explains can no longer
be found (a function was renamed or removed), the build stops and names it rather than
publishing an explanation of code that is not there.

Needs pygments (syntax colours) and, for the computed figure on the start page, the
repository's own requirements (torch, numpy, scipy).
"""
from __future__ import annotations
import datetime
import html
import os
import re
import subprocess
import sys

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, MarkdownLexer, PythonLexer, YamlLexer, TextLexer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pages  # noqa: E402

LEXERS = {".py": PythonLexer, ".sh": BashLexer, ".sbatch": BashLexer, ".yaml": YamlLexer,
          ".yml": YamlLexer, ".md": MarkdownLexer}


# ---- source files -> explained pieces -------------------------------------------------------
class MissingAnchor(RuntimeError):
    pass


def _indent(line):
    return len(line) - len(line.lstrip())


def _pull_back(lines, j, floor):
    """Start a piece at the decorators / comment lines directly above its anchor line (at the
    same indentation, so a trailing comment of the line before stays where it is)."""
    ind = _indent(lines[j])
    while (j - 1 > floor and lines[j - 1].strip().startswith(("@", "#"))
           and _indent(lines[j - 1]) == ind and not lines[j - 1].startswith("#!")):
        j -= 1
    return j


def split(path, notes):
    """[(first line index, end index, note or None)] covering every line of the file in order.
    notes = [(anchor, title, html)]: a piece starts at the first line (after the previous piece)
    whose text begins with `anchor`."""
    with open(os.path.join(ROOT, path)) as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    starts, pos = [], 0
    for anchor, title, body in notes:
        j = next((i for i in range(pos, len(lines)) if lines[i].strip().startswith(anchor)), None)
        if j is None:
            raise MissingAnchor(f"{path}: no line starting with {anchor!r} after line {pos + 1} "
                                f"(the code changed? update the note in preview_code/pages.py)")
        j = _pull_back(lines, j, starts[-1][0] if starts else 0)
        starts.append((j, (anchor, title, body)))
        pos = j + 1
    pieces = []
    first = starts[0][0] if starts else len(lines)
    if first > 0:
        pieces.append((0, first, None))
    for n, (j, note) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        pieces.append((j, end, note))
    return lines, pieces


def _name(anchor):
    m = re.match(r"(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)", anchor)
    if m:
        return m.group(2), (m.group(2) + "()" if m.group(1) == "def" else m.group(2))
    slug = re.sub(r"[^A-Za-z0-9_]+", "-", anchor).strip("-").lower()[:40] or "section"
    return slug, anchor


def code_html(text, path, start):
    lexer = LEXERS.get(os.path.splitext(path)[1], TextLexer)()
    fmt = HtmlFormatter(linenos="inline", linenostart=start, cssclass="hl", wrapcode=True)
    return highlight(text, lexer, fmt)


# ---- page chrome ------------------------------------------------------------------------------
def _git():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "--", "code", "conf", "scripts"],
                               cwd=ROOT).returncode != 0
        return h + (" + uncommitted changes" if dirty else "")
    except Exception:
        return "unknown"


BUILD_INFO = None


def nav(current):
    out = []
    for group in pages.NAV:
        if out:
            out.append('<span class="sep">|</span>')
        for slug in group:
            p = pages.BY_SLUG[slug]
            cls = ' class="here"' if slug == current else ""
            out.append(f'<a href="{slug}.html"{cls}>{html.escape(p["short"])}</a>')
    return "".join(out)


def shell(slug, title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · iclr_code explained</title>
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="pygments.css">
</head>
<body>
<header class="topbar"><div class="topbar-inner">
  <a class="brand" href="index.html">iclr_code <span>explained</span></a>
  <nav class="nav">{nav(slug)}</nav>
</div></header>
<main class="page">
{body}
<div class="foot">Generated by <code>preview_code/build.py</code> from commit {html.escape(BUILD_INFO)}
on {datetime.date.today().isoformat()}. The code shown is read from the repository at build time;
after changing the code, run <code>python preview_code/build.py</code> again.</div>
</main>
</body>
</html>
"""


def pager(slug):
    order = [s for g in pages.NAV for s in g]
    i = order.index(slug)
    parts = ['<div class="pager">']
    if i > 0:
        p = pages.BY_SLUG[order[i - 1]]
        parts.append(f'<a href="{p["slug"]}.html"><span class="dir">← previous</span>{html.escape(p["title"])}</a>')
    if i + 1 < len(order):
        p = pages.BY_SLUG[order[i + 1]]
        parts.append(f'<a class="next" href="{p["slug"]}.html"><span class="dir">next →</span>{html.escape(p["title"])}</a>')
    parts.append("</div>")
    return "".join(parts)


def render_file(spec, used_ids):
    path = spec["path"]
    lines, pieces = split(path, spec.get("notes", []))
    rows, toc = [], []
    for start, end, note in pieces:
        text = "\n".join(lines[start:end]).rstrip("\n")
        if not text.strip():
            continue
        code = code_html(text, path, start + 1)
        span = f"lines {start + 1}–{start + len(text.split(chr(10)))}"
        if note is None:
            body = spec.get("intro", "")
            head = f'<h3>{html.escape(spec.get("intro_title", "The top of the file"))}<span class="lines">{span}</span></h3>'
            rows.append(f'<section class="chunk"><div class="note">{head}{body}</div><div class="code">{code}</div></section>')
            continue
        anchor, title, body = note
        ident, auto_title = _name(anchor)
        stem = os.path.splitext(os.path.basename(path))[0]
        if ident in used_ids:
            ident = f"{stem}-{ident}"
        used_ids.add(ident)
        title = title or auto_title
        toc.append(f'<a href="#{ident}">{html.escape(title)}</a>')
        head = (f'<h3><a class="self" href="#{ident}">{html.escape(title)}</a>'
                f'<span class="lines">{span}</span></h3>')
        rows.append(f'<section class="chunk" id="{ident}"><div class="note">{head}{body}</div>'
                    f'<div class="code">{code}</div></section>')
    n_lines = len(lines)
    about = spec.get("about", "")
    return (f'<section class="file" id="file-{re.sub(r"[^a-z0-9]+", "-", path.lower())}">'
            f'<div class="filehead"><h2><code>{html.escape(path)}</code></h2>'
            f'<span class="meta">{n_lines} lines · shown in full</span></div>'
            + (f'<div class="prose">{about}</div>' if about else "")
            + (f'<nav class="toc">{"".join(toc)}</nav>' if toc else "")
            + "".join(rows) + "</section>")


def render_page(p):
    used = set()
    hero = (f'<div class="hero"><div class="kicker">{html.escape(p.get("kicker", ""))}</div>'
            f'<h1>{html.escape(p["title"])}</h1><p class="lede">{p["lede"]}</p></div>')
    intro = f'<div class="prose">{p["intro"]}</div>' if p.get("intro") else ""
    files = "".join(render_file(f, used) for f in p.get("files", []))
    return shell(p["slug"], p["title"], hero + intro + files + pager(p["slug"]))


def main():
    global BUILD_INFO
    BUILD_INFO = _git()
    pages.prepare()                                  # computes the figures
    light = HtmlFormatter(style="default").get_style_defs(".hl")
    dark = HtmlFormatter(style="github-dark").get_style_defs(".hl")
    with open(os.path.join(HERE, "pygments.css"), "w") as f:
        f.write("/* generated by build.py: pygments 'default' (light) and 'github-dark' (dark) */\n")
        f.write(light + "\n@media (prefers-color-scheme: dark) {\n" + dark + "\n}\n")
    written = []
    for p in pages.PAGES:
        out = os.path.join(HERE, f'{p["slug"]}.html')
        with open(out, "w") as f:
            f.write(render_page(p))
        written.append(out)
    for w in written:
        print("wrote", os.path.relpath(w, ROOT))


if __name__ == "__main__":
    main()
