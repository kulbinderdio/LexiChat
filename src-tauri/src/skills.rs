//! Skills — packaged, on-demand capability recipes (see docs/skills-framework.md).
//!
//! A skill is a folder under `~/.local/share/lexichat/skills/<id>/` containing a `SKILL.md`:
//! YAML-ish frontmatter (`name`, `description`) between `---` fences, then a markdown body of
//! instructions. Only the one-line descriptions sit in the system prompt (cheap); the body is
//! loaded on demand when the model calls `use_skill("<name>")` — progressive disclosure, mirroring
//! `find_tools`. Phase 1 is instructions-only (no bundled resources yet); execution happens through
//! the existing `run_python` sandbox.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredSkill {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Built-in tools this skill needs to be usable (e.g. `run_python`). The skill is only offered
    /// when the active profile enables all of them; empty = instructions-only, offered anywhere.
    #[serde(default)]
    pub requires: Vec<String>,
    /// Free-form category for grouping in the UI (e.g. "Deliverables"). Empty = Uncategorised.
    #[serde(default)]
    pub category: String,
    /// The markdown instructions body — returned to the model by `use_skill`, not in the base prompt.
    #[serde(default)]
    pub body: String,
    /// True for app-shipped skills (re-seeded from constants, not editable in place). Set at load.
    #[serde(default)]
    pub builtin: bool,
    /// Resource filenames bundled with the skill (a template, a helper module). Staged into
    /// /work/skills/ when the skill is loaded so run_python can read them. Set at load.
    #[serde(default)]
    pub resources: Vec<String>,
}

pub fn skill_dir(id: &str) -> PathBuf {
    skills_dir().join(id)
}

/// Reject path-traversal / nested names in a resource filename.
fn safe_resource_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() || name.contains('/') || name.contains('\\') || name == "." || name == ".."
        || name.eq_ignore_ascii_case("SKILL.md")
    {
        return Err(format!("invalid resource name '{name}'"));
    }
    Ok(name.to_string())
}

/// Full disk paths of a skill's resource files (for staging into the sandbox).
pub fn skill_resource_paths(s: &RegisteredSkill) -> Vec<PathBuf> {
    s.resources.iter().map(|r| skill_dir(&s.id).join(r)).collect()
}

/// Add a resource file to a skill's folder. Allowed for built-ins too — a built-in's SKILL.md is
/// re-seeded each launch but resource files persist, which is how you drop your own template into a
/// built-in like `branded-deck` without duplicating it.
pub fn add_resource(id: &str, name: &str, bytes: &[u8]) -> Result<(), String> {
    let name = safe_resource_name(name)?;
    let dir = skill_dir(id);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(name), bytes).map_err(|e| e.to_string())
}

pub fn remove_resource(id: &str, name: &str) -> Result<(), String> {
    let name = safe_resource_name(name)?;
    let f = skill_dir(id).join(name);
    if f.exists() { std::fs::remove_file(&f).map_err(|e| e.to_string())?; }
    Ok(())
}

pub fn read_resource(id: &str, name: &str) -> Result<Vec<u8>, String> {
    let name = safe_resource_name(name)?;
    std::fs::read(skill_dir(id).join(name)).map_err(|e| e.to_string())
}

/// Whether an id belongs to an app-shipped (built-in) skill.
pub fn is_builtin(id: &str) -> bool {
    BUILTIN_SKILLS.iter().any(|(bid, _)| *bid == id)
}

/// Default category for a built-in skill (so the shipped skills group sensibly without hardcoding a
/// `category:` line in all 17 constants). Empty if not a known built-in.
fn builtin_category(id: &str) -> &'static str {
    match id {
        "presentation" | "branded-deck" | "spreadsheet-model" | "branded-report"
            | "invoice-quote" | "fillable-pdf" => "Deliverables",
        "dashboard" | "geospatial-map" | "map" | "data-cleaning" | "chart-styling" => "Data & visuals",
        "literature-review" | "citation-format" | "plain-english" | "meeting-notes"
            | "email-draft" => "Research & writing",
        "local-area-brief" | "api-explainer" => "Domain",
        _ => "",
    }
}

/// Serialise a skill's fields back into `SKILL.md` text (frontmatter + body).
fn to_skill_md(name: &str, description: &str, category: &str, requires: &[String], body: &str) -> String {
    let mut s = String::from("---\n");
    s.push_str(&format!("name: {}\n", name.trim()));
    s.push_str(&format!("description: {}\n", description.trim().replace('\n', " ")));
    if !category.trim().is_empty() {
        s.push_str(&format!("category: {}\n", category.trim()));
    }
    if !requires.is_empty() {
        s.push_str(&format!("requires: [{}]\n", requires.join(", ")));
    }
    s.push_str("---\n");
    s.push_str(body.trim_start());
    if !s.ends_with('\n') { s.push('\n'); }
    s
}

/// Write a custom skill's `SKILL.md` to disk (creating its folder). Built-in ids are rejected —
/// they're re-seeded from constants each launch, so a custom skill must use its own id.
pub fn write_skill(id: &str, name: &str, description: &str, category: &str, requires: &[String], body: &str) -> Result<(), String> {
    let id = id.trim();
    if id.is_empty() { return Err("skill id is empty".into()); }
    if is_builtin(id) { return Err(format!("'{id}' is a built-in skill and can't be overwritten — duplicate it to a new id instead")); }
    let dir = skills_dir().join(id);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("SKILL.md"), to_skill_md(name, description, category, requires, body)).map_err(|e| e.to_string())
}

/// Delete a custom skill's folder. Built-ins can't be deleted.
pub fn delete_skill(id: &str) -> Result<(), String> {
    if is_builtin(id) { return Err(format!("'{id}' is a built-in skill and can't be deleted")); }
    let dir = skills_dir().join(id);
    if dir.exists() { std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?; }
    Ok(())
}

/// Turn a display name into a filesystem-safe skill id, avoiding collisions with existing ids.
pub fn unique_skill_id(name: &str, existing: &[String]) -> String {
    let base: String = name.trim().to_lowercase().chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-");
    let base = if base.is_empty() { "skill".to_string() } else { base };
    if !existing.iter().any(|e| e == &base) && !is_builtin(&base) { return base; }
    (2..).map(|n| format!("{base}-{n}"))
        .find(|c| !existing.iter().any(|e| e == c) && !is_builtin(c))
        .unwrap()
}

pub fn skills_dir() -> PathBuf {
    crate::dirs_path().join("skills")
}

/// Parse a `SKILL.md`: single-line `name:`/`description:` frontmatter between the first two `---`
/// fences, then the body. Falls back to the folder id for a missing name. Returns None if there's
/// no frontmatter block at all.
pub fn parse_skill_md(id: &str, text: &str) -> Option<RegisteredSkill> {
    let rest = text.trim_start().strip_prefix("---")?;
    let end = rest.find("\n---")?;
    let front = &rest[..end];
    let body = rest[end..].trim_start_matches('\n').trim_start_matches("---").trim_start().to_string();

    let mut name = String::new();
    let mut description = String::new();
    let mut category = String::new();
    let mut requires = Vec::new();
    for line in front.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") { name = v.trim().to_string(); }
        else if let Some(v) = line.strip_prefix("description:") { description = v.trim().to_string(); }
        else if let Some(v) = line.strip_prefix("category:") { category = v.trim().to_string(); }
        else if let Some(v) = line.strip_prefix("requires:") {
            requires = v.trim().trim_start_matches('[').trim_end_matches(']')
                .split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
        }
    }
    if name.is_empty() { name = id.to_string(); }
    // Built-ins group by a default category unless their frontmatter overrides it.
    if category.is_empty() { category = builtin_category(id).to_string(); }
    Some(RegisteredSkill { id: id.to_string(), name, description, category, requires, body, builtin: is_builtin(id), resources: Vec::new() })
}

/// Load every skill from disk (each direct subdirectory of the skills dir with a `SKILL.md`).
pub fn load_skills() -> Vec<RegisteredSkill> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(skills_dir()) {
        for e in entries.flatten() {
            if !e.path().is_dir() { continue; }
            let id = e.file_name().to_string_lossy().into_owned();
            if let Ok(text) = std::fs::read_to_string(e.path().join("SKILL.md")) {
                if let Some(mut s) = parse_skill_md(&id, &text) {
                    // Any other file in the skill folder is a resource (a template, helper module…).
                    if let Ok(files) = std::fs::read_dir(e.path()) {
                        let mut res: Vec<String> = files.flatten()
                            .filter(|f| f.path().is_file())
                            .filter_map(|f| f.file_name().to_str().map(String::from))
                            .filter(|n| n != "SKILL.md")
                            .collect();
                        res.sort();
                        s.resources = res;
                    }
                    out.push(s);
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Write the built-in skills to disk at startup. Built-ins are **app-managed**: they're re-seeded
/// (overwritten) each launch so they track the app version. User-authored skills (Phase 2) live in
/// their own folders and are never touched here; to customise a built-in, copy it to a new id.
pub fn seed_builtin_skills() {
    for (id, contents) in BUILTIN_SKILLS {
        let md = skills_dir().join(id).join("SKILL.md");
        if let Some(parent) = md.parent() { let _ = std::fs::create_dir_all(parent); }
        let _ = std::fs::write(&md, contents);
    }
}

const BUILTIN_SKILLS: &[(&str, &str)] = &[
    ("presentation", PRESENTATION_SKILL),
    ("branded-deck", BRANDED_DECK_SKILL),
    ("spreadsheet-model", SPREADSHEET_SKILL),
    ("geospatial-map", GEOSPATIAL_SKILL),
    ("branded-report", BRANDED_REPORT_SKILL),
    ("dashboard", DASHBOARD_SKILL),
    ("map", MAP_SKILL),
    ("local-area-brief", LOCAL_AREA_SKILL),
    ("literature-review", LITERATURE_REVIEW_SKILL),
    ("citation-format", CITATION_FORMAT_SKILL),
    ("plain-english", PLAIN_ENGLISH_SKILL),
    ("meeting-notes", MEETING_NOTES_SKILL),
    ("email-draft", EMAIL_DRAFT_SKILL),
    ("data-cleaning", DATA_CLEANING_SKILL),
    ("chart-styling", CHART_STYLING_SKILL),
    ("fillable-pdf", FILLABLE_PDF_SKILL),
    ("invoice-quote", INVOICE_QUOTE_SKILL),
    ("api-explainer", API_EXPLAINER_SKILL),
];

const PRESENTATION_SKILL: &str = r##"---
name: presentation
description: Build a polished slide deck — shown INLINE in the chat and saved as an editable PowerPoint (.pptx). Use whenever the user asks for slides, a deck, or a presentation.
requires: [run_python]
---
# Building a presentation

Make TWO things so the user sees it immediately AND can edit it. Do NOT produce a plain,
text-only deck — design it.
1. An **inline styled deck** via `create_artifact` (self-contained HTML — this is what the user
   sees and judges, so make it look good).
2. An **editable PowerPoint** saved to `/work/out` via `run_python` + `python-pptx`, themed to match.

**Be efficient — decks are token-heavy and a local model is slow.** Write the slide content from what
you already know; do NOT `web_search`/`fetch_webpage` unless the user explicitly wants researched or
cited facts. Build the .pptx in a SINGLE `run_python` call (not several). Keep to 5–8 slides.

**Attached images (e.g. a logo):** to place an image the USER ATTACHED on the slides, use its
`{{upload:N}}` token (N = attachment order) in the inline HTML deck, and its `/work/uploads/<name>`
path for `add_picture` in the .pptx. (`{{figure:N}}` is only for images you GENERATED this turn.)

## Step 1 — Plan
Outline first: a cover slide, then ONE idea per slide (5–8 slides is ideal). Put the takeaway in the
slide title ("Sales doubled in Q3", not "Q3 sales"). 3–6 short bullets per slide, never a wall of
text. Pick ONE accent colour that fits the topic.

## Step 2 — Charts (if slides need them; you MUST do this BEFORE Step 3)
`{{figure:N}}` in the HTML deck points to charts that ALREADY rendered inline THIS turn — so generate
every chart FIRST, then build the deck. If you create the deck before the charts exist, the images
show as broken squares.
- Generate ALL the charts in one `run_python` call. Make each an OPEN matplotlib figure (one
  `plt.figure()` per chart) and do NOT `plt.close()` them — LexiChat captures open figures inline in
  creation order, and that order is what `{{figure:1}}`, `{{figure:2}}`… map to.
- ALSO `plt.savefig('/work/chartN.png')` for each, so Step 4's .pptx can embed the same image
  (/work persists across calls this turn).
- Only reference `{{figure:1}}`…`{{figure:M}}` for the M charts you actually made — an out-of-range
  number renders as a broken image.
- PHOTOS / ILLUSTRATIONS: if the deck needs photorealistic images, call `generate_image` FIRST (once
  per image, before Step 3). Each is captured inline in creation order — so it's `{{figure:N}}` just
  like a chart — AND its tool result reports a file path `/work/data/generated_image_N.png` that
  Step 4 can `add_picture` into the .pptx. Do NOT re-generate images to reuse them; reuse those
  figure numbers and paths. If you mix charts and photos, `{{figure:N}}` counts BOTH in creation order.
- **CRITICAL — do it all in ONE turn.** After generating the images, keep going and build BOTH the
  inline deck (Step 3) and the .pptx (Step 4) in the SAME response. Do NOT stop and ask the user to
  "run it again" — `{{figure:N}}` resolves ONLY against images generated in the current turn, so if
  you build the HTML deck in a later turn the photos render as blank "unavailable" boxes. (The image
  *files* do persist, so a follow-up .pptx can still `add_picture` them — but the inline deck cannot
  see them.) Budget your steps: generate every image, then immediately assemble both deliverables.

## Step 3 — Inline scrollable deck (create_artifact)
Call `create_artifact` with a COMPLETE self-contained HTML document based on this template. The inline
deck is a SCROLLABLE stack of slide CARDS — every slide is a 16:9 card, one below the next, so ALL
slides are visible by scrolling the preview (do NOT build a one-at-a-time slideshow that hides slides
behind navigation — in the small embedded frame that looks like only one slide exists). Add one
`<section class="slide">` per slide, change `--accent`, keep everything inline (no external URLs).

```html
<!doctype html><html><head><meta charset="utf-8"><style>
:root{--accent:#4f46e5;--bg:#ffffff;--fg:#0f172a;--muted:#64748b;--edge:#e2e8f0}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;background:#eef1f6;padding:20px}
.deck{max-width:960px;margin:0 auto;display:flex;flex-direction:column;gap:20px}
.slide{position:relative;aspect-ratio:16/9;background:var(--bg);color:var(--fg);border:1px solid var(--edge);border-radius:16px;box-shadow:0 6px 22px rgba(15,23,42,.08);padding:6% 7%;display:flex;flex-direction:column;justify-content:center;overflow:hidden}
.slide h1{font-size:clamp(26px,5vw,50px);line-height:1.08;letter-spacing:-.02em;font-weight:800}
.slide h2{font-size:clamp(19px,3.2vw,31px);color:var(--accent);margin-bottom:.55em;letter-spacing:-.01em;font-weight:700}
.slide .sub{font-size:clamp(14px,1.9vw,20px);color:var(--muted);margin-top:.8em}
.slide ul{font-size:clamp(15px,2vw,22px);line-height:1.7;list-style:none;margin-top:.3em}
.slide li{position:relative;padding-left:1.4em;margin:.32em 0}
.slide li:before{content:'▸';position:absolute;left:0;color:var(--accent)}
.slide img.figure{max-width:100%;max-height:52%;border-radius:10px;margin-top:.8em;align-self:center}
.cover{background:linear-gradient(135deg,#eef2ff,#ffffff)}
.cover h1{font-size:clamp(32px,6.5vw,62px)}
.bar{position:absolute;top:0;left:0;height:6px;width:100%;background:var(--accent);border-radius:16px 16px 0 0}
.logo{position:absolute;top:6%;right:6%;height:14%;max-height:60px;width:auto;border-radius:8px}
.pg{position:absolute;bottom:5.5%;right:6%;color:var(--muted);font-size:clamp(11px,1.2vw,14px);font-variant-numeric:tabular-nums}
</style></head><body>
<!-- Slides are CARDS stacked vertically — ALL visible by scrolling; no navigation JS. One <section
     class="slide"> per slide. Optional per-slide logo from an ATTACHED image:
     <img class="logo" src="{{upload:1}}">. Charts/generated photos: <img class="figure" src="{{figure:N}}">. -->
<div class="deck">
  <section class="slide cover"><h1>Deck title</h1><div class="sub">Subtitle · author · date</div><div class="pg">1 / 5</div></section>
  <section class="slide"><div class="bar"></div><h2>Takeaway heading</h2><ul><li>Point one</li><li>Point two</li><li>Point three</li></ul><div class="pg">2 / 5</div></section>
  <section class="slide"><div class="bar"></div><h2>A chart slide</h2><img class="figure" src="{{figure:1}}"><div class="pg">3 / 5</div></section>
</div>
</body></html>
```
Design: confident cover title, generous whitespace, ≤6 bullets, prefer a chart or two-column layout
over a text wall. If `create_artifact` isn't available, skip this step and still do the .pptx.

## Step 4 — Editable PowerPoint (run_python + python-pptx) — MUST MATCH THE INLINE DECK
The .pptx and the inline deck must look like the SAME presentation: same LIGHT background, same indigo
accent, and the SAME slides in the SAME order with the SAME content. Build every slide with the
helpers below (`cover_slide`, `content_slide`, `picture_slide`) — do NOT hand-place textboxes at
custom coordinates (that is what makes text overlap). The theme colours here already match the inline
template's `--bg` / `--accent` / `--fg`; keep them in sync if you change the deck's accent.
CRITICAL: every image in your inline deck MUST also be a `picture_slide` in the .pptx (the two are
built separately — the .pptx does NOT inherit `{{figure:N}}`/`{{upload:N}}`). Image files: charts you
`plt.savefig('/work/chartN.png')` this turn, photos from `generate_image` at
`/work/data/generated_image_N.png`, and an attached logo/image at `/work/uploads/<name>`.
```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image
import os

# Theme — keep in sync with the inline deck (LIGHT background, indigo accent).
ACCENT  = RGBColor(0x4f,0x46,0xe5)
BG      = RGBColor(0xff,0xff,0xff)   # slides are white, like the inline cards
COVERBG = RGBColor(0xee,0xf2,0xff)   # soft indigo tint for the cover
FG      = RGBColor(0x0f,0x17,0x2a)   # dark text on light
MUTED   = RGBColor(0x64,0x74,0x8b)

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)  # 16:9
W, H = prs.slide_width, prs.slide_height

def _slide(bg=BG):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = bg
    return s

def _bar(s):  # accent bar across the top, matching the inline card
    b = s.shapes.add_shape(1, 0, 0, W, Emu(80000)); b.fill.solid()
    b.fill.fore_color.rgb = ACCENT; b.line.fill.background(); b.shadow.inherit = False

def _text(s, text, left, top, width, height, size, color, bold=False, align=PP_ALIGN.LEFT):
    tf = s.shapes.add_textbox(left, top, width, height).text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = 'Calibri'
    return tf

# --- Build slides ONLY with these; each lays out in its own non-overlapping region. ---
def cover_slide(title, subtitle=''):
    s = _slide(COVERBG); _bar(s)
    _text(s, title, Inches(0.9), Inches(2.3), W-Inches(1.8), Inches(2.2), 46, FG, bold=True)
    if subtitle: _text(s, subtitle, Inches(0.9), Inches(4.6), W-Inches(1.8), Inches(1), 22, MUTED)

def content_slide(title, bullets):
    s = _slide(); _bar(s)
    _text(s, title, Inches(0.9), Inches(0.8), W-Inches(1.8), Inches(1.2), 30, ACCENT, bold=True)
    tf = s.shapes.add_textbox(Inches(0.9), Inches(2.1), W-Inches(1.8), H-Inches(2.8)).text_frame
    tf.word_wrap = True
    for j, line in enumerate(bullets):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = '▸  ' + line
        r.font.size = Pt(20); r.font.color.rgb = FG; r.font.name = 'Calibri'; p.space_after = Pt(12)

def picture_slide(path, title=''):
    if not os.path.exists(path):
        print('WARNING: image not found, slide skipped:', path); return
    s = _slide(); _bar(s); top = Inches(0.9)
    if title:
        _text(s, title, Inches(0.9), Inches(0.8), W-Inches(1.8), Inches(1), 30, ACCENT, bold=True)
        top = Inches(2.0)
    iw, ih = Image.open(path).size
    sc = min(int(W-Inches(1.8))/iw, int(H-int(top)-Inches(0.7))/ih)
    w, h = int(iw*sc), int(ih*sc)
    s.shapes.add_picture(path, int((int(W)-w)/2), top, width=w, height=h)

# Same slides, order and content as the inline deck:
cover_slide('Deck title', 'Subtitle · author · date')
content_slide('Takeaway heading', ['Point one', 'Point two', 'Point three'])
# picture_slide('/work/data/generated_image_1.png', 'A generated photo')

prs.save('/work/out/deck.pptx'); print('saved /work/out/deck.pptx')
```

## Rules
- Show the inline deck AND save the .pptx. Quote the real saved path from the tool result — never
  tell the user the file is at /work/out (they can't open that).
- When you describe the inline deck, say the slides are stacked and the user can SCROLL through
  them. Do NOT tell the user to use arrow keys, click, or dot indicators — the deck has no such
  navigation. Only mention images/logos you ACTUALLY placed in the HTML (never claim a logo is on
  the slides unless you added one).
- Parity is required: the .pptx must be the SAME presentation as the inline deck — same LIGHT theme
  and accent, same slides in the same order, same titles/bullets, and every chart/photo/logo shown
  inline must ALSO be a `picture_slide` in the .pptx. A dark .pptx when the inline deck is light, a
  different layout, or missing images/slides, is a bug.
- Build .pptx slides ONLY via the `cover_slide` / `content_slide` / `picture_slide` helpers — do not
  hand-position textboxes at custom coordinates, which causes text to overlap.
- One idea per slide, takeaway titles, ≤6 bullets, consistent colours/fonts. Design it; don't ship
  plain black-on-white text.
"##;

const BRANDED_DECK_SKILL: &str = r##"---
name: branded-deck
description: Build a PowerPoint from YOUR uploaded template so the deck matches your brand (fonts, colours, layouts). Attach a .pptx template to this skill's resources, then ask for slides.
requires: [run_python]
---
# Branded presentation (from your template)

This skill builds the deck on top of a PowerPoint TEMPLATE the user attached as a resource
(Admin → Skills → View this skill → Add file → their .pptx). Opening the template inherits its theme
(fonts, colours) and slide layouts, so every slide is on-brand with no manual styling. If no template
is attached it falls back to a plain deck — tell the user they can attach one for branding.

Build it in run_python:
```python
import glob
from pptx import Presentation
from pptx.util import Inches, Pt

tpl = sorted(glob.glob('/work/skills/*.pptx'))
prs = Presentation(tpl[0]) if tpl else Presentation()   # template theme+layouts, or a blank deck

# Keep the template's masters/layouts/theme but drop any example slides it ships with:
for sid in list(prs.slides._sldIdLst):
    prs.slides._sldIdLst.remove(sid)

# Layouts come from the TEMPLATE — check the names once if unsure which index is which:
# for i, l in enumerate(prs.slide_layouts): print(i, l.name)

def slide(layout_idx, title, bullets=None, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    if s.shapes.title is not None:
        s.shapes.title.text = title
    body = [p for p in s.placeholders if p.placeholder_format.idx != 0]  # non-title placeholders
    if subtitle and body:
        body[0].text = subtitle
    elif bullets and body:
        tf = body[0].text_frame; tf.text = bullets[0]
        for b in bullets[1:]:
            p = tf.add_paragraph(); p.text = b
    return s

slide(0, 'Deck title', subtitle='Subtitle · author · date')                 # title layout
slide(1, 'Section heading', bullets=['Point one', 'Point two', 'Point three'])  # title + content
# add more slides…

prs.save('/work/out/deck.pptx'); print('saved /work/out/deck.pptx')
```

Rules: use the template's own layouts and placeholders — do NOT override fonts/colours (the template
supplies the brand). One idea per slide, ≤6 short bullets, a takeaway in each title. For a chart,
render it with matplotlib and save a PNG to /work, then `slide.shapes.add_picture(...)` (the chart may
be from an earlier call this turn — /work persists across run_python calls within a turn). Save to
/work/out and quote the real saved path from the tool result.
"##;

const SPREADSHEET_SKILL: &str = r##"---
name: spreadsheet-model
description: Build an editable multi-sheet Excel workbook (.xlsx) with real formulas and formatting — budgets, trackers, financial models. Use when the user asks for a spreadsheet, workbook, or Excel model.
requires: [run_python]
---
# Building a spreadsheet / Excel model

You have `openpyxl` in the `run_python` sandbox. Build a REAL, editable workbook (formulas that
recompute, not hardcoded totals) and save it so the user gets an .xlsx file — do not dump a plain
grid of numbers.

## Plan
- Decide the sheets first (e.g. a data/model sheet + a Summary sheet). Put inputs and calculations
  where they belong; totals must be FORMULAS (`=SUM(...)`), never numbers you computed yourself.
- Use clear headers, currency/number formats, and a styled header row.

## Build it
```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()
ws = wb.active; ws.title = "Model"

HEADER_FILL = PatternFill("solid", fgColor="4F46E5")
HEADER_FONT = Font(bold=True, color="FFFFFF")
MONEY = '£#,##0.00'   # or '#,##0.00', '0.0%', etc.
thin = Side(style="thin", color="D0D0D0"); border = Border(*(thin,)*4)

headers = ["Item", "Qty", "Unit price", "Total"]
ws.append(headers)
for c in range(1, len(headers) + 1):
    cell = ws.cell(1, c); cell.fill = HEADER_FILL; cell.font = HEADER_FONT
    cell.alignment = Alignment(horizontal="center")

rows = [["Widget", 10, 2.50], ["Gadget", 4, 12.00], ["Sprocket", 25, 0.80]]
for r0, (item, qty, price) in enumerate(rows, start=2):
    ws.cell(r0, 1, item); ws.cell(r0, 2, qty)
    ws.cell(r0, 3, price).number_format = MONEY
    ws.cell(r0, 4, f"=B{r0}*C{r0}").number_format = MONEY   # a FORMULA, not a computed number

last = len(rows) + 1
tot = ws.cell(last + 1, 1, "Total"); tot.font = Font(bold=True)
ws.cell(last + 1, 4, f"=SUM(D2:D{last})").number_format = MONEY
ws.cell(last + 1, 4).font = Font(bold=True)

# tidy: widths + freeze the header row
for c in range(1, len(headers) + 1):
    ws.column_dimensions[get_column_letter(c)].width = 16
ws.freeze_panes = "A2"

# a second sheet referencing the first
summ = wb.create_sheet("Summary")
summ["A1"] = "Grand total"; summ["B1"] = f"=Model!D{last + 1}"; summ["B1"].number_format = MONEY

wb.save('/work/out/model.xlsx'); print('saved /work/out/model.xlsx')
```

## Rules
- Totals and derived cells are FORMULAS so the workbook stays live when the user edits inputs.
- Style the header row, set number/currency formats, set sensible column widths, freeze the header.
- Add a Summary sheet for anything multi-sheet. Save to `/work/out/<name>.xlsx` and quote the real
  saved path from the tool result — never say the file is at /work/out.
"##;

const GEOSPATIAL_SKILL: &str = r##"---
name: geospatial-map
description: Plot geospatial data as a map figure — points, lines, polygons or a choropleth — with geopandas/matplotlib, shown inline. Use when the user wants to map locations, boundaries, or regional values.
requires: [run_python]
---
# Mapping geospatial data

You have `geopandas` (with `shapely` and `pyproj`) plus `matplotlib` in the `run_python` sandbox.
Plot the data as a map FIGURE — it renders inline automatically.

IMPORTANT: there is NO online basemap offline (contextily/tiles are blocked). You draw the geometry
itself on axes. If the user really needs a street/satellite backdrop, say so and suggest a connected
map tool (e.g. the Mapbox static-map tool); otherwise plot the geometry cleanly.

## Points from coordinates
```python
import geopandas as gpd, pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

df = pd.DataFrame({"name": ["A", "B", "C"], "lon": [-0.13, -0.09, -0.12], "lat": [51.51, 51.52, 51.50], "value": [30, 55, 12]})
gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326")

ax = gdf.plot(column="value", cmap="viridis", markersize=80, legend=True, figsize=(8, 8))
for _, r in gdf.iterrows():
    ax.annotate(r["name"], (r.geometry.x, r.geometry.y), xytext=(4, 4), textcoords="offset points", fontsize=9)
ax.set_title("My points"); ax.set_aspect("equal"); ax.set_axis_off()
plt.tight_layout()
```

## Choropleth from region polygons
If you have polygons (a GeoJSON the user supplied, or geometry fetched from a connected tool), load
them and shade by a value:
```python
gdf = gpd.read_file("/work/uploads/regions.geojson")   # or build from geometry you fetched
gdf = gdf.merge(values_df, on="code")
ax = gdf.plot(column="value", cmap="OrRd", edgecolor="white", linewidth=0.4, legend=True,
              legend_kwds={"label": "Value", "shrink": 0.6}, figsize=(9, 9))
ax.set_title("Regional values"); ax.set_axis_off()
```

## Rules
- Always set a title, use `set_aspect('equal')` (or an appropriate projection), and turn the axis
  off for a clean map. Give choropleths a colorbar/legend with a labelled units.
- Reproject to a metric CRS (e.g. `gdf.to_crs(27700)` for British National Grid) before distance or
  area work; keep EPSG:4326 only for simple lat/lon plotting.
- The figure shows inline automatically — do NOT paste an image URL. Only `plt.savefig('/work/out/map.png')`
  if the user explicitly wants a saved file, then quote the real saved path.
- No online basemap offline — never claim you drew streets/satellite; for that, use a map tool.
"##;

const BRANDED_REPORT_SKILL: &str = r##"---
name: branded-report
description: Produce a polished, house-style report — cover, executive summary, structured sections and sources — shown inline and saveable as HTML/PDF/Word. Use for formal reports and write-ups.
---
# Branded report

Write a well-structured report and render it as a self-contained styled document via `create_artifact`,
so it looks designed and can be saved (HTML) or printed to PDF/Word from the report export.

Structure (adapt to the content):
1. **Cover** — title, subtitle, author/date, an accent band.
2. **Executive summary** — 3–5 sentences at the top: what this is and the key takeaways.
3. **Sections** — clear H2 headings, short paragraphs, a table or chart where it earns its place.
4. **Sources** — every figure traceable to a source, listed at the end with dates.

Styling: inline CSS, ONE accent colour, a clean body font, generous margins, a cover page, styled
headings, ~65-character line length, `font-variant-numeric: tabular-nums` for figures. Embed any
charts made in `run_python` via `{{figure:N}}`. Keep it fully self-contained (no external URLs). Put
a short summary in chat and the full report in the artifact; tell the user they can Save it or print
to PDF/Word.
"##;

const MAP_SKILL: &str = r##"---
name: map
description: Draw an interactive STREET map with data points plotted on it (crime locations, incidents, a set of places) — real OpenStreetMap streets + markers. Use whenever the user wants points shown on a map.
requires: [run_python]
---
# Interactive street map with data points

Build a real slippy map (OpenStreetMap streets) with the points plotted, via `create_artifact` using
Leaflet. Do NOT use geopandas/matplotlib for this — it has no street basemap and draws points on a
blank background.

THE #1 RULE: plot the REAL coordinates. Never use placeholder, example, sample, or made-up lat/lng.
1. First, in `run_python`, extract the exact points from the actual tool result (e.g. the police
   crime JSON) and print them as a JSON array of {lat, lng, label} — the true coordinates of every
   point. If the data was offloaded to /work/data, read it there.
2. Then call `create_artifact` with the template below, PASTING those exact points into the `points`
   array. Set `centre` to the resolved postcode/area centre.

```html
<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>html,body,#map{height:100%;margin:0}</style></head>
<body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// REAL points from the tool result (replace with the actual data — NEVER placeholders):
const points = [ {lat:51.441,lng:0.372,label:"Anti-social behaviour"} /* … all real points … */ ];
const centre = [51.441, 0.372];               // resolved postcode/area centre
const map = L.map('map').setView(centre, 15);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  { maxZoom: 19, attribution: '© OpenStreetMap contributors' }).addTo(map);
L.circleMarker(centre, { radius:8, color:'#111', fillColor:'#fff', fillOpacity:1, weight:3 })
  .addTo(map).bindPopup('Centre');
const grp = L.featureGroup(points.map(p =>
  L.circleMarker([p.lat, p.lng], { radius:5, color:'#e11d48', fillColor:'#e11d48', fillOpacity:.7, weight:1 })
    .bindPopup(p.label))).addTo(map);
if (points.length) map.fitBounds(grp.getBounds().extend(centre), { padding:[30,30] });
</script></body></html>
```

Colour markers by category if useful, keep the postcode centre marked distinctly, and fit the view to
the points. Build the whole map in ONE create_artifact call.

CRITICAL: pass the HTML to the create_artifact TOOL — do NOT write HTML or an <iframe> into your chat
reply (it shows as raw source, not a map). The artifact HTML contains the Leaflet map directly; do not
wrap it in an inner <iframe>.
"##;

const DASHBOARD_SKILL: &str = r##"---
name: dashboard
description: Build an interactive HTML dashboard — KPI cards, charts and a data table in a self-contained page shown inline. Use when the user wants a dashboard or an at-a-glance view of data.
---
# Dashboard

Build a self-contained interactive dashboard via `create_artifact`. Design for scanning: summary
first, detail below.

Include:
- A header and a row of **KPI cards** (big number + label + a small context/delta).
- One or two **charts**. Matplotlib charts: make them in `run_python` first and embed via
  `{{figure:N}}`. Interactive charts: draw with inline `<canvas>`/SVG + a little JS (no external libs).
- A **data table** if there's row-level data — add tiny inline JS for sort/filter.

Style: cards with subtle borders/shadow, a **semantic** colour for good/warning/critical (separate
from the accent), `tabular-nums` for figures, a responsive grid, both light/dark-friendly if easy.
Everything inline, no CDNs. Keep the prose answer in chat; put the dashboard in the artifact.
"##;

const LOCAL_AREA_SKILL: &str = r##"---
name: local-area-brief
description: Produce a UK neighbourhood report in a fixed structure from a postcode or place — map, crime, prices, planning, demographics, services — each figure sourced. Use with the local-data tools.
requires: [run_python]
---
# Local area brief

Given a UK postcode OR a place/area name, produce a sourced neighbourhood report in this FIXED
structure, using the connected local-data tools (postcode lookup, crime, house prices, planning,
census/NOMIS, mapping). NEVER state a figure from memory — every number comes from a tool call this
turn.

Steps:
1. Resolve the area to a point + admin codes (postcode lookup, or geocode a place then find the
   nearest postcode). State exactly what you resolved to.
2. Show a map of the area inline (a map tool) at the top of the report.
3. Gather: crime (most recent month, ~1 mile); sold prices; planning (ONLY for a specific postcode);
   demographics & deprivation (census at DISTRICT level for a borough/council, LSOA for a small area);
   services (schools/GPs).
4. Compute exact counts/percentages in `run_python` from the tool JSON — never eyeball or estimate.

Report template — keep every heading, in this order:
```
# Local Area Report — [AREA]
**Snapshot:** [2–3 sentences]. _Resolved to: [postcode], [ward], [council]._
## Area & Governance
## Crime (data to [month])
## Housing & Prices
## Planning Activity
## Demographics & Deprivation
## Local Services
## Sources & Currency
```
Match each figure's geography to what was asked; label point-based data (crime/planning) as covering
only the immediate area, not the whole borough. List every source with its data date at the end.
"##;

const LITERATURE_REVIEW_SKILL: &str = r##"---
name: literature-review
description: Synthesise scholarly literature on a question — key papers, how they connect, themes and gaps — with citations. Use with academic tools (e.g. OpenAlex/Crossref) for a structured review.
---
# Literature review

Produce a structured synthesis, not just a list. Use connected academic tools to find REAL papers —
never invent citations.

1. Frame the question precisely.
2. Find the seminal / most-cited works and recent key papers; trace what cites or builds on them.
3. Group findings into 3–5 **themes**; for each, state what's established and what's contested.
4. Identify **gaps** and open questions.
5. **Synthesis** — 1–2 paragraphs tying it together and answering the question.

Cite every claim (author, year, and a DOI/source). Flag where sources disagree (e.g. differing
citation counts between databases). End with a references list. If a fact isn't in a retrieved
source, say so rather than asserting it.
"##;

const CITATION_FORMAT_SKILL: &str = r##"---
name: citation-format
description: Format references and in-text citations in APA, Harvard, MLA, Vancouver or Chicago. Use when the user needs citations styled to a specific convention.
---
# Citation formatting

Ask which style if unspecified (default APA 7th). Format BOTH in-text citations and the reference
list to the chosen style, exactly.

Quick reference:
- **APA 7**: in-text (Author, Year); list: Author, A. A. (Year). Title. Source. https://doi.org/xxx
- **Harvard**: (Author Year); list: Author, A. (Year) 'Title', Source, vol(iss), pp.
- **MLA 9**: (Author page); list: Author. "Title." Source, Year.
- **Vancouver**: numbered [1]; list: 1. Author AA. Title. Source. Year;vol(iss):pages.
- **Chicago (notes)**: footnote n; Author, Title (Place: Publisher, Year).

Alphabetise the reference list (or number it, for Vancouver); be consistent with punctuation,
italics and capitalisation for that style. Only format details you were given — never invent authors,
years, DOIs or page numbers; mark anything missing as [details needed].
"##;

const PLAIN_ENGLISH_SKILL: &str = r##"---
name: plain-english
description: Rewrite text into clear, accessible plain English at a target reading age, with structure and alt-text where relevant. Use to simplify jargon-heavy or dense writing.
---
# Plain English

Rewrite for clarity without losing meaning. Default target: reading age ~9–11 (UK GOV.UK style).

- Short sentences (aim under 20 words), one idea each. Active voice.
- Everyday words; define or replace jargon and spell out acronyms on first use.
- Break walls of text into short paragraphs, headings and bullet lists.
- Address the reader as "you"; give direct instructions ("Send the form", not "The form should be sent").
- Keep all numbers and facts accurate — don't add or drop meaning.

If the content is a document or page, also propose clear headings and provide alt-text for any
images. If asked, give the approximate reading level before and after.
"##;

const MEETING_NOTES_SKILL: &str = r##"---
name: meeting-notes
description: Turn a transcript or rough notes into structured minutes — summary, decisions, action items with owners and dates, and open questions. Use for meeting write-ups.
---
# Meeting notes

Turn the input into clean minutes. Do NOT invent anything not in the source; mark an unclear owner as
"(owner?)" and a missing date as "(TBC)".

```
**Meeting:** [title] · **Date:** [if given] · **Attendees:** [if given]

### Summary
2–4 sentences: what happened and why it matters.

### Decisions
- [decision]

### Action items
| Action | Owner | Due |
|---|---|---|
| … | … | … |

### Open questions / follow-ups
- [item]
```
Keep it factual and concise, group related points, and put every commitment in the action table.
"##;

const EMAIL_DRAFT_SKILL: &str = r##"---
name: email-draft
description: Draft a well-structured email in a chosen tone — formal, follow-up, outreach, apology or request. Use when the user asks to write or reply to an email.
---
# Email draft

Ask for the recipient and goal if unclear. Produce a subject line plus a concise, well-structured
email.

Tone presets:
- **formal** — professional, no slang; **follow-up** — brief, reference prior contact;
- **outreach/cold** — short, value-first, one clear ask; **apology** — own it, offer a remedy, don't
  over-explain; **request** — context, specific ask, deadline, thanks.

Structure: greeting → one-line purpose → body (short paragraphs) → clear call to action → sign-off.
Keep it skimmable and as short as possible, with ONE primary ask; match the recipient's formality.
Offer 1–2 subject-line options. If a compose/send-email tool is connected and the user wants to send,
draft first, confirm, then send.
"##;

const DATA_CLEANING_SKILL: &str = r##"---
name: data-cleaning
description: Clean and tidy a messy CSV/Excel dataset with pandas — types, dates, duplicates, missing values, standardised categories — and save the result. Use when data needs preparing before analysis.
requires: [run_python]
---
# Data cleaning

Load the file from `/work/uploads` and produce a tidy dataset with `run_python` + pandas. Show what
you changed — never silently discard data.

1. **Inspect**: shape, dtypes, `head()`, missing-value counts, duplicate count, unique values of key
   columns.
2. **Clean**: strip/normalise strings; parse dates (`pd.to_datetime`); coerce numerics
   (`pd.to_numeric(..., errors='coerce')`); standardise categories (map/replace, consistent case);
   drop or flag exact duplicates; handle missing values explicitly (drop / fill / leave — say which
   and why).
3. **Validate**: re-check dtypes and value ranges; report how many rows were dropped or changed.
4. **Save** the cleaned file to `/work/out/<name>_clean.csv` (or .xlsx) and summarise the changes as
   a short before/after table. Quote the real saved path from the tool result.
"##;

const CHART_STYLING_SKILL: &str = r##"---
name: chart-styling
description: Apply a clean, consistent house style to matplotlib charts — colours, fonts, spacing, labels. Use when the user wants good-looking or on-brand charts.
requires: [run_python]
---
# Chart styling

Before plotting in `run_python`, set a consistent theme, then build the chart with clear labels.

```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (8, 4.5), "figure.dpi": 130,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.8,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "font.size": 11, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": "#374151",
})
ACCENT = "#4F46E5"
# plt.plot(x, y, color=ACCENT, linewidth=2)
# plt.title("State the takeaway, not just a label"); plt.xlabel("X (units)"); plt.ylabel("Y (units)")
plt.tight_layout()
```
Rules: the title states the takeaway; label both axes with units; keep to ≤5 series (sequential or
qualitative palette); direct-label lines or use a clean legend; no chartjunk. The figure renders
inline automatically.
"##;

const FILLABLE_PDF_SKILL: &str = r##"---
name: fillable-pdf
description: Produce a print-ready formatted document (letter, form, certificate, one-pager) as a styled page shown inline and saveable as PDF. Use when the user wants a PDF document.
---
# Print-ready document (PDF)

Build the document as a self-contained styled page via `create_artifact`. It renders inline and can
be saved to PDF via the report export's "PDF…" button (opens in the browser → Save as PDF), which
preserves the styling.

- Print-friendly layout: A4-ish width, clear margins, a header/letterhead, a readable serif or clean
  sans body, sections/fields laid out cleanly. Inline all CSS; add `@media print` rules if useful
  (hide anything on-screen-only, set page margins).
- For form-like documents, lay out labelled fields/lines and fill in the values you were given.

Note: this generates a NEW styled document → PDF. Filling the form fields of an EXISTING fillable PDF
(AcroForm) isn't available offline — say so, and offer to recreate the form as a styled document
instead.
"##;

const INVOICE_QUOTE_SKILL: &str = r##"---
name: invoice-quote
description: Generate a professional invoice or quote from line items — parties, itemised table, subtotal, tax and total — shown inline and saveable as PDF. Use for invoices, quotes or estimates.
---
# Invoice / quote

Build a clean, professional invoice or quote as a styled page via `create_artifact` (saveable as PDF
via the export's PDF button).

Include:
- Header: **INVOICE** or **QUOTE**, a number, issue date, and due date.
- **From** / **To** blocks.
- An itemised table: description, qty, unit price, line total.
- Subtotal, tax (VAT if applicable — state the rate), and a **bold grand total**.
- Payment terms / notes if given.

Right-align money, use `tabular-nums` and a consistent currency format. Compute totals from the line
items (don't hardcode). Only include details the user provided; ask for any missing essentials
(parties, line items, tax rate).
"##;

const API_EXPLAINER_SKILL: &str = r##"---
name: api-explainer
description: Turn an OpenAPI spec into a concise usage cheat-sheet — endpoints, parameters, auth and example requests. Use when the user pastes or points to an API spec and wants to understand or use it.
---
# API explainer

Given an OpenAPI/Swagger spec (pasted text, an uploaded file, or a connected service), produce a
practical cheat-sheet. Parse it in `run_python` if it's large.

- **Overview**: base URL(s), auth scheme (header/query/OAuth), and what the API is for.
- **Operations table**: method + path, operationId, one-line purpose, required params (name · in ·
  type), and a ready-to-run **example request** (curl or a URL) with placeholder values.
- Group related endpoints; call out pagination, rate limits and common gotchas the spec reveals.

Keep it practical — someone should be able to make a real call straight from your cheat-sheet. Don't
invent endpoints or parameters that aren't in the spec.
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_and_body() {
        let md = "---\nname: presentation\ndescription: Build a deck.\n---\n# How\nDo the thing.\n";
        let s = parse_skill_md("presentation", md).unwrap();
        assert_eq!(s.name, "presentation");
        assert_eq!(s.description, "Build a deck.");
        assert!(s.body.starts_with("# How"));
        assert!(s.body.contains("Do the thing."));
    }

    #[test]
    fn name_falls_back_to_id_and_no_frontmatter_is_none() {
        let s = parse_skill_md("myskill", "---\ndescription: x\n---\nbody").unwrap();
        assert_eq!(s.name, "myskill");
        assert!(parse_skill_md("x", "no frontmatter here").is_none());
    }

    #[test]
    fn builtin_presentation_skill_parses() {
        let s = parse_skill_md("presentation", PRESENTATION_SKILL).unwrap();
        assert_eq!(s.name, "presentation");
        assert!(s.description.to_lowercase().contains("powerpoint"));
        assert!(s.body.contains("python-pptx"));
    }

    #[test]
    fn to_skill_md_round_trips_through_the_parser() {
        let md = to_skill_md("my-skill", "Does a thing.", "My Category", &["run_python".into()], "# How\nStep one.\n");
        let s = parse_skill_md("my-skill", &md).unwrap();
        assert_eq!(s.name, "my-skill");
        assert_eq!(s.description, "Does a thing.");
        assert_eq!(s.category, "My Category");
        assert_eq!(s.requires, vec!["run_python"]);
        assert!(s.body.starts_with("# How"));
        // no requires / category lines when empty
        assert!(!to_skill_md("x", "y", "", &[], "body").contains("requires:"));
        assert!(!to_skill_md("x", "y", "", &[], "body").contains("category:"));
    }

    #[test]
    fn builtin_skills_get_default_categories() {
        assert_eq!(parse_skill_md("presentation", PRESENTATION_SKILL).unwrap().category, "Deliverables");
        assert_eq!(parse_skill_md("citation-format", CITATION_FORMAT_SKILL).unwrap().category, "Research & writing");
        assert_eq!(parse_skill_md("dashboard", DASHBOARD_SKILL).unwrap().category, "Data & visuals");
    }

    #[test]
    fn unique_skill_id_slugs_and_avoids_collisions() {
        assert_eq!(unique_skill_id("Gantt Chart!", &[]), "gantt-chart");
        assert_eq!(unique_skill_id("Gantt Chart", &["gantt-chart".into()]), "gantt-chart-2");
        // never collides with a built-in id
        assert_ne!(unique_skill_id("presentation", &[]), "presentation");
    }

    #[test]
    fn requires_is_parsed_and_gates_instructions_only_skills() {
        // A run_python skill declares its requirement; an instructions-only one requires nothing.
        assert_eq!(parse_skill_md("presentation", PRESENTATION_SKILL).unwrap().requires, vec!["run_python"]);
        assert!(parse_skill_md("citation-format", CITATION_FORMAT_SKILL).unwrap().requires.is_empty());
        assert!(parse_skill_md("literature-review", LITERATURE_REVIEW_SKILL).unwrap().requires.is_empty());
        // bracket + comma forms both parse
        let s = parse_skill_md("x", "---\nname: x\ndescription: y\nrequires: [a, b]\n---\nbody here now").unwrap();
        assert_eq!(s.requires, vec!["a", "b"]);
    }

    #[test]
    fn all_builtin_skills_parse_with_valid_frontmatter() {
        for (id, contents) in BUILTIN_SKILLS {
            let s = parse_skill_md(id, contents)
                .unwrap_or_else(|| panic!("built-in skill '{id}' has no frontmatter block"));
            assert!(!s.name.is_empty(), "skill '{id}' missing name");
            assert!(!s.description.is_empty(), "skill '{id}' missing description");
            assert!(s.body.len() > 50, "skill '{id}' body looks empty");
        }
        assert_eq!(BUILTIN_SKILLS.len(), 18);
    }
}
