# LexiChat Skills — Design Doc

Status: proposal · Author: design notes · Target: post-v2.1.x

## 1. Problem & goal

LexiChat can already *do* a lot — a real Python sandbox (`run_python`), file I/O, charts,
OpenAPI/MCP/SPARQL tools, artifacts and report export. What it lacks is a way to teach the model
**how to do a specific kind of task well**, on demand, without paying for that knowledge in every
conversation.

Today the only place to put "how to produce X" guidance is a **profile system prompt** (always-on)
or a **tool description** (always-on). Both are permanently resident in context. A good "build a
polished deck" recipe is 1–2k tokens; a good "financial model in openpyxl" recipe is more. Putting
several of these into every profile would blow the context budget — the exact failure mode the
context-management work (`fit_wire_to_context`) was added to prevent.

**A Skill is a packaged, on-demand capability recipe.** It bundles instructions ("how to make a
great deck"), optional resources (a template, a helper module), and a short description used for
discovery. Its instructions load into context **only when the task calls for it**. This is
`find_tools`, but for *instructions + resources* instead of tools — progressive disclosure applied
to know-how.

### Non-goals
- Skills do **not** grant new runtime powers. A skill can only use tools the profile already has
  (chiefly `run_python`). It packages *how*, not *what*. Security model is unchanged.
- Skills are **not** a replacement for profiles (always-on persona + toolset) or tools (callable
  capabilities). They are a third, orthogonal axis: *task recipes*, additive within any profile.

## 2. Where Skills sit (the three axes)

| Axis | What it is | Lifetime | Example |
|---|---|---|---|
| **Profile** | Persona + enabled toolset | Always-on for the active profile | "OS Data Explorer" |
| **Tool** | A callable capability | Sent per step (narrowed by `find_tools`) | `os_linkeddata_getlinksbyidentifiertype` |
| **Skill** | A task recipe (instructions + resources) | **Loaded on demand** within any profile | "presentation" |

Skills reuse three patterns LexiChat already has:
1. **`find_tools` discovery** — a menu of cheap descriptions; full payload loaded on demand.
2. **The wiki store** — markdown files with frontmatter under `~/.local/share/lexichat/…`.
3. **`run_python` + resource staging** — `stage_python_files` already copies files into `/work`.

## 3. Anatomy of a Skill

A skill is a folder on disk:

```
skills/presentation/
  SKILL.md            # required: frontmatter (discovery) + markdown instructions (the how-to)
  template.pptx       # optional: resources staged into /work/skills/ when the skill loads
  helpers.py          # optional: a helper module the model can import in run_python
```

`SKILL.md` frontmatter mirrors the wiki/memory format:

```markdown
---
name: presentation
description: Build an editable PowerPoint (.pptx) from an outline or data — use when the user asks
  for slides, a deck, or a presentation.
resources: [template.pptx, helpers.py]      # staged into /work/skills/ on load
requires: [run_python]                       # built-ins the skill needs; warn if the profile lacks them
version: 1
---

# Building a presentation

You have `python-pptx` available in run_python. Build a real, editable deck:

1. Start from the bundled template: `Presentation('/work/skills/template.pptx')`.
2. One idea per slide. Title + 3–5 bullets. Never a wall of text.
3. For any chart, render it with matplotlib, save to /work, and `add_picture(...)`.
4. Save to `/work/out/<name>.pptx`.

Example:
```python
from pptx import Presentation
from pptx.util import Inches, Pt
prs = Presentation('/work/skills/template.pptx')
# … build slides …
prs.save('/work/out/deck.pptx')
```

Design rules: <house style, colours, fonts, do/don't…>
```

- **Frontmatter** = the machine-readable part (discovery + wiring).
- **Body** = the human/model-readable recipe, progressively disclosed.
- **Resources** = optional bundled files (templates, helper code, example data).

## 4. Data model & storage

### Disk (source of truth for resources)
```
~/.local/share/lexichat/skills/<skill-id>/SKILL.md
~/.local/share/lexichat/skills/<skill-id>/<resource files…>
```
Mirrors the wiki dir and `allowed_dirs.json` persistence. Rust owns read/write.

### Settings registry (references, like tools)
Skills live once in the global registry and profiles reference them by ID — identical to the
OpenAPI/MCP/SPARQL registry model:

```ts
// AdminPanel.tsx — ToolRegistry
skills: StoredSkill[]            // { id, name, description, requires, resources, enabled }

// Profile
enabledSkillIds: string[]        // which skills this profile exposes
```

On profile switch, `syncServers()` filters the registry to the active profile's enabled skills and
pushes them to Rust via a new `set_skills()` command (mirrors `set_openapi_specs`). Rust holds only
the active profile's enabled skills in `AppState.skills`.

## 5. Discovery & execution (the core flow)

Progressive disclosure via a `use_skill` meta-tool — deterministic, mirrors `find_tools`.

**Base context (cheap):** when a profile has enabled skills, the system prompt gains a small block:
```
AVAILABLE SKILLS — call use_skill("<name>") to load the full instructions before doing one of these:
- presentation: Build an editable PowerPoint from an outline or data.
- spreadsheet-model: Build a formatted multi-sheet Excel workbook with formulas.
```
Cost: ~1 line per skill. The *bodies* are not in context yet.

**On demand:** the model calls `use_skill("presentation")`. In `dispatch_tool` (like the `find_tools`
special-case):
1. Read `SKILL.md` body for that skill.
2. Stage its `resources` into `/work/skills/` (extend `stage_python_files` / dispatch_paths so
   `run_python` can read them).
3. Return the body as the tool result — it enters the conversation, so the model now has the recipe
   and the staged files on its next step.
4. The model executes via `run_python` (+ the staged template/helpers) and writes output to
   `/work/out`, which flows through the existing write-back.

```
model → use_skill("presentation")
 └─ dispatch: read SKILL.md body + stage resources into /work/skills/
     └─ return body as tool result  ──►  model follows recipe via run_python  ──►  /work/out/deck.pptx
```

Some skills are **instructions-only** (no code) — e.g. a writing-style or citation-format skill. For
those, `use_skill` just injects the guidance; the model applies it directly with no run_python.

### Why a meta-tool (not always-inline)
- **Context**: only descriptions are always-on; bodies load on demand → base prompt stays lean.
- **Determinism**: an explicit `use_skill` call is auditable in the debug trace and the per-call
  timer, unlike prose the model may or may not follow.
- **Trimming**: a loaded skill body is a tool-result message, so `fit_wire_to_context` treats it like
  any other — on a very long run it can be elided (and re-loaded) without losing the system prompt.

## 6. Backend changes (Rust)

- `AppState.skills: Mutex<Vec<RegisteredSkill>>` (id, name, description, body, resource paths, requires).
- `skills_dir()` + load/save helpers (mirror the wiki dir).
- Commands: `set_skills`, `get_skills`, `save_skill`, `delete_skill`, `import_skill`, `export_skill`.
- `ollama.rs`:
  - `use_skill_schema()` + the "AVAILABLE SKILLS" preamble builder (only when the profile has skills).
  - In `dispatch_tool`, special-case `use_skill` (like `find_tools`): load body, stage resources into
    the results dir / a `/work/skills` staging path, return the body.
  - Extend `dispatch_paths` so `run_python` can read the staged skill resources.
- Skills are passed into `agent_loop` (like tool groups) so the preamble + `use_skill` are only wired
  when the active profile enables ≥1 skill.

## 7. Frontend changes (React)

- **AdminPanel → new "Skills" tab** (mirrors the OpenAPI/MCP tabs): list registered skills; create/edit
  (name, description, a markdown instructions editor, resource upload); enable per profile in the
  Profiles tab (`enabledSkillIds` checkboxes).
- **Import/Export** a skill as a bundle (zip of `SKILL.md` + resources), exactly like profile
  export — which means **skills are distributable on the website** alongside profiles.
- **App.tsx**: `use_skill` shows as a normal tool badge with its timer; the loaded-skill result can get
  a subtle "📚 skill: presentation" chip in the trace.

## 8. Security & trust

- A skill runs **within** the profile's existing permissions. Its code executes through `run_python`,
  which is already gated by the code-exec permission prompt (and the persistent "Always allow"
  setting). A skill cannot reach a tool the profile hasn't enabled.
- Importing a skill = importing **instructions + code**, same trust surface as importing a profile
  (which already carries a system prompt). The import UI must show the instructions and list bundled
  files, and never auto-run anything. Resources are staged **read-only** into `/work`.
- No new network capability: run_python has no network; a skill can't add one.

## 9. Context budget

- Base cost: `N_enabled_skills × ~15 tokens` (one description line each). Negligible even at 20 skills.
- Loaded cost: one skill body (typically 300–1500 tokens) as a tool result, only while relevant, and
  elidable by `fit_wire_to_context` on long runs. This is the entire reason to use a skill instead of
  stuffing the recipe into the profile prompt.

## 10. How to add a new skill (no code change)

Skills are **data, not code** — adding one never touches Rust/TS or requires a rebuild:

1. Create `SKILL.md` with frontmatter (`name`, `description`, optional `resources`/`requires`) and the
   instructions body.
2. Add any resource files (a template, a `helpers.py`).
3. Install it: drop the folder in `~/.local/share/lexichat/skills/` **or** Admin → Skills → Import.
4. Enable it for the profile(s) that should offer it (Profiles tab).

That's it. The description appears in that profile's "AVAILABLE SKILLS" menu; `use_skill("<name>")`
loads the body + resources on demand. Ship curated skills on the website like profiles.

## 11. Candidate skills (a starter catalog)

Each reuses an existing capability; the skill just packages the know-how.

**Document & deliverable production** (run_python + write-back)
- **presentation** — editable `.pptx` via `python-pptx` (flagship; also an HTML/reveal.js variant via `create_artifact`).
- **spreadsheet-model** — multi-sheet Excel with formulas/formatting via `openpyxl` (budgets, trackers, financial models).
- **branded-report** — house-style HTML/PDF/Word on top of the existing report export (logo, colours, cover page).
- **fillable-pdf / document-template** — fill a template with data (Pillow/reportlab or docx templating).

**Data work** (run_python + pandas/geopandas)
- **data-cleaning** — recipes for messy CSV/Excel (types, dedupe, dates, joins) → tidy output.
- **chart-styling** — a consistent matplotlib theme so every chart matches a house style.
- **geospatial-map** — geopandas recipes (points, choropleths, boundaries) — pairs with the **OS Data Explorer** profile.
- **survey-analysis** — code Likert/cross-tab/segment analysis and summarise.
- **dashboard** — a self-contained interactive HTML dashboard via `create_artifact`.

**Research & writing** (instructions-only or light code)
- **literature-review** — structured synthesis (question → sources → themes → gaps) — pairs with **Research Scout**.
- **citation-format** — APA/Harvard/Vancouver formatting rules.
- **plain-english** — rewrite to a reading age / accessibility (alt-text, headings).
- **meeting-notes** — turn a transcript into decisions/actions/owners.
- **email-draft** — tone/format presets (formal, follow-up, outreach).

**Domain workflows** (pair with connected tools)
- **local-area-brief** — the fixed neighbourhood-report structure, as a reusable skill instead of a giant profile prompt (would shrink the Local Area Checker system prompt — directly relevant to the context work).
- **api-explainer** — given an OpenAPI spec, produce a usage cheat-sheet.
- **invoice-quote** — generate a branded invoice/quote from line items.

> Note the recurring pattern: several *existing* heavy profile prompts (Local Area Checker's fixed
> report template, Research Scout's synthesis rules) are really **skills wearing a profile costume**.
> Migrating them to skills would shrink those system prompts and free context — the same win the
> framework delivers for new capabilities.

## 12. Suggested phasing

- **Phase 1 — runtime.** `use_skill` meta-tool, `AppState.skills`, `set_skills`, resource staging into
  `/work/skills`, the "AVAILABLE SKILLS" preamble. Ship with **one** built-in skill (presentation) and
  `python-pptx` bundled in `prepare-pyodide.mjs`. No UI yet — seed the skill on disk.
- **Phase 2 — authoring & scoping.** Admin → Skills tab (CRUD + markdown editor + resource upload),
  per-profile `enabledSkillIds`, import/export bundles.
- **Phase 3 — distribution & migration.** Website skill downloads (like profiles); migrate the fixed
  parts of Local Area Checker / Research Scout prompts into skills to reclaim context.

## 13. Open questions

- **Auto-suggest vs explicit** — should the model always choose `use_skill`, or should a strong request
  match auto-load a skill? Start explicit (deterministic, debuggable); consider auto-load later.
- **Skill → tool bundling** — should a skill be able to declare *which OpenAPI/MCP tools* it needs and
  auto-enable them? Powerful but couples skills to tools; defer past Phase 2.
- **Versioning/updates** — website skills need an update path (like the draft-release model). Reuse the
  profile import/version convention.
- **Pyodide package availability** — verify `python-pptx` (+ `lxml`, `Pillow`) load cleanly in the
  bundled Pyodide before committing to the presentation flagship.
