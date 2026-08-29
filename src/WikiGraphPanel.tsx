import React, { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// The wiki, drawn. Until now the only way to see what LexiChat remembers was to ask it or to
// open the folder in Finder — the memory was effectively invisible inside the app.
//
// Two kinds of edge, because the wiki has two kinds of relationship and they say different
// things. A LINK is something written down; a RELATED edge is two pages the embedding index
// finds similar. The second is the interesting one: it surfaces pages that belong together
// but were never connected, and near-duplicate pages that should probably be merged.

interface WikiNode {
  path: string;
  folder: string;
  title: string;
  bytes: number;
  chunks: number;
}
interface WikiLink { from: string; to: string }
interface SemanticEdge { a: string; b: string; score: number }
interface WikiGraph {
  nodes: WikiNode[];
  links: WikiLink[];
  related: SemanticEdge[];
  unindexed: boolean;
}

/** Simulation state per node — position and velocity, kept out of React so ticks are cheap. */
interface Body { x: number; y: number; vx: number; vy: number }

// Folder colours, hand-picked rather than generated: evenly spaced in hue but matched in
// weight, so no folder shouts louder than another and all of them sit with the app's indigo.
// Root-level pages take a neutral, since "no folder" is an absence rather than a category.
const FOLDER_COLORS = ["#5B7CFA", "#2FA98E", "#DE9134", "#A468D6", "#DD6382", "#3FA3C4", "#8AA33E", "#C4736A"];
const ROOT_COLOR = "#98A0AE";
const folderColor = (folder: string, folders: string[]) =>
  folder ? FOLDER_COLORS[folders.indexOf(folder) % FOLDER_COLORS.length] : ROOT_COLOR;

export function WikiGraphPanel({ onClose }: { onClose: () => void }) {
  const [graph, setGraph] = useState<WikiGraph | null>(null);
  const [error, setError] = useState("");
  const [minScore, setMinScore] = useState(0.6);
  const [selected, setSelected] = useState<WikiNode | null>(null);
  const [pageText, setPageText] = useState("");
  const [hover, setHover] = useState<WikiNode | null>(null);
  // Right-click menu. `confirming` is a second click before anything is destroyed — deleting
  // a page removes the file outright, so a single stray click must not be enough.
  const [menu, setMenu] = useState<{ node: WikiNode; x: number; y: number; confirming: boolean } | null>(null);
  const [reload, setReload] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bodies = useRef<Map<string, Body>>(new Map());
  const view = useRef({ x: 0, y: 0, k: 1 });          // pan + zoom
  const drag = useRef<{ node?: string; panning?: boolean; lx: number; ly: number } | null>(null);
  const raf = useRef(0);
  // Simulation energy. Kept in a ref so dragging can re-heat a settled layout without
  // restarting the effect — and so it can decay to ~0, which is what makes nodes sit still
  // long enough to be clicked. An always-warm layout drifts under the cursor.
  const alpha = useRef(1);
  const fitted = useRef(false);

  // ── Data ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    invoke<WikiGraph>("get_wiki_graph", { minScore })
      .then(g => { setGraph(g); setError(""); })
      .catch(e => setError(String(e)));
  }, [minScore, reload]);

  useEffect(() => {
    if (!selected) { setPageText(""); return; }
    invoke<string>("read_wiki_page", { path: selected.path })
      .then(setPageText)
      .catch(e => setPageText(String(e)));
  }, [selected]);

  const folders = useMemo(
    () => [...new Set((graph?.nodes ?? []).map(n => n.folder).filter(Boolean))].sort(),
    [graph]
  );

  // ── Simulation ──────────────────────────────────────────────────────────────
  // A small hand-rolled force layout: repulsion between every pair, springs along edges,
  // and a weak pull to centre. At wiki scale (tens of pages) the O(n²) pass is free, and it
  // avoids a graph library for what is ~40 lines of physics.
  useEffect(() => {
    if (!graph) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Seed new nodes on a circle so the first frames expand outward instead of exploding
    // from a single point; keep positions for nodes that already existed.
    const seen = new Set(graph.nodes.map(n => n.path));
    for (const key of [...bodies.current.keys()]) if (!seen.has(key)) bodies.current.delete(key);
    graph.nodes.forEach((n, i) => {
      if (bodies.current.has(n.path)) return;
      const a = (i / Math.max(1, graph.nodes.length)) * Math.PI * 2;
      bodies.current.set(n.path, { x: Math.cos(a) * 160, y: Math.sin(a) * 160, vx: 0, vy: 0 });
    });

    const edges = [
      ...graph.links.map(l => ({ a: l.from, b: l.to, k: 0.010, len: 130 })),
      // Similar pages pull closer the more alike they are, so near-duplicates end up touching.
      ...graph.related.map(r => ({ a: r.a, b: r.b, k: 0.004 * r.score, len: 210 - r.score * 90 })),
    ];

    alpha.current = 1;
    fitted.current = false;
    const step = () => {
      const arr = graph.nodes.map(n => bodies.current.get(n.path)!);
      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const A = arr[i], B = arr[j];
          let dx = B.x - A.x, dy = B.y - A.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) { dx = (Math.random() - .5); dy = (Math.random() - .5); d2 = 1; }
          const f = 5200 / d2;
          const d = Math.sqrt(d2);
          A.vx -= (dx / d) * f; A.vy -= (dy / d) * f;
          B.vx += (dx / d) * f; B.vy += (dy / d) * f;
        }
      }
      for (const e of edges) {
        const A = bodies.current.get(e.a), B = bodies.current.get(e.b);
        if (!A || !B) continue;
        const dx = B.x - A.x, dy = B.y - A.y;
        const d = Math.hypot(dx, dy) || 1;
        const f = (d - e.len) * e.k;
        A.vx += (dx / d) * f; A.vy += (dy / d) * f;
        B.vx -= (dx / d) * f; B.vy -= (dy / d) * f;
      }
      for (const b of arr) {
        b.vx += -b.x * 0.0016; b.vy += -b.y * 0.0016;   // gentle centring
        b.vx *= 0.86; b.vy *= 0.86;                      // damping
        b.x += b.vx * alpha.current; b.y += b.vy * alpha.current;
      }
      alpha.current *= 0.982;
      // Once motion has essentially stopped, frame the graph to the canvas. Doing it here
      // rather than on load means the fit is computed from the settled layout, not the
      // starting circle.
      if (!fitted.current && alpha.current < 0.06) { fitted.current = true; fit(); }
      draw();
      raf.current = requestAnimationFrame(step);
    };

    /** Centre and zoom so the whole graph is comfortably in frame. */
    const fit = () => {
      const pts = graph.nodes.map(n => bodies.current.get(n.path)!).filter(Boolean);
      if (pts.length === 0) return;
      const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      // Padding leaves room for the labels drawn beneath each node.
      const w = canvas.clientWidth - 120, h = canvas.clientHeight - 120;
      // Cap at 1 so fitting only ever pulls a sprawling graph back into frame. Magnifying a
      // small graph to fill the canvas looks broken — a handful of pages should read as a
      // handful of pages, not as giant discs.
      const k = Math.min(1.6, Math.max(0.3, Math.min(w / Math.max(120, maxX - minX), h / Math.max(120, maxY - minY))));
      view.current.k = k;
      view.current.x = -((minX + maxX) / 2) * k;
      view.current.y = -((minY + maxY) / 2) * k;
    };

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr; canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      ctx.translate(w / 2 + view.current.x, h / 2 + view.current.y);
      ctx.scale(view.current.k, view.current.k);

      const pos = (p: string) => bodies.current.get(p);
      const isDim = (p: string) =>
        !!selected && selected.path !== p &&
        !graph.links.some(l => (l.from === selected.path && l.to === p) || (l.to === selected.path && l.from === p)) &&
        !graph.related.some(r => (r.a === selected.path && r.b === p) || (r.b === selected.path && r.a === p));

      // Curve every edge slightly. Straight lines between a hub and its spokes read as a
      // starburst; a consistent bow makes parallel routes separable and the whole thing
      // calmer to look at.
      const curveTo = (A: Body, B: Body) => {
        const dx = B.x - A.x, dy = B.y - A.y;
        ctx.moveTo(A.x, A.y);
        ctx.quadraticCurveTo((A.x + B.x) / 2 - dy * 0.11, (A.y + B.y) / 2 + dx * 0.11, B.x, B.y);
      };

      // Semantic edges first, so written links always read on top of them.
      ctx.lineCap = "round";
      for (const r of graph.related) {
        const A = pos(r.a), B = pos(r.b);
        if (!A || !B) continue;
        const faded = isDim(r.a) && isDim(r.b);
        // Opacity and width both climb with similarity, so a near-duplicate pair is the
        // heaviest thing on the canvas and finds your eye without needing a label.
        const t = Math.min(1, Math.max(0, (r.score - 0.45) / 0.45));
        ctx.strokeStyle = `rgba(99,102,241,${faded ? 0.07 : 0.16 + t * 0.42})`;
        ctx.lineWidth = 1 + t * 5.5;
        ctx.beginPath(); curveTo(A, B); ctx.stroke();
      }
      for (const l of graph.links) {
        const A = pos(l.from), B = pos(l.to);
        if (!A || !B) continue;
        const faded = isDim(l.from) && isDim(l.to);
        ctx.strokeStyle = faded ? "rgba(120,124,136,.13)" : "rgba(108,112,126,.38)";
        ctx.lineWidth = 1.25;
        ctx.beginPath(); curveTo(A, B); ctx.stroke();
      }

      // Panel background, for the halo that keeps labels readable where edges pass behind.
      const bgRaw = getComputedStyle(canvas).backgroundColor;
      const paper = bgRaw === "rgba(0, 0, 0, 0)" ? getComputedStyle(document.body).backgroundColor : bgRaw;

      for (const n of graph.nodes) {
        const b = pos(n.path); if (!b) continue;
        const dim = isDim(n.path);
        const active = selected?.path === n.path || hover?.path === n.path;
        const r = (6 + Math.min(13, Math.sqrt(n.bytes) / 5)) * (active ? 1.15 : 1);
        const color = folderColor(n.folder, folders);

        // Halo under the selected node rather than a hard ring — it reads as emphasis
        // instead of as another edge terminating there.
        if (selected?.path === n.path) {
          ctx.beginPath(); ctx.arc(b.x, b.y, r + 9, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(99,102,241,.16)"; ctx.fill();
        }

        ctx.globalAlpha = dim ? 0.3 : 1;
        ctx.save();
        ctx.shadowColor = "rgba(20,22,34,.24)";
        ctx.shadowBlur = active ? 12 : 6;
        ctx.shadowOffsetY = active ? 3 : 1.5;
        ctx.beginPath(); ctx.arc(b.x, b.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.restore();
        // A ring in the panel colour separates the disc from any edge running behind it.
        ctx.lineWidth = 2; ctx.strokeStyle = paper; ctx.stroke();
        if (active) {
          ctx.beginPath(); ctx.arc(b.x, b.y, r + 3.5, 0, Math.PI * 2);
          ctx.lineWidth = 1.6; ctx.strokeStyle = "#6366f1"; ctx.stroke();
        }

        // Labels are drawn in world space but sized against the zoom, so they stay the same
        // size on screen at any magnification. Without this, zooming in to read a dense
        // cluster inflates the type instead of separating the nodes.
        const k = view.current.k;
        const label = n.title.length > 28 ? n.title.slice(0, 27) + "…" : n.title;
        ctx.font = `${active ? 600 : 400} ${11.5 / k}px Inter, system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.globalAlpha = dim ? 0.28 : 1;
        ctx.lineJoin = "round";
        ctx.lineWidth = 3.5 / k;
        ctx.strokeStyle = paper;
        const ly = b.y + r + 13 / k;
        ctx.strokeText(label, b.x, ly);   // halo, so the label survives an edge behind it
        ctx.fillStyle = getComputedStyle(canvas).color;
        ctx.fillText(label, b.x, ly);
        ctx.globalAlpha = 1;
      }
      ctx.restore();
    };

    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [graph, folders, selected, hover]);

  // ── Pointer ─────────────────────────────────────────────────────────────────
  const toWorld = (e: React.MouseEvent) => {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - rect.width / 2 - view.current.x) / view.current.k,
      y: (e.clientY - rect.top - rect.height / 2 - view.current.y) / view.current.k,
    };
  };
  const nodeAt = (wx: number, wy: number) => {
    if (!graph) return null;
    for (const n of graph.nodes) {
      const b = bodies.current.get(n.path); if (!b) continue;
      const r = 6 + Math.min(13, Math.sqrt(n.bytes) / 5);
      if ((b.x - wx) ** 2 + (b.y - wy) ** 2 <= (r + 5) ** 2) return n;
    }
    return null;
  };

  return (
    <div className="modal-overlay" style={{ zIndex: 200 }} onClick={onClose}>
      <div className="admin-modal" style={{ width: "min(1200px, 96vw)", height: "min(820px, 92vh)", display: "flex", flexDirection: "column" }}
           onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontSize: 15, fontWeight: 600 }}>🧠 Memory Map</span>
          {graph && (
            <span style={{ fontSize: 12, color: "var(--text-tertiary)" }}>
              {graph.nodes.length} pages · {graph.links.length} links · {graph.related.length} related
            </span>
          )}
          <div style={{ flex: 1 }} />
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-secondary)" }}>
            Relatedness
            <input type="range" min={40} max={90} value={Math.round(minScore * 100)}
              onChange={e => setMinScore(Number(e.target.value) / 100)} style={{ width: 110 }} />
            <span className="mono" style={{ fontVariantNumeric: "tabular-nums", width: 30 }}>{minScore.toFixed(2)}</span>
          </label>
          <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }}
            onClick={() => { fitted.current = false; alpha.current = 1; setSelected(null); }}>Reset</button>
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>

        {error && <div style={{ padding: 16, color: "#f87171", fontSize: 13 }}>{error}</div>}

        {graph?.unindexed && (
          <div style={{ padding: "9px 16px", fontSize: 12.5, background: "var(--surface-alt, rgba(99,102,241,.07))", color: "var(--text-secondary)", borderBottom: "1px solid var(--border-light)" }}>
            Showing written links only. Install an embedding model to also see pages that are
            <em> about</em> similar things — <code>ollama pull nomic-embed-text</code>
          </div>
        )}

        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          {/* Canvas */}
          <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
            <canvas
              ref={canvasRef}
              style={{ width: "100%", height: "100%", display: "block", cursor: hover ? "pointer" : "grab", color: "var(--text)" }}
              onMouseDown={e => {
                if (e.button !== 0) return;          // right button belongs to onContextMenu
                const { x, y } = toWorld(e);
                const n = nodeAt(x, y);
                drag.current = n ? { node: n.path, lx: e.clientX, ly: e.clientY }
                                 : { panning: true, lx: e.clientX, ly: e.clientY };
                if (n) alpha.current = Math.max(alpha.current, 0.35);   // let neighbours respond
              }}
              onMouseMove={e => {
                const { x, y } = toWorld(e);
                if (!drag.current) { setHover(nodeAt(x, y)); return; }
                const dx = e.clientX - drag.current.lx, dy = e.clientY - drag.current.ly;
                drag.current.lx = e.clientX; drag.current.ly = e.clientY;
                if (drag.current.panning) {
                  view.current.x += dx; view.current.y += dy;
                } else if (drag.current.node) {
                  const b = bodies.current.get(drag.current.node);
                  if (b) { b.x += dx / view.current.k; b.y += dy / view.current.k; b.vx = 0; b.vy = 0; }
                }
              }}
              onMouseUp={e => {
                // Right-click raises contextmenu *before* mouseup, so without this guard the
                // menu would be opened and then immediately closed by the same gesture.
                if (e.button !== 0) return;
                const moved = drag.current && (Math.abs(e.clientX - drag.current.lx) > 3 || Math.abs(e.clientY - drag.current.ly) > 3);
                if (!moved) {
                  const { x, y } = toWorld(e);
                  const n = nodeAt(x, y);
                  setSelected(n ?? null);
                }
                setMenu(null);
                drag.current = null;
              }}
              onMouseLeave={() => { drag.current = null; setHover(null); }}
              onContextMenu={e => {
                e.preventDefault();
                const { x, y } = toWorld(e);
                const n = nodeAt(x, y);
                const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                setMenu(n ? { node: n, x: e.clientX - rect.left, y: e.clientY - rect.top, confirming: false } : null);
              }}
              onWheel={e => {
                const f = Math.exp(-e.deltaY * 0.0016);
                view.current.k = Math.min(4, Math.max(0.25, view.current.k * f));
              }}
            />
            {/* Right-click menu. Deleting removes a real file, so the destructive item needs a
                second, explicit click and says what would be lost before it does anything. */}
            {menu && (() => {
              const inbound = (graph?.links ?? []).filter(l => l.to === menu.node.path).length;
              const special = menu.node.path === "index.md" ? "the wiki's index — the model reads it to orient itself in a new chat"
                            : menu.node.path === "log.md"   ? "the change log — the record of what was remembered, and when"
                            : null;
              return (
                <div style={{ position: "absolute", left: Math.min(menu.x, 620), top: menu.y, zIndex: 5,
                              width: 236, background: "var(--surface)", border: "1px solid var(--border)",
                              borderRadius: 9, boxShadow: "0 10px 30px rgba(20,22,34,.22)", overflow: "hidden" }}>
                  <div style={{ padding: "9px 12px", borderBottom: "1px solid var(--border-light)" }}>
                    <div style={{ fontSize: 12.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {menu.node.title}
                    </div>
                    <div className="mono" style={{ fontSize: 10.5, color: "var(--text-tertiary)" }}>{menu.node.path}</div>
                  </div>

                  {!menu.confirming ? (
                    <>
                      <button onClick={() => { setSelected(menu.node); setMenu(null); }}
                        style={{ display: "block", width: "100%", textAlign: "left", padding: "8px 12px", fontSize: 12.5,
                                 background: "none", border: "none", cursor: "pointer", color: "var(--text)" }}>
                        Open
                      </button>
                      <button onClick={() => setMenu({ ...menu, confirming: true })}
                        style={{ display: "block", width: "100%", textAlign: "left", padding: "8px 12px", fontSize: 12.5,
                                 background: "none", border: "none", cursor: "pointer", color: "#dc2626",
                                 borderTop: "1px solid var(--border-light)" }}>
                        Delete page…
                      </button>
                    </>
                  ) : (
                    <div style={{ padding: "10px 12px" }}>
                      <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 10 }}>
                        Deletes the file from your wiki folder. This can't be undone.
                        {inbound > 0 && <> <strong>{inbound} page{inbound === 1 ? "" : "s"}</strong> link{inbound === 1 ? "s" : ""} to it — those links will break.</>}
                        {special && <> This is <strong>{special}</strong>.</>}
                      </div>
                      <div style={{ display: "flex", gap: 6 }}>
                        <button className="btn" style={{ flex: 1, fontSize: 11.5, padding: "4px 8px" }}
                          onClick={() => setMenu({ ...menu, confirming: false })}>Cancel</button>
                        <button style={{ flex: 1, fontSize: 11.5, padding: "4px 8px", cursor: "pointer",
                                         background: "#dc2626", color: "#fff", border: "none", borderRadius: 6, fontWeight: 600 }}
                          onClick={async () => {
                            const target = menu.node.path;
                            setMenu(null);
                            const res = await invoke<string>("delete_wiki_page", { path: target }).catch(e => String(e));
                            if (res.startsWith("Deleted")) {
                              if (selected?.path === target) setSelected(null);
                              bodies.current.delete(target);
                              setReload(n => n + 1);   // redraw the graph without it
                            } else {
                              setError(res);
                            }
                          }}>Delete</button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Legend */}
            <div style={{ position: "absolute", left: 14, bottom: 12, display: "flex", flexWrap: "wrap",
                          alignItems: "center", gap: "6px 14px", fontSize: 11, color: "var(--text-secondary)",
                          pointerEvents: "none", background: "var(--surface)", border: "1px solid var(--border-light)",
                          borderRadius: 8, padding: "7px 11px", maxWidth: "calc(100% - 28px)" }}>
              {folders.map(f => (
                <span key={f} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: folderColor(f, folders),
                                 boxShadow: "0 1px 2px rgba(20,22,34,.28)" }} />{f}
                </span>
              ))}
              <span style={{ width: 1, height: 13, background: "var(--border)" }} />
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 18, height: 1.5, borderRadius: 1, background: "rgba(108,112,126,.6)" }} />linked
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 18, height: 4, borderRadius: 2, background: "rgba(99,102,241,.55)" }} />related
                <span style={{ color: "var(--text-tertiary)" }}>(thicker = more alike)</span>
              </span>
            </div>
          </div>

          {/* Reading pane */}
          <div style={{ width: 380, borderLeft: "1px solid var(--border)", display: "flex", flexDirection: "column", flexShrink: 0 }}>
            {!selected ? (
              <div style={{ padding: 20, fontSize: 13, color: "var(--text-tertiary)", lineHeight: 1.6 }}>
                Click a page to read it. Drag to move a page, scroll to zoom, drag the background to pan.
                <p style={{ marginTop: 14 }}>
                  Grey lines are links you wrote. Indigo lines are pages the memory index finds
                  similar — a thick one usually means the two pages overlap and could be merged.
                </p>
              </div>
            ) : (
              <>
                <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border-light)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 10, height: 10, borderRadius: "50%", flexShrink: 0,
                                   background: folderColor(selected.folder, folders),
                                   boxShadow: "0 1px 2px rgba(20,22,34,.28)" }} />
                    <div style={{ fontSize: 14.5, fontWeight: 600, lineHeight: 1.3 }}>{selected.title}</div>
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 2 }}>{selected.path}</div>
                  <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 6 }}>
                    {selected.bytes} bytes · {selected.chunks} indexed chunk{selected.chunks === 1 ? "" : "s"}
                  </div>
                  {graph && (() => {
                    const rel = graph.related
                      .filter(r => r.a === selected.path || r.b === selected.path)
                      .sort((x, y) => y.score - x.score).slice(0, 4);
                    if (rel.length === 0) return null;
                    return (
                      <div style={{ marginTop: 10 }}>
                        <div style={{ fontSize: 10, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: 4 }}>Related</div>
                        {rel.map(r => {
                          const other = r.a === selected.path ? r.b : r.a;
                          const node = graph.nodes.find(n => n.path === other);
                          return (
                            <button key={other}
                              style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", textAlign: "left",
                                       fontSize: 12, padding: "5px 8px", marginBottom: 4, cursor: "pointer",
                                       background: "var(--surface)", border: "1px solid var(--border-light)",
                                       borderRadius: 7, color: "var(--text)" }}
                              onClick={() => node && setSelected(node)}>
                              <span style={{ fontFamily: "monospace", fontSize: 10.5, fontWeight: 600, flexShrink: 0,
                                             padding: "1px 5px", borderRadius: 4,
                                             background: `rgba(99,102,241,${0.10 + (r.score - 0.45) * 0.5})`,
                                             color: "var(--accent)" }}>{r.score.toFixed(2)}</span>
                              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {node?.title ?? other}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    );
                  })()}
                </div>
                <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px", fontSize: 13.5, lineHeight: 1.6 }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{pageText}</ReactMarkdown>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
