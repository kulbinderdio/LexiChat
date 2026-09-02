import { useState, useEffect, useRef } from "react";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ContextItem { label: string; tokens: number; text?: string }
interface StepContext {
  total: number; num_ctx: number;
  system_tokens: number; tools_tokens: number; history_tokens: number;
  schemas: ContextItem[]; messages: ContextItem[];
}

interface DebugStep {
  index: number;
  context?: StepContext;
  schemaNames: string[];
  candidateTotal?: number;
  /** Approx tokens the tool schemas cost this step — re-sent every step, so it is a recurring
   *  cost, not a one-off. */
  toolsTokens?: number;
  llmText?: string;
  durationMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  toolCalls: { name: string; args: string }[];
  toolResults: { name: string; result: string }[];
  tokens: string;
  thinking: string;
}

interface DebugRun {
  id: number;        // display number (RUN #N)
  runId: number;     // backend agent-loop id — groups events to the right run
  steps: DebugStep[];
  totalMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  error?: string;
  done: boolean;
}

const fmtTok = (n?: number) => (n ?? 0).toLocaleString();
const kTok = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));

// What is in the context and what each part costs. The usual finding is that most of it is tool
// schemas the turn never uses — which is a profile setting, not something the prompt can fix.
function ContextBreakdown({ ctx }: { ctx: StepContext }) {
  const [open, setOpen] = useState(false);
  const [showSchemas, setShowSchemas] = useState(false);
  const [showMessages, setShowMessages] = useState(false);
  const over = ctx.num_ctx > 0 && ctx.total > ctx.num_ctx;
  const bar = (label: string, n: number, colour: string) => {
    const pct = ctx.total > 0 ? Math.round((n / ctx.total) * 100) : 0;
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
        <span style={{ width: 78, opacity: 0.7 }}>{label}</span>
        <span style={{ width: 46, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{kTok(n)}</span>
        <span style={{ flex: 1, height: 5, background: "var(--dbg-step-bg)", borderRadius: 3, overflow: "hidden" }}>
          <span style={{ display: "block", width: `${pct}%`, height: "100%", background: colour }} />
        </span>
        <span style={{ width: 30, textAlign: "right", opacity: 0.5 }}>{pct}%</span>
      </div>
    );
  };
  const rows = (items: ContextItem[]) => items.map((it, i) => (
    <details key={i} style={{ fontSize: 11 }}>
      <summary style={{ cursor: it.text ? "pointer" : "default", listStyle: it.text ? undefined : "none",
                        display: "flex", gap: 8, padding: "1px 0" }}>
        <span style={{ width: 46, textAlign: "right", fontVariantNumeric: "tabular-nums", opacity: 0.65 }}>
          {kTok(it.tokens)}
        </span>
        <span style={{ fontFamily: "monospace", opacity: 0.75, overflow: "hidden", textOverflow: "ellipsis",
                       whiteSpace: "nowrap" }}>{it.label}</span>
      </summary>
      {it.text && (
        <pre style={{ margin: "2px 0 6px 54px", padding: 6, background: "var(--dbg-text-bg)", borderRadius: 4,
                      fontSize: 10, whiteSpace: "pre-wrap", wordBreak: "break-word",
                      maxHeight: 260, overflow: "auto" }}>{it.text}</pre>
      )}
    </details>
  ));
  return (
    <div>
      <button onClick={() => setOpen(o => !o)}
        style={{ background: "none", border: "none", cursor: "pointer", padding: "2px 0", fontSize: 11,
                 color: over ? "#e0a458" : "var(--dbg-schemas-color)", display: "flex", alignItems: "center", gap: 4 }}>
        <span style={{ fontSize: 9 }}>{open ? "▼" : "▶"}</span>
        Context {kTok(ctx.total)} tokens
        {over && ` — over the ${kTok(ctx.num_ctx)} limit, re-read each step`}
      </button>
      {open && (
        <div style={{ paddingLeft: 14, paddingBottom: 4, display: "flex", flexDirection: "column", gap: 3 }}>
          {bar("system", ctx.system_tokens, "#7c8cf8")}
          {bar("schemas", ctx.tools_tokens, "#e0a458")}
          {bar("history", ctx.history_tokens, "#5bb98c")}
          <button onClick={() => setShowSchemas(v => !v)}
            style={{ background: "none", border: "none", cursor: "pointer", padding: "3px 0", fontSize: 11,
                     opacity: 0.75, textAlign: "left" }}>
            {showSchemas ? "▼" : "▶"} {ctx.schemas.length} schemas, largest first
          </button>
          {showSchemas && <div>{rows(ctx.schemas)}</div>}
          <button onClick={() => setShowMessages(v => !v)}
            style={{ background: "none", border: "none", cursor: "pointer", padding: "3px 0", fontSize: 11,
                     opacity: 0.75, textAlign: "left" }}>
            {showMessages ? "▼" : "▶"} {ctx.messages.length} messages
          </button>
          {showMessages && <div>{rows(ctx.messages)}</div>}
          {!ctx.schemas.some(x => x.text) && (
            <div style={{ fontSize: 10, opacity: 0.45, paddingTop: 2 }}>
              Sizes only. Turn on “Capture full context” in Settings to see the text itself.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
// token badge (in ↑ / out ↓) shown on a run header or step row
function TokenBadge({ tin, tout }: { tin?: number; tout?: number }) {
  if ((tin ?? 0) === 0 && (tout ?? 0) === 0) return null;
  return (
    <span title={`${fmtTok(tin)} input (prompt) / ${fmtTok(tout)} output (completion) tokens`}
      style={{ fontSize: 9.5, color: "var(--purple)", fontVariantNumeric: "tabular-nums" }}>
      {fmtTok(tin)} in · {fmtTok(tout)} out
    </span>
  );
}

// ── Step row ──────────────────────────────────────────────────────────────────

function StepRow({ step, isLast }: { step: DebugStep; isLast: boolean }) {
  const [open, setOpen] = useState(isLast);
  const [schemasOpen, setSchemasOpen] = useState(false);

  return (
    <div style={{ marginBottom: 4 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", textAlign: "left", background: "var(--dbg-step-bg)",
          border: "1px solid var(--dbg-border)", borderRadius: 6,
          padding: "5px 10px", cursor: "pointer", display: "flex",
          alignItems: "center", gap: 8, fontSize: 12,
        }}
      >
        <span style={{ fontSize: 10, opacity: 0.5 }}>{open ? "▼" : "▶"}</span>
        <span style={{ fontWeight: 600 }}>Step {step.index + 1}</span>
        {step.toolCalls.length > 0 && (
          <span style={{ fontSize: 10, background: "var(--purple)", color: "#fff",
            padding: "1px 6px", borderRadius: 10 }}>
            {step.toolCalls.length} tool{step.toolCalls.length > 1 ? "s" : ""}
          </span>
        )}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <TokenBadge tin={step.tokensIn} tout={step.tokensOut} />
          {step.durationMs !== undefined && (
            <span style={{ fontSize: 10, color: "var(--purple)" }}>{step.durationMs}ms</span>
          )}
        </span>
      </button>

      {open && (
        <div style={{ paddingLeft: 8, paddingTop: 4, display: "flex", flexDirection: "column", gap: 4 }}>
          {step.context && <ContextBreakdown ctx={step.context} />}
          {/* Schemas */}
          {step.schemaNames.length > 0 && (
            <div>
              <button
                onClick={() => setSchemasOpen(o => !o)}
                style={{ background: "none", border: "none", cursor: "pointer", padding: "2px 0",
                  fontSize: 11, color: "var(--dbg-schemas-color)", display: "flex", alignItems: "center", gap: 4 }}
              >
                <span style={{ fontSize: 9 }}>{schemasOpen ? "▼" : "▶"}</span>
                {step.candidateTotal != null && step.candidateTotal > step.schemaNames.length
                  ? `Schemas (selected ${step.schemaNames.length} of ${step.candidateTotal} tools)`
                  : `Schemas (${step.schemaNames.length} tools sent)`}
                {step.toolsTokens != null && (
                  <span style={{ opacity: 0.6 }}
                    title="Approximate tokens the tool definitions cost. They are re-sent every step, so this recurs for each step of the run.">
                    · ~{step.toolsTokens.toLocaleString()} tok/step
                  </span>
                )}
              </button>
              {schemasOpen && (
                <div style={{ paddingLeft: 14, paddingBottom: 4 }}>
                  {step.schemaNames.map(n => (
                    <div key={n} style={{ fontSize: 11, fontFamily: "monospace", opacity: 0.7 }}>{n}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Thinking */}
          {step.thinking && (
            <details style={{ fontSize: 11 }}>
              <summary style={{
                cursor: "pointer", userSelect: "none", opacity: 0.55,
                padding: "2px 0", listStyle: "none", display: "flex", alignItems: "center", gap: 4,
              }}>
                <span style={{ fontSize: 9 }}>▶</span>
                <span>💭 Thinking ({step.thinking.length} chars)</span>
              </summary>
              <div style={{
                background: "var(--dbg-text-bg)", borderRadius: 4, padding: "6px 8px",
                fontSize: 11, fontFamily: "monospace", whiteSpace: "pre-wrap",
                wordBreak: "break-word", maxHeight: 200, overflowY: "auto",
                opacity: 0.6, fontStyle: "italic", marginTop: 4,
                borderLeft: "2px solid var(--dbg-border)",
              }}>
                {step.thinking}
              </div>
            </details>
          )}

          {/* LLM output */}
          {(step.llmText || step.tokens) && (
            <div style={{
              background: "var(--dbg-text-bg)", borderRadius: 4, padding: "6px 8px",
              fontSize: 11, fontFamily: "monospace", whiteSpace: "pre-wrap",
              wordBreak: "break-word", maxHeight: 160, overflowY: "auto",
              opacity: step.llmText ? 1 : 0.6,
            }}>
              {step.llmText || step.tokens || <span style={{ opacity: 0.4 }}>(no text output)</span>}
            </div>
          )}

          {/* Tool calls + results */}
          {step.toolCalls.map((tc, i) => (
            <div key={i}>
              <div style={{
                background: "var(--dbg-tool-bg)", border: "1px solid var(--purple)33",
                borderRadius: 4, padding: "5px 8px",
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--purple)", marginBottom: 2 }}>
                  ⚡ {tc.name}
                </div>
                <pre style={{ margin: 0, fontSize: 10, fontFamily: "monospace",
                  whiteSpace: "pre-wrap", wordBreak: "break-all", opacity: 0.8 }}>
                  {tc.args}
                </pre>
              </div>
              {step.toolResults[i] && (
                <div style={{
                  background: "var(--dbg-result-bg)", borderRadius: 4, padding: "5px 8px",
                  marginTop: 2, fontSize: 10, fontFamily: "monospace",
                  whiteSpace: "pre-wrap", wordBreak: "break-all",
                  maxHeight: 100, overflowY: "auto", opacity: 0.7,
                }}>
                  {step.toolResults[i].result}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Run row ───────────────────────────────────────────────────────────────────

function RunRow({ run }: { run: DebugRun }) {
  const [open, setOpen] = useState(true);

  return (
    <div style={{ marginBottom: 8 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", textAlign: "left", background: "var(--dbg-run-bg)",
          border: "1px solid var(--dbg-border)", borderRadius: 6,
          padding: "6px 10px", cursor: "pointer", display: "flex",
          alignItems: "center", gap: 8, fontSize: 12,
        }}
      >
        <span style={{ fontSize: 10, opacity: 0.5 }}>{open ? "▼" : "▶"}</span>
        <span style={{ fontWeight: 700, fontSize: 11, opacity: 0.6 }}>RUN #{run.id}</span>
        {!run.done && (
          <span style={{ fontSize: 10, color: "#60a5fa" }}>running…</span>
        )}
        {run.done && run.error && (
          <span style={{ fontSize: 10, color: "#f87171" }}>✕ error</span>
        )}
        {run.done && !run.error && (
          <span style={{ fontSize: 10, color: "#4ade80" }}>✓ done</span>
        )}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <TokenBadge tin={run.tokensIn} tout={run.tokensOut} />
          {run.totalMs !== undefined && (
            <span style={{ fontSize: 10, color: "var(--purple)" }}>{run.totalMs}ms</span>
          )}
        </span>
      </button>

      {open && (
        <div style={{ paddingLeft: 8, paddingTop: 4 }}>
          {[...run.steps].sort((a, b) => a.index - b.index).map((step, i, arr) => (
            <StepRow key={step.index} step={step} isLast={i === arr.length - 1} />
          ))}
          {run.error && (
            <div style={{ fontSize: 11, color: "#f87171", padding: "4px 8px" }}>
              {run.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Debug Panel ───────────────────────────────────────────────────────────────

interface Props {
  visible: boolean;
  clearKey?: number;
}

interface BridgeMsg { dir: string; tool: string; label: string; preview: string; }

export function DebugPanel({ visible, clearKey }: Props) {
  const [runs, setRuns] = useState<DebugRun[]>([]);
  const [bridge, setBridge] = useState<BridgeMsg[]>([]);
  const runCounter = useRef(0);
  const activeRunId = useRef(0); // backend run_id of the step currently streaming (for content events)
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (clearKey === undefined) return;
    setRuns([]);
    setBridge([]);
    runCounter.current = 0;
  }, [clearKey]);

  // MCP-App ↔ host postMessage bridge traffic (frontend-only; window event).
  useEffect(() => {
    const h = (e: Event) => {
      const d = (e as CustomEvent).detail as BridgeMsg;
      setBridge(prev => [...prev.slice(-199), d]);
    };
    window.addEventListener("mcp-app-bridge", h);
    return () => window.removeEventListener("mcp-app-bridge", h);
  }, []);

  // Subscribe once on mount (the panel is always mounted, only its rendering is gated by
  // `visible`) so the trace accumulates even while the panel is closed — opening it mid-run
  // then shows the full history instead of nothing.
  useEffect(() => {
    const unsubs: UnlistenFn[] = [];

    const setup = async () => {
      // Update a run identified by its backend run_id (immutably).
      const updateRun = (runId: number, fn: (run: DebugRun) => DebugRun) =>
        setRuns(prev => {
          const i = prev.findIndex(r => r.runId === runId);
          if (i < 0) return prev;
          const next = [...prev];
          next[i] = fn({ ...next[i], steps: [...next[i].steps] });
          return next;
        });

      // Content events (tokens/thinking/tool calls) carry no run_id — attach them to the step
      // currently streaming: the last step of the active run (else the last run that isn't done).
      const updateActiveStep = (fn: (s: DebugStep) => void) =>
        setRuns(prev => {
          let idx = prev.findIndex(r => r.runId === activeRunId.current && !r.done);
          if (idx < 0) for (let k = prev.length - 1; k >= 0; k--) { if (!prev[k].done) { idx = k; break; } }
          if (idx < 0 || prev[idx].steps.length === 0) return prev;
          const run = { ...prev[idx], steps: [...prev[idx].steps] };
          const last = { ...run.steps[run.steps.length - 1] };
          fn(last);
          run.steps[run.steps.length - 1] = last;
          const next = [...prev]; next[idx] = run; return next;
        });

      // New step starting — create the run on first sight of its run_id, else append the step.
      unsubs.push(await listen<{ run_id: number; step: number; schema_names: string[]; candidate_total?: number; tools_tokens?: number }>("debug-step-start", ({ payload }) => {
        activeRunId.current = payload.run_id;
        const newStep: DebugStep = {
          index: payload.step, schemaNames: payload.schema_names, candidateTotal: payload.candidate_total,
          toolsTokens: payload.tools_tokens,
          toolCalls: [], toolResults: [], tokens: "", thinking: "",
        };
        setRuns(prev => {
          const i = prev.findIndex(r => r.runId === payload.run_id);
          if (i < 0) {
            runCounter.current += 1;
            return [...prev, { id: runCounter.current, runId: payload.run_id, steps: [newStep], done: false }];
          }
          const run = { ...prev[i], steps: [...prev[i].steps] };
          if (!run.steps.some(s => s.index === payload.step)) run.steps.push(newStep);
          const next = [...prev]; next[i] = run; return next;
        });
      }));

      unsubs.push(await listen<StepContext & { run_id: number; step: number }>("debug-step-context", ({ payload }) =>
        updateRun(payload.run_id, run => {
          const j = run.steps.findIndex(s => s.index === payload.step);
          if (j >= 0) run.steps[j] = { ...run.steps[j], context: payload };
          return run;
        })));

      unsubs.push(await listen<{ delta: string }>("agent-thinking", ({ payload }) =>
        updateActiveStep(s => { s.thinking = (s.thinking || "") + payload.delta; })));

      unsubs.push(await listen<{ delta: string }>("agent-token", ({ payload }) =>
        updateActiveStep(s => { s.tokens = (s.tokens || "") + payload.delta; })));

      unsubs.push(await listen<{ name: string; args: string }>("agent-tool-call", ({ payload }) =>
        updateActiveStep(s => { s.toolCalls = [...s.toolCalls, { name: payload.name, args: payload.args }]; })));

      unsubs.push(await listen<{ name: string; result: string }>("agent-tool-result", ({ payload }) =>
        updateActiveStep(s => { s.toolResults = [...s.toolResults, { name: payload.name, result: payload.result }]; })));

      // Step done — match by run_id + step index; record duration and per-step tokens.
      unsubs.push(await listen<{ run_id: number; step: number; llm_text: string; duration_ms: number; tokens_in: number; tokens_out: number }>("debug-step-done", ({ payload }) =>
        updateRun(payload.run_id, run => {
          const j = run.steps.findIndex(s => s.index === payload.step);
          if (j >= 0) run.steps[j] = { ...run.steps[j], llmText: payload.llm_text, durationMs: payload.duration_ms, tokensIn: payload.tokens_in, tokensOut: payload.tokens_out, tokens: "" };
          return run;
        })));

      // Run done — close the CORRECT run (by id) with its total time + tokens.
      unsubs.push(await listen<{ run_id: number; total_ms: number; error?: string; tokens_in: number; tokens_out: number }>("debug-run-done", ({ payload }) =>
        updateRun(payload.run_id, run => {
          run.done = true; run.totalMs = payload.total_ms; run.error = payload.error;
          run.tokensIn = payload.tokens_in; run.tokensOut = payload.tokens_out;
          return run;
        })));
    };

    setup();
    return () => { unsubs.forEach(u => u()); };
  }, []);

  // Scroll to bottom on new data
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [runs]);

  if (!visible) return null;

  return (
    <div style={{
      width: 320, minWidth: 260, maxWidth: 400, height: "100%",
      borderLeft: "1px solid var(--dbg-border)",
      background: "var(--dbg-bg)",
      display: "flex", flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "10px 14px", borderBottom: "1px solid var(--dbg-border)",
        display: "flex", alignItems: "center", gap: 8,
        fontSize: 12, fontWeight: 600, opacity: 0.7,
        flexShrink: 0,
      }}>
        <span>🔍</span>
        <span>Agent Trace</span>
        {runs.length > 0 && (
          <button
            onClick={() => setRuns([])}
            style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer",
              fontSize: 10, opacity: 0.4, padding: 0 }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
        {runs.length === 0 && bridge.length === 0 && (
          <div style={{ fontSize: 12, opacity: 0.35, textAlign: "center", marginTop: 40 }}>
            Send a message to see the agent trace.
          </div>
        )}
        {runs.map(run => <RunRow key={run.id} run={run} />)}

        {bridge.length > 0 && (
          <div style={{ marginTop: 12, borderTop: "1px solid var(--dbg-border)", paddingTop: 8 }}>
            <div style={{ display: "flex", alignItems: "center", fontSize: 11, fontWeight: 600, opacity: 0.6, marginBottom: 4 }}>
              <span>🧩 MCP App bridge</span>
              <button onClick={() => setBridge([])}
                style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", fontSize: 10, opacity: 0.4, padding: 0 }}>
                Clear
              </button>
            </div>
            {bridge.map((m, i) => (
              <div key={i} title={m.preview}
                style={{ fontSize: 10, fontFamily: "monospace", opacity: 0.85, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", padding: "1px 0" }}>
                <span style={{ opacity: 0.5 }}>{m.dir === "app→host" ? "▸" : "◂"}</span> {m.label}
              </div>
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
