import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";

// Shapes mirror src-tauri/src/usage.rs (UsageStats) + the system_stats command.
interface Named { name: string; value: number }
interface DayBucket { day: string; input: number; output: number }
interface ProviderTokens { provider: string; prompt: number; completion: number }
interface UsageStats {
  turns: number; prompt_tokens: number; completion_tokens: number;
  images: number; code_runs: number; errors: number;
  by_model: Named[]; by_tool: Named[]; by_day: DayBucket[]; by_provider: ProviderTokens[];
}
interface OllamaModel { name: string; size_vram: number; size: number; expires_at: string }
interface SysStats {
  cpu: number; mem_used: number; mem_total: number; app_mem: number; cores: number;
  models: OllamaModel[]; engine: { supported: boolean; installed: boolean };
}

type Range = "today" | "7d" | "30d" | "all";
const RANGE_SINCE: Record<Range, () => number> = {
  today: () => { const d = new Date(); d.setHours(0, 0, 0, 0); return Math.floor(d.getTime() / 1000); },
  "7d": () => Math.floor(Date.now() / 1000) - 7 * 86400,
  "30d": () => Math.floor(Date.now() / 1000) - 30 * 86400,
  all: () => 0,
};
const PRICE: Record<string, [number, number]> = { openai: [2.5, 10], anthropic: [3, 15] };

function fmt(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
  return String(n);
}
const gb = (bytes: number) => (bytes / 1e9).toFixed(1);
function shortDay(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return isNaN(d.getTime()) ? iso.slice(5) : d.toLocaleDateString(undefined, { weekday: "short" });
}
function keepAlive(iso: string): string {
  const t = new Date(iso).getTime();
  if (isNaN(t)) return "";
  const m = Math.round((t - Date.now()) / 60000);
  return m > 0 ? `keep-alive ${m}m` : "expiring";
}

// Use the app's real design tokens (src/App.css) so the panel matches LexiChat exactly.
const A = "var(--accent)", A2 = "var(--purple)";
const GOOD = "#34c759", WARN = "#ff9f0a"; // Apple-style semantic accents (distinct from --accent)
const card: React.CSSProperties = { background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 14, padding: "14px 16px" };
const lab: React.CSSProperties = { fontSize: 10.5, letterSpacing: ".05em", textTransform: "uppercase", color: "var(--text-tertiary)", fontWeight: 600, marginBottom: 10, display: "flex", justifyContent: "space-between" };
const big: React.CSSProperties = { fontSize: 23, fontWeight: 700, letterSpacing: "-.02em", fontVariantNumeric: "tabular-nums", color: "var(--text)" };
const track: React.CSSProperties = { height: 7, borderRadius: 4, background: "var(--surface3)", overflow: "hidden", marginTop: 8 };

function Ring({ pct, color, value, sub }: { pct: number; color: string; value: string; sub: string }) {
  const C = 2 * Math.PI * 26;
  const off = C * (1 - Math.max(0, Math.min(1, pct / 100)));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <svg width="60" height="60" viewBox="0 0 62 62" style={{ flexShrink: 0 }}>
        <circle cx="31" cy="31" r="26" fill="none" stroke="var(--surface3)" strokeWidth="7" />
        <circle cx="31" cy="31" r="26" fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={off} transform="rotate(-90 31 31)" style={{ transition: "stroke-dashoffset .4s" }} />
      </svg>
      <div><div style={big}>{value}</div><div style={{ fontSize: 11, opacity: .6, marginTop: 2 }}>{sub}</div></div>
    </div>
  );
}

// Live stats as a DOCKED rail (like the Debug panel) so you watch the system while chatting — the
// whole point of "live" is seeing CPU/RAM/model/render-spike as a turn runs. History is a modal
// (below), because it's periodic review and its charts need width a narrow rail can't give.
export function UsageRail({ open, onClose, onOpenHistory }: { open: boolean; onClose: () => void; onOpenHistory: () => void }) {
  const [sys, setSys] = useState<SysStats | null>(null);
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!open) { if (timer.current) { clearInterval(timer.current); timer.current = null; } return; }
    const poll = () => invoke<SysStats>("system_stats").then(setSys).catch(() => {});
    poll();
    timer.current = window.setInterval(poll, 2000);
    return () => { if (timer.current) { clearInterval(timer.current); timer.current = null; } };
  }, [open]);
  if (!open) return null;
  return (
    <div style={{ width: 300, minWidth: 260, maxWidth: 340, height: "100%", flexShrink: 0,
      background: "var(--surface)", borderLeft: "1px solid var(--border)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 14px", borderBottom: "1px solid var(--border)" }}>
        <span style={{ fontWeight: 650, fontSize: 13 }}>📊 Live</span>
        <span style={{ fontSize: 10, fontWeight: 600, color: GOOD, display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 5, height: 5, borderRadius: "50%", background: "currentColor" }} />on-device
        </span>
        <button onClick={onOpenHistory} title="Usage history"
          style={{ marginLeft: "auto", border: 0, background: "none", cursor: "pointer", color: A, fontWeight: 600, fontSize: 12 }}>History</button>
        <button onClick={onClose} title="Close" style={{ border: 0, background: "none", cursor: "pointer", color: "var(--text-secondary)", fontSize: 16, lineHeight: 1, padding: 0 }}>✕</button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
        <LiveView sys={sys} compact />
      </div>
    </div>
  );
}

// Usage history as a focused modal — opened from the Live rail's "History" link.
export function UsageHistoryModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [range, setRange] = useState<Range>("7d");
  const [stats, setStats] = useState<UsageStats | null>(null);
  useEffect(() => {
    if (!open) return;
    setStats(null);
    invoke<UsageStats>("get_usage_stats", { args: { since: RANGE_SINCE[range]() } }).then(setStats).catch(() => setStats(null));
  }, [open, range]);
  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="admin-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 720, width: "100%" }}>
        <div className="admin-header">
          <span className="admin-title">📊 Usage History</span>
          <span style={{ marginLeft: "auto", marginRight: 12, fontSize: 11, fontWeight: 600, color: GOOD, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "currentColor" }} /> On-device only
          </span>
          <button className="btn primary" onClick={onClose}>Done</button>
        </div>
        <div className="admin-scroll" style={{ padding: 16 }}>
          <HistoryView stats={stats} range={range} setRange={setRange} />
        </div>
      </div>
    </div>
  );
}

function LiveView({ sys, compact }: { sys: SysStats | null; compact?: boolean }) {
  if (!sys) return <p style={{ opacity: .6, fontSize: 13 }}>Reading system…</p>;
  const memPct = sys.mem_total ? (sys.mem_used / sys.mem_total) * 100 : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "grid", gridTemplateColumns: compact ? "1fr" : "1fr 1fr", gap: 12 }}>
        <div style={card}>
          <div style={lab}>System CPU</div>
          <Ring pct={sys.cpu} color={sys.cpu > 85 ? WARN : A} value={`${Math.round(sys.cpu)}%`} sub={`${sys.cores} cores`} />
        </div>
        <div style={card}>
          <div style={lab}>Memory</div>
          <Ring pct={memPct} color={memPct > 88 ? WARN : A2} value={`${gb(sys.mem_used)} GB`} sub={`of ${gb(sys.mem_total)} GB`} />
          <div style={{ fontSize: 11, opacity: .6, marginTop: 10 }}>LexiChat process: <b style={{ fontVariantNumeric: "tabular-nums" }}>{gb(sys.app_mem)} GB</b></div>
        </div>
      </div>

      <div style={card}>
        <div style={lab}><span>Models loaded · Ollama</span><span style={{ opacity: .5 }}>resident in memory</span></div>
        {sys.models.length === 0 && <div style={{ fontSize: 12.5, opacity: .55 }}>No model loaded right now — one loads on your next message.</div>}
        {sys.models.map(m => {
          const pct = sys.mem_total ? (m.size_vram / sys.mem_total) * 100 : 0;
          return (
            <div key={m.name} style={{ padding: "9px 0", borderBottom: "1px solid var(--border-light)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontSize: 12.5 }}>
                <span style={{ fontFamily: "ui-monospace,monospace", fontWeight: 600 }}>{m.name}</span>
                <span style={{ opacity: .6, fontSize: 11 }}>{keepAlive(m.expires_at)}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 6 }}>
                <div style={{ ...track, flex: 1, marginTop: 0 }}>
                  <div style={{ height: "100%", width: `${Math.min(100, pct)}%`, background: `linear-gradient(90deg,${A},${A2})`, borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 11, opacity: .7, fontVariantNumeric: "tabular-nums", width: 74, textAlign: "right" }}>{gb(m.size_vram)} GB mem</span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ ...card, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={lab}>Image engine</div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>
            {!sys.engine.supported ? "Not available on this platform"
              : sys.engine.installed ? <span style={{ color: GOOD }}>✓ Installed &amp; ready</span>
              : <span style={{ color: WARN }}>Not installed</span>}
          </div>
        </div>
        <div style={{ fontSize: 11, opacity: .55, textAlign: "right", maxWidth: 240 }}>
          Image models load transiently during a render (not kept in memory), and briefly spike the GPU.
        </div>
      </div>

      <p style={{ fontSize: 11, opacity: .45, textAlign: "center", margin: "2px 0 0" }}>Updates every 2s · read locally, nothing uploaded.</p>
    </div>
  );
}

function HistoryView({ stats, range, setRange }: { stats: UsageStats | null; range: Range; setRange: (r: Range) => void }) {
  const days = stats?.by_day ?? [];
  const maxDay = Math.max(1, ...days.map(d => d.input + d.output));
  const totalTokens = (stats?.prompt_tokens ?? 0) + (stats?.completion_tokens ?? 0);
  const maxTool = Math.max(1, ...(stats?.by_tool ?? []).map(t => t.value));
  const cloudCost = (stats?.by_provider ?? []).reduce((s, p) => {
    const r = PRICE[p.provider]; return r ? s + (p.prompt / 1e6) * r[0] + (p.completion / 1e6) * r[1] : s;
  }, 0);
  const empty = (stats?.turns ?? 0) === 0;

  return (
    <>
      <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
        {(["today", "7d", "30d", "all"] as Range[]).map(r => (
          <button key={r} onClick={() => setRange(r)} style={{
            border: "1px solid var(--border)", background: range === r ? A : "transparent",
            color: range === r ? "#fff" : "inherit", borderColor: range === r ? "transparent" : undefined,
            fontSize: 12, fontWeight: 600, padding: "5px 12px", borderRadius: 100, cursor: "pointer" }}>
            {r === "today" ? "Today" : r === "all" ? "All time" : r === "7d" ? "7 days" : "30 days"}
          </button>
        ))}
      </div>

      {!stats && <p style={{ opacity: .6, fontSize: 13 }}>Loading…</p>}
      {stats && empty && (
        <div style={{ ...card, textAlign: "center", padding: "36px 16px" }}>
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>No usage recorded in this range yet</div>
          <div style={{ fontSize: 13, opacity: .6 }}>Send a few messages and they'll show up here — computed locally, nothing uploaded.</div>
        </div>
      )}

      {stats && !empty && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
            <div style={card}><div style={lab}>Chats</div><div style={big}>{fmt(stats.turns)}</div></div>
            <div style={card}><div style={lab}>Tokens</div><div style={big}>{fmt(totalTokens)}</div>
              <div style={{ fontSize: 11, opacity: .55, marginTop: 4 }}>{fmt(stats.prompt_tokens)} in · {fmt(stats.completion_tokens)} out</div></div>
            <div style={card}><div style={lab}>Images</div><div style={big}>{fmt(stats.images)}</div></div>
            <div style={card}><div style={lab}>Code runs</div><div style={big}>{fmt(stats.code_runs)}</div></div>
          </div>

          <div style={card}>
            <div style={lab}><span>Tokens per day</span><span style={{ opacity: 1 }}><span style={{ color: A }}>■</span> in&nbsp; <span style={{ color: A2 }}>■</span> out</span></div>
            <svg viewBox="0 0 480 140" width="100%" height="150" preserveAspectRatio="none" aria-hidden="true">
              {days.slice(-14).map((d, i, arr) => {
                const n = arr.length, w = Math.min(34, (440 / n) - 6), step = 440 / n, x = 30 + i * step + (step - w) / 2;
                const inH = (d.input / maxDay) * 110, outH = (d.output / maxDay) * 110;
                return (
                  <g key={d.day}>
                    <rect x={x} y={120 - inH} width={w} height={inH} rx="3" fill={A} />
                    <rect x={x} y={120 - inH - outH} width={w} height={outH} rx="3" fill={A2} />
                    {(n <= 10 || i % 2 === 0) && <text x={x + w / 2} y="134" textAnchor="middle" fontSize="9" fill="currentColor" opacity=".5" style={{ fontFamily: "ui-monospace,monospace" }}>{shortDay(d.day)}</text>}
                  </g>
                );
              })}
            </svg>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={card}>
              <div style={lab}>Tokens by model</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                {stats.by_model.slice(0, 6).map(m => (
                  <div key={m.name}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 3 }}>
                      <span style={{ fontFamily: "ui-monospace,monospace" }}>{m.name}</span>
                      <span style={{ opacity: .6, fontVariantNumeric: "tabular-nums" }}>{fmt(m.value)}</span>
                    </div>
                    <div style={{ ...track, marginTop: 0 }}>
                      <div style={{ height: "100%", width: `${(m.value / Math.max(1, stats.by_model[0].value)) * 100}%`, background: `linear-gradient(90deg,${A},${A2})`, borderRadius: 4 }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div style={card}>
              <div style={lab}>Tool usage</div>
              {stats.by_tool.length === 0 && <div style={{ fontSize: 12, opacity: .5 }}>No tools used in this range.</div>}
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {stats.by_tool.slice(0, 6).map(t => (
                  <div key={t.name} style={{ display: "grid", gridTemplateColumns: "110px 1fr 40px", alignItems: "center", gap: 8, fontSize: 11.5 }}>
                    <span style={{ fontFamily: "ui-monospace,monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.name}</span>
                    <div style={{ ...track, marginTop: 0 }}><div style={{ height: "100%", width: `${(t.value / maxTool) * 100}%`, background: A, borderRadius: 4 }} /></div>
                    <span style={{ textAlign: "right", opacity: .6, fontVariantNumeric: "tabular-nums" }}>{fmt(t.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div style={{ ...card, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div><div style={lab}>Estimated cloud cost</div><div style={big}>${cloudCost.toFixed(2)}</div></div>
            <div style={{ fontSize: 12, textAlign: "right", opacity: .75 }}>
              Local models (Ollama): <b style={{ color: GOOD }}>free &amp; private</b><br />
              <span style={{ opacity: .6 }}>{stats.errors} failed turn{stats.errors === 1 ? "" : "s"} in range</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
