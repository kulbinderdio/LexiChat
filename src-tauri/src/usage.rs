// Local, private usage tracking. One append-only record per completed chat turn, stored as JSONL in
// the app data dir (never uploaded — same privacy stance as the rest of LexiChat). Aggregated on
// demand for the Usage panel's History view. Kept dependency-light (plain JSONL, aggregated in
// memory); volumes are modest (one line per turn), and it can move to SQLite later if needed.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

/// One completed chat turn.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct UsageRecord {
    pub ts: i64, // unix seconds (UTC)
    pub model: String,
    pub provider: String, // "ollama" | "openai" | "anthropic"
    #[serde(default)] pub profile: String,
    #[serde(default)] pub prompt_tokens: u64,
    #[serde(default)] pub completion_tokens: u64,
    #[serde(default)] pub duration_ms: u64,
    #[serde(default)] pub steps: u32,
    #[serde(default)] pub tools: BTreeMap<String, u32>, // tool name → calls this turn
    #[serde(default)] pub images: u32,
    #[serde(default)] pub code_runs: u32,
    #[serde(default)] pub error: bool,
}

fn usage_file() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("lexichat").join("usage.jsonl"))
}

/// Append one turn record. Best-effort — never fails a chat over telemetry (which is local anyway).
pub fn append(rec: &UsageRecord) {
    let Some(path) = usage_file() else { return };
    if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
    let Ok(line) = serde_json::to_string(rec) else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Aggregated stats for the History view, over records with `ts >= since` (0 = all time).
#[derive(Serialize, Default)]
pub struct UsageStats {
    pub turns: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub images: u64,
    pub code_runs: u64,
    pub errors: u64,
    pub by_model: Vec<Named>,                 // total tokens per model, desc
    pub by_tool: Vec<Named>,                  // calls per tool, desc
    pub by_day: Vec<DayBucket>,               // chronological
    pub by_provider: Vec<ProviderTokens>,     // for cost estimation
}

#[derive(Serialize, Default)]
pub struct Named { pub name: String, pub value: u64 }
#[derive(Serialize, Default)]
pub struct DayBucket { pub day: String, pub input: u64, pub output: u64 }
#[derive(Serialize, Default)]
pub struct ProviderTokens { pub provider: String, pub prompt: u64, pub completion: u64 }

/// Read + aggregate. Returns empty stats if there's no data yet.
pub fn aggregate(since: i64) -> UsageStats {
    let mut stats = UsageStats::default();
    let Some(path) = usage_file() else { return stats };
    let Ok(text) = std::fs::read_to_string(&path) else { return stats };

    let mut by_model: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_tool: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_day: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut by_provider: BTreeMap<String, (u64, u64)> = BTreeMap::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let Ok(r) = serde_json::from_str::<UsageRecord>(line) else { continue };
        if r.ts < since { continue; }

        stats.turns += 1;
        stats.prompt_tokens += r.prompt_tokens;
        stats.completion_tokens += r.completion_tokens;
        stats.images += r.images as u64;
        stats.code_runs += r.code_runs as u64;
        if r.error { stats.errors += 1; }

        let toks = r.prompt_tokens + r.completion_tokens;
        if !r.model.is_empty() { *by_model.entry(r.model.clone()).or_insert(0) += toks; }
        for (t, n) in &r.tools { *by_tool.entry(t.clone()).or_insert(0) += *n as u64; }
        let day = day_utc(r.ts);
        let e = by_day.entry(day).or_insert((0, 0));
        e.0 += r.prompt_tokens; e.1 += r.completion_tokens;
        if !r.provider.is_empty() {
            let p = by_provider.entry(r.provider.clone()).or_insert((0, 0));
            p.0 += r.prompt_tokens; p.1 += r.completion_tokens;
        }
    }

    stats.by_model = sorted_desc(by_model);
    stats.by_tool = sorted_desc(by_tool);
    stats.by_day = by_day.into_iter().map(|(day, (input, output))| DayBucket { day, input, output }).collect();
    stats.by_provider = by_provider.into_iter().map(|(provider, (prompt, completion))| ProviderTokens { provider, prompt, completion }).collect();
    stats
}

fn sorted_desc(map: BTreeMap<String, u64>) -> Vec<Named> {
    let mut v: Vec<Named> = map.into_iter().map(|(name, value)| Named { name, value }).collect();
    v.sort_by(|a, b| b.value.cmp(&a.value));
    v
}

/// Format a unix timestamp as a UTC calendar day (YYYY-MM-DD) without pulling extra deps beyond the
/// chrono already in the tree.
fn day_utc(ts: i64) -> String {
    use chrono::{TimeZone, Utc};
    Utc.timestamp_opt(ts, 0).single()
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_sums_and_groups() {
        // Aggregation is pure over UsageRecords; test the grouping/sorting logic directly by
        // reconstructing what aggregate() does (the file path is env-dependent, so we don't touch it).
        let recs = vec![
            UsageRecord { ts: 1_000_000, model: "qwen".into(), provider: "ollama".into(), prompt_tokens: 100, completion_tokens: 50,
                tools: BTreeMap::from([("web_search".into(), 2u32)]), code_runs: 1, ..Default::default() },
            UsageRecord { ts: 1_000_000, model: "qwen".into(), provider: "ollama".into(), prompt_tokens: 40, completion_tokens: 10,
                tools: BTreeMap::from([("run_python".into(), 1u32)]), images: 1, ..Default::default() },
            UsageRecord { ts: 1_000_000, model: "gpt-5".into(), provider: "openai".into(), prompt_tokens: 200, completion_tokens: 100,
                error: true, ..Default::default() },
        ];
        // Mirror the reduce in aggregate() to assert the shape.
        let mut by_model: BTreeMap<String, u64> = BTreeMap::new();
        for r in &recs { *by_model.entry(r.model.clone()).or_insert(0) += r.prompt_tokens + r.completion_tokens; }
        let top = sorted_desc(by_model);
        assert_eq!(top[0].name, "gpt-5"); // 300 tokens
        assert_eq!(top[0].value, 300);
        assert_eq!(top[1].name, "qwen");  // 200 tokens
        assert_eq!(top[1].value, 200);
        let total_images: u32 = recs.iter().map(|r| r.images).sum();
        assert_eq!(total_images, 1);
    }

    #[test]
    fn day_formats_utc() {
        assert_eq!(day_utc(0), "1970-01-01");
    }
}
