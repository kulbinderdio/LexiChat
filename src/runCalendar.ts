// Pure date/schedule logic behind the Run History calendar. Kept out of JobsPanel.tsx
// so the missed-slot rules — the part with the real edge cases — can be unit-tested
// without mounting the panel.

import type { JobRun, JobSchedule, ScheduledJob } from "./jobTypes";

/** Runs retained per job by append_run() in jobs.rs. Keep the two in sync. */
export const RETENTION_PER_JOB = 100;

export type CellState = "ran" | "failed" | "missed" | "idle" | "unknown";

const pad2 = (n: number) => String(n).padStart(2, "0");

/** yyyy-mm-dd in LOCAL time (schedules are wall-clock, so UTC keys would drift). */
export const dayKey = (d: Date) =>
  `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

/** Local-midnight Date for a yyyy-mm-dd key. */
export const keyToDate = (k: string) => {
  const [y, m, d] = k.split("-").map(Number);
  return new Date(y, m - 1, d);
};

/**
 * Most recent moment this schedule should have fired, at or before `now`.
 * Mirrors prev_occurrence() in jobs.rs. `null` for Interval/Manual.
 */
export function prevOccurrence(s: JobSchedule, now: Date): Date | null {
  if (s.type === "Daily") {
    const d = new Date(now);
    d.setHours(s.hour ?? 0, s.minute ?? 0, 0, 0);
    if (d > now) d.setDate(d.getDate() - 1);
    return d;
  }
  if (s.type === "Weekly") {
    // JS getDay() is 0=Sun…6=Sat; JobSchedule.weekday is 0=Mon…6=Sun.
    const todayMon = (now.getDay() + 6) % 7;
    const back = (todayMon - (s.weekday ?? 0) + 7) % 7;
    const d = new Date(now);
    d.setDate(d.getDate() - back);
    d.setHours(s.hour ?? 0, s.minute ?? 0, 0, 0);
    if (d > now) d.setDate(d.getDate() - 7);
    return d;
  }
  return null;
}

/**
 * Days in [from, to] on which `job` was expected to fire.
 *
 * `null` means "never draw a missed cell for this job":
 *  - Interval/Manual have no once-a-day slot, so absence says nothing.
 *  - A job that is currently disabled tells us nothing about when it was switched
 *    off, so historical misses would be guesswork.
 *
 * A slot only counts once it has actually passed (today's evening job is not late
 * at noon) and only from the job's creation onwards.
 */
export function expectedDays(
  job: ScheduledJob,
  from: Date,
  to: Date,
  now: Date,
): Set<string> | null {
  const s = job.schedule;
  if (s.type !== "Daily" && s.type !== "Weekly") return null;
  if (!job.enabled) return null;

  const out = new Set<string>();
  const created = new Date(job.created_at);
  const today = dayKey(now);
  const prev = prevOccurrence(s, now);

  const d = new Date(from);
  d.setHours(0, 0, 0, 0);
  const end = new Date(to);
  end.setHours(0, 0, 0, 0);

  while (d <= end) {
    const k = dayKey(d);
    const matches = s.type === "Daily" || ((d.getDay() + 6) % 7) === (s.weekday ?? 0);
    const slot = new Date(d);
    slot.setHours(s.hour ?? 0, s.minute ?? 0, 0, 0);
    // Today only counts once its slot has passed, which prevOccurrence reports.
    const passed = k !== today || (prev != null && dayKey(prev) === today);
    if (matches && passed && slot >= created) out.add(k);
    d.setDate(d.getDate() + 1);
  }
  return out;
}

/**
 * How one job × day cell should render. Absence is only reported as "missed" when
 * we can trust it: past the retention cap an unrecorded day may just be eviction.
 */
export function cellState(args: {
  day: string;
  dayRuns: JobRun[];
  totalRunsForJob: number;
  oldestRetainedDay?: string;
  expected: Set<string> | null;
}): CellState {
  const { day, dayRuns, totalRunsForJob, oldestRetainedDay, expected } = args;
  if (dayRuns.length > 0) {
    return dayRuns.some(r => r.status === "Error") ? "failed" : "ran";
  }
  if (totalRunsForJob >= RETENTION_PER_JOB && oldestRetainedDay && day < oldestRetainedDay) {
    return "unknown";
  }
  return expected?.has(day) ? "missed" : "idle";
}
