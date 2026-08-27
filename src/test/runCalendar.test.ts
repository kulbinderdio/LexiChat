import { describe, it, expect } from "vitest";
import { dayKey, prevOccurrence, expectedDays, cellState } from "../runCalendar";
import type { JobRun, JobSchedule, ScheduledJob } from "../jobTypes";

const job = (over: Partial<ScheduledJob> = {}): ScheduledJob => ({
  id: "j1", name: "Morning brief", enabled: true,
  schedule: { type: "Daily", hour: 7, minute: 0 } as JobSchedule,
  prompt: "", model: "m", system_prompt: null, enabled_builtin_tools: [],
  output_file: null, created_at: "2026-01-01T00:00:00Z", last_run_at: null,
  profile_id: null, profile_name: null, profile_context: null, steps: [],
  catch_up: true, ...over,
});

const run = (startedLocal: string, status: "Success" | "Error" = "Success"): JobRun => ({
  id: Math.random().toString(36), job_id: "j1", job_name: "Morning brief",
  started_at: new Date(startedLocal).toISOString(),
  finished_at: new Date(startedLocal).toISOString(),
  duration_ms: 1000, status, output: "", error: null, trace: [], profile_name: null,
});

// Local-time construction throughout: schedules are wall-clock, so building these
// from UTC strings would drift the assertions by the machine's offset.
const local = (y: number, m: number, d: number, h = 0, min = 0) => new Date(y, m - 1, d, h, min);

describe("prevOccurrence", () => {
  it("returns today's slot once it has passed", () => {
    const now = local(2026, 8, 27, 9, 30);
    expect(prevOccurrence({ type: "Daily", hour: 7, minute: 0 }, now)).toEqual(local(2026, 8, 27, 7, 0));
  });

  it("falls back to yesterday when today's slot is still ahead", () => {
    const now = local(2026, 8, 27, 6, 0);
    expect(prevOccurrence({ type: "Daily", hour: 7, minute: 0 }, now)).toEqual(local(2026, 8, 26, 7, 0));
  });

  it("walks back to the most recent matching weekday", () => {
    // 2026-08-27 is a Thursday; weekday 0 = Monday → 2026-08-24.
    const now = local(2026, 8, 27, 12, 0);
    expect(prevOccurrence({ type: "Weekly", weekday: 0, hour: 7, minute: 0 }, now))
      .toEqual(local(2026, 8, 24, 7, 0));
  });

  it("has no slot for Interval or Manual", () => {
    const now = local(2026, 8, 27, 12, 0);
    expect(prevOccurrence({ type: "Interval", hours: 4 }, now)).toBeNull();
    expect(prevOccurrence({ type: "Manual" }, now)).toBeNull();
  });
});

describe("expectedDays", () => {
  const from = local(2026, 8, 24), to = local(2026, 8, 27);

  it("lists every past day for a daily job", () => {
    const now = local(2026, 8, 27, 9, 0); // today's 07:00 has passed
    const days = expectedDays(job(), from, to, now)!;
    expect([...days].sort()).toEqual(["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"]);
  });

  it("excludes today while its slot is still ahead", () => {
    const now = local(2026, 8, 27, 6, 0); // 07:00 not reached
    const days = expectedDays(job(), from, to, now)!;
    expect(days.has("2026-08-27")).toBe(false);
    expect(days.has("2026-08-26")).toBe(true);
  });

  it("excludes days before the job existed", () => {
    const now = local(2026, 8, 27, 9, 0);
    const days = expectedDays(job({ created_at: local(2026, 8, 26, 12, 0).toISOString() }), from, to, now)!;
    // Created after the 26th's 07:00 slot, so the first real expectation is the 27th.
    expect([...days].sort()).toEqual(["2026-08-27"]);
  });

  it("only matches the scheduled weekday for a weekly job", () => {
    const now = local(2026, 8, 27, 9, 0);
    const days = expectedDays(
      job({ schedule: { type: "Weekly", weekday: 0, hour: 7, minute: 0 } }),
      from, to, now)!;
    expect([...days]).toEqual(["2026-08-24"]); // the Monday
  });

  it("claims nothing for Interval, Manual, or a disabled job", () => {
    const now = local(2026, 8, 27, 9, 0);
    expect(expectedDays(job({ schedule: { type: "Interval", hours: 4 } }), from, to, now)).toBeNull();
    expect(expectedDays(job({ schedule: { type: "Manual" } }), from, to, now)).toBeNull();
    expect(expectedDays(job({ enabled: false }), from, to, now)).toBeNull();
  });
});

describe("cellState", () => {
  const expected = new Set(["2026-08-26", "2026-08-27"]);

  it("is 'ran' when every run that day succeeded", () => {
    expect(cellState({ day: "2026-08-26", dayRuns: [run("2026-08-26T07:00")], totalRunsForJob: 1, expected }))
      .toBe("ran");
  });

  it("is 'failed' when any run that day errored", () => {
    const runs = [run("2026-08-26T07:00"), run("2026-08-26T08:00", "Error")];
    expect(cellState({ day: "2026-08-26", dayRuns: runs, totalRunsForJob: 2, expected })).toBe("failed");
  });

  it("is 'missed' for an expected day with no runs", () => {
    expect(cellState({ day: "2026-08-27", dayRuns: [], totalRunsForJob: 1, expected })).toBe("missed");
  });

  it("is 'idle' for a day that was never scheduled", () => {
    expect(cellState({ day: "2026-08-20", dayRuns: [], totalRunsForJob: 1, expected })).toBe("idle");
  });

  it("does not cry 'missed' for days evicted by the retention cap", () => {
    // At the cap, an unrecorded older day may simply have been evicted.
    expect(cellState({
      day: "2026-08-26", dayRuns: [], totalRunsForJob: 100,
      oldestRetainedDay: "2026-08-27", expected,
    })).toBe("unknown");
  });

  it("still reports misses inside the retained window at the cap", () => {
    expect(cellState({
      day: "2026-08-27", dayRuns: [], totalRunsForJob: 100,
      oldestRetainedDay: "2026-08-26", expected,
    })).toBe("missed");
  });
});

describe("dayKey", () => {
  it("keys by local date, not UTC", () => {
    expect(dayKey(local(2026, 8, 27, 23, 30))).toBe("2026-08-27");
    expect(dayKey(local(2026, 1, 5, 0, 15))).toBe("2026-01-05");
  });
});
