import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Every built-in tool the app SENDS must have a checkbox in the Admin panel.
 *
 * These are two hand-maintained lists in different files, and they drifted: App.tsx sent
 * compose_email, fetch_webpage and create_artifact, none of which appeared in AdminPanel's
 * BUILTIN_TOOLS. Because a profile enables a tool unless it records `false`, and an absent
 * tool can never be recorded, those three could not be turned off from anywhere in the UI —
 * a profile with every box unticked still shipped ~920 tokens of their schemas on every step.
 */
const read = (f: string) => readFileSync(resolve(__dirname, "..", f), "utf8");

function sentTools(): string[] {
  // Tool schemas in App.tsx, excluding the wiki set (gated by its own wikiEnabled switch).
  return [...read("App.tsx").matchAll(/function: \{ name: "([a-z_]+)"/g)]
    .map(m => m[1])
    .filter(n => !n.startsWith("wiki_"));
}

function adminTools(): string[] {
  const block = read("AdminPanel.tsx").match(/const BUILTIN_TOOLS = \[(.*?)\n\];/s);
  if (!block) throw new Error("BUILTIN_TOOLS not found in AdminPanel.tsx");
  return [...block[1].matchAll(/name: "([a-z_]+)"/g)].map(m => m[1]);
}

describe("built-in tool coverage", () => {
  it("every tool sent to the model can be toggled in the Admin panel", () => {
    const missing = sentTools().filter(n => !adminTools().includes(n));
    expect(missing, `sent but not toggleable: ${missing.join(", ")}`).toEqual([]);
  });

  it("the Admin panel lists no tool the app never sends", () => {
    const sent = sentTools();
    const stale = adminTools().filter(n => !sent.includes(n));
    expect(stale, `a checkbox that controls nothing: ${stale.join(", ")}`).toEqual([]);
  });
});
