import { describe, it, expect } from "vitest";
import { withErrorShim, withCorsScripts } from "../App";

// A model-authored artifact that throws used to render as a silent blank white frame — no clue for
// the user or for us. These cover the injection that turns that into a visible error.
describe("withErrorShim", () => {
  it("installs the shim inside <head>, after the doctype", () => {
    const out = withErrorShim(`<!DOCTYPE html><html><head><title>x</title></head><body></body></html>`);
    // The doctype must stay first — putting a <script> before it flips the document into quirks
    // mode, which breaks the `html,body{height:100%}` layout every map artifact relies on.
    expect(out.indexOf("<!DOCTYPE html>")).toBe(0);
    expect(out).toContain("__lexiArtifact");
    // Installed before the page's own content, so it catches errors from scripts that follow.
    expect(out.indexOf("__lexiArtifact")).toBeLessThan(out.indexOf("<title>"));
  });

  it("falls back to <body>, then <html>, then prepends for a bare fragment", () => {
    const body = withErrorShim(`<!DOCTYPE html><html><body><p>hi</p></body></html>`);
    expect(body.indexOf("<!DOCTYPE html>")).toBe(0);
    expect(body.indexOf("__lexiArtifact")).toBeLessThan(body.indexOf("<p>"));

    const htmlOnly = withErrorShim(`<html><p>hi</p></html>`);
    expect(htmlOnly.indexOf("__lexiArtifact")).toBeLessThan(htmlOnly.indexOf("<p>"));

    const fragment = withErrorShim(`<div>hi</div>`);
    expect(fragment.indexOf("__lexiArtifact")).toBeLessThan(fragment.indexOf("<div>"));
  });

  it("injects exactly once", () => {
    const out = withErrorShim(`<!DOCTYPE html><html><head></head><body></body></html>`);
    // The marker appears once in the post() payload the shim sends.
    expect(out.split("__lexiArtifact:").length - 1).toBe(1);
  });

  it("carries the per-frame token and announces readiness", () => {
    const out = withErrorShim(`<!DOCTYPE html><html><head></head><body></body></html>`, "tok-123");
    expect(out).toContain(`"tok-123"`);
    // Absence of this ping in the parent is what proves scripts never ran at all.
    expect(out).toContain(`post("ready"`);
  });

  it("keeps two frames' reports distinguishable", () => {
    const a = withErrorShim(`<html><head></head></html>`, "tok-a");
    const b = withErrorShim(`<html><head></head></html>`, "tok-b");
    expect(a).toContain(`"tok-a"`);
    expect(a).not.toContain(`"tok-b"`);
    expect(b).toContain(`"tok-b"`);
  });
});

describe("withCorsScripts", () => {
  // Without crossorigin, an exception thrown inside Leaflet (a cross-origin script) is sanitised by
  // the browser to a bare "Script error." with no message or line. Both allowed CDNs send
  // `access-control-allow-origin: *`, so opting in recovers the real message.
  it("marks unpkg and jsdelivr scripts crossorigin", () => {
    const out = withCorsScripts(
      `<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>` +
      `<script src="https://cdn.jsdelivr.net/npm/leaflet/dist/leaflet.js"></script>`);
    expect(out.match(/crossorigin="anonymous"/g)).toHaveLength(2);
    expect(out).toContain(`src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"`);
  });

  it("leaves other hosts alone — crossorigin on a host without CORS headers would BLOCK it", () => {
    const src = `<script src="https://example.com/app.js"></script>`;
    expect(withCorsScripts(src)).toBe(src);
  });

  it("leaves inline scripts and an existing crossorigin untouched", () => {
    const inline = `<script>const a = 1;</script>`;
    expect(withCorsScripts(inline)).toBe(inline);
    const already = `<script crossorigin="use-credentials" src="https://unpkg.com/x.js"></script>`;
    expect(withCorsScripts(already)).toBe(already);
  });

  it("does not touch the leaflet stylesheet <link>", () => {
    const link = `<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>`;
    expect(withCorsScripts(link)).toBe(link);
  });
});
