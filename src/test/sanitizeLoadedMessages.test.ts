import { describe, it, expect } from "vitest";
import { sanitizeLoadedMessages } from "../App";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const msg = (m: any) => m;

describe("sanitizeLoadedMessages", () => {
  it("clears streaming/status so a reopened chat shows no eternal thinking dots", () => {
    const out = sanitizeLoadedMessages([
      msg({ id: "1", role: "assistant", text: "partial answer", streaming: true, status: "Selecting tools…" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].streaming).toBeUndefined();
    expect(out[0].status).toBeUndefined();
    expect(out[0].text).toBe("partial answer"); // the text it did produce is kept
  });

  it("drops an assistant message interrupted with nothing renderable", () => {
    // This is the exact shape that rendered forever-spinning dots: streaming, no text, no tools.
    const out = sanitizeLoadedMessages([
      msg({ id: "1", role: "user", text: "hi" }),
      msg({ id: "2", role: "assistant", text: "", streaming: true }),
    ]);
    expect(out.map(m => m.id)).toEqual(["1"]);
  });

  it("keeps a tool-only assistant message (no text but has tool calls)", () => {
    const out = sanitizeLoadedMessages([
      msg({ id: "1", role: "assistant", text: "", streaming: true, toolCalls: [{ name: "read_file", args: "{}" }] }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].streaming).toBeUndefined();
  });

  it("keeps an assistant message that produced only an artifact or image", () => {
    const out = sanitizeLoadedMessages([
      msg({ id: "1", role: "assistant", text: "", artifact: { title: "T", html: "<p>x</p>" } }),
      msg({ id: "2", role: "assistant", text: "", toolImages: ["data:image/png;base64,AAAA"] }),
    ]);
    expect(out.map(m => m.id)).toEqual(["1", "2"]);
  });

  it("leaves a normal, finished conversation untouched", () => {
    const input = [
      msg({ id: "1", role: "user", text: "hi" }),
      msg({ id: "2", role: "assistant", text: "hello" }),
    ];
    expect(sanitizeLoadedMessages(input)).toEqual(input);
  });
});
