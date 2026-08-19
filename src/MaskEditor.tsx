import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

/**
 * Paint a mask over an attached image to mark the region to edit. The user brushes (red) over the
 * area to change; on save we export a black/white PNG at the image's NATURAL resolution (white =
 * edit) which the backend inpaints — so only the painted region changes and the rest of the photo
 * stays pixel-identical. Passing an empty string back clears any existing mask.
 */
export function MaskEditor({ path, onSave, onClose }: {
  path: string;
  onSave: (maskDataUrl: string) => void;
  onClose: () => void;
}) {
  const imgCanvas = useRef<HTMLCanvasElement>(null);   // base image (display res)
  const overlay = useRef<HTMLCanvasElement>(null);     // red strokes (display res), receives input
  const mask = useRef<HTMLCanvasElement | null>(null); // offscreen b/w mask at natural res
  const img = useRef<HTMLImageElement | null>(null);
  const drawing = useRef(false);
  const paintedRef = useRef(false);

  const [brush, setBrush] = useState(36);
  const [erase, setErase] = useState(false);
  const [ready, setReady] = useState(false);
  const [painted, setPainted] = useState(false);
  const [dims, setDims] = useState({ w: 0, h: 0, scale: 1 });

  useEffect(() => {
    let cancelled = false;
    invoke<string>("read_image_data_url", { path }).then(url => {
      if (cancelled || !url) return;
      const im = new Image();
      im.onload = () => {
        if (cancelled) return;
        img.current = im;
        const maxW = 560, maxH = 480;
        const scale = Math.min(maxW / im.naturalWidth, maxH / im.naturalHeight, 1);
        const w = Math.round(im.naturalWidth * scale);
        const h = Math.round(im.naturalHeight * scale);
        setDims({ w, h, scale });
        const mc = document.createElement("canvas");
        mc.width = im.naturalWidth; mc.height = im.naturalHeight;
        const mx = mc.getContext("2d")!;
        mx.fillStyle = "#000"; mx.fillRect(0, 0, mc.width, mc.height);
        mask.current = mc;
        setReady(true);
      };
      im.src = url;
    });
    return () => { cancelled = true; };
  }, [path]);

  // Draw the base image once the canvas is sized.
  useEffect(() => {
    if (!ready || !img.current || !imgCanvas.current) return;
    const ctx = imgCanvas.current.getContext("2d")!;
    ctx.clearRect(0, 0, dims.w, dims.h);
    ctx.drawImage(img.current, 0, 0, dims.w, dims.h);
  }, [ready, dims]);

  const pointAt = (e: React.PointerEvent) => {
    const r = overlay.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const paint = (x: number, y: number) => {
    const ov = overlay.current!.getContext("2d")!;
    const mx = mask.current!.getContext("2d")!;
    const s = dims.scale || 1;
    if (erase) {
      ov.save(); ov.globalCompositeOperation = "destination-out";
      ov.beginPath(); ov.arc(x, y, brush, 0, Math.PI * 2); ov.fill(); ov.restore();
      mx.fillStyle = "#000";
    } else {
      ov.globalCompositeOperation = "source-over";
      ov.fillStyle = "rgba(239,68,68,0.5)";
      ov.beginPath(); ov.arc(x, y, brush, 0, Math.PI * 2); ov.fill();
      mx.fillStyle = "#fff";
      if (!paintedRef.current) { paintedRef.current = true; setPainted(true); }
    }
    mx.beginPath(); mx.arc(x / s, y / s, brush / s, 0, Math.PI * 2); mx.fill();
  };

  const clear = () => {
    overlay.current?.getContext("2d")!.clearRect(0, 0, dims.w, dims.h);
    const mc = mask.current;
    if (mc) { const mx = mc.getContext("2d")!; mx.fillStyle = "#000"; mx.fillRect(0, 0, mc.width, mc.height); }
    paintedRef.current = false;
    setPainted(false);
  };

  const save = () => {
    if (!paintedRef.current || !mask.current) { onSave(""); onClose(); return; }
    onSave(mask.current.toDataURL("image/png"));
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div onClick={e => e.stopPropagation()}
        style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 18, maxWidth: 640 }}>
        <div style={{ fontWeight: 600, marginBottom: 4, color: "var(--text)" }}>Mark the area to edit</div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 12 }}>
          Paint over the part of the image you want changed. Everything you don't paint stays exactly as-is.
        </div>

        <div style={{ position: "relative", width: dims.w, height: dims.h, margin: "0 auto",
          borderRadius: 8, overflow: "hidden", background: "var(--surface2)" }}>
          <canvas ref={imgCanvas} width={dims.w} height={dims.h}
            style={{ position: "absolute", inset: 0, display: "block" }} />
          <canvas ref={overlay} width={dims.w} height={dims.h}
            style={{ position: "absolute", inset: 0, display: "block", cursor: "crosshair", touchAction: "none" }}
            onPointerDown={e => { (e.target as Element).setPointerCapture(e.pointerId); drawing.current = true; const p = pointAt(e); paint(p.x, p.y); }}
            onPointerMove={e => { if (!drawing.current) return; const p = pointAt(e); paint(p.x, p.y); }}
            onPointerUp={() => { drawing.current = false; }}
            onPointerLeave={() => { drawing.current = false; }} />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 14, flexWrap: "wrap" }}>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "var(--text-secondary)" }}>
            Brush
            <input type="range" min={8} max={90} value={brush} onChange={e => setBrush(+e.target.value)} />
          </label>
          <button className="btn" onClick={() => setErase(v => !v)}
            style={{ background: erase ? "var(--accent)" : "var(--surface2)", color: erase ? "#fff" : "var(--text)" }}>
            {erase ? "Erasing" : "Erase"}
          </button>
          <button className="btn" onClick={clear} style={{ background: "var(--surface2)", color: "var(--text)" }}>Clear</button>
          <div style={{ flex: 1 }} />
          <button className="btn" onClick={onClose} style={{ background: "var(--surface2)", color: "var(--text)" }}>Cancel</button>
          <button className="btn" onClick={save} disabled={!painted}
            style={{ background: painted ? "var(--accent)" : "var(--surface3)", color: "#fff", opacity: painted ? 1 : 0.6 }}>
            {painted ? "Use this region" : "Nothing painted"}
          </button>
        </div>
      </div>
    </div>
  );
}
