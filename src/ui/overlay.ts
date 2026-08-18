// Canvas overlay: draggable JWildfire-style transform triangles + pan/zoom.
// Each triangle visualizes an xform's affine: O = translation (c,f),
// X = O + (a,d), Y = O + (b,e). Dragging a handle edits the affine directly.
import { App, XFORM_COLORS } from './common';

const HANDLE_R = 14; // hit radius in CSS px

type DragState =
  | { kind: 'handle'; xi: number; vertex: 0 | 1 | 2 }
  | { kind: 'pan'; startX: number; startY: number; startCX: number; startCY: number }
  | null;

export function buildOverlay(app: App, overlay: HTMLCanvasElement, wrap: HTMLElement) {
  const ctx = overlay.getContext('2d')!;
  let visible = true;
  let drag: DragState = null;
  let dpr = window.devicePixelRatio || 1;

  const flameOf = () => app.flame;
  /** the selected escape-time layer (its view is navigated instead of the flame's; no triangles) */
  const esc = () => app.escapeLayer?.escape ?? null;
  /** what pan/zoom edit: the escape layer's view or the flame camera (same field names) */
  const viewOf = () => esc() ?? flameOf();

  function ppu(): number {
    return 0.25 * Math.min(overlay.width, overlay.height) * viewOf().zoom;
  }

  // world -> device px
  function w2s(x: number, y: number): [number, number] {
    const f = flameOf();
    const ox = x - f.centerX;
    const oy = y - f.centerY;
    const ca = Math.cos(f.rotation), sa = Math.sin(f.rotation);
    const rx = ox * ca - oy * sa;
    const ry = ox * sa + oy * ca;
    // +y is down on screen (flam3/JWildfire convention; matches the render kernel)
    return [rx * ppu() + overlay.width / 2, ry * ppu() + overlay.height / 2];
  }

  // device px -> world
  function s2w(px: number, py: number): [number, number] {
    const e = esc();
    if (e) {
      // escape layers: +y up, world = centre + R(rotation)·screen offset (escapeRenderer.ts)
      const dx = (px - overlay.width / 2) / ppu(), dy = (overlay.height / 2 - py) / ppu();
      const cs = Math.cos(e.rotation), sn = Math.sin(e.rotation);
      return [e.centerX + cs * dx - sn * dy, e.centerY + sn * dx + cs * dy];
    }
    const f = flameOf();
    const rx = (px - overlay.width / 2) / ppu();
    const ry = (py - overlay.height / 2) / ppu();
    const ca = Math.cos(-f.rotation), sa = Math.sin(-f.rotation);
    return [rx * ca - ry * sa + f.centerX, rx * sa + ry * ca + f.centerY];
  }

  function triangleOf(xi: number): [number, number][] {
    const ly = app.activeLayer;
    const x = xi === -1 ? ly.final : ly.xforms[xi];
    if (!x) return [];
    const [a, b, c, d, e, fq] = x.affine;
    return [
      [c, fq],           // O
      [a + c, d + fq],   // X
      [b + c, e + fq],   // Y
    ];
  }

  function draw() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!visible || esc()) return;
    const ly = app.activeLayer;
    const items: number[] = ly.xforms.map((_, i) => i);
    if (ly.final) items.push(-1);
    for (const xi of items) {
      const selected = app.selected === xi;
      const color = xi === -1 ? '#ffffff' : XFORM_COLORS[xi % XFORM_COLORS.length];
      const tri = triangleOf(xi).map(([wx, wy]) => w2s(wx, wy));
      if (tri.length < 3) continue;
      ctx.lineWidth = (selected ? 2 : 1) * dpr;
      ctx.strokeStyle = color;
      ctx.globalAlpha = selected ? 0.95 : 0.42;
      ctx.setLineDash(selected ? [] : [5 * dpr, 5 * dpr]);
      ctx.beginPath();
      ctx.moveTo(tri[0][0], tri[0][1]);
      ctx.lineTo(tri[1][0], tri[1][1]);
      ctx.lineTo(tri[2][0], tri[2][1]);
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      // Handles: O square, X circle, Y circle
      const labels = ['O', 'X', 'Y'];
      tri.forEach(([hx, hy], vi) => {
        ctx.beginPath();
        const r = (selected ? 5.5 : 4) * dpr;
        if (vi === 0) {
          ctx.rect(hx - r, hy - r, r * 2, r * 2);
        } else {
          ctx.arc(hx, hy, r, 0, Math.PI * 2);
        }
        ctx.fillStyle = color;
        ctx.fill();
        if (selected) {
          ctx.fillStyle = '#000';
          ctx.font = `${8 * dpr}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(labels[vi], hx, hy);
        }
      });
      ctx.globalAlpha = 1;
    }
  }

  function hitTest(px: number, py: number): { xi: number; vertex: 0 | 1 | 2 } | null {
    const ly = app.activeLayer;
    const items: number[] = ly.xforms.map((_, i) => i);
    if (ly.final) items.push(-1);
    // Prefer the selected xform's handles
    items.sort((a, b) => (a === app.selected ? -1 : 0) - (b === app.selected ? -1 : 0));
    const rr = HANDLE_R * dpr;
    for (const xi of items) {
      const tri = triangleOf(xi).map(([wx, wy]) => w2s(wx, wy));
      for (let vi = 0; vi < 3; vi++) {
        const dx = tri[vi][0] - px, dy = tri[vi][1] - py;
        if (dx * dx + dy * dy < rr * rr) return { xi, vertex: vi as 0 | 1 | 2 };
      }
    }
    return null;
  }

  function evPos(e: PointerEvent): [number, number] {
    const rect = overlay.getBoundingClientRect();
    return [
      ((e.clientX - rect.left) / rect.width) * overlay.width,
      ((e.clientY - rect.top) / rect.height) * overlay.height,
    ];
  }

  overlay.addEventListener('pointerdown', (e) => {
    const [px, py] = evPos(e);
    const hit = visible && !esc() ? hitTest(px, py) : null;
    if (hit) {
      if (app.selected !== hit.xi) app.select(hit.xi);
      drag = { kind: 'handle', xi: hit.xi, vertex: hit.vertex };
    } else {
      drag = {
        kind: 'pan', startX: px, startY: py,
        startCX: viewOf().centerX, startCY: viewOf().centerY,
      };
    }
    overlay.setPointerCapture(e.pointerId);
  });

  overlay.addEventListener('pointermove', (e) => {
    const [px, py] = evPos(e);
    if (!drag) {
      overlay.style.cursor = visible && !esc() && hitTest(px, py) ? 'grab' : 'crosshair';
      return;
    }
    if (drag.kind === 'pan') {
      const e = esc();
      const scale = ppu();
      const dx = (px - drag.startX) / scale;
      const dy = (py - drag.startY) / scale;
      if (e) {
        const cs = Math.cos(e.rotation), sn = Math.sin(e.rotation);
        e.centerX = drag.startCX - (cs * dx + sn * dy);
        e.centerY = drag.startCY - (sn * dx - cs * dy);
      } else {
        const f = flameOf();
        const ca = Math.cos(-f.rotation), sa = Math.sin(-f.rotation);
        f.centerX = drag.startCX - (dx * ca - dy * sa);
        f.centerY = drag.startCY - (dx * sa + dy * ca);
      }
      app.commit('overlay-view');
      draw();
      return;
    }
    // Handle drag → edit affine
    const ly = app.activeLayer;
    const x = drag.xi === -1 ? ly.final : ly.xforms[drag.xi];
    if (!x) return;
    const [wx, wy] = s2w(px, py);
    const [a, b, , d, e2] = x.affine;
    if (drag.vertex === 0) {
      x.affine[2] = wx;
      x.affine[5] = wy;
    } else if (drag.vertex === 1) {
      x.affine[0] = wx - x.affine[2];
      x.affine[3] = wy - x.affine[5];
      void a; void d;
    } else {
      x.affine[1] = wx - x.affine[2];
      x.affine[4] = wy - x.affine[5];
      void b; void e2;
    }
    app.commit('overlay');
    draw();
  });

  const endDrag = () => { drag = null; };
  overlay.addEventListener('pointerup', endDrag);
  overlay.addEventListener('pointercancel', endDrag);

  overlay.addEventListener('wheel', (e) => {
    e.preventDefault();
    const f = viewOf();
    const [px, py] = evPos(e as unknown as PointerEvent);
    const [wxBefore, wyBefore] = s2w(px, py);
    const factor = Math.pow(1.0016, -e.deltaY);
    f.zoom = Math.min(esc() ? 1e12 : 64, Math.max(0.02, f.zoom * factor)); // escape views zoom deep (f32 blurs past ~1e5)
    const [wxAfter, wyAfter] = s2w(px, py);
    f.centerX += wxBefore - wxAfter;
    f.centerY += wyBefore - wyAfter;
    app.commit('overlay-view');
    draw();
  }, { passive: false });

  const sync = () => {
    dpr = window.devicePixelRatio || 1;
    overlay.width = app.renderer.width;
    overlay.height = app.renderer.height;
    draw();
  };

  app.on('flame', draw);
  app.on('comp', draw);
  app.on('select', draw);
  app.on('tone', draw);
  app.on('preview', draw);

  return {
    draw,
    sync,
    setVisible(v: boolean) { visible = v; draw(); },
    get visible() { return visible; },
  };
}
