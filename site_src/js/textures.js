// Procedural 16×16 pixel-art block textures, drawn on canvas at load time (no texture pack).
// Shared by the 3D replay (as three.js textures) and the gallery's isometric thumbnails.
// Deterministic: a seeded RNG per texture, so every page draws the same pixels.
import { typeColor } from './core-adapter.js';

const S = 16;

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hash = s => { let h = 2166136261; for (const c of String(s)) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };
const clamp = v => Math.max(0, Math.min(255, Math.round(v)));
const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const mul = (c, f) => [c[0] * f, c[1] * f, c[2] * f];

/** Paint a size×size canvas from fn(x, y, rnd) -> [r, g, b, a?]. */
function paint(name, fn, size = S) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  const rnd = rng(hash(name));
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const p = fn(x, y, rnd);
    const i = (y * size + x) * 4;
    img.data[i] = clamp(p[0]); img.data[i + 1] = clamp(p[1]); img.data[i + 2] = clamp(p[2]);
    img.data[i + 3] = p.length > 3 ? clamp(p[3]) : 255;
  }
  g.putImageData(img, 0, 0);
  c.dataset.name = name;
  return c;
}
const noisy = (c, rnd, amp) => { const f = 1 + (rnd() - 0.5) * amp; return mul(c, f); };

// ---------- block faces ----------
const STONE = [128, 129, 131];
function stone() {
  return paint('stone', (x, y, r) => {
    let c = noisy(STONE, r, 0.16);
    if (r() < 0.09) c = mul(c, 0.8);
    if (r() < 0.04) c = mul(c, 1.15);
    return c;
  });
}
const BRICK = [152, 74, 54], MORTAR = [176, 166, 152];
function brick() {
  const tone = rng(hash('brick-tone'));
  const tones = Array.from({ length: 16 }, () => 0.85 + tone() * 0.3);
  return paint('brick', (x, y, r) => {
    const row = y >> 2, off = row % 2 ? 4 : 0;
    if (y % 4 === 3 || (x + off) % 8 === 7) return noisy(MORTAR, r, 0.08);
    const brickIdx = row * 2 + (((x + off) >> 3) % 2);
    let c = mul(BRICK, tones[brickIdx % 16]);
    c = noisy(c, r, 0.12);
    if (y % 4 === 0) c = mul(c, 1.1);
    return c;
  });
}
const PLANK = [176, 136, 82];
function plank(name = 'plank', base = PLANK) {
  return paint(name, (x, y, r) => {
    const row = y >> 2;
    if (y % 4 === 3) return mul(base, 0.62);
    const seam = (row * 7 + 3) % 16;
    if (x === seam) return mul(base, 0.7);
    let c = mul(base, 1 - row % 2 * 0.06);
    c = noisy(c, r, 0.1);
    if (r() < 0.1) c = mul(c, 0.9);
    return c;
  });
}
const BARK = [96, 74, 46], RING_A = [184, 148, 92], RING_B = [158, 122, 72];
function logSide() {
  const cr = rng(hash('bark-cols'));
  const cols = Array.from({ length: S }, () => 0.8 + cr() * 0.35);
  return paint('wood-side', (x, y, r) => {
    let c = mul(BARK, cols[x]);
    if ((x === 3 || x === 10) && r() < 0.7) c = mul(c, 0.75);
    return noisy(c, r, 0.1);
  });
}
function logTop() {
  return paint('wood-top', (x, y, r) => {
    const d = Math.max(Math.abs(x - 7.5), Math.abs(y - 7.5));
    if (d > 6.5) return noisy(BARK, r, 0.12);
    return noisy(Math.floor(d) % 2 ? RING_A : RING_B, r, 0.06);
  });
}
const ROOF = [64, 92, 140];
function roof() {
  return paint('roof', (x, y, r) => {
    const row = y >> 2, off = row % 2 ? 4 : 0;
    let c = mul(ROOF, 1.12 - (y % 4) * 0.07);
    if (y % 4 === 3) c = mul(ROOF, 0.6);
    else if ((x + off) % 8 === 0) c = mul(c, 0.72);
    return noisy(c, r, 0.08);
  });
}
function glass() {
  return paint('glass', (x, y) => {
    if (x === 0 || y === 0 || x === S - 1 || y === S - 1) return [222, 238, 246, 255];
    const streak = (x + y === 10 && x > 2 && x < 8) || (x + y === 13 && x > 6 && x < 11) || (x + y === 6 && x > 1 && x < 4);
    if (streak) return [255, 255, 255, 200];
    return [175, 215, 238, 46];
  });
}
const DOOR = [142, 96, 54];
/** variant: 'single' | 'top' | 'bottom' (a 2-high door is a top over a bottom) */
function door(variant) {
  return paint('door-' + variant, (x, y, r) => {
    const frame = x === 0 || x === S - 1 || (variant !== 'bottom' && y === 0) || (variant !== 'top' && y === S - 1);
    if (frame) return noisy(mul(DOOR, 0.62), r, 0.06);
    const win = variant === 'top' ? (y >= 3 && y <= 8) : variant === 'single' ? (y >= 2 && y <= 5) : false;
    if (win && (x >= 3 && x <= 6 || x >= 9 && x <= 12)) return (x + y) % 7 === 0 ? [235, 246, 252] : [150, 196, 222];
    if ((variant === 'bottom' || variant === 'single') && x === 12 && (y === 9 || y === 10)) return [62, 62, 66];
    if ((variant === 'bottom' && (y === 4 || y === 11)) || (variant === 'top' && y === 12)) return mul(DOOR, 0.7);
    let c = mul(DOOR, x % 4 === 0 ? 0.82 : 1);
    return noisy(c, r, 0.1);
  });
}
const GRASS = [94, 156, 58], DIRT = [134, 96, 66];
function grassTop() { return paint('grass-top', (x, y, r) => { let c = noisy(GRASS, r, 0.22); if (r() < 0.07) c = mul(c, 1.18); return c; }); }
function grassSide() {
  const dr = rng(hash('grass-drips'));
  const drip = Array.from({ length: S }, () => 2 + Math.floor(dr() * 3));
  return paint('grass-side', (x, y, r) => (y < drip[x] ? noisy(GRASS, r, 0.2) : noisy(DIRT, r, 0.18)));
}
function dirt() { return paint('dirt', (x, y, r) => { let c = noisy(DIRT, r, 0.2); if (r() < 0.06) c = mul(c, 0.75); return c; }); }

const CHEST = [164, 112, 46];
function chestFace(face) {
  return paint('chest-' + face, (x, y, r) => {
    const edge = x === 0 || y === 0 || x === S - 1 || y === S - 1;
    if (edge) return noisy(mul(CHEST, 0.45), r, 0.08);
    if (face !== 'top' && y === 5) return mul(CHEST, 0.45);
    if (face === 'front' && x >= 7 && x <= 8 && y >= 4 && y <= 7) return y === 7 ? [120, 120, 126] : [198, 198, 206];
    let c = mul(CHEST, (face === 'top' ? x : y) % 4 === 0 ? 0.88 : 1);
    return noisy(c, r, 0.08);
  });
}

function generic(type) {
  const base = hex(typeColor(type));
  return paint('generic-' + type, (x, y, r) => {
    if (x === 0 || y === 0 || x === S - 1 || y === S - 1) return mul(base, 0.72);
    return noisy(base, r, 0.18);
  });
}

// ---------- registry ----------
let cache = null;
function build() {
  const st = stone(), br = brick(), pl = plank(), ro = roof(), gl = glass();
  const side = logSide(), top = logTop();
  const d1 = door('single'), dt = door('top'), db = door('bottom');
  return {
    stone: { top: st, side: st, bottom: st },
    brick: { top: br, side: br, bottom: br },
    plank: { top: pl, side: pl, bottom: pl },
    roof: { top: ro, side: ro, bottom: ro },
    glass: { top: gl, side: gl, bottom: gl },
    wood: { top, side, bottom: top },
    door: { top: pl, side: d1, bottom: pl },
    'door:top': { top: pl, side: dt, bottom: pl },
    'door:bottom': { top: pl, side: db, bottom: pl },
    grass: { top: grassTop(), side: grassSide(), bottom: dirt() },
    chest: { top: chestFace('top'), side: chestFace('side'), front: chestFace('front'), bottom: chestFace('top') },
  };
}
/** Faces {top, side, bottom} (canvases) for a block type; unknown types get a tinted generic face. */
export function blockFaces(type) {
  cache ??= build();
  if (!cache[type]) { const g = generic(type); cache[type] = { top: g, side: g, bottom: g }; }
  return cache[type];
}
export const isTransparent = type => type === 'glass';

/**
 * Door variant for a door block, from whether a door sits directly above/below it.
 * has(x, y, z) -> block type or undefined.
 */
export function doorKey(has, x, y, z) {
  const up = has(x, y, z + 1) === 'door', down = has(x, y, z - 1) === 'door';
  if (up && !down) return 'door:bottom';
  if (down && !up) return 'door:top';
  if (up && down) return 'door:top';
  return 'door';
}

/** A tiny CSS background (data URL) of a block's side face, for legend swatches. */
const swatchCache = {};
export function swatchUrl(type) {
  return (swatchCache[type] ??= blockFaces(type).side.toDataURL());
}

// ---------- agent skins (8×8 faces) ----------
const SKIN = [226, 182, 140];
const HAIRS = [[70, 46, 28], [36, 30, 28], [150, 96, 44], [196, 160, 92], [92, 60, 40], [120, 40, 30], [60, 60, 66], [170, 120, 70]];
export function headFaces(i) {
  const hair = HAIRS[i % HAIRS.length];
  const p = (name, fn) => paint(`head${i}-${name}`, fn, 8);
  return {
    front: p('front', (x, y, r) => {
      if (y < 2 || (y === 2 && (x === 0 || x === 7))) return noisy(hair, r, 0.12);
      if (y === 4 && (x === 1 || x === 6)) return [245, 245, 245];
      if (y === 4 && (x === 2 || x === 5)) return [52, 74, 140];
      if (y === 6 && (x === 3 || x === 4)) return mul(SKIN, 0.7);
      return noisy(SKIN, r, 0.05);
    }),
    side: p('side', (x, y, r) => (y < 3 || (y < 5 && x > 4) ? noisy(hair, r, 0.12) : noisy(SKIN, r, 0.05))),
    back: p('back', (x, y, r) => (y < 6 ? noisy(hair, r, 0.12) : noisy(SKIN, r, 0.05))),
    top: p('top', (x, y, r) => noisy(hair, r, 0.12)),
    bottom: p('bottom', (x, y, r) => noisy(SKIN, r, 0.05)),
  };
}
/** Light grey noise to multiply with a material colour (cloth look). */
let clothCache = null;
export function clothTexture() {
  return (clothCache ??= paint('cloth', (x, y, r) => { const v = 225 + (r() - 0.5) * 50; return [v, v, v]; }, 8));
}
