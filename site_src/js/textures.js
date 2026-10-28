// Procedural 16×16 pixel-art block textures, drawn on canvas on first use (no texture pack).
// Shared by the 3D replay (as three.js textures) and the gallery's isometric thumbnails.
// Deterministic: a seeded RNG per texture, so every page draws the same pixels.
//
// Log v4 items (blockswarm/catalog.py, logged in header.shop.catalog) get a texture derived from
// their family + colour: planks grain, log bark/rings, stone noise, brick pattern in the item colour,
// soft wool, flat concrete, matte terracotta, framed translucent glass / stained glass / panes,
// metal sheen, door panels, a glowing sea lantern, a bookshelf, hay. The v1–v3 block types (wood,
// plank, stone, brick, glass, roof, door) keep their original hand-made textures.
// Generated lazily and cached per item (keyed by catalog vs legacy, since "stone" / "glass" exist in both).
import { itemInfo, typeColor } from './core-adapter.js';

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
const hex = h => (/^#[0-9a-f]{6}$/i.test(h || '') ? [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)] : [128, 128, 128]);
const mul = (c, f) => [c[0] * f, c[1] * f, c[2] * f];
const mix = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
const lum = c => (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
/** Darken light colours / lighten dark ones (mortar, grout, frames that stay visible on any base). */
const contrast = (c, f) => (lum(c) > 0.45 ? mul(c, 1 - f) : mix(c, [255, 255, 255], f * 0.8));

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
const tones = (name, n, lo, hi) => { const r = rng(hash(name)); return Array.from({ length: n }, () => lo + r() * (hi - lo)); };

// ---------- legacy (log v1–v3) faces ----------
const STONE = [128, 129, 131];
function stone(name = 'stone', base = STONE, amp = 0.16) {
  return paint(name, (x, y, r) => {
    let c = noisy(base, r, amp);
    if (r() < 0.09) c = mul(c, 0.8);
    if (r() < 0.04) c = mul(c, 1.15);
    return c;
  });
}
const BRICK = [152, 74, 54], MORTAR = [176, 166, 152];
function redBrick(name = 'brick', base = BRICK, mortar = MORTAR) {
  const tn = tones(name + '-tone', 16, 0.85, 1.15);
  return paint(name, (x, y, r) => {
    const row = y >> 2, off = row % 2 ? 4 : 0;
    if (y % 4 === 3 || (x + off) % 8 === 7) return noisy(mortar, r, 0.08);
    const brickIdx = row * 2 + (((x + off) >> 3) % 2);
    let c = mul(base, tn[brickIdx % 16]);
    c = noisy(c, r, 0.12);
    if (y % 4 === 0) c = mul(c, 1.1);
    return c;
  });
}
const PLANK = [176, 136, 82];
function planks(name = 'plank', base = PLANK, grain = name !== 'plank') {
  return paint(name, (x, y, r) => {
    const row = y >> 2;
    if (y % 4 === 3) return mul(base, 0.62);
    const seam = (row * 7 + 3) % 16;
    if (x === seam) return mul(base, 0.7);
    let c = mul(base, 1 - row % 2 * 0.06);
    if (grain && (x * 3 + row * 5) % 11 === 0) c = mul(c, 0.9); // grain streaks
    c = noisy(c, r, 0.1);
    if (r() < 0.1) c = mul(c, 0.9);
    return c;
  });
}
const BARK = [96, 74, 46], RING_A = [184, 148, 92], RING_B = [158, 122, 72];
function logSide(name = 'wood-side', bark = BARK) {
  const cols = tones(name === 'wood-side' ? 'bark-cols' : name + '-cols', S, 0.8, 1.15);
  return paint(name, (x, y, r) => {
    let c = mul(bark, cols[x]);
    if ((x === 3 || x === 10) && r() < 0.7) c = mul(c, 0.75);
    return noisy(c, r, 0.1);
  });
}
function logTop(name = 'wood-top', bark = BARK, ringA = RING_A, ringB = RING_B) {
  return paint(name, (x, y, r) => {
    const d = Math.max(Math.abs(x - 7.5), Math.abs(y - 7.5));
    if (d > 6.5) return noisy(bark, r, 0.12);
    return noisy(Math.floor(d) % 2 ? ringA : ringB, r, 0.06);
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
/** kind: 'clear' | 'stained' | 'pane' (clear glass with a cross of muntins). */
function glass(name = 'glass', tint = null, kind = 'clear') {
  const frame = tint ? [...mix(tint, [255, 255, 255], 0.35), 255] : [222, 238, 246, 255];
  const fill = tint ? [...tint, 120] : [175, 215, 238, 46];
  return paint(name, (x, y) => {
    if (x === 0 || y === 0 || x === S - 1 || y === S - 1) return frame;
    if (kind === 'pane' && (x === 7 || x === 8 || y === 7 || y === 8)) return [214, 230, 238, 235];
    const streak = (x + y === 10 && x > 2 && x < 8) || (x + y === 13 && x > 6 && x < 11) || (x + y === 6 && x > 1 && x < 4);
    if (streak) return [255, 255, 255, 200];
    return fill;
  });
}
const DOOR = [142, 96, 54];
/** variant: 'single' | 'top' | 'bottom' (a 2-high door is a top over a bottom); iron: rivets, no grain. */
function door(name, variant, base = DOOR, iron = false) {
  return paint(name + '-' + variant, (x, y, r) => {
    const frame = x === 0 || x === S - 1 || (variant !== 'bottom' && y === 0) || (variant !== 'top' && y === S - 1);
    if (frame) return noisy(mul(base, iron ? 0.7 : 0.62), r, 0.06);
    const win = variant === 'top' ? (y >= 3 && y <= 8) : variant === 'single' ? (y >= 2 && y <= 5) : false;
    if (win && (x >= 3 && x <= 6 || x >= 9 && x <= 12)) return (x + y) % 7 === 0 ? [235, 246, 252] : [150, 196, 222];
    if ((variant === 'bottom' || variant === 'single') && x === 12 && (y === 9 || y === 10)) return iron ? [70, 70, 74] : [62, 62, 66];
    if (iron && (x === 2 || x === 13) && y % 5 === 2) return mul(base, 0.6);
    if ((variant === 'bottom' && (y === 4 || y === 11)) || (variant === 'top' && y === 12)) return mul(base, 0.7);
    const c = mul(base, !iron && x % 4 === 0 ? 0.82 : 1);
    return noisy(c, r, iron ? 0.04 : 0.1);
  });
}
const GRASS = [94, 156, 58], DIRT = [134, 96, 66];
function grassTop() { return paint('grass-top', (x, y, r) => { let c = noisy(GRASS, r, 0.22); if (r() < 0.07) c = mul(c, 1.18); return c; }); }
function grassSide() {
  const drip = tones('grass-drips', S, 2, 5).map(Math.floor);
  return paint('grass-side', (x, y, r) => (y < drip[x] ? noisy(GRASS, r, 0.2) : noisy(DIRT, r, 0.18)));
}
function dirt(name = 'dirt', base = DIRT) { return paint(name, (x, y, r) => { let c = noisy(base, r, 0.2); if (r() < 0.06) c = mul(c, 0.75); return c; }); }

const CHEST = [164, 112, 46];
function chestFace(face) {
  return paint('chest-' + face, (x, y, r) => {
    const edge = x === 0 || y === 0 || x === S - 1 || y === S - 1;
    if (edge) return noisy(mul(CHEST, 0.45), r, 0.08);
    if (face !== 'top' && y === 5) return mul(CHEST, 0.45);
    if (face === 'front' && x >= 7 && x <= 8 && y >= 4 && y <= 7) return y === 7 ? [120, 120, 126] : [198, 198, 206];
    const c = mul(CHEST, (face === 'top' ? x : y) % 4 === 0 ? 0.88 : 1);
    return noisy(c, r, 0.08);
  });
}

// ---------- catalog (log v4) styles ----------
/** Brick / stone-brick pattern in the item colour; bw × bh bricks, mortar from the base. */
function bricks(name, base, bw = 8, bh = 4) {
  const tn = tones(name + '-tone', 32, 0.88, 1.12);
  const mortar = contrast(base, 0.32);
  return paint(name, (x, y, r) => {
    const row = Math.floor(y / bh), off = row % 2 ? bw / 2 : 0;
    if (y % bh === bh - 1 || (x + off) % bw === bw - 1) return noisy(mortar, r, 0.06);
    let c = mul(base, tn[(row * 3 + Math.floor((x + off) / bw)) % 32]);
    if (y % bh === 0) c = mul(c, 1.07);
    return noisy(c, r, 0.1);
  });
}
function mossify(c, x, y, r) { return (hash(x * 31 + y * 7) % 5 === 0 || r() < 0.12) ? mix(c, [86, 120, 52], 0.55) : c; }
function cobble(name, base, mossy) {
  const pr = rng(hash(name + '-pts'));
  const pts = Array.from({ length: 9 }, () => [pr() * 16, pr() * 16, 0.82 + pr() * 0.3]);
  return paint(name, (x, y, r) => {
    let d1 = 1e9, d2 = 1e9, best = 0;
    for (let i = 0; i < pts.length; i++) {
      for (const [ox, oy] of [[0, 0], [16, 0], [-16, 0], [0, 16], [0, -16]]) {
        const d = Math.hypot(x + 0.5 - pts[i][0] - ox, y + 0.5 - pts[i][1] - oy);
        if (d < d1) { d2 = d1; d1 = d; best = i; } else if (d < d2) d2 = d;
      }
    }
    let c = d2 - d1 < 1.1 ? mul(base, 0.6) : mul(base, pts[best][2]);
    c = noisy(c, r, 0.1);
    return mossy ? mossify(c, x, y, r) : c;
  });
}
function polished(name, base) {
  return paint(name, (x, y, r) => {
    if (x === 0 || y === 0) return mul(base, 1.12);
    if (x === S - 1 || y === S - 1) return mul(base, 0.78);
    return noisy(base, r, 0.05);
  });
}
function tiles(name, base, n = 2) {
  const step = S / n, tn = tones(name + '-t', n * n, 0.9, 1.1);
  return paint(name, (x, y, r) => {
    if (x % step === step - 1 || y % step === step - 1) return mul(base, 0.68);
    if (x % step === 0 || y % step === 0) return mul(base, 1.08);
    return noisy(mul(base, tn[Math.floor(y / step) * n + Math.floor(x / step)]), r, 0.07);
  });
}
function sandstone(name, base, face) {
  return paint(name + '-' + face, (x, y, r) => {
    if (face === 'top') return noisy(mul(base, 1.04), r, 0.06);
    let c = base;
    if (y < 3) c = mul(base, 1.06);
    else if (y === 3 || y === 12) c = mul(base, 0.86);
    else if (y > 12) c = mul(base, 0.95);
    return noisy(c, r, 0.07);
  });
}
function wool(name, base) {
  return paint(name, (x, y, r) => {
    let c = noisy(base, r, 0.09);
    if ((x + 2 * y) % 7 === 0) c = mul(c, 0.94); // fibre
    if ((2 * x + y) % 9 === 0) c = mul(c, 1.04);
    return c;
  });
}
function concrete(name, base) { return paint(name, (x, y, r) => noisy(base, r, 0.035)); }
function terracotta(name, base) {
  return paint(name, (x, y, r) => {
    let c = noisy(mul(base, 0.92), r, 0.1);
    if (r() < 0.05) c = mul(c, 0.86);
    return c;
  });
}
function metal(name, base) {
  return paint(name, (x, y, r) => {
    if (x === 0 || y === 0) return mix(base, [255, 255, 255], 0.35);
    if (x === S - 1 || y === S - 1) return mul(base, 0.68);
    const d = (x + (S - 1 - y)) / (2 * (S - 1)); // 0 bottom-left → 1 top-right
    let c = mul(base, 0.86 + 0.24 * d);
    if (Math.abs(x - y) <= 1 && x > 3 && x < 12) c = mix(c, [255, 255, 255], 0.3); // sheen streak
    return noisy(c, r, 0.04);
  });
}
function lantern(name, base) {
  const bright = mix(base, [255, 255, 255], 0.55);
  return paint(name, (x, y, r) => {
    if (x === 0 || y === 0 || x === S - 1 || y === S - 1) return mul(base, 0.75);
    const cx = Math.abs(x - 7.5), cy = Math.abs(y - 7.5);
    if (cx < 1.5 || cy < 1.5) return mix(bright, [255, 255, 255], 0.5);
    const d = Math.hypot(cx, cy) / 10;
    return noisy(mix(bright, base, d), r, 0.05);
  });
}
const BOOKS = [[150, 44, 40], [52, 84, 150], [60, 120, 64], [170, 140, 60], [110, 60, 120], [190, 190, 180], [92, 60, 40]];
function bookshelf(name, wood) {
  const br = rng(hash(name + '-books'));
  const cols = Array.from({ length: 32 }, () => BOOKS[Math.floor(br() * BOOKS.length)]);
  const tall = Array.from({ length: 32 }, () => br() < 0.3);
  return paint(name, (x, y, r) => {
    if (y === 0 || y === 7 || y === 8 || y === 15) return noisy(mul(wood, y === 7 || y === 15 ? 0.7 : 1), r, 0.08);
    const shelf = y < 8 ? 0 : 1, top = shelf ? 9 : 1;
    if (x === 0 || x === 15) return mul(wood, 0.8);
    const book = Math.floor((x - 1) / 2) + shelf * 16;
    if ((x - 1) % 2 === 1 && x % 5 === 0) return mul(wood, 0.45); // gap
    if (!tall[book] && y === top) return mul(wood, 0.45);
    return noisy(cols[book], r, 0.1);
  });
}
function hay(name, base, face) {
  return paint(name + '-' + face, (x, y, r) => {
    if (face === 'top') { const d = Math.hypot(x - 7.5, y - 7.5); return noisy(mul(base, 1.05 - (Math.floor(d) % 3 === 0 ? 0.12 : 0)), r, 0.12); }
    if (y === 3 || y === 12) return mul(base, 0.55); // binding bands
    return noisy(mul(base, x % 3 === 0 ? 0.88 : 1.08), r, 0.12);
  });
}
function prismarine(name, base) {
  const alt = mix(base, [70, 140, 160], 0.35);
  return paint(name, (x, y, r) => noisy(mix(base, alt, (Math.sin(x * 1.3 + y * 0.7) + Math.sin(y * 1.9 - x * 0.4) + 2) / 4), r, 0.1));
}

/** Texture style for a catalog item, from its family and id. */
function styleOf(it) {
  const id = it.id, f = it.family;
  if (it.door || f === 'door') return 'door';
  if (f === 'glass') return id.includes('tinted') ? 'tinted' : id.includes('pane') ? 'pane' : id.includes('stained') ? 'stained' : 'glass';
  if (f === 'wood') return id.endsWith('_log') || id.endsWith('_wood') ? 'log' : 'planks';
  if (f === 'wool' || f === 'concrete' || f === 'terracotta') return f;
  if (id === 'bookshelf') return 'bookshelf';
  if (id === 'hay_block') return 'hay';
  if (id === 'sea_lantern') return 'lantern';
  if (f === 'metal' || id === 'copper_block') return 'metal';
  if (id.includes('tiles') || id === 'purpur_block' || id === 'cut_copper') return 'tiles';
  if (id === 'bricks') return 'redbrick';
  if (id.includes('bricks')) return 'bricks';
  if (id.includes('cobble')) return 'cobble';
  if (f === 'sandstone') return id.startsWith('smooth') ? 'polished' : 'sandstone';
  if (id.startsWith('polished') || id.startsWith('smooth') || id === 'quartz_block' || id.startsWith('cut_')) return 'polished';
  if (f === 'earth' || id === 'packed_mud') return 'dirt';
  if (f === 'ocean') return 'prismarine';
  return 'stone';
}

function catalogFaces(it, variant) {
  const base = hex(it.color), name = 'item-' + it.id, same = c => ({ top: c, side: c, bottom: c });
  switch (styleOf(it)) {
    case 'planks': return same(planks(name, base));
    case 'log': {
      const pl = itemInfo(it.id.replace(/_(log|wood)$/, '_planks'));
      const ring = pl && !pl.legacy ? hex(pl.color) : mix(base, [230, 200, 150], 0.5);
      const top = logTop(name + '-top', base, ring, mul(ring, 0.85));
      return { top, side: logSide(name + '-side', base), bottom: top };
    }
    case 'stone': { const c = paint(name, (x, y, r) => { let p = noisy(base, r, 0.16); if (r() < 0.08) p = mul(p, 0.82); if (r() < 0.04) p = mul(p, 1.14); return it.id.includes('mossy') ? mossify(p, x, y, r) : p; }); return same(c); }
    case 'cobble': return same(cobble(name, base, it.id.includes('mossy')));
    case 'bricks': {
      const c = bricks(name, base, 8, 4);
      return same(it.id.includes('mossy') ? paint(name + '-moss', (x, y, r) => {
        const p = c.getContext('2d').getImageData(x, y, 1, 1).data; return mossify([p[0], p[1], p[2]], x, y, r);
      }) : c);
    }
    case 'redbrick': return same(redBrick(name, base, mix(base, [210, 200, 186], 0.6)));
    case 'polished': return same(polished(name, base));
    case 'tiles': return same(tiles(name, base, it.id === 'cut_copper' ? 2 : it.id === 'purpur_block' ? 2 : 4));
    case 'sandstone': { const top = sandstone(name, base, 'top'); return { top, side: sandstone(name, base, 'side'), bottom: top }; }
    case 'wool': return same(wool(name, base));
    case 'concrete': return same(concrete(name, base));
    case 'terracotta': return same(terracotta(name, base));
    case 'glass': return same(glass(name));
    case 'pane': return same(glass(name, null, 'pane'));
    case 'stained': return same(glass(name, base, 'stained'));
    case 'tinted': return same(paint(name, (x, y, r) => (x === 0 || y === 0 || x === S - 1 || y === S - 1 ? mul(base, 1.6) : noisy(base, r, 0.25))));
    case 'metal': return same(metal(name, base));
    case 'lantern': return same(lantern(name, base));
    case 'bookshelf': {
      const wood = hex((itemInfo('oak_planks') || {}).color || '#b8945f');
      const top = planks(name + '-top', wood);
      return { top, side: bookshelf(name, wood), bottom: top };
    }
    case 'hay': { const top = hay(name, base, 'top'); return { top, side: hay(name, base, 'side'), bottom: top }; }
    case 'prismarine': return same(prismarine(name, base));
    case 'dirt': return same(dirt(name, base));
    case 'door': {
      const iron = it.id.includes('iron'), top = planks(name + '-edge', base);
      return { top, side: door(name, variant || 'single', base, iron), bottom: top };
    }
    default: return same(stone(name, base));
  }
}

// ---------- registry ----------
const cache = new Map();
let legacyCache = null;
function legacyFaces() {
  if (legacyCache) return legacyCache;
  const st = stone(), br = redBrick(), pl = planks(), ro = roof(), gl = glass();
  const side = logSide(), top = logTop();
  legacyCache = {
    stone: { top: st, side: st, bottom: st },
    brick: { top: br, side: br, bottom: br },
    plank: { top: pl, side: pl, bottom: pl },
    roof: { top: ro, side: ro, bottom: ro },
    glass: { top: gl, side: gl, bottom: gl },
    wood: { top, side, bottom: top },
    door: { top: pl, side: door('door', 'single'), bottom: pl },
    'door:top': { top: pl, side: door('door', 'top'), bottom: pl },
    'door:bottom': { top: pl, side: door('door', 'bottom'), bottom: pl },
  };
  return legacyCache;
}
const INTERNAL = {
  grass: () => ({ top: grassTop(), side: grassSide(), bottom: dirt() }),
  chest: () => ({ top: chestFace('top'), side: chestFace('side'), front: chestFace('front'), bottom: chestFace('top') }),
  counter: () => { const p = planks('counter', [168, 122, 72]); return { top: p, side: p, bottom: p }; },
  // market stall (gallery thumbnails): striped awning on top, a plank counter with an awning fringe on the sides
  shop: () => {
    const P = [168, 122, 72], RED = [209, 73, 91], WHITE = [244, 241, 234];
    const top = paint('shop-top', (x, y, r) => noisy(Math.floor(x / 4) % 2 ? WHITE : RED, r, 0.05));
    const side = paint('shop-side', (x, y, r) => {
      if (y < 4) return noisy(Math.floor(x / 4) % 2 ? WHITE : RED, r, 0.05);
      if (y === 4) return mul(RED, 0.7);
      if (x < 2 || x > 13) return noisy(mul(P, 0.6), r, 0.08);
      if (y > 8) return noisy(mul(P, y === 9 ? 0.75 : 1), r, 0.1);
      return y === 8 && x > 3 && x < 7 ? [246, 208, 61] : [40, 34, 30];
    });
    return { top, side, bottom: side };
  },
};
/** [v5] A team's market stall: the awning striped in the team colour (cached per colour). */
export function shopFaces(color) {
  const k = 'shop:' + color;
  if (cache.has(k)) return cache.get(k);
  const C = hex(color), WHITE = [244, 241, 234];
  const P = [168, 122, 72];
  const top = paint('shop-top-' + color, (x, y, r) => noisy(Math.floor(x / 4) % 2 ? WHITE : C, r, 0.05));
  const side = paint('shop-side-' + color, (x, y, r) => {
    if (y < 4) return noisy(Math.floor(x / 4) % 2 ? WHITE : C, r, 0.05);
    if (y === 4) return mul(C, 0.7);
    if (x < 2 || x > 13) return noisy(mul(P, 0.6), r, 0.08);
    if (y > 8) return noisy(mul(P, y === 9 ? 0.75 : 1), r, 0.1);
    return y === 8 && x > 3 && x < 7 ? [246, 208, 61] : [40, 34, 30];
  });
  const f = { top, side, bottom: side, awning: top };
  cache.set(k, f);
  return f;
}
function generic(type) {
  const base = hex(typeColor(type));
  const g = paint('generic-' + type, (x, y, r) => {
    if (x === 0 || y === 0 || x === S - 1 || y === S - 1) return mul(base, 0.72);
    return noisy(base, r, 0.18);
  });
  return { top: g, side: g, bottom: g };
}
const splitVariant = v => { const i = String(v).lastIndexOf(':'); return i > 0 ? [v.slice(0, i), v.slice(i + 1)] : [String(v), null]; };

/**
 * Faces {top, side, bottom[, front]} (canvases) for a block type or door variant ("<type>:top" /
 * "<type>:bottom"), or an internal name (grass, chest, counter). Items come from the catalog of the
 * loaded log (v4) or the legacy v1–v3 set; unknown types get a tinted generic face.
 */
export function blockFaces(variant) {
  if (INTERNAL[variant]) { const k = 'I:' + variant; if (!cache.has(k)) cache.set(k, INTERNAL[variant]()); return cache.get(k); }
  const [type, part] = splitVariant(variant);
  const it = itemInfo(type);
  const k = (it.legacy ? 'L:' : 'C:') + variant;
  if (cache.has(k)) return cache.get(k);
  let faces;
  if (it.legacy) {
    const L = legacyFaces();
    faces = L[variant] || L[type] || generic(type);
  } else faces = catalogFaces(it, part);
  cache.set(k, faces);
  return faces;
}
/** Doesn't block sight: glass, panes, stained glass (v4 catalog flag) / "glass" (v1–v3). Accepts door variants. */
export const isTransparent = variant => !!itemInfo(splitVariant(variant)[0]).transparent;
export const isDoor = variant => !!itemInfo(splitVariant(variant)[0]).door;

/**
 * Door variant for a door block, from whether a door sits directly above/below it.
 * has(x, y, z) -> block type or undefined. Returns "<type>" | "<type>:top" | "<type>:bottom".
 */
export function doorKey(has, x, y, z) {
  const t = has(x, y, z);
  const isD = v => v !== undefined && !!itemInfo(v).door;
  const up = isD(has(x, y, z + 1)), down = isD(has(x, y, z - 1));
  if (up && !down) return t + ':bottom';
  if (down) return t + ':top';
  return t;
}

/** A tiny CSS background (data URL) of a block's side face, for legend swatches. */
const swatchCache = new Map();
export function swatchUrl(type) {
  const it = itemInfo(type), k = (it.legacy ? 'L:' : 'C:') + type;
  if (!swatchCache.has(k)) swatchCache.set(k, blockFaces(type).side.toDataURL());
  return swatchCache.get(k);
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
