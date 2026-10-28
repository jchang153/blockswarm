// Isometric 2D-canvas rendering of a build (gallery thumbnails). Cheap: no WebGL context per card.
// Uses the same procedural textures as the 3D replay. View from the south-east, like the replay camera.
import { blockFaces, doorKey, isTransparent, isDoor, shopFaces } from './textures.js';
import { teamColor } from './core-adapter.js';

// [v5] grass top with the team-battle line painted along its east (edge 'e') or west (edge 'w') edge
const lineCache = new Map();
function grassLine(edge) {
  if (lineCache.has(edge)) return lineCache.get(edge);
  const base = blockFaces('grass');
  const c = document.createElement('canvas'); c.width = c.height = 16;
  const g = c.getContext('2d');
  g.drawImage(base.top, 0, 0);
  g.fillStyle = '#fff8d6'; g.fillRect(edge === 'e' ? 14 : 0, 0, 2, 16);
  const f = { ...base, top: c };
  lineCache.set(edge, f);
  return f;
}

/**
 * Draw blocks on a canvas (sized to its CSS box × devicePixelRatio).
 * build = {final_blocks: [[x, y, z, type], ...], grid: {width, depth, height}, chest_pos: [x, y, z] | null,
 *          shop_pos: [x, y, z] | null (log v4), shops: {team: [x, y, z]} | null and divider_x (log v5 team battle;
 *          team_order optional)}. Item types resolve through the catalog registered in core
 *          (the caller sets it per episode with setCatalog for v4 summaries).
 */
export function drawIso(canvas, build, { pad = 0.08, island = 2 } = {}) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.max(1, Math.round(canvas.clientWidth * dpr)), ch = Math.max(1, Math.round(canvas.clientHeight * dpr));
  canvas.width = cw; canvas.height = ch;
  const g = canvas.getContext('2d');
  g.imageSmoothingEnabled = false;

  const grid = build.grid || { width: 16, depth: 16, height: 8 };
  const cells = new Map();
  const k = (x, y, z) => x + ',' + y + ',' + z;
  for (const [x, y, z, t] of build.final_blocks || []) cells.set(k(x, y, z), t);
  const cp = Array.isArray(build.chest_pos) ? build.chest_pos : null;
  if (cp) cells.set(k(cp[0], cp[1], cp[2]), '__chest');
  const teamShops = build.shops && typeof build.shops === 'object' ? build.shops : null;
  const teams = teamShops ? { names: Array.isArray(build.team_order) ? build.team_order : Object.keys(teamShops) } : null;
  if (teamShops) {
    for (const t in teamShops) { const p = teamShops[t]; if (Array.isArray(p)) cells.set(k(p[0], p[1], p[2]), '__shop:' + t); }
  } else {
    const sp = Array.isArray(build.shop_pos) ? build.shop_pos : null;
    if (sp) cells.set(k(sp[0], sp[1], sp[2]), '__shop');
  }
  const div = teamShops && typeof build.divider_x === 'number' ? build.divider_x : null;

  // ground island: bounding box of the build (+ margin), clamped to the grid
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const key of cells.keys()) {
    const [x, y] = key.split(',').map(Number);
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  if (!isFinite(x0)) { x0 = 0; y0 = 0; x1 = grid.width - 1; y1 = grid.depth - 1; }
  x0 = Math.max(0, x0 - island); y0 = Math.max(0, y0 - island);
  x1 = Math.min(grid.width - 1, x1 + island); y1 = Math.min(grid.depth - 1, y1 + island);
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++)
    cells.set(k(x, y, -1), div === null ? 'grass' : x === div - 1 ? 'grass:e' : x === div ? 'grass:w' : 'grass');

  // projection (unit a = 1): +x → right-down, +y → right-up, +z → up
  const EX = [1, 0.5], EY = [1, -0.5], EZ = [0, -1.18];
  const P = (x, y, z) => [x * EX[0] + y * EY[0] + z * EZ[0], x * EX[1] + y * EY[1] + z * EZ[1]];
  let zMax = -1;
  for (const key of cells.keys()) zMax = Math.max(zMax, +key.split(',')[2]);
  const corners = [P(x0, y0, -1), P(x1 + 1, y0, -1), P(x0, y1 + 1, -1), P(x1 + 1, y1 + 1, -1),
    P(x0, y0, zMax + 1), P(x1 + 1, y0, zMax + 1), P(x0, y1 + 1, zMax + 1), P(x1 + 1, y1 + 1, zMax + 1)];
  const bx0 = Math.min(...corners.map(c => c[0])), bx1 = Math.max(...corners.map(c => c[0]));
  const by0 = Math.min(...corners.map(c => c[1])), by1 = Math.max(...corners.map(c => c[1]));
  const s = Math.min(cw * (1 - 2 * pad) / (bx1 - bx0), ch * (1 - 2 * pad) / (by1 - by0));
  const ox = (cw - (bx1 - bx0) * s) / 2 - bx0 * s, oy = (ch - (by1 - by0) * s) / 2 - by0 * s;
  const S = (x, y, z) => { const p = P(x, y, z); return [ox + p[0] * s, oy + p[1] * s]; };
  const vec = v => [v[0] * s, v[1] * s];
  const ex = vec(EX), ey = vec(EY), ez = vec(EZ);

  const special = t => t === '__chest' || t.startsWith('__shop') || t.startsWith('grass');
  const opaqueAt = (x, y, z) => { const t = cells.get(k(x, y, z)); return t !== undefined && (special(t) || !isTransparent(t)); };
  const typeAt = (x, y, z) => cells.get(k(x, y, z));
  const face = (img, o, u, v, shade) => {
    g.setTransform(u[0] / 16, u[1] / 16, v[0] / 16, v[1] / 16, o[0], o[1]);
    g.drawImage(img, -0.2, -0.2, 16.4, 16.4);
    if (shade) { g.fillStyle = `rgba(0,0,0,${shade})`; g.fillRect(-0.2, -0.2, 16.4, 16.4); }
  };

  const list = [...cells.entries()].map(([key, t]) => { const [x, y, z] = key.split(',').map(Number); return { x, y, z, t }; });
  list.sort((a, b) => (a.x - a.y + a.z) - (b.x - b.y + b.z) || a.z - b.z);
  const neg = v => [-v[0], -v[1]];
  for (const { x, y, z, t } of list) {
    const isChest = t === '__chest';
    const faces = isChest ? blockFaces('chest') : t === '__shop' ? blockFaces('shop') : t.startsWith('__shop:') ? shopFaces(teamColor(teams, t.slice(7)))
      : t === 'grass' ? blockFaces('grass') : t.startsWith('grass:') ? grassLine(t.slice(6))
      : blockFaces(isDoor(t) ? doorKey(typeAt, x, y, z) : t);
    const glassy = !special(t) && isTransparent(t);
    const skip = (nx, ny, nz) => opaqueAt(nx, ny, nz) || (glassy && typeAt(nx, ny, nz) === t);
    // south face (y = y plane), seen from -y; texture u → +x, v → down
    if (!skip(x, y - 1, z)) face(isChest ? faces.front : faces.side, S(x, y, z + 1), ex, neg(ez), glassy ? 0.05 : 0.28);
    // east face (x + 1 plane); u → +y
    if (!skip(x + 1, y, z)) face(faces.side, S(x + 1, y, z + 1), ey, neg(ez), glassy ? 0.03 : 0.14);
    // top face
    if (!skip(x, y, z + 1)) face(faces.top, S(x, y, z + 1), ex, ey, 0);
  }
  g.setTransform(1, 0, 0, 1, 0, 0);
}
