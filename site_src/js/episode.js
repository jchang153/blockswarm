// Replay page: three.js scene + transport + chat/ticker panels + results card.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  loadEpisodeIndex, loadEpisodeLog, posAtTime, agentPoseAt, describeEvent, agentColors, agentTotals, finalScore, key,
  finalResults, purchaseOf, leaderboard, deployTopK, itemInfo,
} from './core-adapter.js';
import { blockFaces, doorKey, isTransparent, isDoor, headFaces, clothTexture, swatchUrl } from './textures.js';
import {
  $, esc, fmtClock, fmtDuration, fmtScore, scoreBand, SCORE_COLORS, agentLabel, shortModel, reasonLabel,
  prefersReducedMotion, initTheme, onThemeChange, isDark, toast,
} from './ui.js';

initTheme();

const SPEEDS = [1, 2, 5, 10, 30];
const R = {
  ep: null, log: null, built: null, agents: [], colors: {}, models: {},
  pos: 0, time: 0, tEnd: 1, playing: false, speed: 5,
  showInterior: false, showNames: true, orbit: false, showNotes: false,
  chatItems: [], chatShown: -1, resultsShownAuto: false,
};

// ---------------------------------------------------------------- three.js scene
const sceneEl = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.className = 'gl';
sceneEl.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.maxPolarAngle = Math.PI * 0.49;
controls.autoRotateSpeed = 0.6;

const hemi = new THREE.HemisphereLight(0xdcefff, 0x6d5a3c, 1.5);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff3df, 2.3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
sun.shadow.radius = 3;
scene.add(sun, sun.target);
const amb = new THREE.AmbientLight(0xffffff, 0.25);
scene.add(amb);

// log (x, y, z), z up → three (x, z, -y), y up. Blocks are centred on cells; agents stand on the cell floor.
const toThree = (x, y, z) => new THREE.Vector3(x + 0.5, z + 0.5, -(y + 0.5));
const feetToThree = (x, y, z) => new THREE.Vector3(x + 0.5, z, -(y + 0.5));

function tex(canvas, repeat) {
  const t = new THREE.CanvasTexture(canvas);
  t.magFilter = THREE.NearestFilter;
  t.minFilter = repeat ? THREE.NearestMipmapLinearFilter : THREE.NearestFilter;
  t.generateMipmaps = !!repeat;
  t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); }
  return t;
}
const texCache = new Map();
const cachedTex = c => { if (!texCache.has(c)) texCache.set(c, tex(c)); return texCache.get(c); };

const matCache = new Map();
/** Six materials in BoxGeometry face order (+x, -x, +y top, -y bottom, +z south/front, -z north). */
function blockMaterials(variant) {
  if (matCache.has(variant)) return matCache.get(variant);
  const f = blockFaces(variant === '__chest' ? 'chest' : variant);
  const glass = isTransparent(variant);
  const m = c => new THREE.MeshLambertMaterial(glass
    ? { map: cachedTex(c), transparent: true, depthWrite: false, side: THREE.DoubleSide }
    : { map: cachedTex(c) });
  const side = m(f.side), top = m(f.top), bottom = m(f.bottom);
  const mats = [side, side, top, bottom, f.front ? m(f.front) : side, side];
  matCache.set(variant, mats);
  return mats;
}

const world = { root: null, blocks: new Map(), ghosts: new Map(), avatars: new Map(), chest: null, shop: null, popping: new Set() };
const boxGeo = new THREE.BoxGeometry(1, 1, 1);

function dims() { const h = R.log.header, g = h.grid || h.config; return { W: g.width, D: g.depth, H: g.height }; }

function buildStatic() {
  if (world.root) scene.remove(world.root);
  world.root = new THREE.Group(); scene.add(world.root);
  world.blocks.clear(); world.ghosts.clear(); world.avatars.clear(); world.popping.clear();
  const { W, D } = dims();
  // grass island: one grass layer over two dirt layers
  const g = blockFaces('grass');
  const grassTop = tex(g.top, [W, D]);
  const sideX = tex(g.side, [D, 1]), sideZ = tex(g.side, [W, 1]);
  const dirtTop = tex(g.bottom, [W, D]);
  const L = t => new THREE.MeshLambertMaterial({ map: t });
  const slab = new THREE.Mesh(new THREE.BoxGeometry(W, 1, D), [L(sideX), L(sideX), L(grassTop), L(dirtTop), L(sideZ), L(sideZ)]);
  slab.position.set(W / 2, -0.5, -D / 2);
  slab.receiveShadow = true;
  world.root.add(slab);
  const dX = tex(g.bottom, [D, 2]), dZ = tex(g.bottom, [W, 2]);
  const dirtSlab = new THREE.Mesh(new THREE.BoxGeometry(W, 2, D), [L(dX), L(dX), L(dirtTop), L(dirtTop), L(dZ), L(dZ)]);
  dirtSlab.position.set(W / 2, -2, -D / 2);
  world.root.add(dirtSlab);
  // sun + shadow frustum sized to the grid
  const c = new THREE.Vector3(W / 2, 0, -D / 2);
  sun.position.set(c.x - W * 0.75, 24, c.z + D * 0.55); // from the south-west: south faces lit, shadows fall north-east
  sun.target.position.copy(c);
  const r = Math.max(W, D) * 0.9;
  Object.assign(sun.shadow.camera, { left: -r, right: r, top: r, bottom: -r, near: 1, far: 90 });
  sun.shadow.camera.updateProjectionMatrix();
  buildChest();
  buildShop();
  buildAvatars();
  applyTheme();
}

// ---------------------------------------------------------------- sky + theme
function gradientTexture(stops) {
  const c = document.createElement('canvas'); c.width = 2; c.height = 256;
  const g = c.getContext('2d'); const grd = g.createLinearGradient(0, 0, 0, 256);
  stops.forEach(([o, col]) => grd.addColorStop(o, col));
  g.fillStyle = grd; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const SKY = {
  light: { stops: [[0, '#5fa8e8'], [0.62, '#a9d4f5'], [1, '#e4f2fb']], fog: '#cfe6f7', hemi: [0xdcefff, 0x6d5a3c, 1.5], sun: [0xfff3df, 2.3], amb: 0.25 },
  dark: { stops: [[0, '#0b1426'], [0.6, '#1d2d4d'], [1, '#3a4a6a']], fog: '#2a3854', hemi: [0x9fb4e0, 0x2a2418, 1.0], sun: [0xc9d6ff, 1.5], amb: 0.22 },
};
let skyTex = null;
function applyTheme() {
  const s = isDark() ? SKY.dark : SKY.light;
  if (skyTex) skyTex.dispose();
  skyTex = gradientTexture(s.stops);
  scene.background = skyTex;
  const r = R.log ? Math.max(dims().W, dims().D) : 16;
  scene.fog = new THREE.Fog(s.fog, r * 3, r * 9);
  hemi.color.set(s.hemi[0]); hemi.groundColor.set(s.hemi[1]); hemi.intensity = s.hemi[2];
  sun.color.set(s.sun[0]); sun.intensity = s.sun[1];
  amb.intensity = s.amb;
  sceneEl.style.background = `linear-gradient(${s.stops.map(x => x[1]).join(',')})`;
}
onThemeChange(applyTheme);

// ---------------------------------------------------------------- chest
function buildChest() {
  world.chest = null;
  const h = R.log.header;
  const cp = h.chest_pos || (h.config && h.config.chest_pos);
  if (!Array.isArray(cp)) return;
  const mats = blockMaterials('__chest');
  const g = new THREE.Group();
  g.position.copy(feetToThree(cp[0], cp[1], cp[2]));
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.875, 0.62, 0.875), mats);
  body.position.y = 0.31; body.castShadow = body.receiveShadow = true; g.add(body);
  const hinge = new THREE.Group(); hinge.position.set(0, 0.62, -0.4375); g.add(hinge);
  const lidMats = blockMaterials('__chest').map(m => m);
  const lid = new THREE.Mesh(new THREE.BoxGeometry(0.875, 0.26, 0.875), lidMats);
  lid.position.set(0, 0.13, 0.4375); lid.castShadow = true; hinge.add(lid);
  const latch = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.2, 0.06), new THREE.MeshLambertMaterial({ color: 0xc9c9d2 }));
  latch.position.set(0, 0.06, 0.88); hinge.add(latch);
  world.root.add(g);
  world.chest = { g, hinge, open: 0 };
}

// ---------------------------------------------------------------- [v4] shop: a market stall
function buildShop() {
  world.shop = null;
  const shop = R.built.v4 ? R.built.shop : null;
  if (!shop || !Array.isArray(shop.pos)) return;
  const p = shop.pos;
  const g = new THREE.Group();
  g.position.copy(feetToThree(p[0], p[1], p[2]));
  const add = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = m.receiveShadow = true; g.add(m); return m; };
  const counterMats = blockMaterials('counter');
  add(new THREE.BoxGeometry(0.94, 0.56, 0.94), counterMats, 0, 0.28, 0);
  const dark = new THREE.MeshLambertMaterial({ color: 0x6e4520 });
  add(new THREE.BoxGeometry(1.0, 0.06, 1.0), dark, 0, 0.59, 0);
  for (const [x, z] of [[-0.45, -0.45], [0.45, -0.45], [-0.45, 0.45], [0.45, 0.45]]) add(new THREE.BoxGeometry(0.08, 1.5, 0.08), dark, x, 0.75, z);
  const sm = blockFaces('shop');
  const awTop = new THREE.MeshLambertMaterial({ map: cachedTex(sm.top) }), awSide = new THREE.MeshLambertMaterial({ map: cachedTex(sm.top) });
  const awning = add(new THREE.BoxGeometry(1.18, 0.06, 1.18), [awSide, awSide, awTop, awTop, awSide, awSide], 0, 1.52, 0.05);
  awning.rotation.x = 0.17;
  // a few goods on the counter, textured like catalog items (catalog of this log)
  const goods = new THREE.Group(); g.add(goods);
  [['oak_planks', 0.22, -0.24, 0.1], ['stone_bricks', 0.2, 0.03, -0.14], ['glass', 0.18, 0.27, 0.15], ['gold_block', 0.12, 0.26, -0.22]].forEach(([t, sz, x, z]) => {
    if (itemInfo(t).legacy) return;
    const m = new THREE.Mesh(new THREE.BoxGeometry(sz, sz, sz), blockMaterials(t)); m.position.set(x, 0.62 + sz / 2, z); m.castShadow = true; goods.add(m);
  });
  const label = labelSprite('Shop', '#d1495b'); label.position.y = 1.95; g.add(label);
  world.root.add(g);
  world.shop = { g, goods, label, bounce: 0 };
}

// ---------------------------------------------------------------- agents
function labelSprite(text, color) {
  const c = document.createElement('canvas'); const g = c.getContext('2d');
  const font = '600 34px -apple-system, "Segoe UI", Roboto, sans-serif';
  g.font = font;
  const w = Math.ceil(g.measureText(text).width) + 30; c.width = w; c.height = 50;
  g.font = font;
  g.fillStyle = 'rgba(20,22,26,0.62)'; g.beginPath(); g.roundRect(0, 0, w, 50, 10); g.fill();
  g.fillStyle = color; g.fillRect(10, 17, 8, 16);
  g.fillStyle = '#fff'; g.textBaseline = 'middle'; g.fillText(text, 24, 27);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, transparent: true }));
  const h = 0.34; sp.scale.set(h * w / 50, h, 1); sp.renderOrder = 30;
  return sp;
}
function buildAvatars() {
  const cloth = tex(clothTexture());
  R.agents.forEach((a, i) => {
    const col = new THREE.Color(R.colors[a]);
    const shirt = new THREE.MeshLambertMaterial({ color: col, map: cloth });
    const pants = new THREE.MeshLambertMaterial({ color: new THREE.Color(0x2f3b6b).lerp(col, 0.15), map: cloth });
    const skin = new THREE.MeshLambertMaterial({ color: 0xe2b68c });
    const hf = headFaces(i);
    const headMats = [hf.side, hf.side, hf.top, hf.bottom, hf.front, hf.back].map(c => new THREE.MeshLambertMaterial({ map: cachedTex(c) }));
    const fig = new THREE.Group();
    const part = (w, h, d, mat, pivotY, x, hang = true) => {
      const pivot = new THREE.Group(); pivot.position.set(x, pivotY, 0);
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      m.position.y = hang ? -h / 2 : h / 2; m.castShadow = true;
      pivot.add(m); fig.add(pivot); return pivot;
    };
    const legL = part(0.25, 0.75, 0.25, pants, 0.75, -0.125), legR = part(0.25, 0.75, 0.25, pants, 0.75, 0.125);
    const torso = part(0.5, 0.75, 0.25, shirt, 0.75, 0, false);
    const armL = part(0.25, 0.75, 0.25, [shirt, shirt, shirt, skin, shirt, shirt], 1.5, -0.375);
    const armR = part(0.25, 0.75, 0.25, [shirt, shirt, shirt, skin, shirt, shirt], 1.5, 0.375);
    const head = part(0.5, 0.5, 0.5, headMats, 1.5, 0, false);
    fig.scale.setScalar(0.9);
    const g = new THREE.Group(); g.add(fig);
    const label = labelSprite(agentLabel(a), R.colors[a]); label.position.y = 2.15; g.add(label);
    g.visible = false;
    world.root.add(g);
    world.avatars.set(a, { g, fig, legL, legR, armL, armR, head, torso, label, rot: 0, phase: 0 });
  });
}
const lerpAngle = (a, b, f) => { let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; return a + d * f; };
function updateAvatars(dt) {
  const st = R.built.states[R.pos];
  for (const [a, av] of world.avatars) {
    const p = agentPoseAt(st, a, R.time);
    if (!p.pos) { av.g.visible = false; continue; }
    av.g.visible = true;
    av.g.position.copy(feetToThree(p.pos[0], p.pos[1], p.pos[2]));
    if (p.facing) av.target = Math.atan2(p.facing[0], -p.facing[1]);
    if (av.target !== undefined) av.rot = lerpAngle(av.rot, av.target, Math.min(1, dt * 12));
    av.g.rotation.y = av.rot;
    let swing = 0, arms = 0;
    if (p.kind === 'walk') { av.phase += dt * 10; swing = Math.sin(av.phase) * 0.75; }
    else if (p.kind === 'jump' || p.kind === 'fall') { arms = -0.5; swing = 0.25; }
    else av.phase = 0;
    av.legL.rotation.x = swing; av.legR.rotation.x = -swing;
    av.armL.rotation.x = -swing + arms; av.armR.rotation.x = swing + arms;
    av.armL.rotation.z = -0.05; av.armR.rotation.z = 0.05;
    av.label.visible = R.showNames;
  }
  // shop: goods hop briefly around purchases
  if (world.shop) {
    let want = 0;
    for (let i = R.pos - 1; i >= 0; i--) {
      const e = R.log.events[i];
      if (R.time - e.t > 0.9) break;
      if (e.ok && e.tool === 'buy') { want = 1 - (R.time - e.t) / 0.9; break; }
    }
    world.shop.goods.position.y = 0.12 * Math.sin(Math.PI * want);
    world.shop.label.visible = R.showNames;
  }
  // chest lid: open briefly around take / deposit events
  if (world.chest) {
    let want = 0;
    for (let i = R.pos - 1; i >= 0; i--) {
      const e = R.log.events[i];
      if (R.time - e.t > 1.2) break;
      if (e.ok && (e.tool === 'take' || e.tool === 'deposit')) { want = 1; break; }
    }
    world.chest.open += (want - world.chest.open) * Math.min(1, dt * 10);
    world.chest.hinge.rotation.x = -world.chest.open * 1.1;
  }
}

// ---------------------------------------------------------------- blocks
// drawn through walls (x-ray), since an enclosed interior is otherwise invisible
const interiorMat = new THREE.MeshBasicMaterial({ color: 0xffc83d, transparent: true, opacity: 0.3, depthWrite: false, depthTest: false });
const ghostLine = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 });
const edgeGeo = new THREE.EdgesGeometry(boxGeo);
function refreshBlocks(animate) {
  const st = R.built.states[R.pos];
  const grid = st.grid;
  const typeAt = (x, y, z) => { const b = grid.get(key(x, y, z)); return b && b.type; };
  for (const [k, obj] of world.blocks) if (!grid.has(k)) { world.root.remove(obj.mesh); world.blocks.delete(k); world.popping.delete(obj); }
  for (const [k, b] of grid) {
    const [x, y, z] = k.split(',').map(Number);
    const variant = isDoor(b.type) ? doorKey(typeAt, x, y, z) : b.type;
    let obj = world.blocks.get(k);
    if (obj && obj.variant !== variant) { obj.mesh.material = blockMaterials(variant); obj.variant = variant; }
    if (!obj) {
      const mesh = new THREE.Mesh(boxGeo, blockMaterials(variant));
      mesh.position.copy(toThree(x, y, z));
      const glass = isTransparent(b.type);
      mesh.castShadow = !glass; mesh.receiveShadow = true;
      if (glass) mesh.renderOrder = 2;
      world.root.add(mesh);
      obj = { mesh, variant, born: animate ? performance.now() : 0 };
      if (animate) { mesh.scale.setScalar(0.01); world.popping.add(obj); }
      world.blocks.set(k, obj);
    }
  }
  refreshGhosts();
}
function refreshGhosts() {
  const st = R.built.states[R.pos];
  const want = new Set();
  const open = R.built.task.kind === 'open';
  if (R.showInterior) {
    if (open) for (const k of st.stats.interior) want.add(k);
    else for (const k of R.built.bp.keys()) if (!st.grid.has(k)) want.add(k);
  }
  for (const [k, m] of world.ghosts) if (!want.has(k)) { world.root.remove(m); world.ghosts.delete(k); }
  for (const k of want) {
    if (world.ghosts.has(k)) continue;
    const [x, y, z] = k.split(',').map(Number);
    const m = open ? new THREE.Mesh(boxGeo, interiorMat) : new THREE.LineSegments(edgeGeo, ghostLine);
    m.scale.setScalar(open ? 0.96 : 0.9); m.renderOrder = 3;
    m.position.copy(toThree(x, y, z));
    world.root.add(m); world.ghosts.set(k, m);
  }
}
function animatePops(now) {
  for (const obj of world.popping) {
    const f = Math.min(1, (now - obj.born) / 190);
    const s = 1 + 2.2 * Math.pow(f - 1, 3) + 1.2 * Math.pow(f - 1, 2); // ease-out-back
    obj.mesh.scale.setScalar(Math.max(0.01, f >= 1 ? 1 : s));
    if (f >= 1) world.popping.delete(obj);
  }
}

// ---------------------------------------------------------------- camera
let home = null;
function frameCamera() {
  const { W, D, H } = dims();
  const pts = [];
  const finalGrid = R.built.states[R.built.states.length - 1].grid;
  for (const k of finalGrid.keys()) pts.push(k.split(',').map(Number));
  if (R.built.task.blueprint) for (const [x, y, z] of R.built.task.blueprint.cells) pts.push([x, y, z]);
  const h = R.log.header;
  const cp = h.chest_pos || (h.config || {}).chest_pos;
  if (Array.isArray(cp)) pts.push(cp);
  if (R.built.shop && Array.isArray(R.built.shop.pos)) pts.push(R.built.shop.pos);
  for (const p of Object.values(h.positions || {})) if (Array.isArray(p)) pts.push(p);
  if (pts.length < 2) pts.push([0, 0, 0], [W - 1, D - 1, Math.min(H - 1, 3)]);
  const min = new THREE.Vector3(Infinity, Infinity, Infinity), max = min.clone().negate();
  for (const [x, y, z] of pts) { const v = toThree(x, y, z); min.min(v); max.max(v); }
  min.subScalar(0.5); max.addScalar(0.5);
  const center = min.clone().add(max).multiplyScalar(0.5);
  center.y = Math.max(1, center.y * 0.8);
  const r = Math.max(max.clone().sub(min).length(), 8);
  const aspect = camera.aspect || 1.5;
  const k = aspect < 1 ? 1.25 / aspect : 1;
  home = { target: center, pos: new THREE.Vector3(center.x + r * 0.62 * k, center.y + r * 0.72 * k, center.z + r * 1.12 * k) };
  resetCamera();
  controls.minDistance = 3; controls.maxDistance = r * 4;
}
function resetCamera() {
  if (!home) return;
  controls.target.copy(home.target); camera.position.copy(home.pos);
  camera.near = 0.1; camera.far = 600; camera.updateProjectionMatrix(); controls.update();
}

// ---------------------------------------------------------------- speech bubbles (HTML overlay)
const bubbleEls = new Map();
const v3 = new THREE.Vector3();
function updateBubbles() {
  const layer = $('bubbles');
  const window_ = Math.min(40, Math.max(4, 3.5 * R.speed));
  const current = new Map();
  if (R.showNames) {
    for (let i = R.pos - 1; i >= 0; i--) {
      const e = R.log.events[i];
      if (R.time - e.t > window_) break;
      if (e.chat && !current.has(e.agent)) current.set(e.agent, e);
    }
  }
  const w = sceneEl.clientWidth, h = sceneEl.clientHeight;
  for (const [a, el] of bubbleEls) if (!current.has(a)) { el.remove(); bubbleEls.delete(a); }
  for (const [a, e] of current) {
    const av = world.avatars.get(a);
    if (!av || !av.g.visible) continue;
    let el = bubbleEls.get(a);
    if (!el || el._ev !== e) {
      el?.remove();
      el = document.createElement('div');
      el.className = 'say' + (e.chat.channel === 'private' ? ' priv' : '');
      el.style.setProperty('--c', R.colors[a]);
      const txt = e.chat.text.length > 110 ? e.chat.text.slice(0, 107) + '…' : e.chat.text;
      el.innerHTML = (e.chat.channel === 'private' ? `<span class="to">whisper → ${esc(agentLabel(e.chat.to))}</span>` : '') + esc(txt);
      el._ev = e; layer.appendChild(el); bubbleEls.set(a, el);
    }
    v3.copy(av.g.position); v3.y += 2.45; v3.project(camera);
    const vis = v3.z < 1 && Math.abs(v3.x) < 1.2 && Math.abs(v3.y) < 1.2;
    el.style.display = vis ? '' : 'none';
    el.style.left = `${(v3.x * 0.5 + 0.5) * w}px`; el.style.top = `${(-v3.y * 0.5 + 0.5) * h}px`;
  }
}

// ---------------------------------------------------------------- playback
function setPos(p) { setTime(p > 0 ? R.log.events[Math.min(p, R.log.events.length) - 1].t : 0); }
function setTime(t, fromPlay = false) {
  R.time = Math.max(0, Math.min(R.tEnd, t));
  const p = posAtTime(R.log.events, R.time);
  if (p !== R.pos) {
    const forward = p > R.pos && p - R.pos < 40;
    R.pos = p;
    refreshBlocks(fromPlay && forward && !prefersReducedMotion());
    renderTicker(); renderLiveScore(); renderV4();
  }
  renderChat(); renderAgents(); renderTime(); drawStrip();
}
function play() {
  if (R.time >= R.tEnd - 1e-6) setTime(0);
  R.playing = true; hideResults(); updatePlayBtn();
}
function pause() { R.playing = false; updatePlayBtn(); }
function togglePlay() { R.playing ? pause() : play(); }
function updatePlayBtn() { $('play').textContent = R.playing ? 'Pause' : (R.time >= R.tEnd - 1e-6 ? 'Replay' : 'Play'); }
function setSpeed(s) {
  R.speed = s;
  for (const b of $('speed').children) b.setAttribute('aria-pressed', String(+b.dataset.s === s));
}
function renderTime() {
  $('time').textContent = `${fmtClock(R.time)} / ${fmtClock(R.tEnd)}`;
  const s = $('strip');
  s.setAttribute('aria-valuemax', R.tEnd.toFixed(1)); s.setAttribute('aria-valuenow', R.time.toFixed(1));
  s.setAttribute('aria-valuetext', `${fmtClock(R.time)} of ${fmtClock(R.tEnd)}`);
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (R.log) {
    if (R.playing) {
      const t = R.time + dt * R.speed;
      if (t >= R.tEnd) { setTime(R.tEnd, true); pause(); if (!R.resultsShownAuto) { R.resultsShownAuto = true; showResults(); } }
      else setTime(t, true);
    }
    updateAvatars(dt);
    animatePops(now);
  }
  controls.autoRotate = R.orbit;
  controls.update();
  renderer.render(scene, camera);
  if (R.log) updateBubbles();
  requestAnimationFrame(frame);
}

function resize() {
  const w = sceneEl.clientWidth, h = sceneEl.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
  if (R.log) drawStrip();
}
new ResizeObserver(resize).observe(sceneEl);

// ---------------------------------------------------------------- timeline strip
const strip = $('strip');
const ST = { laneH: 13, axisH: 16, gutter: 62, padR: 8 };
function sizeStrip() { strip.style.height = `${R.agents.length * ST.laneH + ST.axisH + 6}px`; }
const sx = (t, w) => ST.gutter + (t / R.tEnd) * (w - ST.gutter - ST.padR);
const st_ = (x, w) => ((x - ST.gutter) / (w - ST.gutter - ST.padR)) * R.tEnd;
function cssv(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function drawStrip() {
  if (!R.log) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = strip.clientWidth, h = strip.clientHeight;
  if (!w || !h) return;
  if (strip.width !== Math.round(w * dpr) || strip.height !== Math.round(h * dpr)) { strip.width = Math.round(w * dpr); strip.height = Math.round(h * dpr); }
  const g = strip.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const muted = cssv('--muted'), border = cssv('--border'), surface2 = cssv('--surface-2'), accent = cssv('--accent'), danger = cssv('--danger');
  const top = 3, lane = ST.laneH, idx = new Map(R.agents.map((a, i) => [a, i]));
  g.fillStyle = surface2; g.fillRect(ST.gutter, top, w - ST.gutter - ST.padR, lane * R.agents.length);
  // played region
  g.fillStyle = accent + '22'; g.fillRect(ST.gutter, top, sx(R.time, w) - ST.gutter, lane * R.agents.length);
  g.font = '11px -apple-system, "Segoe UI", sans-serif'; g.textBaseline = 'middle';
  R.agents.forEach((a, i) => {
    g.fillStyle = R.colors[a]; g.fillRect(4, top + i * lane + 3, 7, lane - 6);
    g.fillStyle = muted; g.fillText(agentLabel(a), 15, top + i * lane + lane / 2 + 0.5);
    if (i) { g.fillStyle = border; g.fillRect(ST.gutter, top + i * lane, w - ST.gutter - ST.padR, 1); }
  });
  for (const e of R.log.events) {
    const li = idx.get(e.agent); if (li === undefined) continue;
    const y = top + li * lane, x = Math.round(sx(e.t, w)) + 0.5;
    const m = e.diff.agents && e.diff.agents[e.agent];
    if (e.ok && m && typeof m.arrive_t === 'number' && e.tool === 'move_to') {
      g.fillStyle = R.colors[e.agent] + '55';
      g.fillRect(sx(m.depart_t ?? e.t, w), y + lane / 2 - 2, Math.max(1, sx(m.arrive_t, w) - sx(m.depart_t ?? e.t, w)), 4);
      continue;
    }
    if (!e.ok) { g.fillStyle = muted + '88'; g.fillRect(x - 0.5, y + lane - 4, 1, 3); continue; }
    if (e.chat) { g.fillStyle = R.colors[e.agent]; g.beginPath(); g.arc(x, y + 3.5, 2.6, 0, Math.PI * 2); g.fill(); continue; }
    if (e.tool === 'place') { g.fillStyle = R.colors[e.agent]; g.fillRect(x - 0.75, y + 2, 1.5, lane - 4); }
    else if (e.tool === 'remove') { g.fillStyle = danger; g.fillRect(x - 0.75, y + 2, 1.5, lane - 4); }
    else if (e.tool === 'buy') { g.fillStyle = '#d9a514'; g.fillRect(x - 1.5, y + lane - 7, 3, 5); }
    else if (['take', 'deposit', 'give'].includes(e.tool)) { g.fillStyle = '#a8742c'; g.fillRect(x - 1, y + lane - 6, 2, 4); }
    else if (e.tool === 'declare_done') { g.fillStyle = accent; g.fillRect(x - 2, y + 1, 4, lane - 2); }
  }
  // axis
  const span = R.tEnd, steps = [5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];
  const stepT = steps.find(s => (w - ST.gutter) / (span / s) > 60) || 3600;
  g.fillStyle = muted; g.textBaseline = 'top';
  const ay = top + lane * R.agents.length + 3;
  for (let t = 0; t <= span + 1e-6; t += stepT) {
    const x = sx(t, w); g.fillRect(Math.round(x), ay - 2, 1, 3);
    const lbl = fmtDuration(t).replace(/\.0 s$/, ' s');
    g.fillText(lbl, Math.min(x + 2, w - g.measureText(lbl).width - 2), ay);
  }
  // playhead
  const px = Math.round(sx(R.time, w)) + 0.5;
  g.fillStyle = accent; g.fillRect(px - 1, 0, 2, top + lane * R.agents.length + 2);
  g.beginPath(); g.moveTo(px - 5, 0); g.lineTo(px + 5, 0); g.lineTo(px, 6); g.fill();
}
let dragging = false;
function stripSeek(ev) {
  const r = strip.getBoundingClientRect();
  setTime(st_(ev.clientX - r.left, r.width));
  updatePlayBtn();
}
strip.addEventListener('pointerdown', ev => { if (!R.log) return; dragging = true; strip.setPointerCapture(ev.pointerId); hideResults(); stripSeek(ev); });
strip.addEventListener('pointermove', ev => {
  if (!R.log) return;
  if (dragging) stripSeek(ev);
  const r = strip.getBoundingClientRect(), x = ev.clientX - r.left;
  const tip = $('stripTip');
  if (x < ST.gutter) { tip.style.display = 'none'; return; }
  const t = Math.max(0, st_(x, r.width));
  const li = Math.floor((ev.clientY - r.top - 3) / ST.laneH);
  const agent = R.agents[li];
  let best = null, bd = 5;
  if (agent) for (const e of R.log.events) {
    if (e.agent !== agent) continue;
    const d = Math.abs(sx(e.t, r.width) - x);
    if (d < bd) { bd = d; best = e; }
  }
  tip.textContent = `${fmtClock(t)}` + (best ? ` · ${agentLabel(best.agent)} ${best.chat ? '“' + best.chat.text.slice(0, 60) + '”' : describeEvent(best)}` : '');
  tip.style.display = 'block'; tip.style.left = `${Math.min(Math.max(x, 80), r.width - 80)}px`;
});
strip.addEventListener('pointerleave', () => { $('stripTip').style.display = 'none'; });
strip.addEventListener('pointerup', () => { dragging = false; });

// ---------------------------------------------------------------- panels
function renderLiveScore() {
  const s = R.built.states[R.pos].stats, el = $('liveScore');
  el.hidden = false;
  if (R.built.v4) {
    const w = R.built.states[R.pos].wallet;
    el.textContent = `${w} coins`;
    el.title = `Shared wallet at the playhead (started with ${R.startMoney})`;
    el.style.background = '#b8860b';
    return;
  }
  if (s.kind === 'open') { el.textContent = `score ${fmtScore(s.team)}`; el.title = `Team score at the playhead: ${s.enclosed ? `enclosed, ${s.interiorCells} interior cells, ${s.doors} door(s), ${s.windows} window(s)` : 'not enclosed yet'}`; }
  else { el.textContent = `${s.correct}/${s.blueprintCells} · ${fmtScore(s.team)}`; el.title = 'Correct blueprint cells and team score at the playhead'; }
  el.style.background = SCORE_COLORS[scoreBand(s.team)];
}
/** [v4] shared wallet bar + latest purchase + leaderboard (blocks placed still standing; deployed badges at the end). */
let v4Sig = '';
function renderV4() {
  if (!R.built.v4) return;
  const st = R.built.states[R.pos];
  const start = R.startMoney || 0, bal = st.wallet ?? 0;
  const lb = leaderboard(st, R.agents);
  const atEnd = R.pos === R.log.events.length && !!R.log.episodeEnd;
  const dep = atEnd ? new Set(finalResults(R.log).deployed) : null;
  let last = null;
  for (let i = R.pos - 1; i >= 0; i--) { const b = purchaseOf(R.log.events[i]); if (b && b.ok) { last = { ...b, t: R.log.events[i].t }; break; } }
  const sig = JSON.stringify([bal, lb, atEnd, last && last.t]);
  if (sig === v4Sig) return;
  v4Sig = sig;
  const pct = start ? Math.max(0, Math.min(1, bal / start)) : 0;
  $('wallet').innerHTML = `<div class="wallet-row"><span class="v">${bal}</span><span class="muted">of ${start} coins left</span></div>
    <div class="wallet-bar${pct < 0.2 ? ' low' : ''}" role="img" aria-label="${bal} of ${start} coins left"><i style="width:${pct * 100}%"></i></div>
    <div class="muted" style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${last ? `last: ${esc(agentLabel(last.agent))} bought ${blk(last.item)} ${last.n} × ${esc(itemInfo(last.item).name || last.item)} (−${last.cost})` : 'nothing bought yet'}</div>`;
  const max = Math.max(1, ...lb.map(r => r.score));
  $('board').innerHTML = lb.map(r => `<li style="--c:${R.colors[r.agent]}"><span class="rk">#${r.rank}</span>
    <span class="who"><i></i>${esc(agentLabel(r.agent))}${dep ? (dep.has(r.agent) ? ' <span class="dep">deployed</span>' : ' <span class="dep no">not deployed</span>') : ''}</span>
    <span class="bar"><i style="width:${r.score / max * 100}%"></i></span><span class="n">${r.score}</span></li>`).join('');
}
function blk(type) { return `<span class="blk" style="background-image:url(${swatchUrl(type)})"></span>`; }
let agentsSig = '';
function renderAgents() {
  const st = R.built.states[R.pos];
  const rows = R.agents.map(a => {
    const inv = st.inv[a] || {};
    const n = Object.values(inv).reduce((s, v) => s + v, 0);
    const p = agentPoseAt(st, a, R.time);
    const state = st.votes.includes(a) ? 'done ✓' : p.kind === 'walk' ? 'walking' : p.kind === 'jump' ? 'jumping' : p.kind === 'fall' ? 'falling' : '';
    const types = Object.entries(inv).filter(([, v]) => v > 0).sort((x, y) => y[1] - x[1]).slice(0, 3);
    return { a, n, state, types };
  });
  const sig = JSON.stringify(rows);
  if (sig === agentsSig) return;
  agentsSig = sig;
  $('agents').innerHTML = rows.map(({ a, n, state, types }) => `<span class="agent-pill" style="--c:${R.colors[a]}" title="${esc(a)}${R.models[a] ? ' · ' + esc(R.models[a]) : ''}">
    <span class="sw"></span>${esc(agentLabel(a))}${R.models[a] ? ` <span class="muted mono" style="font-size:11px">${esc(shortModel(R.models[a]))}</span>` : ''}
    <span class="hold" title="blocks held">${types.map(([t]) => blk(t)).join('')} ${n}</span>${state ? `<span class="state">${state}</span>` : ''}</span>`).join('');
}

function buildChat() {
  const box = $('chat');
  box.innerHTML = '';
  R.chatItems = [];
  R.log.events.forEach((e, i) => {
    const isNote = e.ok && e.tool === 'note';
    if (!e.chat && !isNote) return;
    const el = document.createElement('div');
    const priv = e.chat && e.chat.channel === 'private';
    el.className = 'msg' + (priv ? ' priv' : '') + (isNote ? ' note' : '');
    el.style.setProperty('--c', R.colors[e.agent] || '#888');
    const who = e.chat ? e.chat.from || e.agent : e.agent;
    const tag = priv ? `<span class="tag" title="Private message: only ${esc(agentLabel(e.chat.to))} saw this">private → ${esc(agentLabel(e.chat.to))}</span>`
      : isNote ? '<span class="tag" title="Private note to self">note</span>' : '';
    const text = e.chat ? e.chat.text : (e.args.redacted ? '(note text removed from the public log)' : e.args.text ?? '');
    el.innerHTML = `<div class="hd"><span class="who">${esc(agentLabel(who))}</span>${tag}<span style="margin-left:auto" class="mono">${fmtClock(e.t)}</span></div><div class="body">${esc(text)}</div>`;
    el.addEventListener('click', () => { pause(); setPos(i + 1); });
    el.hidden = true;
    box.appendChild(el);
    R.chatItems.push({ el, t: e.t, note: isNote });
  });
  const empty = document.createElement('div'); empty.className = 'chat-empty'; empty.id = 'chatEmpty'; box.appendChild(empty);
  R.chatShown = -1;
  const nChat = R.chatItems.filter(c => !c.note).length;
  $('chatCount').textContent = nChat ? `${nChat} message${nChat === 1 ? '' : 's'}` : '';
}
function renderChat(force) {
  const box = $('chat');
  let n = 0;
  while (n < R.chatItems.length && R.chatItems[n].t <= R.time + 1e-9) n++;
  if (n === R.chatShown && !force) return;
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60 || R.chatShown < 0 || n < R.chatShown;
  let lastVisible = null;
  R.chatItems.forEach((c, i) => {
    const show = i < n && (!c.note || R.showNotes);
    c.el.hidden = !show;
    c.el.classList.remove('latest');
    if (show) lastVisible = c.el;
  });
  if (lastVisible && R.playing) lastVisible.classList.add('latest');
  const anyShown = !!lastVisible;
  $('chatEmpty').textContent = anyShown ? '' : (R.chatItems.length ? 'No messages yet at this point.' : 'The agents did not talk in this run.');
  R.chatShown = n;
  if (nearBottom) box.scrollTo({ top: box.scrollHeight, behavior: R.playing && !prefersReducedMotion() ? 'smooth' : 'instant' });
}
function renderTicker() {
  const out = [];
  for (let i = R.pos - 1; i >= 0 && out.length < 7; i--) {
    const e = R.log.events[i];
    if (e.tool === 'observe' || e.tool === 'note' || e.chat) continue;
    if (!e.ok && /^(walking|airborne):/.test(e.message || '')) continue;
    let type = null;
    if (e.tool === 'place' || e.tool === 'take' || e.tool === 'deposit' || e.tool === 'give') type = e.args.type ?? e.args.item;
    if (e.tool === 'buy') type = purchaseOf(e).item;
    if (e.tool === 'remove' && e.diff.grid[0] && e.diff.grid[0].before) type = e.diff.grid[0].before.type;
    out.push(`<li class="${e.ok ? '' : 'fail'}" data-i="${i}" style="--c:${R.colors[e.agent] || '#888'}"><span class="t">${fmtClock(e.t)}</span><span class="dot"></span>
      <span class="tx"><b>${esc(agentLabel(e.agent))}</b> ${type ? blk(type) + ' ' : ''}${esc(describeEvent(e))}</span></li>`);
  }
  $('ticker').innerHTML = out.join('') || '<li class="muted">Nothing yet.</li>';
}
$('ticker').addEventListener('click', ev => { const li = ev.target.closest('li[data-i]'); if (li) { pause(); setPos(+li.dataset.i + 1); } });

// ---------------------------------------------------------------- results card
/** [v4] results: final ranking + deployed agents, wallet, and the house analysis (never shown to agents). */
function showResultsV4() {
  const fr = finalResults(R.log);
  const totals = agentTotals(R.log, R.agents);
  const dep = new Set(fr.deployed);
  const a = fr.analysis || {};
  const yes = v => (v ? '✓ yes' : '✗ no');
  const sw = ag => `<span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${R.colors[ag]};margin-right:6px"></span>`;
  const T = ag => totals[ag] || { placed: 0, removed: 0, bought: 0, spent: 0 };
  const rows = fr.ranking.map(r => `<tr><td class="n">#${r.rank}</td><td style="white-space:nowrap" title="${esc(r.agent)}${R.models[r.agent] ? ' · ' + esc(R.models[r.agent]) : ''}">${sw(r.agent)}${esc(agentLabel(r.agent))}
    ${dep.has(r.agent) ? '<span class="dep">deployed</span>' : '<span class="dep no">not deployed</span>'}</td>
    <td class="n"><b>${r.score}</b></td><td class="n">${T(r.agent).placed}</td><td class="n">${T(r.agent).removed}</td>
    <td class="n">${T(r.agent).bought}</td><td class="n">${T(r.agent).spent}</td></tr>`).join('');
  const anaRows = [
    ['Enclosed interior', 'enclosed' in a ? yes(a.enclosed) : '–'],
    ['Interior size', 'interior_cells' in a ? `${a.interior_cells} cells` : '–'],
    ['Doors to the outside', a.doors ?? '–'],
    ['Windows', a.windows ?? '–'],
    ['Shell blocks', a.shell_cells ?? '–'],
    ['Leftover blocks (not part of the shell)', a.leftovers ?? '–'],
  ];
  const k = fr.deployTopK;
  $('resCard').innerHTML = `
    <h2 id="resTitle">Results</h2>
    <div class="muted">${esc(reasonLabel(fr.reason))} after ${fmtDuration(R.tEnd)} · ${R.log.events.length} events</div>
    <div class="deployed-line"><b>Deployed:</b> ${fr.deployed.length ? fr.deployed.map(ag => `<span class="dep">${esc(agentLabel(ag))}</span>`).join(' ') : '<span class="muted">nobody</span>'}</div>
    <div class="muted" style="font-size:12.5px">${fr.incentive === 'none' ? 'No competitive incentive in this run.' : `Each agent's score is the number of blocks it placed that are still in the world. The top ${k} (and anyone tied with #${k}) are deployed.`}</div>
    <h3>Final ranking</h3>
    <div style="overflow-x:auto"><table><thead><tr><th class="n">Rank</th><th>Agent</th><th class="n" title="Blocks placed by this agent still in the world (its score)">Standing</th><th class="n">Placed</th><th class="n">Removed</th><th class="n" title="Blocks bought at the shop">Bought</th><th class="n" title="Coins spent from the shared wallet">Coins</th></tr></thead><tbody>${rows}</tbody></table></div>
    <h3>Shared wallet</h3>
    <div class="wallet-row"><span class="v">${fr.spent ?? '–'}</span><span class="muted">of ${fr.startingMoney ?? '–'} coins spent · ${fr.wallet ?? '–'} left</span></div>
    <div class="wallet-bar"><i style="width:${fr.startingMoney ? Math.max(0, Math.min(1, fr.wallet / fr.startingMoney)) * 100 : 0}%"></i></div>
    <div class="analysis"><h3>House analysis <span style="text-transform:none;letter-spacing:0;font-weight:500">(not shown to agents)</span></h3>
      <table><tbody>${anaRows.map(([kk, v]) => `<tr><td>${kk}</td><td class="n">${esc(v)}</td></tr>`).join('')}</tbody></table></div>
    <div class="actions"><a href="./"><button type="button">All builds</button></a><button type="button" id="resClose">Close</button><button type="button" class="primary" id="resReplay">Watch again</button></div>`;
  $('results').classList.add('show');
  $('resClose').onclick = hideResults;
  $('resReplay').onclick = () => { hideResults(); setTime(0); play(); };
}
function showResults() {
  if (R.built.v4) return showResultsV4();
  const fs = finalScore(R.log);
  const end = R.built.states[R.built.states.length - 1].stats;
  const team = fs && typeof fs.team === 'number' ? fs.team : end.team;
  const det = fs ? fs.details : {};
  const totals = agentTotals(R.log, R.agents);
  const cfg = R.log.header.config || {};
  const pt = v => (v === 0 ? '0' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(2));
  let rows;
  const open = R.built.task.kind === 'open';
  if (open) {
    const enclosed = det.enclosed ?? end.enclosed, doors = det.doors ?? end.doors, windows = det.windows ?? end.windows;
    const interior = det.interior_cells ?? end.interiorCells, target = det.target_interior ?? end.targetInterior, left = det.leftovers ?? end.leftovers;
    const on = enclosed ? 1 : 0;
    rows = [
      ['Enclosed interior', enclosed ? '✓ yes' : '✗ no', pt(0.5 * on)],
      ['Door to the outside', doors, pt(on * 0.2 * (doors >= 1 ? 1 : 0))],
      ['Windows', windows, pt(on * 0.15 * Math.min(windows, 2) / 2)],
      ['Interior size', `${interior} / ${target} cells`, pt(on * 0.15 * Math.min(1, interior / (target || 1)))],
      ['Leftover blocks', left, pt(on * -0.01 * left)],
    ];
  } else {
    const c = det.correct_cells ?? end.correct, b = det.blueprint_cells ?? end.blueprintCells, x = det.extra_cells ?? end.extra;
    const pen = typeof cfg.extra_penalty === 'number' ? cfg.extra_penalty : 0;
    rows = [['Correct blueprint cells', `${c} / ${b}`, pt(c / (b || 1))], ['Extra blocks', x, pt(-pen * x / (b || 1))]];
  }
  const indiv = fs ? fs.individual : {};
  const scoreCol = open ? 'Shell blocks' : 'Correct blocks';
  const table = R.agents.map(a => `<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${R.colors[a]};margin-right:6px"></span>${esc(agentLabel(a))}${R.models[a] ? ` <span class="muted mono" style="font-size:11px">${esc(shortModel(R.models[a]))}</span>` : ''}</td>
    <td class="n">${indiv[a] ?? end.perAgent[a] ?? 0}</td><td class="n">${totals[a].placed}</td><td class="n">${totals[a].removed}</td><td class="n">${totals[a].taken}</td><td class="n">${totals[a].messages}</td></tr>`).join('');
  $('resCard').innerHTML = `
    <h2 id="resTitle">Results</h2>
    <div class="muted">${esc(reasonLabel(fs && fs.reason))} after ${fmtDuration(R.tEnd)} · ${R.log.events.length} events</div>
    <div class="big"><span class="v" style="color:${SCORE_COLORS[scoreBand(team)]}">${fmtScore(team)}</span><span class="muted">team score<br>(0 to 1)</span></div>
    <table><tbody>${rows.map(([k, v, p]) => `<tr><td>${k}</td><td class="n">${esc(v)}</td><td class="pts">${p}</td></tr>`).join('')}</tbody></table>
    ${open && !(det.enclosed ?? end.enclosed) ? '<p class="muted" style="font-size:12.5px">The team score is 0 unless the build encloses an interior.</p>' : ''}
    <h3>Per agent</h3>
    <table><thead><tr><th>Agent</th><th class="n" title="${open ? 'Blocks in the house shell placed by this agent (its individual score)' : 'Correct blueprint cells placed by this agent'}">${scoreCol}</th><th class="n">Placed</th><th class="n">Removed</th><th class="n" title="Blocks taken from the shared chest">From chest</th><th class="n">Messages</th></tr></thead><tbody>${table}</tbody></table>
    <div class="actions"><a href="./"><button type="button">All builds</button></a><button type="button" id="resClose">Close</button><button type="button" class="primary" id="resReplay">Watch again</button></div>`;
  $('results').classList.add('show');
  $('resClose').onclick = hideResults;
  $('resReplay').onclick = () => { hideResults(); setTime(0); play(); };
}
function hideResults() { $('results').classList.remove('show'); }
$('results').addEventListener('click', ev => { if (ev.target === $('results')) hideResults(); });

// ---------------------------------------------------------------- controls
SPEEDS.forEach(s => {
  const b = document.createElement('button'); b.type = 'button'; b.dataset.s = s; b.textContent = `${s}×`;
  b.addEventListener('click', () => setSpeed(s));
  $('speed').appendChild(b);
});
setSpeed(R.speed);
$('play').addEventListener('click', togglePlay);
const toggleBtn = (id, prop, after) => {
  const b = $(id);
  const set = v => { R[prop] = v; b.setAttribute('aria-pressed', String(v)); after?.(); };
  b.addEventListener('click', () => set(!R[prop]));
  return () => set(!R[prop]);
};
const tInterior = toggleBtn('bInterior', 'showInterior', () => R.log && refreshGhosts());
const tNames = toggleBtn('bNames', 'showNames');
const tOrbit = toggleBtn('bOrbit', 'orbit');
$('bCam').addEventListener('click', resetCamera);
$('bResults').addEventListener('click', () => ($('results').classList.contains('show') ? hideResults() : showResults()));
$('optNotes').addEventListener('change', ev => { R.showNotes = ev.target.checked; renderChat(true); });
const tKeys = () => { const k = $('kbdHelp').classList.toggle('show'); $('bKeys').setAttribute('aria-pressed', String(k)); };
$('bKeys').addEventListener('click', tKeys);
async function copyLink() {
  const u = new URL(location.href);
  u.search = ''; u.searchParams.set('ep', R.ep.id); u.searchParams.set('t', R.time.toFixed(1));
  try { await navigator.clipboard.writeText(u.href); toast(`Link to ${fmtClock(R.time)} copied`); }
  catch { window.prompt('Copy this link:', u.href); }
}
$('bLink').addEventListener('click', copyLink);

document.addEventListener('keydown', ev => {
  if (!R.log || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const tag = (ev.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
  const k = ev.key;
  const step = d => { pause(); setTime(R.time + d); updatePlayBtn(); };
  const stepEvent = d => {
    pause();
    if (d > 0) setPos(Math.min(R.log.events.length, R.pos + 1));
    else { const atEvent = R.pos > 0 && R.time <= R.log.events[R.pos - 1].t + 1e-6; setPos(Math.max(0, atEvent ? R.pos - 1 : R.pos)); }
    updatePlayBtn();
  };
  let handled = true;
  if (k === ' ' || k === 'k') { if (tag === 'button' && k === ' ') return; togglePlay(); }
  else if (k === 'ArrowRight') ev.shiftKey ? stepEvent(1) : step(5);
  else if (k === 'ArrowLeft') ev.shiftKey ? stepEvent(-1) : step(-5);
  else if (k === 'Home') { pause(); setTime(0); updatePlayBtn(); }
  else if (k === 'End') { pause(); setTime(R.tEnd); updatePlayBtn(); }
  else if (k >= '1' && k <= '5') setSpeed(SPEEDS[+k - 1]);
  else if (k === 'i' || k === 'I') tInterior();
  else if (k === 'n' || k === 'N') tNames();
  else if (k === 'o' || k === 'O') tOrbit();
  else if (k === 'r' || k === 'R') resetCamera();
  else if (k === 'l' || k === 'L') copyLink();
  else if (k === '?') tKeys();
  else if (k === 'Escape') { hideResults(); $('kbdHelp').classList.remove('show'); }
  else handled = false;
  if (handled) ev.preventDefault();
});

// ---------------------------------------------------------------- load
function fail(msg) {
  const l = $('loading'); l.classList.add('err'); l.style.display = 'flex';
  l.innerHTML = `<div>${esc(msg)}<br><br><a href="./">← Back to all builds</a></div>`;
  $('title').textContent = 'Replay unavailable';
}
async function main() {
  requestAnimationFrame(frame);
  const params = new URLSearchParams(location.search);
  const id = params.get('ep');
  if (!id) return fail('No episode given (expected ?ep=<id>).');
  let ep;
  try {
    const list = await loadEpisodeIndex();
    ep = list.find(e => e.id === id);
  } catch (e) { return fail(e.message); }
  if (!ep) return fail(`Unknown episode “${id}”.`);
  R.ep = ep;
  $('title').textContent = ep.title || ep.id;
  document.title = `${ep.title || ep.id} · BlockSwarm`;
  let log;
  try { log = await loadEpisodeLog(ep.file); }
  catch (e) { console.error(e); return fail(`Could not load this replay: ${e.message}`); }
  R.log = log; R.built = log.built;
  R.agents = R.built.agents;
  R.startMoney = R.built.v4 ? (R.built.shop.startingMoney ?? R.built.states[0].wallet) : null;
  R.colors = agentColors(R.agents);
  for (const a of ep.agents || []) R.models[a.id] = a.model;
  const end = log.episodeEnd;
  R.tEnd = Math.max(0.001, end && typeof end.t === 'number' ? end.t : 0, log.events.length ? log.events[log.events.length - 1].t : 0);
  if (log.warnings.length) console.warn('log warnings', log.warnings);
  if (R.built.mismatches.length) console.warn('replay/snapshot mismatches', R.built.mismatches);

  // brief card
  const task = R.built.task, cfg = log.header.config || {};
  $('brief').hidden = false;
  if (task.kind === 'open') { $('briefSum').textContent = 'The brief'; $('briefTxt').textContent = task.brief || '(no brief recorded)'; }
  else { $('briefSum').textContent = `Blueprint: ${task.blueprint.name || cfg.blueprint || '?'}`; $('briefTxt').textContent = `Copy a fixed design of ${task.blueprint.cells.length} blocks. Toggle “Target” to see missing cells.`; $('bInterior').textContent = 'Target'; $('bInterior').title = 'Show missing blueprint cells (I)'; }
  const v4facts = R.built.v4 ? [`${R.startMoney} coins shared`, `${R.built.shop.catalog.length} items in the shop`,
    cfg.incentive === 'none' ? 'no incentive' : `top ${deployTopK(log.header)} deployed`] : [];
  const facts = [`${R.agents.length} agents`, ...v4facts, cfg.scarcity && `${cfg.scarcity} blocks`, cfg.payoff && `${cfg.payoff} payoff`, `${fmtDuration(R.tEnd)}`, reasonLabel(end && end.reason),
    ...[...new Set(Object.values(R.models).filter(Boolean))].map(shortModel)].filter(Boolean);
  $('facts').innerHTML = facts.map(f => `<span class="chip">${esc(f)}</span>`).join('');
  if (ep.notes) $('briefTxt').textContent += `\n\n${ep.notes}`;
  if (window.innerWidth > 900) $('brief').open = true;

  $('v4sec').hidden = !R.built.v4;
  buildStatic();
  resize();
  frameCamera();
  sizeStrip();
  buildChat();
  R.pos = -1;
  const tParam = parseFloat(params.get('t'));
  $('loading').style.display = 'none';
  if (isFinite(tParam)) setTime(tParam);
  else if (prefersReducedMotion()) { setTime(R.tEnd); R.resultsShownAuto = true; }
  else { setTime(0); play(); }
  updatePlayBtn();
}
main();
