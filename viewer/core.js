/**
 * BlockSwarm viewer core — shared, DOM-free and three.js-free logic for replaying episode logs
 * (docs/log-format.md, log versions 1–3, blueprint and open tasks). Used by viewer/index.html and
 * the static gallery site. Plain ES module; also runs under Node (>= 18) for tests.
 *
 * Coordinates are log coordinates throughout: x east, y north, z up (z = 0 is the ground layer).
 * Cell keys are strings "x,y,z" (see `key`).
 *
 * ── Parsing + replay ────────────────────────────────────────────────────────────────────────
 * createLogBuilder() -> builder
 *     Incremental reader. Feed records in file order; it reconstructs state after every event.
 *     builder.addChunk(textOrBytes, {skipErrors}?)  append raw file content (string or Uint8Array;
 *                                     UTF-8 safe); only complete lines are parsed, a partial last line
 *                                     is kept. skipErrors: bad lines after the header become warnings.
 *                                     Returns {events, snapshots, ended} counts added by this call.
 *     builder.flush()                 parse a buffered final line that has no trailing newline.
 *     builder.addLine(line, lineNo)   parse + add one line (blank lines ignored).
 *     builder.addRecord(rec, lineNo)  add one already-parsed record.
 *     builder.log   = {header, task, events, snapshots, episodeEnd, warnings}
 *     builder.built = {states, bp, task, agents, mismatches, verified, snapshotCount, snapsAtPos}
 *       states[p]  = state after the first p events (states[0] = start):
 *                    {grid: Map<key,{type,placed_by,placed_at|placed_at_round}>, chest: {type:n},
 *                     inv: {agent:{type:n}}, votes: [agent], actions: {agent:n},
 *                     positions: {agent:[x,y,z]}, moves: {agent: last diff.agents entry (+ landZ/raiseAt
 *                     when a block was placed under a jumping agent)}, stats: see scoring below}
 *       bp         = Map<key, type> of blueprint cells (empty for open tasks)
 *       mismatches = [{pos, t, seq, problems: [string]}] snapshots that disagree with the replay
 *       snapsAtPos = Map<p, [snapshot]>
 *     Errors (bad JSON, missing/invalid header) throw Error with a readable "Line N: ..." message.
 * parseLog(text) -> log      Whole-file parse; log.built holds the replay. Throws if there is no header.
 * buildStates(log) -> built  Same object as log.built (kept for older callers).
 * checkSnapshot(snapshot, state) -> [problem strings]
 * taskOf(header) -> {kind: 'open'|'blueprint', brief, blueprint|null, targetInterior}
 *
 * ── Scoring (matches docs/log-format.md › Scoring) ────────────────────────────────────────────
 * blueprintStats(grid, bp, agents, extraPenalty)
 *     -> {kind:'blueprint', correct, wrong, extra, perAgent, team, blueprintCells}
 * openStats(grid, dims{width,depth,height}, chestKey|null, agents, targetInterior)
 *     -> {kind:'open', enclosed, interior:[key], interiorCells, shellCells, doors, windows,
 *         leftovers, leftoverSet:Set<key>, team, perAgent, targetInterior}
 *
 * ── Time + bodies ─────────────────────────────────────────────────────────────────────────────
 * posAtTime(events, t) -> number of events with event.t <= t (binary search)
 * walkPosition(move, t) -> [x, y, z] (floats) of an agent along a diff.agents entry at time t:
 *     walks follow `path` (straight steps 1, diagonal √2), jumps hop HOP_HEIGHT blocks in a
 *     parabola (landing on a block placed underneath mid-hop), falls drop along their path.
 * agentPoseAt(state, agent, t) -> {pos:[x,y,z], cell:[x,y,z], moving, kind:'walk'|'jump'|'fall'|null,
 *     until: arrive_t|null, facing:[dx,dy]|null} — feet position (cell coords; feet at z), for drawing.
 *
 * ── Presentation helpers (no DOM) ─────────────────────────────────────────────────────────────
 * describeEvent(event) -> short text for feeds ("place stone at (4, 5, 0)", "walks to (…) (1.9 s)",
 *     "jumps", "places wood under itself at …", "remove … — agent_1 fell 1", ...)
 * typeColor(type) -> '#rrggbb' (fixed palette for known block types, stable hash colour otherwise)
 * agentColors(agents) -> {agent: '#rrggbb'} in order (8-colour palette, repeats after 8)
 * TYPE_COLORS, AGENT_COLORS, HOP_HEIGHT, key(x, y, z)
 */

const TYPE_COLORS = {
  wood: '#9a6a3f', stone: '#9aa1a8', brick: '#b8533c', glass: '#8fd3f4',
  plank: '#dcb878', roof: '#3d5f91', door: '#2e9a74',
};
const AGENT_COLORS = ['#3b7dd8', '#f28e2b', '#4bab5a', '#d1495b', '#9467bd', '#17a2b8', '#e0b000', '#e377c2'];
const EXTRA_TYPE_COLORS = ['#c2185b', '#7cb342', '#5e35b1', '#00897b', '#f4511e', '#6d4c41'];

const key = (x, y, z) => x + ',' + y + ',' + z;

function addDelta(obj, t, d) {
  const v = (obj[t] || 0) + d;
  if (v === 0) delete obj[t]; else obj[t] = v;
}

/** Normalise the task: header.task {kind, brief, blueprint} (newer) or top-level header.blueprint (older). */
function taskOf(header) {
  const t = header.task && typeof header.task === 'object' ? header.task : null;
  const c = header.config || {};
  const blueprint = (t && t.blueprint) || header.blueprint || null;
  let kind = t && t.kind ? t.kind : (blueprint ? 'blueprint' : (c.task || 'open'));
  if (kind === 'blueprint' && !blueprint) kind = 'open';
  return { kind, brief: (t && t.brief) || c.brief || null, blueprint: kind === 'blueprint' ? blueprint : null,
    targetInterior: typeof c.target_interior === 'number' ? c.target_interior : 12 };
}

function blueprintStats(grid, bp, agents, extraPenalty) {
  let correct = 0, wrong = 0, extra = 0;
  const perAgent = {};
  for (const a of agents) perAgent[a] = 0;
  for (const [k, b] of grid) {
    const want = bp.get(k);
    if (want === undefined) extra++;
    else if (want === b.type) { correct++; perAgent[b.placed_by] = (perAgent[b.placed_by] || 0) + 1; }
    else wrong++;
  }
  const n = bp.size;
  const team = n ? Math.max(0, (correct - extraPenalty * extra) / n) : 0;
  return { kind: 'blueprint', correct, wrong, extra, perAgent, team: Math.round(team * 1e6) / 1e6, blueprintCells: n };
}

/** Open task scoring (docs/log-format.md › Scoring › open). dims = {width, depth, height}; chestKey = "x,y,z" or null. */
function openStats(grid, dims, chestKey, agents, targetInterior) {
  const W = dims.width, D = dims.depth, H = dims.height;
  const idx = (x, y, z) => (z * D + y) * W + x;
  const N = W * D * H;
  const solid = new Uint8Array(N);
  for (const k of grid.keys()) {
    const [x, y, z] = k.split(',').map(Number);
    if (x >= 0 && x < W && y >= 0 && y < D && z >= 0 && z < H) solid[idx(x, y, z)] = 1;
  }
  if (chestKey) {
    const [x, y, z] = chestKey.split(',').map(Number);
    if (x >= 0 && x < W && y >= 0 && y < D && z >= 0 && z < H) solid[idx(x, y, z)] = 1;
  }
  const NB = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  // outside air: flood from empty boundary cells (x/y faces and the top layer; z = 0 sits on the ground)
  const outside = new Uint8Array(N);
  const stack = [];
  for (let z = 0; z < H; z++) for (let y = 0; y < D; y++) for (let x = 0; x < W; x++) {
    if (x === 0 || y === 0 || x === W - 1 || y === D - 1 || z === H - 1) {
      const i = idx(x, y, z);
      if (!solid[i] && !outside[i]) { outside[i] = 1; stack.push(x, y, z); }
    }
  }
  while (stack.length) {
    const z = stack.pop(), y = stack.pop(), x = stack.pop();
    for (const [dx, dy, dz] of NB) {
      const nx = x + dx, ny = y + dy, nz = z + dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= W || ny >= D || nz >= H) continue;
      const j = idx(nx, ny, nz);
      if (!solid[j] && !outside[j]) { outside[j] = 1; stack.push(nx, ny, nz); }
    }
  }
  // enclosed empty regions; interior = the largest (ties: first found in x, y, z scan order)
  const seen = new Uint8Array(N);
  let best = [];
  for (let z = 0; z < H; z++) for (let y = 0; y < D; y++) for (let x = 0; x < W; x++) {
    const i = idx(x, y, z);
    if (solid[i] || outside[i] || seen[i]) continue;
    const region = []; seen[i] = 1; const st = [x, y, z];
    while (st.length) {
      const cz = st.pop(), cy = st.pop(), cx = st.pop();
      region.push([cx, cy, cz]);
      for (const [dx, dy, dz] of NB) {
        const nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= W || ny >= D || nz >= H) continue;
        const j = idx(nx, ny, nz);
        if (!solid[j] && !outside[j] && !seen[j]) { seen[j] = 1; st.push(nx, ny, nz); }
      }
    }
    if (region.length > best.length) best = region;
  }
  const enclosed = best.length >= 2;
  const interiorSet = new Set(enclosed ? best.map(([x, y, z]) => key(x, y, z)) : []);
  const shell = new Set(), leftovers = new Set();
  let doors = 0, windows = 0;
  const perAgent = {};
  for (const a of agents) perAgent[a] = 0;
  for (const [k, b] of grid) {
    const [x, y, z] = k.split(',').map(Number);
    // shell: any of the 26 neighbours is interior
    let inShell = false;
    for (let dx = -1; dx <= 1 && !inShell; dx++) for (let dy = -1; dy <= 1 && !inShell; dy++) for (let dz = -1; dz <= 1; dz++) {
      if ((dx || dy || dz) && interiorSet.has(key(x + dx, y + dy, z + dz))) { inShell = true; break; }
    }
    if (!inShell) { leftovers.add(k); continue; }
    shell.add(k);
    perAgent[b.placed_by] = (perAgent[b.placed_by] || 0) + 1;
    if (b.type !== 'door' && b.type !== 'glass') continue;
    // door / window: face contact with both interior and outside air
    let faceInterior = false, faceOutside = false;
    for (const [dx, dy, dz] of NB) {
      const nx = x + dx, ny = y + dy, nz = z + dz;
      if (interiorSet.has(key(nx, ny, nz))) faceInterior = true;
      if (nx >= 0 && ny >= 0 && nz >= 0 && nx < W && ny < D && nz < H) { if (outside[idx(nx, ny, nz)]) faceOutside = true; }
      else if (nz >= H && nx >= 0 && ny >= 0 && nx < W && ny < D) faceOutside = true; // open sky above the top layer
    }
    if (faceInterior && faceOutside) { if (b.type === 'door') doors++; else windows++; }
  }
  let team = 0;
  if (enclosed) {
    team = 0.5 + 0.2 * (doors >= 1 ? 1 : 0) + 0.15 * Math.min(windows, 2) / 2
      + 0.15 * Math.min(1, best.length / (targetInterior || 1)) - 0.01 * leftovers.size;
    team = Math.round(Math.min(1, Math.max(0, team)) * 1e6) / 1e6;
  }
  return { kind: 'open', enclosed, interior: [...interiorSet], interiorCells: interiorSet.size, shellCells: shell.size,
    doors, windows, leftovers: leftovers.size, leftoverSet: leftovers, team, perAgent, targetInterior };
}

/**
 * Incremental log reader + replay. Feed it records (or lines) in file order; it keeps
 *   log   = {header, task, events, snapshots, episodeEnd, warnings}
 *   built = {states, bp, task, agents, mismatches, verified, snapshotCount, snapsAtPos}
 * where states[p] is the state after the first p events (diffs applied in order; snapshots only checked).
 * Used both for whole files and for live following (records appended as they arrive).
 */
function createLogBuilder() {
  const log = { header: null, task: null, events: [], snapshots: [], episodeEnd: null, warnings: [] };
  const built = { states: [], bp: new Map(), task: null, agents: [], mismatches: [], verified: 0, snapshotCount: 0, snapsAtPos: new Map() };
  let lastId = -Infinity, lastT = -Infinity;
  const idToPos = new Map();
  let cur = null, computeStats = null;

  function startFromHeader(header, lineNo) {
    const g = header.grid || header.config || {};
    if (!(g.width > 0 && g.depth > 0 && g.height > 0)) throw new Error(`Line ${lineNo}: header has no valid grid size (grid.width/depth/height).`);
    const task = taskOf(header);
    if (task.kind === 'blueprint' && !Array.isArray(task.blueprint.cells)) throw new Error(`Line ${lineNo}: header blueprint has no cells list.`);
    log.header = header; log.task = task; built.task = task;
    const agents = header.agents || (header.config && header.config.agents) || [];
    built.agents = agents;
    const extraPenalty = header.config && typeof header.config.extra_penalty === 'number' ? header.config.extra_penalty : 0;
    if (task.blueprint) for (const [x, y, z, t] of task.blueprint.cells) built.bp.set(key(x, y, z), t);
    const cp = header.chest_pos || (header.config && header.config.chest_pos);
    const chestKey = Array.isArray(cp) ? key(cp[0], cp[1], cp[2]) : null;
    computeStats = (grid) => task.kind === 'open'
      ? openStats(grid, g, chestKey, agents, task.targetInterior)
      : blueprintStats(grid, built.bp, agents, extraPenalty);
    const inv = {};
    for (const a of agents) inv[a] = { ...((header.inventories || {})[a] || {}) };
    for (const a in (header.inventories || {})) if (!inv[a]) inv[a] = { ...header.inventories[a] };
    const actions = {};
    for (const a of agents) actions[a] = 0;
    // [v3] bodies: current cell per agent (position jumps to `to` immediately) + last move for animation
    cur = { grid: new Map(), chest: { ...(header.chest || {}) }, inv, votes: [], actions,
            positions: { ...(header.positions || {}) }, moves: {} };
    pushState();
  }

  function pushState() {
    const st = { ...cur, stats: computeStats(cur.grid) };
    built.states.push(st);
  }

  function checkAt(p, s) {
    const problems = checkSnapshot(s, built.states[p]);
    if (problems.length) built.mismatches.push({ pos: p, t: s.t, seq: s.seq, problems });
    else built.verified++;
  }

  function applyEvent(e) {
    const d = e.diff;
    let { grid, chest, inv, votes, actions, positions, moves } = cur;
    if (d.grid.length) {
      grid = new Map(grid);
      for (const g of d.grid) {
        const [x, y, z] = g.pos;
        if (g.after) grid.set(key(x, y, z), g.after); else grid.delete(key(x, y, z));
      }
    }
    if (Object.keys(d.chest).length) {
      chest = { ...chest };
      for (const t in d.chest) chest[t] = (chest[t] || 0) + d.chest[t];
    }
    if (Object.keys(d.inventory).length) {
      inv = { ...inv };
      for (const a in d.inventory) {
        inv[a] = { ...(inv[a] || {}) };
        for (const t in d.inventory[a]) addDelta(inv[a], t, d.inventory[a][t]);
      }
    }
    if (Object.keys(d.agents).length) {
      positions = { ...positions }; moves = { ...moves };
      for (const a in d.agents) {
        const m = d.agents[a];
        if (!m || !m.to) continue;
        positions[a] = m.to;
        const prev = moves[a];
        if (!m.jump && !m.fall && prev && prev.jump && typeof prev.arrive_t === 'number' && e.t <= prev.arrive_t + 1e-9) {
          // a block placed into the jumper's feet cell mid-hop: keep the hop, land on top of the new block
          moves[a] = { ...prev, to: m.to, landZ: m.to[2], raiseAt: e.t, arrive_t: m.arrive_t ?? prev.arrive_t };
        } else moves[a] = m;
      }
    }
    // failed calls while walking / airborne "use nothing up" (log v3); every other action-cost call counts, even failed ones
    const walkingFail = !e.ok && typeof e.message === 'string' && /^(walking|airborne):/.test(e.message);
    if (e.cost === 'action' && !walkingFail) actions = { ...actions, [e.agent]: (actions[e.agent] || 0) + 1 };
    if (e.tool === 'declare_done' && e.ok && !votes.includes(e.agent)) votes = [...votes, e.agent];
    cur = { grid, chest, inv, votes, actions, positions, moves };
    pushState();
  }

  /** Add one parsed record. Throws Error with a readable message for contract violations. */
  function addRecord(rec, lineNo) {
    if (!rec || typeof rec !== 'object' || Array.isArray(rec)) throw new Error(`Line ${lineNo}: expected a JSON object.`);
    if (!log.header && rec.type !== 'header') throw new Error(`Line ${lineNo}: the first record must have "type": "header" (found ${JSON.stringify(rec.type)}).`);
    switch (rec.type) {
      case 'header':
        if (log.header) { log.warnings.push(`Line ${lineNo}: extra header ignored.`); break; }
        startFromHeader(rec, lineNo); break;
      case 'event': {
        if (typeof rec.event_id === 'number' && rec.event_id <= lastId) log.warnings.push(`Line ${lineNo}: event_id ${rec.event_id} is not increasing.`);
        if (typeof rec.event_id === 'number') lastId = rec.event_id;
        if (typeof rec.t !== 'number') {
          // tolerate logs without t (e.g. log v1): fall back to one event per second
          rec.t = Number.isFinite(lastT) ? lastT + 1 : 0;
          if (!log.warnings.some(w => w.includes('no "t"'))) log.warnings.push(`Line ${lineNo}: event has no "t"; using 1 s per event.`);
        }
        if (rec.t < lastT) { log.warnings.push(`Line ${lineNo}: t decreases (${rec.t} < ${lastT}).`); rec.t = lastT; }
        lastT = rec.t;
        rec.diff = rec.diff || {};
        rec.diff.grid = rec.diff.grid || [];
        rec.diff.chest = rec.diff.chest || {};
        rec.diff.inventory = rec.diff.inventory || {};
        rec.diff.agents = rec.diff.agents || {};
        rec.args = rec.args || {};
        log.events.push(rec);
        idToPos.set(rec.event_id, log.events.length);
        applyEvent(rec);
        break;
      }
      case 'snapshot': {
        log.snapshots.push(rec); built.snapshotCount++;
        // a snapshot is written after the event it names, so that state already exists
        const p = rec.after_event_id === -1 ? 0 : idToPos.get(rec.after_event_id);
        if (p === undefined) { built.mismatches.push({ pos: null, t: rec.t, seq: rec.seq, problems: [`snapshot refers to unknown event_id ${rec.after_event_id}`] }); break; }
        if (!built.snapsAtPos.has(p)) built.snapsAtPos.set(p, []);
        built.snapsAtPos.get(p).push(rec);
        checkAt(p, rec);
        break;
      }
      case 'episode_end': log.episodeEnd = rec; break;
      default: break; // unknown record types are ignored by contract
    }
  }

  /** Parse and add one text line (blank lines ignored). */
  function addLine(line, lineNo) {
    line = line.trim();
    if (!line) return;
    let rec;
    try { rec = JSON.parse(line); }
    catch (e) {
      const snippet = line.length > 80 ? line.slice(0, 80) + '…' : line;
      throw new Error(`Line ${lineNo} is not valid JSON (${e.message}).\n  ${snippet}`);
    }
    addRecord(rec, lineNo);
  }

  // raw file content in arbitrary chunks (live following): only complete lines are parsed
  let pending = '', lineNo = 0, decoder = null;
  function addChunk(chunk, opts = {}) {
    if (typeof chunk !== 'string') { decoder = decoder || new TextDecoder(); chunk = decoder.decode(chunk, { stream: true }); }
    const before = { e: log.events.length, s: log.snapshots.length };
    pending += chunk;
    const parts = pending.split('\n');
    pending = parts.pop();
    for (const line of parts) {
      try { addLine(line, ++lineNo); }
      catch (err) { if (!opts.skipErrors || !log.header) throw err; log.warnings.push(err.message); }
    }
    return { events: log.events.length - before.e, snapshots: log.snapshots.length - before.s, ended: !!log.episodeEnd };
  }
  function flush() {
    if (decoder) { pending += decoder.decode(); decoder = null; }
    const rest = pending; pending = '';
    if (rest.trim()) addLine(rest, ++lineNo);
  }

  return { log, built, addRecord, addLine, addChunk, flush };
}

/** Parse a whole JSONL text. Returns the log (with .built attached). Throws a readable Error. */
function parseLog(text) {
  const b = createLogBuilder();
  b.addChunk(text); b.flush();
  if (!b.log.header) throw new Error('The file is empty or has no header record.');
  Object.defineProperty(b.log, 'built', { value: b.built, enumerable: false });
  return b.log;
}
/** Kept for callers/tests: the replay of a parsed log. */
function buildStates(log) { return log.built; }

function checkSnapshot(snap, st) {
  const out = [];
  if (Array.isArray(snap.blocks)) {
    const seen = new Set();
    for (const [x, y, z, t, by] of snap.blocks) {
      const k = key(x, y, z); seen.add(k);
      const b = st.grid.get(k);
      if (!b) out.push(`block ${t} at (${k}) missing in replay`);
      else if (b.type !== t || (by !== undefined && b.placed_by !== by)) out.push(`block at (${k}): replay ${b.type}/${b.placed_by}, snapshot ${t}/${by}`);
    }
    for (const k of st.grid.keys()) if (!seen.has(k)) out.push(`replay has a block at (${k}) the snapshot lacks`);
  }
  const norm = o => { const r = {}; for (const t in (o || {})) if (o[t]) r[t] = o[t]; return r; };
  const cmp = (label, a, b) => {
    const A = norm(a), B = norm(b);
    for (const t of new Set([...Object.keys(A), ...Object.keys(B)]))
      if ((A[t] || 0) !== (B[t] || 0)) out.push(`${label} ${t}: replay ${A[t] || 0}, snapshot ${B[t] || 0}`);
  };
  if (snap.chest) cmp('chest', st.chest, snap.chest);
  if (snap.inventories) for (const a in snap.inventories) cmp(`${a} inventory`, st.inv[a], snap.inventories[a]);
  if (snap.actions_used) cmp('actions used by', st.actions, snap.actions_used);
  if (Array.isArray(snap.done_votes)) {
    const A = [...st.votes].sort().join(','), B = [...snap.done_votes].sort().join(',');
    if (A !== B) out.push(`done votes: replay [${A}], snapshot [${B}]`);
  }
  if (snap.positions) for (const a in snap.positions) {
    const A = (st.positions[a] || []).join(','), B = (snap.positions[a] || []).join(',');
    if (A !== B) out.push(`${a} position: replay (${A}), snapshot (${B})`);
  }
  // scores: our own computation vs. what the engine logged
  const S = st.stats, det = snap.score_details || {};
  const num = k => (typeof det[k] === 'number' || typeof det[k] === 'boolean') ? det[k] : snap[k];
  const want = S.kind === 'open'
    ? { enclosed: S.enclosed, interior_cells: S.interiorCells, shell_cells: S.shellCells, doors: S.doors, windows: S.windows, leftovers: S.leftovers }
    : { correct_cells: S.correct, extra_cells: S.extra };
  for (const k in want) { const v = num(k); if (v !== undefined && v !== want[k]) out.push(`${k}: replay ${want[k]}, logged ${v}`); }
  if (typeof snap.team_score === 'number' && Math.abs(snap.team_score - S.team) > 1e-5) out.push(`team_score: replay ${S.team}, logged ${snap.team_score}`);
  if (snap.individual_scores) for (const a in snap.individual_scores)
    if ((S.perAgent[a] || 0) !== snap.individual_scores[a]) out.push(`individual score ${a}: replay ${S.perAgent[a] || 0}, logged ${snap.individual_scores[a]}`);
  return out.slice(0, 8);
}

/** Number of events with t <= time (binary search; events sorted by t). */
function posAtTime(events, time) {
  let lo = 0, hi = events.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (events[m].t <= time + 1e-9) lo = m + 1; else hi = m; }
  return lo;
}

function describeEvent(e) {
  const a = e.args || {};
  const pos = ('x' in a) ? `(${a.x}, ${a.y}, ${a.z})` : '';
  const target = a.agent ?? a.to ?? a.target ?? a.recipient;
  switch (e.tool) {
    case 'place': {
      const m = e.diff.agents && e.diff.agents[e.agent];
      return m && m.to ? `places ${a.type ?? '?'} under itself at ${pos}` : `place ${a.type ?? '?'} at ${pos}`;
    }
    case 'remove': {
      const g = e.diff.grid[0];
      const b = g && g.before;
      const falls = Object.entries(e.diff.agents || {}).filter(([, m]) => m && m.fall && m.from && m.to)
        .map(([who, m]) => `${who} fell ${m.from[2] - m.to[2]}`);
      return `remove ${b ? b.type + ' ' : ''}at ${pos}` + (b && b.placed_by && b.placed_by !== e.agent ? ` (placed by ${b.placed_by})` : '') + (falls.length ? ` — ${falls.join(', ')}` : '');
    }
    case 'take': return `take ${a.n ?? ''} ${a.type ?? ''} from chest`;
    case 'deposit': return `deposit ${a.n ?? ''} ${a.type ?? ''} into chest`;
    case 'give': return `give ${a.n ?? ''} ${a.type ?? ''} → ${target ?? '?'}`;
    case 'say': return 'says (public)';
    case 'whisper': return `whispers → ${(e.chat && e.chat.to) || target || '?'}`;
    case 'note': return 'writes a note';
    case 'jump': return 'jumps';
    case 'pillar_up': return `pillars up${a.type ? ' on ' + a.type : ''}` + (e.diff.agents && e.diff.agents[e.agent] && e.diff.agents[e.agent].to ? ` to (${e.diff.agents[e.agent].to.join(', ')})` : '');
    case 'move_to': {
      const m = e.diff.agents && e.diff.agents[e.agent];
      const to = m && m.to ? m.to : ('x' in a ? [a.x, a.y, a.z] : null);
      return `walks to ${to ? '(' + to.join(', ') + ')' : JSON.stringify(a)}` + (m && typeof m.arrive_t === 'number' && typeof m.depart_t === 'number' ? ` (${(m.arrive_t - m.depart_t).toFixed(1)} s)` : '');
    }
    case 'observe': return 'observes';
    case 'declare_done': return 'declares done';
    default: return `${e.tool} ${Object.keys(a).length ? JSON.stringify(a) : ''}`;
  }
}
/** [v3] Where an agent is at time t given its last walk (path weighted: straight 1, diagonal sqrt 2). Returns [x, y, z] floats. */
const HOP_HEIGHT = 1.25;
function walkPosition(m, t) {
  if (m.jump) {
    // small parabolic hop in place; if a block was placed underneath mid-hop, land on top of it
    const f = m.arrive_t > m.depart_t ? Math.min(1, Math.max(0, (t - m.depart_t) / (m.arrive_t - m.depart_t))) : 1;
    if (f >= 1) return m.to;
    const base = m.from || m.to;
    let z = base[2] + 4 * HOP_HEIGHT * f * (1 - f);
    if (typeof m.landZ === 'number' && t >= m.raiseAt) z = Math.max(z, m.landZ);
    return [base[0], base[1], z];
  }
  const path = m.path && m.path.length ? m.path : [m.from || m.to, m.to];
  if (!(typeof m.depart_t === 'number' && typeof m.arrive_t === 'number') || t >= m.arrive_t || path.length < 2) return m.to;
  if (t <= m.depart_t) return path[0];
  const segLen = [];
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    const d = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1], path[i][2] - path[i - 1][2]);
    segLen.push(d); total += d;
  }
  let dist = total * (t - m.depart_t) / (m.arrive_t - m.depart_t);
  for (let i = 0; i < segLen.length; i++) {
    if (dist <= segLen[i] || i === segLen.length - 1) {
      const f = segLen[i] ? Math.min(1, dist / segLen[i]) : 1, A = path[i], B = path[i + 1];
      return [A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f];
    }
    dist -= segLen[i];
  }
  return m.to;
}

/** Pose of `agent` at time t in a reconstructed state (see header). */
function agentPoseAt(state, agent, t) {
  const m = state.moves[agent];
  const cell = state.positions[agent] || null;
  if (!m) return { pos: cell, cell, moving: false, kind: null, until: null, facing: null };
  const pos = walkPosition(m, t);
  const moving = typeof m.arrive_t === 'number' && t < m.arrive_t;
  const kind = m.jump ? 'jump' : m.fall ? 'fall' : 'walk';
  let facing = null;
  if (moving && kind === 'walk') {
    const ahead = walkPosition(m, Math.min(m.arrive_t, t + 0.05));
    const dx = ahead[0] - pos[0], dy = ahead[1] - pos[1], n = Math.hypot(dx, dy);
    if (n > 1e-6) facing = [dx / n, dy / n];
  }
  return { pos, cell, moving, kind: moving ? kind : null, until: moving ? m.arrive_t : null, facing };
}

const typeColorCache = {};
function typeColor(t) {
  if (TYPE_COLORS[t]) return TYPE_COLORS[t];
  if (!typeColorCache[t]) {
    let h = 0; for (const c of String(t)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    typeColorCache[t] = EXTRA_TYPE_COLORS[h % EXTRA_TYPE_COLORS.length];
  }
  return typeColorCache[t];
}
function agentColors(agents) {
  const out = {};
  agents.forEach((a, i) => { out[a] = AGENT_COLORS[i % AGENT_COLORS.length]; });
  return out;
}

export {
  createLogBuilder, parseLog, buildStates, checkSnapshot, taskOf,
  blueprintStats, openStats,
  posAtTime, walkPosition, agentPoseAt, HOP_HEIGHT,
  describeEvent, typeColor, agentColors, TYPE_COLORS, AGENT_COLORS, key,
};
