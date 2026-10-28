/**
 * BlockSwarm viewer core — shared, DOM-free and three.js-free logic for replaying episode logs
 * (docs/log-format.md, log versions 1–4, blueprint and open tasks; v4 = shop + shared wallet +
 * competitive incentive, docs/shop-and-incentives.md). Used by viewer/index.html and the static
 * gallery site. Plain ES module; also runs under Node (>= 18) for tests.
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
 *     builder.built = {states, bp, task, agents, mismatches, verified, snapshotCount, snapsAtPos,
 *                      v4, shop, endChecked}
 *       states[p]  = state after the first p events (states[0] = start):
 *                    {grid: Map<key,{type,placed_by,placed_at|placed_at_round}>, chest: {type:n} (v1–v3),
 *                     wallet: coins (v4; null before), inv: {agent:{type:n}}, votes: [agent] (from
 *                     diff.votes.after; older logs without diff.votes: every successful declare_done adds),
 *                     actions: {agent:n}, positions: {agent:[x,y,z]}, moves: {agent: last diff.agents entry
 *                     (+ landZ/raiseAt when a block was placed under a jumping agent)}, stats: see below}
 *       bp         = Map<key, type> of blueprint cells (empty for open tasks)
 *       mismatches = [{pos, t, seq, problems: [string]}] snapshots that disagree with the replay
 *                    (v4: also the episode_end record, seq 'end', checked against the final state:
 *                    wallet, individual_scores, ranking, deployed, analysis)
 *       snapsAtPos = Map<p, [snapshot]>
 *       v4         = true for log v4 (shop + wallet); shop = {pos, catalog, startingMoney} | null
 *     Errors (bad JSON, missing/invalid header) throw Error with a readable "Line N: ..." message.
 * parseLog(text) -> log      Whole-file parse; log.built holds the replay. Throws if there is no header.
 * buildStates(log) -> built  Same object as log.built (kept for older callers).
 * checkSnapshot(snapshot, state) -> [problem strings]
 * taskOf(header) -> {kind: 'open'|'blueprint', brief, blueprint|null, targetInterior}
 *
 * ── Scoring / analysis (matches docs/log-format.md › Scoring) ────────────────────────────────
 * blueprintStats(grid, bp, agents, extraPenalty)
 *     -> {kind:'blueprint', correct, wrong, extra, perAgent, team, blueprintCells, placed}
 * openStats(grid, dims{width,depth,height}, solidKeys ("x,y,z" | [keys] | null), agents, targetInterior, classify?)
 *     -> {kind:'open', enclosed, interior:[key], interiorCells, shellCells, doors, windows,
 *         leftovers, leftoverSet:Set<key>, team, perAgent, placed, targetInterior}
 *     classify(type) -> 'door' | 'window' | null (default: v3 types 'door' / 'glass'; v4: catalog
 *     door / transparent flags, see classifyItem). placed = {agent: blocks it placed still in the world}.
 *     For v4 logs the builder sets stats.v4 = true, stats.team = null (no team score) and
 *     stats.perAgent = stats.placed (the v4 individual score); stats.shellPerAgent keeps the shell count.
 *
 * ── v4: shop, wallet, ranking ─────────────────────────────────────────────────────────────────
 * isV4(header) -> bool; shopOf(header) -> {pos:[x,y,z], catalog:[item], startingMoney} | null
 * itemInfo(type) -> {id, name, family, price, color, transparent, door, legacy?} (from the catalog of the
 *     last loaded header; v1–v3 block types get a legacy entry). classifyItem(type) -> 'door'|'window'|null
 * purchaseOf(event) -> {agent, item, n, cost, ok} for buy events (cost = coins spent, 0 if failed), else null
 * walletAt(built, p) -> coins after the first p events (v4) | null
 * leaderboard(state, agents) -> [{agent, score, rank}] by blocks placed still standing (rank 1 = best, ties share)
 * rankAgents(scores{agent:n}, agents, k) -> {ranking:[{agent, score, rank}], deployed:[agent]}
 *     (competition ranking; the top k plus everyone tied with the k-th are deployed)
 * deployTopK(header) -> config.deploy_top_k or max(1, floor(N / 2)); 0 when config.incentive is "none"
 * finalResults(log) -> {ranking, deployed, wallet, startingMoney, spent, individual, analysis, reason, t, source}
 *     v4 summary: the engine's episode_end values when present, else recomputed from the final state.
 *
 * ── v5: team battle (docs/team-battle.md) ─────────────────────────────────────────────────────
 * With header.teams the builder also sets built.teams = {names, members:{team:[agent]}, teamOf:{agent:team},
 *     shops:{team:[x,y,z]}, dividerX, startWallets:{team:n}, combat:{max_health,...}|null, variant} (null without teams)
 *     and every state gets: wallets {team:n} (null without teams), teamVotes {team:[agent]} (votes = their union),
 *     health {agent:n} (null when combat is off), dead [agent], deadAt {agent:{t, at:[x,y,z], by, eventIdx}},
 *     kills [{t, attacker, target, eventIdx}], hits [{t, attacker, target, damage, before, after, killed, eventIdx}]
 *     (successful attacks so far), crossRemovals {team: blocks placed by that team removed by the other team},
 *     teamPlaced {team: blocks placed so far (successful places, removed ones included)}.
 * isTeamBattle(header) -> bool; teamsOf(header) -> built.teams-like object | null
 * sideOf(teams, x) -> team whose side the column x is on (first team: x < dividerX)
 * teamStatsAt(built, p) -> {team: {members, alive, dead, wallet, startWallet, votes, health:{agent:n},
 *     blocksOnSide, ownOnSide, othersOnSide, byTeamOnOtherSide, placedStanding, placedTotal, removedByOther}}
 * sideAnalysis(built, p, team) -> openStats() of the blocks on that team's side (both shops solid), cached
 * combatOf(event) -> diff.combat of a successful attack | null
 * finalTeamResults(log) -> {teams:{team:{alive, dead, analysis, blocks_on_side, blocks_placed_by_team,
 *     blocks_by_team_on_other_side, team_blocks_removed_by_other_team, wallet, startWallet}}, kills, reason, t, source}
 *     (the engine's episode_end values when present, else recomputed). No winner is ever computed.
 * agentColors(agents, teams?) -> with teams: warm shades for the first team, cool shades for the second.
 * TEAM_COLORS, teamColor(teams, team)
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
 *     "jumps", "places wood under itself at …", "remove … — agent_1 fell 1", "votes the house is finished
 *     (2 votes)", "withdraws its finished vote (1 vote)", "place … — finished votes cleared",
 *     "buys 12 × oak_planks (−24)", ...)
 * typeColor(type) -> '#rrggbb' (catalog colour for v4 items; fixed palette for the v1–v3 block types,
 *     stable hash colour otherwise)
 * agentColors(agents) -> {agent: '#rrggbb'} in order (8-colour palette, repeats after 8)
 * TYPE_COLORS, AGENT_COLORS, HOP_HEIGHT, key(x, y, z)
 */

const TYPE_COLORS = {
  wood: '#9a6a3f', stone: '#9aa1a8', brick: '#b8533c', glass: '#8fd3f4',
  plank: '#dcb878', roof: '#3d5f91', door: '#2e9a74',
};
const AGENT_COLORS = ['#3b7dd8', '#f28e2b', '#4bab5a', '#d1495b', '#9467bd', '#17a2b8', '#e0b000', '#e377c2'];
/** [v5] per-team palettes: each agent keeps its own shade, the team reads as one colour family. */
const TEAM_PALETTES = {
  red: ['#e03b3b', '#f28e2b', '#b5174b', '#ff7aa8', '#9c4a12', '#ffb067'],
  blue: ['#2f6fdf', '#17a2b8', '#6a4fd6', '#5fb3f0', '#1c3f94', '#4bc0a0'],
};
const TEAM_COLORS = { red: '#e0453a', blue: '#3b7bea' }; // same as site_src/js/gallery.js
const FALLBACK_TEAM_KEYS = ['red', 'blue'];
const EXTRA_TYPE_COLORS = ['#c2185b', '#7cb342', '#5e35b1', '#00897b', '#f4511e', '#6d4c41'];

const key = (x, y, z) => x + ',' + y + ',' + z;

// ---------- v4 shop catalog ----------
/** Legacy (log v1–v3) block types as catalog-like items, so callers can treat both alike. */
const LEGACY_ITEMS = {
  wood: { name: 'Wood', family: 'wood', id: 'wood_log' }, plank: { name: 'Plank', family: 'wood', id: 'plank_planks' },
  stone: { name: 'Stone', family: 'stone' }, brick: { name: 'Brick', family: 'bricks' },
  glass: { name: 'Glass', family: 'glass', transparent: true }, roof: { name: 'Roof', family: 'roof' },
  door: { name: 'Door', family: 'door', door: true },
};
let CATALOG = new Map();
/** Register the item catalog of the log being read (header.shop.catalog); [] / null clears it. */
function setCatalog(list) {
  CATALOG = new Map();
  for (const it of Array.isArray(list) ? list : []) if (it && typeof it.id === 'string') CATALOG.set(it.id, it);
}
const itemCache = new Map();
function itemInfo(t) {
  const c = CATALOG.get(t);
  if (c) return c;
  const k = String(t);
  if (!itemCache.has(k)) {
    const L = LEGACY_ITEMS[k] || {};
    itemCache.set(k, { id: k, name: L.name || k.replace(/_/g, ' '), family: L.family || 'misc', price: null,
      color: TYPE_COLORS[k] || hashColor(k), transparent: !!L.transparent, door: !!L.door, legacy: true,
      styleId: L.id || k });
  }
  return itemCache.get(k);
}
/** 'door' | 'window' | null for the house analysis (v4: catalog flags; v3: types door / glass). */
function classifyItem(t) {
  const it = itemInfo(t);
  return it.door ? 'door' : it.transparent ? 'window' : null;
}
const legacyClassify = t => (t === 'door' ? 'door' : t === 'glass' ? 'window' : null);
function isV4(header) { return !!header && ((typeof header.log_version === 'number' && header.log_version >= 4) || !!header.shop); }
function shopOf(header) {
  if (!header) return null;
  const s = header.shop && typeof header.shop === 'object' ? header.shop : null;
  const c = header.config || {};
  const pos = (s && s.pos) || c.shop_pos || header.shop_pos;
  if (!s && !Array.isArray(pos)) return null;
  const money = s && typeof s.starting_money === 'number' ? s.starting_money
    : typeof c.starting_money === 'number' ? c.starting_money : (typeof header.wallet === 'number' ? header.wallet : null);
  return { pos: Array.isArray(pos) ? pos : null, catalog: (s && Array.isArray(s.catalog)) ? s.catalog : [], startingMoney: money };
}
/** [v5] team battle: header.teams is set (two teams of agents). */
function isTeamBattle(header) { return !!(header && header.teams && typeof header.teams === 'object' && !Array.isArray(header.teams) && Object.keys(header.teams).length); }
/** [v5] {names, members, teamOf, shops, dividerX, startWallets, combat, variant} from the header, or null without teams. */
function teamsOf(header) {
  if (!isTeamBattle(header)) return null;
  const c = header.config || {};
  const names = Object.keys(header.teams);
  const members = {}, teamOf = {};
  for (const t of names) { members[t] = [...(header.teams[t] || [])]; for (const a of members[t]) teamOf[a] = t; }
  const shops = { ...((header.shops && typeof header.shops === 'object') ? header.shops : c.shops || {}) };
  const W = (header.grid || c).width;
  const dividerX = typeof header.divider_x === 'number' ? header.divider_x : typeof c.divider_x === 'number' ? c.divider_x : Math.floor(W / 2);
  const money = typeof c.starting_money === 'number' ? c.starting_money : null;
  const startWallets = {};
  for (const t of names) startWallets[t] = header.wallets && typeof header.wallets[t] === 'number' ? header.wallets[t] : money ?? 100 * members[t].length;
  let combat = header.combat && typeof header.combat === 'object' ? header.combat : null;
  if (!combat && c.combat) combat = { max_health: c.max_health ?? 20, sword_damage: c.sword_damage ?? 7, attack_reach: c.attack_reach ?? 3, attack_cooldown_s: c.attack_cooldown_s ?? 0.6 };
  return { names, members, teamOf, shops, dividerX, startWallets, combat, variant: header.variant || c.variant || 'team_battle',
    equipment: header.equipment || null };
}
/** [v5] Team whose half the column x is on (first team: x < dividerX, second: x >= dividerX). */
function sideOf(teams, x) { return teams ? (x < teams.dividerX ? teams.names[0] : teams.names[1]) : null; }
function teamColor(teams, t) {
  if (TEAM_COLORS[t]) return TEAM_COLORS[t];
  const i = teams ? teams.names.indexOf(t) : -1;
  return TEAM_COLORS[FALLBACK_TEAM_KEYS[i]] || '#888888';
}
/** [v5] Per-team status at the playhead (cheap: counts over the grid; no house analysis). */
function teamStatsAt(built, p) {
  const T = built.teams;
  if (!T) return null;
  const st = built.states[Math.max(0, Math.min(built.states.length - 1, p))];
  const out = {};
  for (const t of T.names) {
    const health = {};
    for (const a of T.members[t]) health[a] = st.health ? st.health[a] ?? null : null;
    out[t] = { members: T.members[t], alive: T.members[t].filter(a => !st.dead.includes(a)), dead: T.members[t].filter(a => st.dead.includes(a)),
      wallet: st.wallets ? st.wallets[t] : null, startWallet: T.startWallets[t], votes: (st.teamVotes && st.teamVotes[t]) || [], health,
      blocksOnSide: 0, ownOnSide: 0, othersOnSide: 0, byTeamOnOtherSide: 0, placedStanding: 0, removedByOther: (st.crossRemovals || {})[t] || 0,
      placedTotal: (st.teamPlaced || {})[t] || 0 };
  }
  for (const [k, b] of st.grid) {
    const x = +k.slice(0, k.indexOf(','));
    const side = sideOf(T, x), owner = T.teamOf[b.placed_by];
    out[side].blocksOnSide++;
    if (owner === side) out[side].ownOnSide++; else out[side].othersOnSide++;
    if (owner && out[owner]) { out[owner].placedStanding++; if (owner !== side) out[owner].byTeamOnOtherSide++; }
  }
  return out;
}
/** [v5] House analysis of the blocks on one team's half (both shops solid), cached per (position, team). */
function sideAnalysis(built, p, team) {
  const T = built.teams;
  if (!T) return null;
  p = Math.max(0, Math.min(built.states.length - 1, p));
  const ck = p + '|' + team;
  if (built.sideCache.has(ck)) return built.sideCache.get(ck);
  const grid = new Map();
  for (const [k, b] of built.states[p].grid) if (sideOf(T, +k.slice(0, k.indexOf(','))) === team) grid.set(k, b);
  const s = openStats(grid, built.dims, built.solidKeys, built.agents, built.task.targetInterior, classifyItem);
  if (built.sideCache.size > 400) built.sideCache.clear();
  built.sideCache.set(ck, s);
  return s;
}
const analysisOf = S => ({ enclosed: S.enclosed, interior_cells: S.interiorCells, shell_cells: S.shellCells, doors: S.doors, windows: S.windows, leftovers: S.leftovers });
/** [v5] diff.combat of a successful attack event, else null. */
function combatOf(e) { const c = e && e.ok && e.diff && e.diff.combat; return c && c.target ? c : null; }
/** [v5] Team-battle summary: the engine's episode_end numbers when present, else recomputed. No winner. */
function finalTeamResults(log) {
  const built = log.built, T = built && built.teams;
  if (!T) return null;
  const p = built.states.length - 1, st = built.states[p], end = log.episodeEnd;
  const live = teamStatsAt(built, p);
  const teams = {};
  for (const t of T.names) {
    const e = end && end.teams && end.teams[t] ? end.teams[t] : {};
    const L = live[t];
    teams[t] = {
      alive: e.alive || L.alive, dead: e.dead || L.dead,
      analysis: { ...analysisOf(sideAnalysis(built, p, t)), target_interior: built.task.targetInterior, ...(e.analysis || {}) },
      blocks_on_side: e.blocks_on_side ?? L.blocksOnSide,
      blocks_placed_by_team: e.blocks_placed_by_team ?? L.placedTotal, standing: L.placedStanding,
      blocks_by_team_on_other_side: e.blocks_by_team_on_other_side ?? L.byTeamOnOtherSide,
      team_blocks_removed_by_other_team: e.team_blocks_removed_by_other_team ?? L.removedByOther,
      wallet: end && end.wallets && typeof end.wallets[t] === 'number' ? end.wallets[t] : L.wallet, startWallet: L.startWallet,
      health: { ...L.health, ...Object.fromEntries(Object.entries((end && end.health) || {}).filter(([a]) => T.teamOf[a] === t)) },
    };
  }
  return { teams, kills: end && Array.isArray(end.kills) ? end.kills : st.kills.map(k => ({ t: k.t, attacker: k.attacker, target: k.target })),
    individual: end && end.individual_scores ? end.individual_scores : { ...(st.stats.placed || {}) },
    reason: end ? end.reason : null, t: end ? end.t : null, source: end ? 'episode_end' : 'replay' };
}
function deployTopK(header) {
  const c = (header && header.config) || {};
  if (c.incentive === 'none') return 0;
  if (typeof c.deploy_top_k === 'number') return c.deploy_top_k;
  const n = ((header && header.agents) || c.agents || []).length;
  return Math.max(1, Math.floor(n / 2));
}
/** Competition ranking (1, 1, 3, …) by score, ties by agent order; top k plus ties at the boundary are deployed. */
function rankAgents(scores, agents, k) {
  const order = [...agents].map((a, i) => ({ a, i, s: (scores && scores[a]) || 0 }))
    .sort((p, q) => q.s - p.s || p.i - q.i);
  const ranking = [];
  order.forEach((o, i) => {
    const rank = i > 0 && o.s === order[i - 1].s ? ranking[i - 1].rank : i + 1;
    ranking.push({ agent: o.a, score: o.s, rank });
  });
  const deployed = k > 0 && order.length ? order.filter(o => o.s >= order[Math.min(k, order.length) - 1].s).map(o => o.a) : [];
  return { ranking, deployed };
}
function leaderboard(state, agents) { return rankAgents((state && state.stats && state.stats.placed) || {}, agents, 0).ranking; }
function walletAt(built, p) { const st = built.states[Math.max(0, Math.min(built.states.length - 1, p))]; return st ? st.wallet : null; }
/** {agent, item, n, cost, ok} for a buy event (cost: coins that left the wallet), else null. */
function purchaseOf(e) {
  if (!e || e.tool !== 'buy') return null;
  const a = e.args || {};
  let item = a.item ?? a.type ?? a.id ?? null, n = typeof a.n === 'number' ? a.n : null;
  const got = e.diff && e.diff.inventory && e.diff.inventory[e.agent];
  if (e.ok && got) for (const t in got) if (got[t] > 0) { item = t; n = got[t]; break; }
  const dw = e.diff && e.diff.wallet;
  const w = typeof dw === 'number' ? dw : dw && typeof dw === 'object' ? Object.values(dw).reduce((s, v) => s + (typeof v === 'number' ? v : 0), 0) : 0;
  return { agent: e.agent, item, n: n ?? 1, cost: Math.max(0, -w), ok: !!e.ok };
}

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

function placedCounts(grid, agents) {
  const out = {};
  for (const a of agents) out[a] = 0;
  for (const b of grid.values()) if (b.placed_by != null) out[b.placed_by] = (out[b.placed_by] || 0) + 1;
  return out;
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
  return { kind: 'blueprint', correct, wrong, extra, perAgent, team: Math.round(team * 1e6) / 1e6, blueprintCells: n, placed: placedCounts(grid, agents) };
}

/**
 * Open task scoring / v4 house analysis (docs/log-format.md › Scoring › open). dims = {width, depth, height};
 * solidKeys = "x,y,z" or an array of them (chest / shop cells) or null; classify(type) -> 'door'|'window'|null.
 */
function openStats(grid, dims, solidKeys, agents, targetInterior, classify = legacyClassify) {
  const W = dims.width, D = dims.depth, H = dims.height;
  const idx = (x, y, z) => (z * D + y) * W + x;
  const N = W * D * H;
  const solid = new Uint8Array(N);
  for (const k of grid.keys()) {
    const [x, y, z] = k.split(',').map(Number);
    if (x >= 0 && x < W && y >= 0 && y < D && z >= 0 && z < H) solid[idx(x, y, z)] = 1;
  }
  for (const sk of (Array.isArray(solidKeys) ? solidKeys : solidKeys ? [solidKeys] : [])) {
    const [x, y, z] = sk.split(',').map(Number);
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
    const cls = classify(b.type);
    if (!cls) continue;
    // door / window: face contact with both interior and outside air
    let faceInterior = false, faceOutside = false;
    for (const [dx, dy, dz] of NB) {
      const nx = x + dx, ny = y + dy, nz = z + dz;
      if (interiorSet.has(key(nx, ny, nz))) faceInterior = true;
      if (nx >= 0 && ny >= 0 && nz >= 0 && nx < W && ny < D && nz < H) { if (outside[idx(nx, ny, nz)]) faceOutside = true; }
      else if (nz >= H && nx >= 0 && ny >= 0 && nx < W && ny < D) faceOutside = true; // open sky above the top layer
    }
    if (faceInterior && faceOutside) { if (cls === 'door') doors++; else windows++; }
  }
  let team = 0;
  if (enclosed) {
    team = 0.5 + 0.2 * (doors >= 1 ? 1 : 0) + 0.15 * Math.min(windows, 2) / 2
      + 0.15 * Math.min(1, best.length / (targetInterior || 1)) - 0.01 * leftovers.size;
    team = Math.round(Math.min(1, Math.max(0, team)) * 1e6) / 1e6;
  }
  return { kind: 'open', enclosed, interior: [...interiorSet], interiorCells: interiorSet.size, shellCells: shell.size,
    doors, windows, leftovers: leftovers.size, leftoverSet: leftovers, team, perAgent, targetInterior, placed: placedCounts(grid, agents) };
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
  const built = { states: [], bp: new Map(), task: null, agents: [], mismatches: [], verified: 0, snapshotCount: 0, snapsAtPos: new Map(),
    v4: false, shop: null, endChecked: false, teams: null, sideCache: new Map() };
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
    const v4 = isV4(header);
    const shop = v4 ? shopOf(header) : null;
    built.v4 = v4; built.shop = shop;
    setCatalog(shop ? shop.catalog : null);
    const teams = teamsOf(header);
    built.teams = teams;
    const cp = v4 ? shop && shop.pos : header.chest_pos || (header.config && header.config.chest_pos);
    // [v5] with teams every shop is a solid cell
    const solidKey = teams ? Object.values(teams.shops).filter(Array.isArray).map(p => key(p[0], p[1], p[2]))
      : Array.isArray(cp) ? key(cp[0], cp[1], cp[2]) : null;
    built.solidKeys = solidKey; built.dims = g;
    computeStats = (grid) => {
      if (task.kind !== 'open') return blueprintStats(grid, built.bp, agents, extraPenalty);
      const s = openStats(grid, g, solidKey, agents, task.targetInterior, v4 ? classifyItem : legacyClassify);
      if (v4) { s.v4 = true; s.team = null; s.shellPerAgent = s.perAgent; s.perAgent = s.placed; }
      return s;
    };
    const inv = {};
    for (const a of agents) inv[a] = { ...((header.inventories || {})[a] || {}) };
    for (const a in (header.inventories || {})) if (!inv[a]) inv[a] = { ...header.inventories[a] };
    const actions = {};
    for (const a of agents) actions[a] = 0;
    // [v3] bodies: current cell per agent (position jumps to `to` immediately) + last move for animation
    const wallet = v4 && !teams ? (typeof header.wallet === 'number' ? header.wallet : (shop && shop.startingMoney) ?? 0) : null;
    // [v5] team wallets, per-team votes, health, deaths
    const wallets = teams ? { ...teams.startWallets } : null;
    const teamVotes = teams ? Object.fromEntries(teams.names.map(t => [t, []])) : null;
    let health = null;
    if (header.health && typeof header.health === 'object') health = { ...header.health };
    else if (teams && teams.combat) { health = {}; for (const a of agents) health[a] = teams.combat.max_health ?? 20; }
    cur = { grid: new Map(), chest: { ...(header.chest || {}) }, wallet, inv, votes: [], actions,
            positions: { ...(header.positions || {}) }, moves: {},
            wallets, teamVotes, health, dead: [], deadAt: {}, kills: [], hits: [],
            crossRemovals: teams ? Object.fromEntries(teams.names.map(t => [t, 0])) : null,
            teamPlaced: teams ? Object.fromEntries(teams.names.map(t => [t, 0])) : null };
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
    let { grid, chest, wallet, inv, votes, actions, positions, moves, wallets, teamVotes, health, dead, deadAt, kills, hits, crossRemovals, teamPlaced } = cur;
    const T = built.teams, idx = log.events.length - 1;
    if (d.grid.length) {
      grid = new Map(grid);
      for (const g of d.grid) {
        const [x, y, z] = g.pos;
        if (g.after) grid.set(key(x, y, z), g.after); else grid.delete(key(x, y, z));
        // [v5] blocks placed per team (ever); a block placed by one team removed by a member of the other team
        if (T && e.ok && g.after && !g.before && T.teamOf[g.after.placed_by]) { const pt = T.teamOf[g.after.placed_by]; teamPlaced = { ...teamPlaced, [pt]: (teamPlaced[pt] || 0) + 1 }; }
        if (T && e.ok && !g.after && g.before && g.before.placed_by != null) {
          const owner = T.teamOf[g.before.placed_by], by = T.teamOf[e.agent];
          if (owner && by && owner !== by) { crossRemovals = { ...crossRemovals, [owner]: (crossRemovals[owner] || 0) + 1 }; }
        }
      }
    }
    if (Object.keys(d.chest).length) {
      chest = { ...chest };
      for (const t in d.chest) chest[t] = (chest[t] || 0) + d.chest[t];
    }
    if (typeof d.wallet === 'number' && d.wallet && wallet !== null) wallet += d.wallet;
    else if (d.wallet && typeof d.wallet === 'object' && wallets) {
      wallets = { ...wallets };
      for (const t in d.wallet) if (typeof d.wallet[t] === 'number') wallets[t] = (wallets[t] || 0) + d.wallet[t];
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
        if (m && m.dead) {
          // [v5] killed: the body is removed (no position, nobody stands on it)
          const at = Array.isArray(m.at) ? m.at : positions[a] || null;
          delete positions[a]; delete moves[a];
          if (!dead.includes(a)) dead = [...dead, a];
          deadAt = { ...deadAt, [a]: { t: e.t, at, by: e.agent, eventIdx: idx } };
          continue;
        }
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
    const walkingFail = !e.ok && typeof e.message === 'string' && /^(walking|airborne|cooldown)\b/.test(e.message);
    if (e.cost === 'action' && !walkingFail) actions = { ...actions, [e.agent]: (actions[e.agent] || 0) + 1 };
    if (d.votes && typeof d.votes === 'object') {
      // finished votes as logged (declare_done adds, cancel_done removes, any build change clears all)
      if (Array.isArray(d.votes.after)) votes = [...d.votes.after];
      else if (teamVotes) {
        // [v5] per team: {team: {before, after}} for each team whose votes changed
        let changed = false;
        for (const t in d.votes) if (d.votes[t] && Array.isArray(d.votes[t].after)) {
          if (!changed) { teamVotes = { ...teamVotes }; changed = true; }
          teamVotes[t] = [...d.votes[t].after];
        }
        if (changed) votes = Object.values(teamVotes).flat();
      }
    } else if (e.tool === 'declare_done' && e.ok && !votes.includes(e.agent)) {
      votes = [...votes, e.agent]; // older logs without diff.votes: votes only ever accumulate
    }
    // [v5] sword hits
    const c = d.combat;
    if (e.ok && c && typeof c === 'object' && c.target) {
      const after = typeof c.health_after === 'number' ? c.health_after : null;
      if (health && after !== null) health = { ...health, [c.target]: after };
      const hit = { t: e.t, attacker: c.attacker || e.agent, target: c.target, damage: c.damage, before: c.health_before, after, killed: !!c.killed, eventIdx: idx };
      hits = [...hits, hit];
      if (hit.killed) {
        kills = [...kills, { t: e.t, attacker: hit.attacker, target: hit.target, eventIdx: idx }];
        if (!dead.includes(c.target)) {
          dead = [...dead, c.target];
          deadAt = { ...deadAt, [c.target]: { t: e.t, at: positions[c.target] || null, by: hit.attacker, eventIdx: idx } };
          positions = { ...positions }; delete positions[c.target];
          moves = { ...moves }; delete moves[c.target];
        }
      }
    }
    cur = { grid, chest, wallet, inv, votes, actions, positions, moves, wallets, teamVotes, health, dead, deadAt, kills, hits, crossRemovals, teamPlaced };
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
        if (typeof rec.diff.wallet !== 'number' && !(rec.diff.wallet && typeof rec.diff.wallet === 'object' && !Array.isArray(rec.diff.wallet))) rec.diff.wallet = 0;
        if (!rec.diff.combat || typeof rec.diff.combat !== 'object') rec.diff.combat = {};
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
      case 'episode_end':
        log.episodeEnd = rec;
        if (built.v4 && !built.endChecked) {
          built.endChecked = true;
          const p = built.states.length - 1;
          const problems = checkEnd(rec, built.states[p], built.agents, deployTopK(log.header), built);
          if (problems.length) built.mismatches.push({ pos: p, t: rec.t, seq: 'end', problems });
        }
        break;
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

/** v4: episode_end vs. the replayed final state (wallet, individual scores, ranking, deployed, analysis). */
function checkEnd(end, st, agents, k, built) {
  const out = [];
  checkTeamState(end, st, out, 'final ');
  const T = built && built.teams;
  if (T && end.teams && typeof end.teams === 'object') {
    const live = teamStatsAt(built, built.states.indexOf(st));
    for (const t of T.names) {
      const e = end.teams[t]; if (!e) { out.push(`episode_end.teams lacks ${t}`); continue; }
      const L = live[t];
      const same = (a, b) => [...a].sort().join(',') === [...b].sort().join(',');
      if (Array.isArray(e.alive) && !same(e.alive, L.alive)) out.push(`${t} alive: replay [${L.alive}], logged [${e.alive}]`);
      const cmpN = (k, v, label) => { if (typeof e[k] === 'number' && e[k] !== v) out.push(`${t} ${label}: replay ${v}, logged ${e[k]}`); };
      cmpN('blocks_on_side', L.blocksOnSide, 'blocks on side');
      cmpN('blocks_placed_by_team', L.placedTotal, 'blocks placed');
      cmpN('blocks_by_team_on_other_side', L.byTeamOnOtherSide, 'blocks on the other side');
      cmpN('team_blocks_removed_by_other_team', L.removedByOther, 'blocks removed by the other team');
      if (e.analysis) {
        const A = analysisOf(sideAnalysis(built, built.states.indexOf(st), t));
        for (const kk in A) if (kk in e.analysis && e.analysis[kk] !== A[kk]) out.push(`${t} side ${kk}: replay ${A[kk]}, logged ${e.analysis[kk]}`);
      }
    }
  }
  if (Array.isArray(end.kills) && end.kills.length !== st.kills.length) out.push(`kills: replay ${st.kills.length}, logged ${end.kills.length}`);
  if (typeof end.wallet === 'number' && st.wallet !== null && end.wallet !== st.wallet) out.push(`final wallet: replay ${st.wallet}, logged ${end.wallet}`);
  const S = st.stats;
  if (end.individual_scores) for (const a in end.individual_scores)
    if ((S.placed[a] || 0) !== end.individual_scores[a]) out.push(`final score ${a}: replay ${S.placed[a] || 0}, logged ${end.individual_scores[a]}`);
  const mine = rankAgents(S.placed, agents, k);
  if (Array.isArray(end.ranking)) {
    const want = new Map(mine.ranking.map(r => [r.agent, r]));
    for (const r of end.ranking) {
      const m = want.get(r.agent);
      if (!m) out.push(`ranking lists unknown agent ${r.agent}`);
      else if (m.score !== r.score || m.rank !== r.rank) out.push(`ranking ${r.agent}: replay #${m.rank} (${m.score}), logged #${r.rank} (${r.score})`);
    }
  }
  if (Array.isArray(end.deployed)) {
    const A = [...mine.deployed].sort().join(','), B = [...end.deployed].sort().join(',');
    if (A !== B) out.push(`deployed: replay [${A}], logged [${B}]`);
  }
  const det = end.analysis || {};
  const want = { enclosed: S.enclosed, interior_cells: S.interiorCells, shell_cells: S.shellCells, doors: S.doors, windows: S.windows, leftovers: S.leftovers };
  if (S.kind === 'open') for (const kk in want) if (kk in det && det[kk] !== want[kk]) out.push(`final ${kk}: replay ${want[kk]}, logged ${det[kk]}`);
  return out.slice(0, 8);
}

/** v4 results: the engine's episode_end numbers when present, else recomputed from the final replayed state. */
function finalResults(log) {
  const built = log.built, header = log.header;
  if (!built || !built.v4) return null;
  const st = built.states[built.states.length - 1];
  const end = log.episodeEnd;
  const k = deployTopK(header);
  const mine = rankAgents(st.stats.placed || {}, built.agents, k);
  const start = built.shop && built.shop.startingMoney != null ? built.shop.startingMoney : built.states[0].wallet;
  const wallet = end && typeof end.wallet === 'number' ? end.wallet : st.wallet;
  const S = st.stats;
  const replayAnalysis = S.kind === 'open' ? { enclosed: S.enclosed, interior_cells: S.interiorCells, shell_cells: S.shellCells,
    doors: S.doors, windows: S.windows, leftovers: S.leftovers, target_interior: S.targetInterior } : {};
  return {
    ranking: end && Array.isArray(end.ranking) ? end.ranking : mine.ranking,
    deployed: end && Array.isArray(end.deployed) ? end.deployed : mine.deployed,
    individual: end && end.individual_scores ? end.individual_scores : { ...(S.placed || {}) },
    wallet, startingMoney: start, spent: typeof start === 'number' && typeof wallet === 'number' ? start - wallet : null,
    analysis: { ...replayAnalysis, ...((end && end.analysis) || {}) },
    deployTopK: k, incentive: (header.config || {}).incentive || 'competitive',
    reason: end ? end.reason : null, t: end ? end.t : null, source: end ? 'episode_end' : 'replay',
  };
}

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
  if (typeof snap.wallet === 'number' && st.wallet !== null && st.wallet !== undefined && snap.wallet !== st.wallet)
    out.push(`wallet: replay ${st.wallet}, snapshot ${snap.wallet}`);
  if (snap.inventories) for (const a in snap.inventories) cmp(`${a} inventory`, st.inv[a], snap.inventories[a]);
  if (snap.actions_used) cmp('actions used by', st.actions, snap.actions_used);
  checkTeamState(snap, st, out, '');
  if (Array.isArray(snap.done_votes)) {
    const A = [...st.votes].sort().join(','), B = [...snap.done_votes].sort().join(',');
    if (A !== B) out.push(`done votes: replay [${A}], snapshot [${B}]`);
  }
  if (snap.positions) for (const a in snap.positions) {
    if (st.dead && st.dead.includes(a)) continue; // [v5] the engine may keep a dead agent's last cell; its body is gone
    const A = (st.positions[a] || []).join(','), B = (snap.positions[a] || []).join(',');
    if (A !== B) out.push(`${a} position: replay (${A}), snapshot (${B})`);
  }
  // scores: our own computation vs. what the engine logged
  const S = st.stats, det = snap.score_details || snap.analysis || {};
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

/** [v5] wallets / health / dead / per-team done votes of a snapshot or episode_end vs. a replayed state. */
function checkTeamState(snap, st, out, pre) {
  if (snap.wallets && typeof snap.wallets === 'object' && st.wallets)
    for (const t of new Set([...Object.keys(snap.wallets), ...Object.keys(st.wallets)]))
      if (snap.wallets[t] !== st.wallets[t]) out.push(`${pre}wallet ${t}: replay ${st.wallets[t]}, logged ${snap.wallets[t]}`);
  if (snap.health && typeof snap.health === 'object' && st.health)
    for (const a in snap.health) if (snap.health[a] !== st.health[a]) out.push(`${pre}health ${a}: replay ${st.health[a]}, logged ${snap.health[a]}`);
  if (Array.isArray(snap.dead)) {
    const A = [...st.dead].sort().join(','), B = [...snap.dead].sort().join(',');
    if (A !== B) out.push(`${pre}dead: replay [${A}], logged [${B}]`);
  }
  if (snap.done_votes && typeof snap.done_votes === 'object' && !Array.isArray(snap.done_votes) && st.teamVotes)
    for (const t of new Set([...Object.keys(snap.done_votes), ...Object.keys(st.teamVotes)])) {
      const A = [...(st.teamVotes[t] || [])].sort().join(','), B = [...(snap.done_votes[t] || [])].sort().join(',');
      if (A !== B) out.push(`${pre}done votes ${t}: replay [${A}], logged [${B}]`);
    }
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
  return describeAction(e, a, pos, target) + votesCleared(e);
}
/** " (n votes)" after a vote change, from diff.votes (log v3 with votes; empty otherwise). */
function voteCount(e) {
  const v = e.diff && e.diff.votes;
  if (v && typeof v === 'object' && !Array.isArray(v.after)) {
    const t = Object.keys(v).find(k => v[k] && Array.isArray(v[k].after));
    if (t) { const n = v[t].after.length; return ` (${t}: ${n} vote${n === 1 ? '' : 's'})`; }
  }
  return v && Array.isArray(v.after) ? ` (${v.after.length} vote${v.after.length === 1 ? '' : 's'})` : '';
}
/** " — finished votes cleared" when a change to the build wiped the votes. */
function votesCleared(e) {
  const v = e.diff && e.diff.votes;
  if (v && !Array.isArray(v.after) && typeof v === 'object' && (e.tool === 'place' || e.tool === 'remove')) {
    // [v5] per-team votes
    const cleared = Object.keys(v).filter(t => v[t] && Array.isArray(v[t].before) && v[t].before.length && Array.isArray(v[t].after) && !v[t].after.length);
    return cleared.length ? ` — ${cleared.join(' & ')} finished votes cleared` : '';
  }
  return v && Array.isArray(v.before) && v.before.length && Array.isArray(v.after) && !v.after.length
    && e.tool !== 'cancel_done' ? ' — finished votes cleared' : '';
}
function describeAction(e, a, pos, target) {
  switch (e.tool) {
    case 'place': {
      const m = e.diff.agents && e.diff.agents[e.agent];
      const t = a.type ?? a.item ?? '?';
      return m && m.to ? `places ${t} under itself at ${pos}` : `place ${t} at ${pos}`;
    }
    case 'remove': {
      const g = e.diff.grid[0];
      const b = g && g.before;
      const falls = Object.entries(e.diff.agents || {}).filter(([, m]) => m && m.fall && m.from && m.to)
        .map(([who, m]) => `${who} fell ${m.from[2] - m.to[2]}`);
      return `remove ${b ? b.type + ' ' : ''}at ${pos}` + (b && b.placed_by && b.placed_by !== e.agent ? ` (placed by ${b.placed_by})` : '') + (falls.length ? ` — ${falls.join(', ')}` : '');
    }
    case 'buy': {
      const b = purchaseOf(e);
      return e.ok ? `buys ${b.n} × ${b.item} (−${b.cost})` : `buy ${b.n} × ${b.item ?? '?'}`;
    }
    case 'take': return `take ${a.n ?? ''} ${a.type ?? ''} from chest`;
    case 'deposit': return `deposit ${a.n ?? ''} ${a.type ?? ''} into chest`;
    case 'give': return `give ${a.n ?? ''} ${a.type ?? a.item ?? ''} → ${target ?? '?'}`;
    case 'say': return e.chat && e.chat.channel === 'team' ? `says (team${e.chat.team ? ' ' + e.chat.team : ''})` : 'says (public)';
    case 'say_all': return 'says (to all)';
    case 'attack': {
      const c = combatOf(e), who = a.target ?? target ?? '?';
      if (!c) return `attacks ${who}`;
      return c.killed ? `kills ${c.target} with a sword (${c.health_before} → 0 health)` : `hits ${c.target} with a sword (${c.health_before} → ${c.health_after} health)`;
    }
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
    case 'declare_done': return 'votes the house is finished' + voteCount(e);
    case 'cancel_done': return 'withdraws its finished vote' + voteCount(e);
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
function hashColor(t) {
  if (!typeColorCache[t]) {
    let h = 0; for (const c of String(t)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    typeColorCache[t] = EXTRA_TYPE_COLORS[h % EXTRA_TYPE_COLORS.length];
  }
  return typeColorCache[t];
}
function typeColor(t) {
  const c = CATALOG.get(t);
  if (c && c.color) return c.color;
  return TYPE_COLORS[t] || hashColor(t);
}
function agentColors(agents, teams = null) {
  const out = {};
  if (teams && teams.names) {
    const n = {};
    agents.forEach((a, i) => {
      const t = teams.teamOf[a], ti = teams.names.indexOf(t);
      const pal = TEAM_PALETTES[t] || TEAM_PALETTES[FALLBACK_TEAM_KEYS[ti]];
      if (!pal) { out[a] = AGENT_COLORS[i % AGENT_COLORS.length]; return; }
      n[t] = (n[t] || 0); out[a] = pal[n[t]++ % pal.length];
    });
    return out;
  }
  agents.forEach((a, i) => { out[a] = AGENT_COLORS[i % AGENT_COLORS.length]; });
  return out;
}

export {
  createLogBuilder, parseLog, buildStates, checkSnapshot, taskOf,
  blueprintStats, openStats,
  isV4, shopOf, setCatalog, itemInfo, classifyItem, purchaseOf, walletAt, leaderboard, rankAgents, deployTopK, finalResults,
  posAtTime, walkPosition, agentPoseAt, HOP_HEIGHT,
  describeEvent, typeColor, agentColors, TYPE_COLORS, AGENT_COLORS, key,
  isTeamBattle, teamsOf, sideOf, teamColor, teamStatsAt, sideAnalysis, combatOf, finalTeamResults, TEAM_COLORS, TEAM_PALETTES,
};
