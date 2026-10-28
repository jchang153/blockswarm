// Thin seam between the public site and the shared replay logic.
// `core.js` is viewer/core.js, copied next to this file by scripts/export_site.py, so the site and
// the developer viewer share one implementation of log parsing, replay, scoring and animation.
// Everything site-specific (where data lives, gzip, per-agent totals) stays in this file.
export {
  createLogBuilder, parseLog, taskOf, openStats, blueprintStats,
  posAtTime, walkPosition, agentPoseAt, describeEvent, typeColor, agentColors, key,
  isV4, shopOf, setCatalog, itemInfo, purchaseOf, walletAt, leaderboard, rankAgents, deployTopK, finalResults,
  isTeamBattle, teamsOf, sideOf, teamColor, teamStatsAt, sideAnalysis, combatOf, finalTeamResults, TEAM_COLORS,
} from './core.js';
import { parseLog, purchaseOf, combatOf } from './core.js';

let configPromise = null;
/** site-config.json → {title, subtitle, data_base_url}; data_base_url defaults to ./data/. */
export function siteConfig() {
  configPromise ??= fetch('site-config.json', { cache: 'no-cache' })
    .then(r => (r.ok ? r.json() : {}))
    .catch(() => ({}))
    .then(c => {
      let base = typeof c.data_base_url === 'string' && c.data_base_url ? c.data_base_url : './data/';
      if (!base.endsWith('/')) base += '/';
      return { title: 'BlockSwarm', subtitle: '', ...c, data_base_url: base };
    });
  return configPromise;
}

/** Resolve a data file name against the configured base (absolute URLs pass through). */
export async function dataUrl(file) {
  const { data_base_url } = await siteConfig();
  return new URL(file, new URL(data_base_url, location.href)).href;
}

export async function loadEpisodeIndex() {
  const url = await dataUrl('episodes.json');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load the episode list (${r.status} ${r.statusText}) from ${url}`);
  const list = await r.json();
  return Array.isArray(list) ? list : (list.episodes || []);
}

/**
 * Fetch a log and return its text. Works for gzip (checked by magic bytes, decompressed with
 * DecompressionStream) and for plain JSONL, so it does not matter whether the host also sets
 * Content-Encoding (in which case the browser has already decompressed it).
 */
export async function fetchLogText(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (${r.status} ${r.statusText})`);
  const buf = new Uint8Array(await r.arrayBuffer());
  onProgress?.(buf.length);
  if (buf.length >= 2 && buf[0] === 0x1f && buf[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decompress gzip (no DecompressionStream). Please use a current Chrome, Edge, Firefox or Safari.');
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return await new Response(stream).text();
  }
  return new TextDecoder().decode(buf);
}

export async function loadEpisodeLog(file) {
  return parseLog(await fetchLogText(await dataUrl(file)));
}

/** Per-agent totals over the first `upTo` events: placed, removed, taken from chest (v3), bought / coins spent (v4), messages, notes,
 *  [v5] sword hits dealt / taken, damage dealt / taken, kills. */
export function agentTotals(log, agents, upTo = log.events.length) {
  const out = {};
  for (const a of agents) out[a] = { placed: 0, removed: 0, taken: 0, deposited: 0, bought: 0, spent: 0, messages: 0, notes: 0, failed: 0,
    hits: 0, hitsTaken: 0, dealt: 0, received: 0, kills: 0 };
  for (let i = 0; i < upTo; i++) {
    const e = log.events[i], s = out[e.agent];
    if (!s) continue;
    if (!e.ok) { s.failed++; continue; }
    if (e.tool === 'place') s.placed++;
    else if (e.tool === 'remove') s.removed++;
    else if (e.tool === 'take') for (const v of Object.values(e.diff.chest)) { if (v < 0) s.taken -= v; }
    else if (e.tool === 'deposit') for (const v of Object.values(e.diff.chest)) { if (v > 0) s.deposited += v; }
    else if (e.tool === 'buy') { const b = purchaseOf(e); s.bought += b.n; s.spent += b.cost; }
    else if (e.tool === 'note') s.notes++;
    const c = combatOf(e);
    if (c) {
      s.hits++; s.dealt += c.damage || 0; if (c.killed) s.kills++;
      const v = out[c.target]; if (v) { v.hitsTaken++; v.received += c.damage || 0; }
    }
    if (e.chat) s.messages++;
  }
  return out;
}

/** The engine's final numbers: episode_end (or last snapshot) team score + details (log v2 kept them top-level). */
export function finalScore(log) {
  const src = log.episodeEnd || log.snapshots[log.snapshots.length - 1] || null;
  if (!src) return null;
  const det = { ...(src.score_details || {}) };
  for (const k of ['correct_cells', 'blueprint_cells', 'extra_cells']) if (!(k in det) && k in src) det[k] = src[k];
  return { team: src.team_score, details: det, individual: src.individual_scores || {}, reason: log.episodeEnd ? log.episodeEnd.reason : null, t: src.t };
}
