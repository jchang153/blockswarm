// Gallery: episode cards with isometric thumbnails of the final build.
import { siteConfig, loadEpisodeIndex, setCatalog } from './core-adapter.js';
import { drawIso } from './iso.js';
import { swatchUrl } from './textures.js';
import { $, esc, fmtDuration, fmtScore, scoreBand, shortModel, reasonLabel, initTheme, agentLabel } from './ui.js';

initTheme();
let episodes = [];

const taskLabel = t => (t === 'open' ? 'Open brief' : t === 'blueprint' ? 'Blueprint' : t || '–');
const modelsOf = ep => (ep.models && ep.models.length ? ep.models : [...new Set((ep.agents || []).map(a => a.model).filter(Boolean))]);

/** Log v4 cards: who got deployed, coins spent, the house analysis (never shown to agents), top items. */
function chipsForV4(ep) {
  const a = ep.analysis || {}, out = [];
  const dep = ep.deployed || [];
  out.push(`<span class="chip dep-chip" title="Agents with the most blocks still standing (top ${ep.deploy_top_k ?? '?'}, ties included)">deployed: ${dep.length ? dep.map(x => esc(agentLabel(x))).join(', ') : 'nobody'}</span>`);
  if (ep.wallet && typeof ep.wallet.spent === 'number') out.push(`<span class="chip" title="Coins spent from the shared wallet">${ep.wallet.spent}/${ep.wallet.start} coins spent</span>`);
  if ('enclosed' in a) out.push(`<span class="chip ${a.enclosed ? 'ok' : 'no'}" title="House analysis (not shown to agents)">${a.enclosed ? '✓ enclosed' : '✗ not enclosed'}</span>`);
  if (a.enclosed && 'interior_cells' in a) out.push(`<span class="chip" title="House analysis (not shown to agents)">interior ${a.interior_cells}</span>`);
  if ('doors' in a) out.push(`<span class="chip ${a.doors ? 'ok' : 'no'}">${a.doors} door${a.doors === 1 ? '' : 's'}</span>`);
  if ('windows' in a) out.push(`<span class="chip ${a.windows ? 'ok' : 'no'}">${a.windows} window${a.windows === 1 ? '' : 's'}</span>`);
  if (a.leftovers) out.push(`<span class="chip no">${a.leftovers} leftover${a.leftovers === 1 ? '' : 's'}</span>`);
  return out.join('');
}
function itemsFor(ep) {
  if (!ep.v4 || !(ep.items_used || []).length) return '';
  setCatalog(ep.catalog || null);
  const more = (ep.item_types || 0) - ep.items_used.length;
  return `<div class="chips">${ep.items_used.map(([t, n]) => {
    const it = (ep.catalog || []).find(c => c.id === t);
    return `<span class="chip" title="${esc(it ? it.name : t)}"><span class="blk" style="background-image:url(${swatchUrl(t)})"></span>${esc(t.replace(/_/g, ' '))} ${n}</span>`;
  }).join('')}${more > 0 ? `<span class="chip no">+${more} more</span>` : ''}</div>`;
}

function chipsFor(ep) {
  if (ep.v4) return chipsForV4(ep);
  const d = ep.score_details || {};
  const out = [];
  if (ep.task === 'open') {
    if ('enclosed' in d) out.push(`<span class="chip ${d.enclosed ? 'ok' : 'no'}">${d.enclosed ? '✓ enclosed' : '✗ not enclosed'}</span>`);
    if ('interior_cells' in d) out.push(`<span class="chip">interior ${d.interior_cells}${d.target_interior ? '/' + d.target_interior : ''}</span>`);
    if ('doors' in d) out.push(`<span class="chip ${d.doors ? 'ok' : 'no'}">${d.doors} door${d.doors === 1 ? '' : 's'}</span>`);
    if ('windows' in d) out.push(`<span class="chip ${d.windows ? 'ok' : 'no'}">${d.windows} window${d.windows === 1 ? '' : 's'}</span>`);
    if (d.leftovers) out.push(`<span class="chip no">${d.leftovers} leftover${d.leftovers === 1 ? '' : 's'}</span>`);
  } else if ('correct_cells' in d) {
    out.push(`<span class="chip ok">${d.correct_cells}/${d.blueprint_cells} correct</span>`);
    if (d.extra_cells) out.push(`<span class="chip no">${d.extra_cells} extra</span>`);
  }
  return out.join('');
}

function winnerBadge(ep) {
  const top = (ep.ranking || []).filter(r => r.rank === 1);
  if (!top.length) return '';
  const names = top.map(r => agentLabel(r.agent)).join(' & ');
  return `<span class="winner" title="Most blocks still standing: ${esc(names)} (${top[0].score})">★ ${esc(names)} · ${top[0].score}</span>`;
}

function card(ep) {
  const a = document.createElement('a');
  a.className = 'ep-card';
  a.href = `episode.html?ep=${encodeURIComponent(ep.id)}`;
  const models = modelsOf(ep);
  const title = ep.title || ep.id;
  const sub = ep.task === 'blueprint' ? `blueprint “${ep.blueprint || '?'}”` : (ep.brief ? 'open brief' : 'open task');
  a.innerHTML = `
    <div class="thumb"><canvas aria-hidden="true"></canvas>
      <span class="kind">${esc(taskLabel(ep.task))}</span>
      ${ep.v4 ? winnerBadge(ep) : `<span class="score s${scoreBand(ep.team_score)}" title="Team score (0–1)">${fmtScore(ep.team_score)}</span>`}</div>
    <div class="ep-body">
      <div class="ep-title">${esc(title)}</div>
      <div class="chips">${models.length ? models.map(m => `<span class="chip model" title="${esc(m)}">${esc(shortModel(m))}</span>`).join('') : '<span class="chip model no">model not recorded</span>'}</div>
      <div class="chips">${chipsFor(ep)}</div>
      ${itemsFor(ep)}
      <div class="ep-meta">
        <span>${(ep.agents || []).length} agents</span>
        <span>${fmtDuration(ep.duration_s)}</span>
        <span>${esc(reasonLabel(ep.end_reason))}</span>
        <span>${ep.chat_count ?? 0} messages</span>
        ${ep.scarcity ? `<span>${esc(ep.scarcity)} blocks</span>` : ''}
        ${ep.v4 && ep.incentive && ep.incentive !== 'competitive' ? `<span>incentive: ${esc(ep.incentive)}</span>` : ''}
      </div>
    </div>`;
  a.setAttribute('aria-label', ep.v4 ? `${title}, ${sub}, deployed: ${(ep.deployed || []).map(agentLabel).join(', ') || 'nobody'}`
    : `${title}, ${sub}, team score ${fmtScore(ep.team_score)}`);
  a._ep = ep;
  return a;
}

// draw thumbnails lazily as cards scroll into view
const io = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
  for (const en of entries) if (en.isIntersecting) { io.unobserve(en.target); drawThumb(en.target); }
}, { rootMargin: '200px' }) : null;
function drawThumb(cardEl) {
  try { setCatalog(cardEl._ep.v4 ? cardEl._ep.catalog || [] : null); drawIso(cardEl.querySelector('canvas'), cardEl._ep); } catch (e) { console.warn('thumbnail failed', cardEl._ep.id, e); }
}

function render() {
  const task = $('fTask').value, model = $('fModel').value, sort = $('fSort').value;
  let list = episodes.filter(ep => (!task || ep.task === task) && (!model || modelsOf(ep).includes(model) || (model === '__none' && !modelsOf(ep).length)));
  const num = v => (typeof v === 'number' ? v : -Infinity);
  const cmp = {
    score: (a, b) => num(b.team_score) - num(a.team_score),
    date: (a, b) => String(b.created || '').localeCompare(String(a.created || '')),
    duration: (a, b) => num(b.duration_s) - num(a.duration_s),
    'duration-asc': (a, b) => num(a.duration_s) - num(b.duration_s),
  }[sort] || (() => 0);
  list = list.slice().sort((a, b) => cmp(a, b) || String(a.id).localeCompare(String(b.id)));
  const grid = $('grid');
  grid.innerHTML = '';
  for (const ep of list) {
    const c = card(ep);
    grid.appendChild(c);
    if (io) io.observe(c); else drawThumb(c);
  }
  $('count').textContent = `${list.length} of ${episodes.length} run${episodes.length === 1 ? '' : 's'}`;
  const params = new URLSearchParams();
  if (task) params.set('task', task);
  if (model) params.set('model', model);
  if (sort !== 'score') params.set('sort', sort);
  history.replaceState(null, '', params.toString() ? '?' + params : location.pathname);
}

function redrawAll() { for (const c of $('grid').children) if (c._ep && c.querySelector('canvas').width > 1) drawThumb(c); }
let resizeT = 0;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(redrawAll, 200); });

async function main() {
  const cfg = await siteConfig();
  if (cfg.title) { $('siteTitle').textContent = cfg.title; $('heroTitle').textContent = cfg.title; document.title = `${cfg.title} · Builds`; }
  if (cfg.subtitle) $('heroSub').textContent = cfg.subtitle;
  if (cfg.generated_at) $('footGen').textContent = `Built ${cfg.generated_at.slice(0, 10)}.`;
  try { episodes = await loadEpisodeIndex(); }
  catch (e) {
    const el = $('empty'); el.hidden = false;
    el.innerHTML = `<h2>No episodes found</h2><p class="muted">${esc(e.message)}</p>
      <p>Build the site with <code>python3 scripts/export_site.py runs/demo/*.jsonl --out site</code> and serve it over http,
      e.g. <code>python3 -m http.server -d site 8000</code>. Opening the files directly from disk does not work.</p>`;
    return;
  }
  if (!episodes.length) { const el = $('empty'); el.hidden = false; el.innerHTML = '<h2>No episodes yet</h2><p class="muted">The episode list is empty.</p>'; }
  const tasks = [...new Set(episodes.map(e => e.task).filter(Boolean))].sort();
  $('fTask').innerHTML += tasks.map(t => `<option value="${esc(t)}">${esc(taskLabel(t))}</option>`).join('');
  const models = [...new Set(episodes.flatMap(modelsOf))].sort();
  $('fModel').innerHTML += models.map(m => `<option value="${esc(m)}">${esc(shortModel(m))}</option>`).join('')
    + (episodes.some(e => !modelsOf(e).length) ? '<option value="__none">not recorded</option>' : '');
  const q = new URLSearchParams(location.search);
  for (const [id, k] of [['fTask', 'task'], ['fModel', 'model'], ['fSort', 'sort']]) {
    const v = q.get(k);
    if (v && [...$(id).options].some(o => o.value === v)) $(id).value = v;
  }
  for (const id of ['fTask', 'fModel', 'fSort']) $(id).addEventListener('change', render);
  render();
}
main();
