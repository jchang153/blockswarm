// Small shared DOM helpers: escaping, formatting, theme toggle.
export const $ = id => document.getElementById(id);
export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function fmtDuration(s) {
  if (typeof s !== 'number' || !isFinite(s)) return '–';
  s = Math.max(0, s);
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return m ? `${m}:${String(r).padStart(2, '0')}` : `${s.toFixed(s < 10 ? 1 : 0)} s`;
}
export const fmtClock = s => `${Math.floor(Math.max(0, s) / 60)}:${(Math.max(0, s) % 60).toFixed(1).padStart(4, '0')}`;
export const fmtScore = v => (typeof v === 'number' ? v.toFixed(2) : '–');
/** Score band 0–4 for badge colours. */
export const scoreBand = v => (typeof v !== 'number' ? 0 : v >= 0.9 ? 4 : v >= 0.65 ? 3 : v >= 0.35 ? 2 : v > 0 ? 1 : 0);
export const SCORE_COLORS = ['#8a8f96', '#c5582a', '#c08a12', '#6a9a1e', '#2f8f3a'];
export const agentLabel = a => String(a).replace(/^agent_(\d+)$/, 'Agent $1');
/** "anthropic/claude-x-y" → "claude-x-y" */
export const shortModel = m => (m ? String(m).split('/').pop() : '');
export const REASONS = { all_done: 'all voted done', time_limit: 'time limit', action_limit: 'action limit', aborted: 'aborted' };
export const reasonLabel = r => (r ? REASONS[r] || r : 'incomplete');

const reducedMotionQ = window.matchMedia('(prefers-reduced-motion: reduce)');
export const prefersReducedMotion = () => reducedMotionQ.matches;

// ---------- theme: auto (system) / light / dark, remembered per browser ----------
function storedTheme() { try { return localStorage.getItem('bs-theme'); } catch { return null; } }
function storeTheme(v) { try { v ? localStorage.setItem('bs-theme', v) : localStorage.removeItem('bs-theme'); } catch { /* storage blocked */ } }
const darkQ = window.matchMedia('(prefers-color-scheme: dark)');
export const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === 'dark' : darkQ.matches;
};
const themeListeners = [];
export const onThemeChange = fn => themeListeners.push(fn);
function applyTheme(v) {
  if (v === 'light' || v === 'dark') document.documentElement.dataset.theme = v;
  else delete document.documentElement.dataset.theme;
  const b = document.getElementById('themeBtn');
  if (b) {
    b.textContent = v === 'dark' ? '☾' : v === 'light' ? '☀' : '◐';
    b.title = `Theme: ${v || 'system'} (click to change)`;
    b.setAttribute('aria-label', b.title);
  }
  themeListeners.forEach(fn => fn());
}
export function initTheme() {
  applyTheme(storedTheme());
  darkQ.addEventListener('change', () => themeListeners.forEach(fn => fn()));
  const b = document.getElementById('themeBtn');
  if (b) b.addEventListener('click', () => {
    const cur = storedTheme();
    const next = cur === null ? (darkQ.matches ? 'light' : 'dark') : cur === 'dark' ? 'light' : null;
    storeTheme(next); applyTheme(next);
  });
}

let toastTimer = 0;
export function toast(msg) {
  let el = document.querySelector('.toast');
  if (!el) { el = document.createElement('div'); el.className = 'toast'; el.setAttribute('role', 'status'); document.body.appendChild(el); }
  el.textContent = msg; el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 1800);
}
