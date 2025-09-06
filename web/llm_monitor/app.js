(function(){
  const API_BASE = 'http://127.0.0.1:8765/api/llm';

  const qs = (sel) => document.querySelector(sel);
  const ce = (tag, cls) => { const el = document.createElement(tag); if (cls) el.className = cls; return el; };

  const state = {
    items: [],
    selectedId: null,
    timer: null,
    tiles: [], // recent ids for dynamic strip
    follow: true,
    tilesSet: new Set(),
  };

  async function fetchSessions() {
    const traceId = qs('#traceIdInput').value.trim();
    const params = new URLSearchParams();
    if (traceId) params.set('trace_id', traceId);
    const url = `${API_BASE}/sessions?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load sessions: ${res.status}`);
    const data = await res.json();
    state.items = data.items || [];
    renderSessions();
  }

  async function fetchDetail(id) {
    const res = await fetch(`${API_BASE}/sessions/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(`Failed to load detail: ${res.status}`);
    const data = await res.json();
    renderDetail(id, data);
  }

  function shortTS(s) { // 2025-09-05T16:50:25.353750 -> 2025-09-05T16:50:25
    if (!s) return '';
    const i = s.indexOf('.');
    return i > 0 ? s.slice(0, i) : s;
  }

  function prettyPrintJSON(value) {
    try {
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') {
        const trimmed = value.trim();
        if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
          try {
            const parsed = JSON.parse(trimmed);
            return JSON.stringify(parsed, null, 2);
          } catch (_) {
            // fall through
          }
        }
        return value;
      }
      if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
      }
      return String(value);
    } catch (e) {
      try {
        return JSON.stringify(value, null, 2);
      } catch (_) {
        return String(value ?? '');
      }
    }
  }

  function renderSessions() {
    const wrap = qs('#sessions');
    wrap.innerHTML = '';
    state.items.forEach((it) => {
      const row = ce('div', `session session--${it.status || 'pending'}`);
      row.addEventListener('click', () => {
        state.selectedId = it.trace_id;
        fetchDetail(it.trace_id).catch(console.error);
      });
      const tr = ce('div', 'session__trace');
      tr.textContent = it.trace_id || '';
      const info = ce('div', 'session__info');
      const t = ce('div', 'session__time'); t.textContent = shortTS(it.started_at || '');
      const model = ce('div', 'session__model'); model.textContent = it.model_name || '';
      const tokens = ce('div', 'session__tokens'); tokens.textContent = (it.total_tokens ?? '') + '';
      info.appendChild(t); info.appendChild(model); info.appendChild(tokens);
      row.appendChild(tr); row.appendChild(info);
      wrap.appendChild(row);
    });
  }

  async function updateStats() {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (!res.ok) return;
      const data = await res.json();
      const total = data && typeof data.total === 'number' ? data.total : 0;
      const tokens = data && typeof data.total_tokens === 'number' ? data.total_tokens : 0;
      const elTotal = qs('#statTotal');
      const elTokens = qs('#statTokens');
      if (elTotal) elTotal.textContent = String(total);
      if (elTokens) elTokens.textContent = String(tokens);
    } catch (_) {}
  }

  function renderDetail(id, data) {
    qs('#detailEmpty').classList.add('hidden');
    qs('#detail').classList.remove('hidden');
    qs('#traceId').textContent = id;

    const c = data.content || {};
    const m = data.metadata || {};

    qs('#detailMeta').textContent = `${m.status || ''} · ${m.provider || ''} · ${m.model_name || ''}`;
    qs('#requestJson').textContent = prettyPrintJSON(c.request || {});
    qs('#responseJson').textContent = prettyPrintJSON(c.response || {});
    qs('#timingJson').textContent = prettyPrintJSON(c.timing || {});
    qs('#metadataJson').textContent = prettyPrintJSON(m || {});
    qs('#contentJson').textContent = prettyPrintJSON(c || {});
  }

  function renderDynamicStrip() {
    const strip = qs('#dynamicStrip');
    strip.innerHTML = '';
    state.tiles.slice(-200).forEach((t) => {
      const el = ce('div', `tile tile--${t.status || 'pending'}`);
      el.title = t.trace_id;
      const idEl = ce('div', 'tile__id'); idEl.textContent = t.trace_id;
      const metaEl = ce('div', 'tile__meta'); metaEl.textContent = `${t.model_name || ''} · ${t.status || ''}`;
      el.appendChild(idEl); el.appendChild(metaEl);
      el.addEventListener('click', () => {
        state.selectedId = t.trace_id;
        fetchDetail(t.trace_id).catch(console.error);
      });
      strip.appendChild(el);
    });
    // auto scroll to end
    strip.scrollLeft = strip.scrollWidth;
  }

  async function addTileById(id, silent = false) {
    // fetch one item meta via sessions?trace_id=...
    try {
      const url = `${API_BASE}/sessions?` + new URLSearchParams({ trace_id: id }).toString();
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      const item = (data.items || [])[0];
      if (!item) return;
      state.tilesSet.add(item.trace_id);
      const idx = state.tiles.findIndex((t) => t.trace_id === item.trace_id);
      const isNew = idx < 0;
      if (isNew) {
        state.tiles.push(item);
      } else {
        state.tiles[idx] = { ...state.tiles[idx], ...item };
      }
      renderDynamicStrip();
      if (state.follow && !silent && isNew) {
        state.selectedId = item.trace_id;
        fetchDetail(item.trace_id).catch(console.error);
      }
    } catch (e) {
      // ignore
    }
  }

  async function refreshTiles() {
    try {
      const ids = state.tiles.slice(-100).map(t => t.trace_id);
      for (const id of ids) {
        await addTileById(id, true);
      }
      if (state.follow && state.selectedId) {
        await fetchDetail(state.selectedId).catch(() => {});
      }
    } catch (_) {}
  }

  function bindUI() {
    qs('#reloadBtn').addEventListener('click', () => { fetchSessions().catch(console.error); updateStats().catch?.(()=>{}); });
    const leftReload = qs('#leftReloadBtn');
    if (leftReload) leftReload.addEventListener('click', () => { fetchSessions().catch(console.error); updateStats().catch?.(()=>{}); });
    qs('#traceIdInput').addEventListener('keyup', (e) => {
      if (e.key === 'Enter') { fetchSessions().catch(console.error); updateStats().catch?.(()=>{}); }
    });

    const followToggle = qs('#followToggle');
    state.follow = !!followToggle.checked;
    followToggle.addEventListener('change', () => { state.follow = !!followToggle.checked; });

    // Map vertical wheel to horizontal scroll inside dynamic strip
    const strip = qs('#dynamicStrip');
    if (strip) {
      strip.addEventListener('wheel', (e) => {
        if (e.deltaY !== 0) {
          e.preventDefault();
          strip.scrollLeft += e.deltaY;
        }
      }, { passive: false });
    }
  }

  async function init() {
    bindUI();
    await fetchSessions();
    await updateStats();
    // Subscribe SSE for realtime updates
    try {
      const sse = new EventSource(`${API_BASE}/stream`);
      sse.onmessage = async (ev) => {
        try {
          const data = JSON.parse(ev.data || '{}');
          if (data && data.type === 'update') {
            try {
              const d = await fetch(`${API_BASE}/sessions`).then(r => r.json());
              const newest = (d.items || [])[0];
              if (newest && newest.trace_id) await addTileById(String(newest.trace_id));
            } catch (_) {}
            await refreshTiles();
            await updateStats();
          } else if (data && data.type === 'new' && data.id) {
            await addTileById(String(data.id));
            await updateStats();
          }
        } catch (e) {
          // ignore
        }
      };
      sse.onerror = () => {
        // Browser will try to reconnect automatically; optional: backoff UI
      };
    } catch (e) {
      // SSE not available; fallback to polling only
    }
  }

  window.LLMMonitorApp = { init };
})();


