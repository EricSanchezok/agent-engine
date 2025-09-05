(function(){
  const API_BASE = 'http://127.0.0.1:8765/api/llm';

  const qs = (sel) => document.querySelector(sel);
  const ce = (tag, cls) => { const el = document.createElement(tag); if (cls) el.className = cls; return el; };

  const state = {
    items: [],
    selectedId: null,
    timer: null,
  };

  async function fetchSessions() {
    const status = qs('#statusSelect').value.trim();
    const model = qs('#modelInput').value.trim();
    const provider = qs('#providerInput').value.trim();
    const q = qs('#searchInput').value.trim();
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (model) params.set('model', model);
    if (provider) params.set('provider', provider);
    if (q) params.set('q', q);
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

  function renderSessions() {
    const wrap = qs('#sessions');
    wrap.innerHTML = '';
    state.items.forEach((it) => {
      const row = ce('div', 'session');
      row.addEventListener('click', () => {
        state.selectedId = it.trace_id;
        fetchDetail(it.trace_id).catch(console.error);
      });
      const t = ce('div', 'session__time');
      t.textContent = it.started_at || '';
      const title = ce('div', 'session__title');
      title.innerHTML = `<span class="mono">${it.trace_id}</span> · ${it.model_name || ''}`;
      const status = ce('div', `session__status session__status--${it.status || 'pending'}`);
      status.textContent = it.status || '';
      row.appendChild(t); row.appendChild(title); row.appendChild(status);
      wrap.appendChild(row);
    });
  }

  function renderDetail(id, data) {
    qs('#detailEmpty').classList.add('hidden');
    qs('#detail').classList.remove('hidden');
    qs('#traceId').textContent = id;

    const c = data.content || {};
    const m = data.metadata || {};

    qs('#detailMeta').textContent = `${m.status || ''} · ${m.provider || ''} · ${m.model_name || ''}`;
    qs('#requestJson').textContent = JSON.stringify(c.request || {}, null, 2);
    qs('#responseJson').textContent = JSON.stringify(c.response || {}, null, 2);
    qs('#timingJson').textContent = JSON.stringify(c.timing || {}, null, 2);
    qs('#metadataJson').textContent = JSON.stringify(m || {}, null, 2);
    qs('#contentJson').textContent = JSON.stringify(c || {}, null, 2);
  }

  function bindUI() {
    qs('#reloadBtn').addEventListener('click', () => fetchSessions().catch(console.error));
    qs('#statusSelect').addEventListener('change', () => fetchSessions().catch(console.error));
    qs('#modelInput').addEventListener('change', () => fetchSessions().catch(console.error));
    qs('#providerInput').addEventListener('change', () => fetchSessions().catch(console.error));
    qs('#searchInput').addEventListener('keyup', (e) => {
      if (e.key === 'Enter') fetchSessions().catch(console.error);
    });
    const toggle = qs('#pollToggle');
    const intervalSel = qs('#pollInterval');
    function startPoll() {
      stopPoll();
      state.timer = setInterval(() => fetchSessions().catch(console.error), parseInt(intervalSel.value, 10));
    }
    function stopPoll() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }
    toggle.addEventListener('change', () => { toggle.checked ? startPoll() : stopPoll(); });
    intervalSel.addEventListener('change', () => { if (toggle.checked) startPoll(); });
    // start by default
    if (toggle.checked) startPoll();
  }

  async function init() {
    bindUI();
    await fetchSessions();
    // Subscribe SSE for realtime updates
    try {
      const sse = new EventSource(`${API_BASE}/stream`);
      sse.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data || '{}');
          if (data && data.type === 'update') {
            fetchSessions().catch(console.error);
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


