(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // Basic state
  const state = {
    sessionId: crypto.randomUUID(),
    ws: null,
    connected: false,
    patientId: null,
    updating: false,
  };

  const els = {
    newPatientBtn: $('#newPatientBtn'),
    connectBtn: $('#connectBtn'),
    updateBtn: $('#updateBtn'),
    statusText: $('#statusText'),
    messages: $('#messages'),
    inputBox: $('#inputBox'),
    sendBtn: $('#sendBtn'),
    modal: $('#modal'),
    modalClose: $('#modalClose'),
    patientSelect: $('#patientSelect'),
    patientConfirm: $('#patientConfirm'),
  };

  function setStatus(text) { els.statusText.textContent = text; }
  function setComposerEnabled(enabled) {
    els.inputBox.disabled = !enabled;
    els.sendBtn.disabled = !enabled;
  }

  function wsConnect() {
    const url = new URL(window.location.href);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = '/ws';
    url.searchParams.set('session_id', state.sessionId);
    const ws = new WebSocket(url.href);
    state.ws = ws;
    ws.onopen = () => setStatus('WebSocket connected');
    ws.onclose = () => setStatus('WebSocket closed');
    ws.onerror = () => setStatus('WebSocket error');
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        handleServerEvent(msg);
      } catch {}
    };
  }

  function handleServerEvent(msg) {
    const t = msg.type;
    if (t === 'block_create') {
      renderBlockCreate(msg.block);
    } else if (t === 'block_update') {
      renderBlockUpdate(msg.block_id, msg.llm_response);
    } else if (t === 'block_complete') {
      renderBlockComplete(msg.block_id, msg.result);
    } else if (t === 'error') {
      console.warn('Server error:', msg.message);
    }
  }

  function renderBlockCreate(block) {
    const container = document.createElement('div');
    container.className = 'block';
    container.dataset.blockId = block.id;
    const header = document.createElement('div');
    header.className = 'block-header';
    const title = document.createElement('div');
    title.textContent = block.header || (block.block_type === 'user' ? 'User' : 'Agent');
    const toggle = document.createElement('button');
    toggle.textContent = '▼';
    toggle.style.background = 'transparent';
    toggle.onclick = () => {
      body.style.display = body.style.display === 'none' ? '' : 'none';
    };
    header.appendChild(title);
    header.appendChild(toggle);
    const body = document.createElement('div');
    body.className = 'block-body';
    const pre = document.createElement('pre');
    pre.textContent = '';
    body.appendChild(pre);
    const result = document.createElement('div');
    result.className = 'block-result';
    result.textContent = '';
    container.appendChild(header);
    container.appendChild(body);
    container.appendChild(result);
    els.messages.appendChild(container);
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  function renderBlockUpdate(blockId, llmResponse) {
    const node = els.messages.querySelector(`[data-block-id="${blockId}"]`);
    if (!node) return;
    const pre = node.querySelector('.block-body pre');
    pre.textContent = typeof llmResponse === 'string' ? llmResponse : JSON.stringify(llmResponse, null, 2);
  }

  function renderBlockComplete(blockId, result) {
    const node = els.messages.querySelector(`[data-block-id="${blockId}"]`);
    if (!node) return;
    const resultNode = node.querySelector('.block-result');
    resultNode.textContent = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
  }

  async function modalOpenPatients() {
    els.modal.classList.remove('hidden');
    els.patientSelect.innerHTML = '';
    try {
      const r = await fetch('/api/patients');
      const data = await r.json();
      const list = Array.isArray(data.patients) ? data.patients : [];
      for (const pid of list) {
        const opt = document.createElement('option');
        opt.value = pid; opt.textContent = pid;
        els.patientSelect.appendChild(opt);
      }
      if (list.length > 0) {
        els.patientSelect.value = list[0];
      }
    } catch (e) {
      console.error(e);
    }
  }

  async function connectSelectedPatient() {
    const patientId = els.patientSelect.value;
    if (!patientId) return;
    try {
      const r = await fetch('/api/connect_patient', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId, patient_id: patientId })
      });
      const data = await r.json();
      if (data.ok) {
        state.patientId = patientId;
        setComposerEnabled(true);
        els.updateBtn.disabled = false;
        setStatus(`Connected to ${patientId}`);
      }
    } catch (e) {
      console.error(e);
    } finally {
      els.modal.classList.add('hidden');
    }
  }

  async function triggerIngestionUpdate() {
    if (state.updating) return;
    state.updating = true;
    els.updateBtn.disabled = true;
    try {
      const r = await fetch('/api/ingestion/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId })
      });
      const data = await r.json();
      if (data.ok) {
        setStatus(`Ingestion wrote ${data.written_count} events`);
      }
    } catch (e) {
      console.error(e);
    } finally {
      state.updating = false;
      els.updateBtn.disabled = false;
    }
  }

  function sendUserMessage() {
    const text = els.inputBox.value.trim();
    if (!text || !state.ws) return;
    state.ws.send(JSON.stringify({ type: 'user_message', text }));
    // Also render as a user block locally
    const blockId = crypto.randomUUID();
    renderBlockCreate({ id: blockId, header: 'User', block_type: 'user' });
    renderBlockComplete(blockId, text);
    els.inputBox.value = '';
  }

  // Events
  els.connectBtn.addEventListener('click', modalOpenPatients);
  els.modalClose.addEventListener('click', () => els.modal.classList.add('hidden'));
  els.patientConfirm.addEventListener('click', connectSelectedPatient);
  els.updateBtn.addEventListener('click', triggerIngestionUpdate);
  els.sendBtn.addEventListener('click', sendUserMessage);
  els.inputBox.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendUserMessage(); });

  // Init
  setComposerEnabled(false);
  wsConnect();
})();


