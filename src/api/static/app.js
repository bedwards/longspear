/* Longspear Web UI — Application Logic */

const API_BASE = '';  // Same origin

const PERSONAS = {
    heather_cox_richardson: {
        name: 'Heather Cox Richardson',
        short: 'HCR',
        gradient: 'persona-a-gradient',
    },
    nate_b_jones: {
        name: 'Nate B Jones',
        short: 'NBJ',
        gradient: 'persona-b-gradient',
    },
};

// ── State ────────────────────────────────────
let currentMode = 'debate';
let isStreaming = false;

// ── DOM helpers ──────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

// ── Navigation ───────────────────────────────
$$('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (isStreaming) return;
        const mode = btn.dataset.mode;
        currentMode = mode;
        $$('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        hide($('#debate-view'));
        hide($('#chat-view'));
        hide($('#monitor-view'));
        show($(`#${mode}-view`));
        if (mode === 'monitor') refreshMonitor();
    });
});

// ── Health check ─────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        const dot = $('#status-dot .dot');
        const text = $('#status-dot .status-text');
        dot.className = 'dot ' + data.status;
        text.textContent = data.status === 'healthy' ? 'All systems go' : data.status;
    } catch {
        const dot = $('#status-dot .dot');
        const text = $('#status-dot .status-text');
        dot.className = 'dot error';
        text.textContent = 'Offline';
    }
}
setInterval(checkHealth, 30000);
checkHealth();

// ── Message rendering ────────────────────────
function createMessage(personaSlug, isUser = false) {
    const div = document.createElement('div');
    div.className = 'message' + (isUser ? ' user-message' : '');

    if (!isUser && personaSlug) {
        const p = PERSONAS[personaSlug] || { name: personaSlug, short: '?', gradient: 'persona-a-gradient' };
        div.innerHTML = `
            <div class="message-header">
                <div class="message-avatar ${p.gradient}">${p.short}</div>
                <div class="message-name">${p.name}</div>
            </div>
            <div class="message-body streaming"><span class="cursor"></span></div>
        `;
    } else {
        div.innerHTML = `<div class="message-body">${''}</div>`;
    }

    return div;
}

function appendToken(msgEl, token) {
    const body = msgEl.querySelector('.message-body');
    const cursor = body.querySelector('.cursor');
    if (cursor) {
        body.insertBefore(document.createTextNode(token), cursor);
    } else {
        body.appendChild(document.createTextNode(token));
    }
}

function finishMessage(msgEl) {
    const body = msgEl.querySelector('.message-body');
    body.classList.remove('streaming');
    const cursor = body.querySelector('.cursor');
    if (cursor) cursor.remove();
}

function showSources(container, sources) {
    if (!sources || sources.length === 0) return;
    const div = document.createElement('div');
    div.className = 'message-sources';
    sources.slice(0, 5).forEach(s => {
        const tag = document.createElement('span');
        tag.className = 'source-tag';
        tag.textContent = s.title ? `${s.title}` : s.url || 'source';
        if (s.date) tag.textContent += ` (${s.date})`;
        div.appendChild(tag);
    });
    container.appendChild(div);
}

// ── Debate mode ──────────────────────────────
$('#debate-send').addEventListener('click', startDebate);
$('#debate-question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startDebate(); }
});

async function startDebate() {
    const input = $('#debate-question');
    const question = input.value.trim();
    if (!question || isStreaming) return;

    isStreaming = true;
    $('#debate-send').disabled = true;
    input.value = '';

    const container = $('#debate-messages');

    // User question
    const userMsg = createMessage(null, true);
    userMsg.querySelector('.message-body').textContent = question;
    container.appendChild(userMsg);

    let currentMsg = null;

    try {
        const res = await fetch(`${API_BASE}/debate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.type === 'turn_start') {
                    currentMsg = createMessage(data.persona);
                    container.appendChild(currentMsg);
                    currentMsg.scrollIntoView({ behavior: 'smooth', block: 'end' });
                } else if (data.type === 'token' && currentMsg) {
                    appendToken(currentMsg, data.content);
                } else if (data.type === 'turn_end' && currentMsg) {
                    finishMessage(currentMsg);
                    currentMsg = null;
                } else if (data.type === 'error') {
                    const errDiv = document.createElement('div');
                    errDiv.className = 'message';
                    errDiv.innerHTML = `<div class="message-body" style="color: #ef4444;">Error: ${data.message}</div>`;
                    container.appendChild(errDiv);
                }
            }
        }
    } catch (err) {
        const errDiv = document.createElement('div');
        errDiv.className = 'message';
        errDiv.innerHTML = `<div class="message-body" style="color: #ef4444;">Connection error: ${err.message}</div>`;
        container.appendChild(errDiv);
    }

    isStreaming = false;
    $('#debate-send').disabled = false;
}

// ── Chat mode ────────────────────────────────
$('#chat-send').addEventListener('click', startChat);
$('#chat-question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startChat(); }
});

async function startChat() {
    const input = $('#chat-question');
    const question = input.value.trim();
    const persona = $('#chat-persona').value;
    if (!question || isStreaming) return;

    isStreaming = true;
    $('#chat-send').disabled = true;
    input.value = '';

    const container = $('#chat-messages');

    // User message
    const userMsg = createMessage(null, true);
    userMsg.querySelector('.message-body').textContent = question;
    container.appendChild(userMsg);

    // Persona response
    const respMsg = createMessage(persona);
    container.appendChild(respMsg);
    respMsg.scrollIntoView({ behavior: 'smooth', block: 'end' });

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, persona, stream: true }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.type === 'meta') {
                    // Could display source info
                } else if (data.type === 'token') {
                    appendToken(respMsg, data.content);
                } else if (data.type === 'done') {
                    finishMessage(respMsg);
                } else if (data.type === 'error') {
                    finishMessage(respMsg);
                    respMsg.querySelector('.message-body').textContent = `Error: ${data.message}`;
                }
            }
        }
    } catch (err) {
        finishMessage(respMsg);
        respMsg.querySelector('.message-body').textContent = `Connection error: ${err.message}`;
    }

    isStreaming = false;
    $('#chat-send').disabled = false;
}

// ── Monitor mode ─────────────────────────────
$('#monitor-refresh').addEventListener('click', refreshMonitor);

async function refreshMonitor() {
    // Health
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        let html = '';
        for (const [svc, status] of Object.entries(data.services)) {
            const cls = status === 'healthy' ? 'healthy' : 'unhealthy';
            html += `<div><span class="key">${svc}:</span> <span class="val ${cls}">${status}</span></div>`;
        }
        $('#monitor-health').innerHTML = html;
    } catch {
        $('#monitor-health').innerHTML = '<span class="unhealthy">Unable to connect</span>';
    }

    // Stats
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();
        let html = '';
        for (const [backend, models] of Object.entries(data.counts)) {
            html += `<div style="margin-bottom: 8px;"><strong>${backend}</strong></div>`;
            for (const [model, count] of Object.entries(models)) {
                html += `<div><span class="key">&nbsp;&nbsp;${model}:</span> <span class="val">${count.toLocaleString()} docs</span></div>`;
            }
        }
        $('#monitor-stats').innerHTML = html || '<span class="key">No data</span>';
    } catch {
        $('#monitor-stats').innerHTML = '<span class="unhealthy">Unable to load</span>';
    }

    // Personas
    try {
        const res = await fetch(`${API_BASE}/personas`);
        const data = await res.json();
        let html = '';
        for (const p of data) {
            html += `<div><span class="val">${p.name}</span> <span class="key">(${p.slug})</span></div>`;
        }
        $('#monitor-personas').innerHTML = html;
    } catch {
        $('#monitor-personas').innerHTML = '<span class="unhealthy">Unable to load</span>';
    }
}
