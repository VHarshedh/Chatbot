let currentSessionId = null;
let benchmarkPrompts = [];

// DOM Elements
const chatThread = document.getElementById("chat-thread");
const welcomeHero = document.getElementById("welcome-hero");
const messagesContainer = document.getElementById("messages-container");
const chatForm = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt-input");
const sessionList = document.getElementById("session-list");
const benchmarkList = document.getElementById("benchmark-list");
const newSessionBtn = document.getElementById("new-session-btn");
const modelNameDisplay = document.getElementById("model-name-display");
const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
const sidebar = document.getElementById("sidebar");

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  setupTextareaAutoGrow();
  await loadModelInfo();
  await loadBenchmarkPrompts();
  await loadSessions();
});

// Auto-grow textarea
function setupTextareaAutoGrow() {
  promptInput.addEventListener("input", () => {
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 180) + "px";
  });

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });
}

// Fetch model metadata
async function loadModelInfo() {
  try {
    const res = await fetch("/api/model_info");
    if (res.ok) {
      const data = await res.json();
      modelNameDisplay.textContent = data.model_name;
    }
  } catch (err) {
    console.error("Error loading model info:", err);
  }
}

// Fetch and render benchmark prompts
async function loadBenchmarkPrompts() {
  try {
    const res = await fetch("/api/benchmark_prompts");
    if (!res.ok) return;
    benchmarkPrompts = await res.json();

    benchmarkList.innerHTML = "";
    benchmarkPrompts.forEach((bp) => {
      const item = document.createElement("div");
      item.className = "benchmark-item";
      item.innerHTML = `
        <div class="benchmark-header">
          <span class="benchmark-domain">${escapeHtml(bp.domain)}</span>
        </div>
        <div class="benchmark-title">${escapeHtml(bp.title)}</div>
      `;
      item.addEventListener("click", () => selectBenchmarkPrompt(bp));
      benchmarkList.appendChild(item);
    });

    // Wire up the welcome hero cards as well
    document.querySelectorAll(".domain-card").forEach((card) => {
      card.addEventListener("click", () => {
        const id = card.getAttribute("data-prompt-id");
        const found = benchmarkPrompts.find((p) => p.id === id);
        if (found) selectBenchmarkPrompt(found);
      });
    });
  } catch (err) {
    console.error("Error loading benchmark prompts:", err);
  }
}

function selectBenchmarkPrompt(bp) {
  promptInput.value = bp.prompt;
  promptInput.focus();
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 180) + "px";
}

// Session Management
async function loadSessions() {
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const sessions = await res.json();

    sessionList.innerHTML = "";
    if (sessions.length === 0) {
      sessionList.innerHTML = `<div style="font-size:0.75rem; color:var(--text-muted); padding:8px 0;">No past sessions</div>`;
      return;
    }

    sessions.forEach((s) => {
      const el = document.createElement("div");
      el.className = `session-item ${currentSessionId === s.id ? "active" : ""}`;
      el.innerHTML = `
        <span class="session-title-text" title="${escapeHtml(s.title)}">${escapeHtml(s.title)}</span>
        <button class="session-delete-btn" title="Delete session">&times;</button>
      `;
      el.addEventListener("click", (e) => {
        if (e.target.classList.contains("session-delete-btn")) return;
        switchSession(s.id);
      });
      el.querySelector(".session-delete-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(s.id);
      });
      sessionList.appendChild(el);
    });
  } catch (err) {
    console.error("Error loading sessions:", err);
  }
}

async function switchSession(sessionId) {
  currentSessionId = sessionId;
  welcomeHero.style.display = "none";
  chatThread.innerHTML = "";

  try {
    const res = await fetch(`/api/sessions/${sessionId}/messages`);
    if (!res.ok) return;
    const messages = await res.json();

    messages.forEach((msg) => {
      appendMessage(msg.role, msg.content, { timestamp: msg.timestamp });
    });
    scrollToBottom();
    await loadSessions();
  } catch (err) {
    console.error("Error switching session:", err);
  }
}

async function deleteSession(sessionId) {
  try {
    const res = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
    if (res.ok) {
      if (currentSessionId === sessionId) {
        startNewSession();
      } else {
        await loadSessions();
      }
    }
  } catch (err) {
    console.error("Error deleting session:", err);
  }
}

function startNewSession() {
  currentSessionId = null;
  chatThread.innerHTML = "";
  welcomeHero.style.display = "block";
  loadSessions();
  promptInput.focus();
}

newSessionBtn.addEventListener("click", startNewSession);

// Chat Submission
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;

  // Clear input
  promptInput.value = "";
  promptInput.style.height = "auto";
  welcomeHero.style.display = "none";

  // Append user message
  appendMessage("user", text);
  scrollToBottom();

  // Show typing indicator
  const indicator = showTypingIndicator();
  scrollToBottom();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId,
        mode: "single_shot"
      })
    });

    indicator.remove();

    if (!response.ok) {
      const err = await response.json();
      appendMessage("assistant", `**Error**: ${err.detail || "Server error occurred."}`);
      return;
    }

    const data = await response.json();
    currentSessionId = data.session_id;

    if (data.status === "success") {
      appendMessage("assistant", data.response, {
        latency: data.latency_seconds,
        model: data.model
      });
    } else {
      appendMessage("assistant", `**Error**: ${data.error || "Generation failed"}`);
    }

    await loadSessions();
    scrollToBottom();
  } catch (err) {
    indicator.remove();
    appendMessage("assistant", `**Network Error**: Could not reach backend API (${err.message})`);
    scrollToBottom();
  }
});

// Message Rendering
function appendMessage(role, content, meta = {}) {
  const item = document.createElement("div");
  item.className = `message-item ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = renderMarkdown(content);

  item.appendChild(bubble);

  if (role === "assistant" && meta.latency !== undefined) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "message-meta";
    metaDiv.innerHTML = `
      <span class="latency-pill">⚡ ${meta.latency}s</span>
      <span>${escapeHtml(meta.model || "Gemma4")}</span>
      <span>Pure Single-Shot</span>
    `;
    item.appendChild(metaDiv);
  }

  chatThread.appendChild(item);
}

function showTypingIndicator() {
  const item = document.createElement("div");
  item.className = "message-item assistant";
  item.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatThread.appendChild(item);
  return item;
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Lightweight Markdown Renderer
function renderMarkdown(raw) {
  if (!raw) return "";
  let html = raw;

  // Code blocks: ```lang ... ```
  html = html.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
  });

  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, (match, code) => `<code>${escapeHtml(code)}</code>`);

  // Bold: **text**
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Italics: *text*
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Paragraphs
  const paragraphs = html.split(/\n{2,}/);
  return paragraphs
    .map((p) => {
      p = p.trim();
      if (p.startsWith("<pre>")) return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
}

function escapeHtml(text) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// Mobile sidebar toggle
if (toggleSidebarBtn) {
  toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });
}
