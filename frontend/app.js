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
const evalHistoryList = document.getElementById("eval-history-list");
const runFlagshipEvalBtn = document.getElementById("run-flagship-eval-btn");
const newSessionBtn = document.getElementById("new-session-btn");
const modelNameDisplay = document.getElementById("model-name-display");
const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
const sidebar = document.getElementById("sidebar");

// Category Display Metadata
const CATEGORY_META = {
  syntax_and_imports: {
    label: "1. Syntax & Core Imports (ASE, NumPy, Matplotlib)",
    max: 15
  },
  site_identification: {
    label: "2. Interfacial Site Identification (M_sur, CUS, O_adb, O_br)",
    max: 20
  },
  density_and_rdfs: {
    label: "3. Density Profiles & Pair RDFs (z-binning, g(r))",
    max: 20
  },
  kinetics_and_survival: {
    label: "4. Surface Kinetics & Survival Probability P(t)",
    max: 25
  },
  model_averaging_and_outputs: {
    label: "5. DP Averaging, 2×2 Summary Figure & Text Report",
    max: 20
  }
};

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  setupTextareaAutoGrow();
  setupEventListeners();
  await loadModelInfo();
  await loadBenchmarkPrompts();
  await loadEvaluationHistory();
  await loadSessions();
});

// Setup global event listeners
function setupEventListeners() {
  if (runFlagshipEvalBtn) {
    runFlagshipEvalBtn.addEventListener("click", handleRunFlagshipEvaluation);
  }
}

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
      if (modelNameDisplay) {
        modelNameDisplay.textContent = data.model_name;
      }
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

    if (benchmarkList) {
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
    }

    // Wire up welcome hero cards
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

// =============================================================================
// Zero-Shot Script Accuracy Evaluation
// =============================================================================

async function handleRunFlagshipEvaluation() {
  if (!runFlagshipEvalBtn) return;

  const originalBtnHtml = runFlagshipEvalBtn.innerHTML;
  runFlagshipEvalBtn.disabled = true;
  runFlagshipEvalBtn.innerHTML = `
    <span class="typing-dot" style="display:inline-block; margin-right:4px;"></span>
    Evaluating Gemma4...
  `;

  welcomeHero.style.display = "none";

  // Live progress bubble in the chat thread with live timer & streaming code
  const assistantItem = document.createElement("div");
  assistantItem.className = "message-item assistant";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  bubble.innerHTML = `
    <div class="stream-header-bar">
      <span class="live-timer-badge">⏳ Starting Flagship Evaluation... 0.0s</span>
      <span class="live-model-pill">batiai/gemma4-e2b:q4 • Zero-Shot Evaluation</span>
    </div>
    <details class="thinking-details" style="display:none;" open>
      <summary><em>🧠 Thinking Process (<span class="thinking-word-count">0</span> words)</em></summary>
      <div class="thinking-stream-content"></div>
    </details>
    <div class="content-stream-container"></div>
    <span class="streaming-cursor"></span>
  `;

  assistantItem.appendChild(bubble);
  chatThread.appendChild(assistantItem);
  scrollToBottom();

  const timerBadge = bubble.querySelector(".live-timer-badge");
  const thinkingDetails = bubble.querySelector(".thinking-details");
  const thinkingWordCount = bubble.querySelector(".thinking-word-count");
  const thinkingContent = bubble.querySelector(".thinking-stream-content");
  const contentContainer = bubble.querySelector(".content-stream-container");
  const cursor = bubble.querySelector(".streaming-cursor");

  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    timerBadge.textContent = `⏳ Live Benchmark Generation... ${elapsed}s`;
  }, 100);

  try {
    const response = await fetch("/api/evaluate_zero_shot/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });

    if (!response.ok) {
      clearInterval(timerInterval);
      if (cursor) cursor.remove();
      const err = await response.json().catch(() => ({ detail: "Evaluation streaming failed." }));
      contentContainer.innerHTML = `<span style="color:var(--color-danger)"><strong>Error:</strong> ${escapeHtml(err.detail || "Evaluation failed.")}</span>`;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let thinkingAccumulated = "";
    let contentAccumulated = "";
    let finalEvalData = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        const cleanedPart = part.trim();
        if (!cleanedPart.startsWith("data:")) continue;
        const jsonStr = cleanedPart.replace(/^data:\s*/, "").trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "session") {
            currentSessionId = event.session_id;
          } else if (event.type === "thinking") {
            thinkingDetails.style.display = "block";
            thinkingAccumulated += event.delta;
            thinkingContent.textContent = thinkingAccumulated;
            const words = thinkingAccumulated.split(/\s+/).filter(Boolean).length;
            thinkingWordCount.textContent = words;
            scrollToBottom();
          } else if (event.type === "content") {
            contentAccumulated += event.delta;
            const isCodeUnclosed = ((contentAccumulated.match(/```/g) || []).length % 2 === 1);
            if (cursor) cursor.style.display = isCodeUnclosed ? "none" : "inline-block";
            contentContainer.innerHTML = renderMarkdown(contentAccumulated, true);
            scrollToBottom();
          } else if (event.type === "eval_complete") {
            finalEvalData = event;
          } else if (event.type === "done") {
            clearInterval(timerInterval);
            if (cursor) cursor.remove();
            const elapsed = event.latency_seconds !== undefined ? event.latency_seconds.toFixed(1) : ((Date.now() - startTime) / 1000).toFixed(1);
            timerBadge.textContent = `⚡ Generated in ${elapsed}s • Scoring AST...`;
            if (thinkingAccumulated) {
              thinkingDetails.open = false;
            }
            contentContainer.innerHTML = renderMarkdown(contentAccumulated, false);
            scrollToBottom();
          }
        } catch (err) {
          console.warn("SSE parse error in flagship eval:", err);
        }
      }
    }

    clearInterval(timerInterval);
    if (cursor) cursor.remove();

    if (finalEvalData) {
      currentSessionId = finalEvalData.session_id;
      timerBadge.textContent = `⚡ Completed (${finalEvalData.score}/100 Grade ${finalEvalData.grade})`;
      renderEvaluationReport(finalEvalData);
      await loadEvaluationHistory();
      await loadSessions();
      scrollToBottom();
    }
  } catch (err) {
    clearInterval(timerInterval);
    if (cursor) cursor.remove();
    appendMessage("assistant", `**Network Error**: Failed to run evaluation (${escapeHtml(err.message)}).`);
    scrollToBottom();
  } finally {
    runFlagshipEvalBtn.disabled = false;
    runFlagshipEvalBtn.innerHTML = originalBtnHtml;
  }
}

// Fetch and render previous evaluations in sidebar
async function loadEvaluationHistory() {
  if (!evalHistoryList) return;

  try {
    const res = await fetch("/api/evaluations");
    if (!res.ok) return;
    const evals = await res.json();

    evalHistoryList.innerHTML = "";
    if (evals.length === 0) {
      evalHistoryList.innerHTML = `<div style="font-size:0.75rem; color:var(--text-muted); padding:6px 0;">No benchmark runs yet</div>`;
      return;
    }

    evals.forEach((ev) => {
      const item = document.createElement("div");
      item.className = "eval-history-item";

      const dateStr = ev.created_at ? formatShortDate(ev.created_at) : "Recent";
      const gradeClass = `grade-${ev.grade || "F"}`;

      item.innerHTML = `
        <div class="eval-history-left">
          <span class="eval-history-title">Eval #${ev.id} • ${ev.score}/100</span>
          <span class="eval-history-time">${dateStr}</span>
        </div>
        <span class="eval-history-badge ${gradeClass}">${ev.grade || "F"}</span>
      `;

      item.addEventListener("click", () => {
        welcomeHero.style.display = "none";
        renderEvaluationReport(ev);
        scrollToBottom();
      });

      evalHistoryList.appendChild(item);
    });
  } catch (err) {
    console.error("Error loading evaluation history:", err);
  }
}

// Render the 100-point accuracy report card
function renderEvaluationReport(ev) {
  const container = document.createElement("div");
  container.className = "message-item assistant";

  const card = document.createElement("div");
  card.className = "evaluation-report-card";

  const grade = ev.grade || "F";
  const score = ev.score !== undefined ? ev.score : 0;
  const gradeClass = `grade-${grade}`;
  const latency = ev.latency_seconds ? `${ev.latency_seconds.toFixed(1)}s` : "N/A";
  const modelName = ev.model || ev.model_name || "batiai/gemma4-e2b:q4";
  const report = ev.evaluation_report || {};
  const passedChecks = ev.passed_checks || report.passed_checks || [];
  const failedChecks = ev.failed_checks || report.failed_checks || [];
  const summaryText = ev.summary || report.summary || "Zero-shot script benchmark completed.";
  const catScores = ev.category_scores || {};
  const codeText = ev.generated_code || "";

  // 1. Header
  let html = `
    <div class="eval-card-header">
      <div class="score-badge-container">
        <div class="score-circle">
          <span class="score-num">${score}</span>
          <span class="score-denom">/ 100</span>
        </div>
        <div class="score-text-details">
          <h3>Interfacial Water Dynamics Generator</h3>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="grade-pill ${gradeClass}">Grade ${grade}</span>
            <span style="font-size:0.75rem; color:var(--text-muted);">Model: ${escapeHtml(modelName)}</span>
          </div>
        </div>
      </div>
      <div class="message-meta">
        <span class="latency-pill">⚡ ${latency}</span>
        <span>Zero-Shot Baseline</span>
      </div>
    </div>
  `;

  // 2. Summary
  html += `
    <div style="font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">
      ${escapeHtml(summaryText)}
    </div>
  `;

  // 3. Category Progress Bars
  html += `<div class="category-bars-grid">`;
  for (const [key, meta] of Object.entries(CATEGORY_META)) {
    const catData = catScores[key] || { score: 0, max: meta.max };
    const curScore = catData.score !== undefined ? catData.score : 0;
    const maxScore = catData.max || meta.max;
    const pct = catData.pct !== undefined ? catData.pct : Math.round((curScore / maxScore) * 100);

    html += `
      <div class="cat-row">
        <div class="cat-meta">
          <span>${meta.label}</span>
          <span><strong>${curScore}</strong> / ${maxScore} pts (${pct}%)</span>
        </div>
        <div class="cat-bar-bg">
          <div class="cat-bar-fill" style="width: ${Math.min(pct, 100)}%;"></div>
        </div>
      </div>
    `;
  }
  html += `</div>`;

  // 4. Diagnostic Checklist Accordion
  html += `
    <details class="eval-details" open>
      <summary>
        <span>Diagnostic Evaluation Checklist (${passedChecks.length} Passed, ${failedChecks.length} Failed)</span>
      </summary>
      <div class="eval-details-content eval-checks-accordion">
  `;

  passedChecks.forEach((chk) => {
    html += `
      <div class="check-item">
        <span class="check-icon passed">✓</span>
        <span style="color:var(--text-primary);">${escapeHtml(chk)}</span>
      </div>
    `;
  });

  failedChecks.forEach((chk) => {
    html += `
      <div class="check-item">
        <span class="check-icon failed">✗</span>
        <span style="color:var(--text-secondary);">${escapeHtml(chk)}</span>
      </div>
    `;
  });

  html += `
      </div>
    </details>
  `;

  // 5. Extracted Python Script Accordion
  const codeLines = codeText ? codeText.split("\n").length : 0;
  html += `
    <details class="eval-details">
      <summary>
        <span>Extracted Python Script (${codeLines} lines)</span>
      </summary>
      <div class="code-header-bar">
        <span>Zero-Shot Generated Implementation</span>
        <button class="btn-copy-code" type="button">Copy Code</button>
      </div>
      <div class="eval-details-content" style="padding:0;">
        <pre style="margin:0; border:none; border-radius:0;"><code class="language-python">${escapeHtml(codeText || "# No code extracted from response")}</code></pre>
      </div>
    </details>
  `;

  card.innerHTML = html;

  // Copy button handler
  const copyBtn = card.querySelector(".btn-copy-code");
  if (copyBtn && codeText) {
    copyBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        await navigator.clipboard.writeText(codeText);
        copyBtn.textContent = "Copied!";
        setTimeout(() => { copyBtn.textContent = "Copy Code"; }, 2000);
      } catch (err) {
        copyBtn.textContent = "Error";
      }
    });
  }

  container.appendChild(card);
  chatThread.appendChild(container);
}

// Format short date helper
function formatShortDate(dateStr) {
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return dateStr;
  }
}

// =============================================================================
// Chat & Session Management
// =============================================================================

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

// Chat Submission with Real-Time Token Streaming
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;

  promptInput.value = "";
  promptInput.style.height = "auto";
  welcomeHero.style.display = "none";

  appendMessage("user", text);
  scrollToBottom();

  // Create streaming assistant message bubble
  const assistantItem = document.createElement("div");
  assistantItem.className = "message-item assistant";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  bubble.innerHTML = `
    <div class="stream-header-bar">
      <span class="live-timer-badge">⏳ Starting inference... 0.0s</span>
      <span class="live-model-pill">batiai/gemma4-e2b:q4 • Live Stream</span>
    </div>
    <details class="thinking-details" style="display:none;" open>
      <summary><em>🧠 Thinking Process (<span class="thinking-word-count">0</span> words)</em></summary>
      <div class="thinking-stream-content"></div>
    </details>
    <div class="content-stream-container"></div>
    <span class="streaming-cursor"></span>
  `;

  assistantItem.appendChild(bubble);
  chatThread.appendChild(assistantItem);
  scrollToBottom();

  const timerBadge = bubble.querySelector(".live-timer-badge");
  const thinkingDetails = bubble.querySelector(".thinking-details");
  const thinkingWordCount = bubble.querySelector(".thinking-word-count");
  const thinkingContent = bubble.querySelector(".thinking-stream-content");
  const contentContainer = bubble.querySelector(".content-stream-container");
  const cursor = bubble.querySelector(".streaming-cursor");

  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    timerBadge.textContent = `⏳ Streaming... ${elapsed}s`;
  }, 100);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId,
        mode: "single_shot"
      })
    });

    if (!response.ok) {
      clearInterval(timerInterval);
      if (cursor) cursor.remove();
      const err = await response.json().catch(() => ({ detail: "Server streaming failed." }));
      contentContainer.innerHTML = `<span style="color:var(--color-danger)"><strong>Error:</strong> ${escapeHtml(err.detail || "Request failed.")}</span>`;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let thinkingAccumulated = "";
    let contentAccumulated = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // Keep uncompleted event in buffer

      for (const part of parts) {
        const cleanedPart = part.trim();
        if (!cleanedPart.startsWith("data:")) continue;
        const jsonStr = cleanedPart.replace(/^data:\s*/, "").trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "session") {
            currentSessionId = event.session_id;
          } else if (event.type === "thinking") {
            thinkingDetails.style.display = "block";
            thinkingAccumulated += event.delta;
            thinkingContent.textContent = thinkingAccumulated;
            const words = thinkingAccumulated.split(/\s+/).filter(Boolean).length;
            thinkingWordCount.textContent = words;
            scrollToBottom();
          } else if (event.type === "content") {
            contentAccumulated += event.delta;
            const isCodeUnclosed = ((contentAccumulated.match(/```/g) || []).length % 2 === 1);
            if (cursor) cursor.style.display = isCodeUnclosed ? "none" : "inline-block";
            contentContainer.innerHTML = renderMarkdown(contentAccumulated, true);
            scrollToBottom();
          } else if (event.type === "done") {
            clearInterval(timerInterval);
            if (cursor) cursor.remove();
            const elapsed = event.latency_seconds !== undefined ? event.latency_seconds.toFixed(1) : ((Date.now() - startTime) / 1000).toFixed(1);
            timerBadge.textContent = `⚡ ${elapsed}s`;

            if (contentAccumulated) {
              contentContainer.innerHTML = renderMarkdown(contentAccumulated, false);
            }
            if (thinkingAccumulated) {
              thinkingDetails.open = false; // Collapse thinking process upon completion
            }

            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";
            metaDiv.style.marginTop = "8px";
            metaDiv.innerHTML = `
              <span class="latency-pill">⚡ ${elapsed}s</span>
              <span>batiai/gemma4-e2b:q4</span>
              <span>Live Single-Shot</span>
            `;
            bubble.appendChild(metaDiv);

            await loadSessions();
            scrollToBottom();
          } else if (event.type === "error") {
            clearInterval(timerInterval);
            if (cursor) cursor.remove();
            contentContainer.innerHTML += `<div style="color:var(--color-danger); margin-top:8px;"><strong>Error:</strong> ${escapeHtml(event.error)}</div>`;
          }
        } catch (err) {
          console.warn("SSE chunk parsing error:", err);
        }
      }
    }

    clearInterval(timerInterval);
    if (cursor) cursor.remove();
  } catch (err) {
    clearInterval(timerInterval);
    if (cursor) cursor.remove();
    contentContainer.innerHTML = `<span style="color:var(--color-danger)"><strong>Network Error:</strong> Could not connect to API (${escapeHtml(err.message)})</span>`;
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

// Lightweight Markdown & Scientific LaTeX Renderer with KaTeX Studio
function renderMarkdown(raw, isStreaming = false) {
  if (!raw) return "";
  let text = raw;

  // 1. Protect existing HTML details blocks (e.g. from session history)
  const detailsBlocks = [];
  text = text.replace(/<details[\s\S]*?<\/details>/gi, (match) => {
    const token = `__DETAILS_BLOCK_${detailsBlocks.length}__`;
    detailsBlocks.push(match);
    return token;
  });

  // 2. Check for unclosed triple-backtick code fence during streaming
  const fences = (text.match(/```/g) || []).length;
  const isCodeUnclosed = fences % 2 === 1;
  if (isCodeUnclosed) {
    text += "\n```";
  }

  // 3. Extract and protect code blocks: ```lang ... ```
  const codeBlocks = [];
  text = text.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, (match, lang, code, offset, fullStr) => {
    const language = (lang || "code").toLowerCase();
    const isLastBlock = (offset + match.length) >= fullStr.length;
    const isCurrentlyGenerating = isStreaming && isCodeUnclosed && isLastBlock;

    const badgeHtml = isCurrentlyGenerating
      ? `<span class="live-generating-badge"><span class="pulse-indicator"></span> Generating ${escapeHtml(language)} code...</span>`
      : `<button class="copy-code-btn" type="button" title="Copy code" onclick="copyCodeFromBlock(this)">Copy</button>`;

    const cursorHtml = isCurrentlyGenerating ? `<span class="code-streaming-cursor"></span>` : "";

    const html = `
      <div class="code-block-wrapper ${isCurrentlyGenerating ? "is-generating" : ""}">
        <div class="code-header">
          <div class="code-header-left">
            <span class="code-lang-icon">${getLanguageIcon(language)}</span>
            <span class="code-lang-name">${escapeHtml(language.toUpperCase())}</span>
          </div>
          <div class="code-header-right">
            ${badgeHtml}
          </div>
        </div>
        <pre><code class="language-${escapeHtml(language)}">${escapeHtml(code.trim())}${cursorHtml}</code></pre>
      </div>
    `;
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(html);
    return token;
  });

  // 4. Extract and protect inline code: `code`
  const inlineCodes = [];
  text = text.replace(/`([^`\n]+)`/g, (match, code) => {
    const html = `<code>${escapeHtml(code)}</code>`;
    const token = `__INLINE_CODE_${inlineCodes.length}__`;
    inlineCodes.push(html);
    return token;
  });

  // 5. Extract and protect display math: $$ ... $$ or \[ ... \]
  const displayMaths = [];
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (match, tex) => {
    const rendered = renderKatex(tex, true);
    const token = `__DISPLAY_MATH_${displayMaths.length}__`;
    displayMaths.push(rendered);
    return token;
  });
  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (match, tex) => {
    const rendered = renderKatex(tex, true);
    const token = `__DISPLAY_MATH_${displayMaths.length}__`;
    displayMaths.push(rendered);
    return token;
  });

  // 6. Extract and protect inline math: \( ... \) or $ ... $
  const inlineMaths = [];
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (match, tex) => {
    const rendered = renderKatex(tex, false);
    const token = `__INLINE_MATH_${inlineMaths.length}__`;
    inlineMaths.push(rendered);
    return token;
  });

  text = text.replace(/\$([^\$\n]+?)\$/g, (match, tex) => {
    const trimmed = tex.trim();
    // Exclude empty or plain currency numbers like $100 or $19.99
    if (!trimmed || /^\d+(?:,\d{3})*(?:\.\d+)?$/.test(trimmed)) {
      return match;
    }
    const rendered = renderKatex(trimmed, false);
    const token = `__INLINE_MATH_${inlineMaths.length}__`;
    inlineMaths.push(rendered);
    return token;
  });

  // 7. Markdown Headings
  text = text.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  text = text.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  text = text.replace(/^# (.*$)/gim, "<h1>$1</h1>");

  // 8. Bold: **text**
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // 9. Italics: *text*
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  // 10. Format paragraphs
  const paragraphs = text.split(/\n{2,}/);
  let rendered = paragraphs
    .map((p) => {
      p = p.trim();
      if (!p) return "";
      if (p.startsWith("__CODE_BLOCK_") && p.endsWith("__")) return p;
      if (p.startsWith("__DISPLAY_MATH_") && p.endsWith("__")) return p;
      if (p.startsWith("__DETAILS_BLOCK_") && p.endsWith("__")) return p;
      if (p.startsWith("<h1") || p.startsWith("<h2") || p.startsWith("<h3")) return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    })
    .filter(Boolean)
    .join("");

  // 11. Restore tokens in reverse order
  displayMaths.forEach((mathHtml, i) => {
    rendered = rendered.replace(new RegExp(`<p>__DISPLAY_MATH_${i}__</p>|__DISPLAY_MATH_${i}__`, "g"), mathHtml);
  });

  inlineMaths.forEach((mathHtml, i) => {
    rendered = rendered.replace(new RegExp(`__INLINE_MATH_${i}__`, "g"), mathHtml);
  });

  inlineCodes.forEach((codeHtml, i) => {
    rendered = rendered.replace(new RegExp(`__INLINE_CODE_${i}__`, "g"), codeHtml);
  });

  codeBlocks.forEach((blockHtml, i) => {
    rendered = rendered.replace(new RegExp(`<p>__CODE_BLOCK_${i}__</p>|__CODE_BLOCK_${i}__`, "g"), blockHtml);
  });

  detailsBlocks.forEach((detailsHtml, i) => {
    rendered = rendered.replace(new RegExp(`<p>__DETAILS_BLOCK_${i}__</p>|__DETAILS_BLOCK_${i}__`, "g"), detailsHtml);
  });

  return rendered;
}

function renderKatex(tex, isDisplay = false) {
  const cleanTex = tex.trim();
  if (!cleanTex) return "";

  if (typeof window !== "undefined" && window.katex && typeof window.katex.renderToString === "function") {
    try {
      return window.katex.renderToString(cleanTex, {
        displayMode: isDisplay,
        throwOnError: false,
        strict: false
      });
    } catch (err) {
      console.warn("KaTeX render error for formula:", cleanTex, err);
    }
  }

  // Graceful fallback if KaTeX is loading or unavailable
  const wrapperClass = isDisplay ? "math-fallback-block" : "math-fallback-inline";
  return `<span class="${wrapperClass}" data-tex="${escapeHtml(cleanTex)}"><code>${escapeHtml(cleanTex)}</code></span>`;
}

// Re-render any fallback math elements once KaTeX is initialized
function hydrateMathFormulas(container = document.body) {
  if (!window.katex || !container) return;
  const fallbacks = container.querySelectorAll(".math-fallback-inline, .math-fallback-block");
  fallbacks.forEach((el) => {
    const isDisplay = el.classList.contains("math-fallback-block");
    const tex = el.getAttribute("data-tex") || el.textContent;
    try {
      const renderedHtml = window.katex.renderToString(tex, {
        displayMode: isDisplay,
        throwOnError: false,
        strict: false
      });
      const temp = document.createElement("span");
      temp.innerHTML = renderedHtml;
      el.replaceWith(temp.firstElementChild || temp);
    } catch (err) {}
  });
}

// Check every 150ms up to 3s on startup until KaTeX is ready
let katexPollCount = 0;
const katexPoll = setInterval(() => {
  katexPollCount++;
  if (window.katex || katexPollCount > 20) {
    clearInterval(katexPoll);
    if (window.katex) {
      hydrateMathFormulas(document.body);
    }
  }
}, 150);

function getLanguageIcon(lang) {
  const map = {
    python: "🐍",
    py: "🐍",
    bash: "💻",
    sh: "💻",
    json: "📦",
    sql: "🗄️",
    yaml: "⚙️",
    yml: "⚙️",
    javascript: "⚡",
    js: "⚡",
    fortran: "🔬",
    f90: "🔬"
  };
  return map[lang.toLowerCase()] || "📄";
}

window.copyCodeFromBlock = function(btn) {
  const wrapper = btn.closest(".code-block-wrapper");
  if (!wrapper) return;
  const codeEl = wrapper.querySelector("code");
  if (!codeEl) return;

  const clone = codeEl.cloneNode(true);
  const cursor = clone.querySelector(".code-streaming-cursor");
  if (cursor) cursor.remove();

  navigator.clipboard.writeText(clone.innerText.trim()).then(() => {
    const originalText = btn.textContent;
    btn.textContent = "Copied! ✓";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove("copied");
    }, 2000);
  }).catch(() => {
    btn.textContent = "Copied!";
  });
};

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
