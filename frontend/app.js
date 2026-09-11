let currentSessionId = null;
let benchmarkPrompts = [];
let currentEngineMode = "zero_shot";

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

// Mode Switcher Elements
const btnModeZeroShot = document.getElementById("btn-mode-zero-shot");
const btnModeAgenticRag = document.getElementById("btn-mode-agentic-rag");
const engineStatusPill = document.getElementById("engine-status-pill");
const engineStatusText = document.getElementById("engine-status-text");
const sidebarModeBadge = document.getElementById("sidebar-mode-badge");
const sidebarStatusDot = document.getElementById("sidebar-status-dot");
const sidebarModeLabel = document.getElementById("sidebar-mode-label");
const sidebarModeDesc = document.getElementById("sidebar-mode-desc");

// Settings Drawer Elements
const settingsToggleBtn = document.getElementById("settings-toggle-btn");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const btnCloseSettingsFooter = document.getElementById("btn-close-settings-footer");
const settingsDrawer = document.getElementById("settings-drawer");
const settingsDrawerBackdrop = document.getElementById("settings-drawer-backdrop");
const drawerModePill = document.getElementById("drawer-mode-pill");
const drawerModePillText = document.getElementById("drawer-mode-pill-text");
const drawerModeExplanation = document.getElementById("drawer-mode-explanation");
const drawerModeTraits = document.getElementById("drawer-mode-traits");
const drawerBtnZeroShot = document.getElementById("drawer-btn-zero-shot");
const drawerBtnAgentic = document.getElementById("drawer-btn-agentic");
const drawerModelName = document.getElementById("drawer-model-name");

// ─── Toast Notification System ───────────────────────────────────────────────
function showToast(message, type = "info", duration = 3000) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const icons = { success: "✓", error: "✗", info: "ℹ", warning: "⚠" };
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${icons[type] || "ℹ"}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("toast-exit");
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─── Theme Management (Dark, Light, High Contrast) ──────────────────────────
function applyTheme(themeName, showNotification = false) {
  const validThemes = ["dark", "light", "high-contrast"];
  const targetTheme = validThemes.includes(themeName) ? themeName : "dark";

  document.documentElement.setAttribute("data-theme", targetTheme);
  localStorage.setItem("scicompute_theme", targetTheme);

  // Update theme option cards inside the settings drawer
  document.querySelectorAll(".theme-option-card").forEach((card) => {
    const cardTheme = card.getAttribute("data-theme-value");
    const isActive = cardTheme === targetTheme;
    card.classList.toggle("active", isActive);
    card.setAttribute("aria-checked", isActive ? "true" : "false");
  });

  if (showNotification) {
    const labels = {
      dark: "Dark Mode",
      light: "Light Mode",
      "high-contrast": "High Contrast Mode (WCAG AAA)"
    };
    showToast(`Switched to ${labels[targetTheme] || targetTheme}`, "info", 2000);
  }
}

function initThemeToggle() {
  const saved = localStorage.getItem("scicompute_theme");
  if (saved) {
    applyTheme(saved, false);
  } else if (window.matchMedia("(prefers-contrast: more)").matches) {
    applyTheme("high-contrast", false);
  } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
    applyTheme("light", false);
  } else {
    applyTheme("dark", false);
  }

  // Bind click & keyboard handlers to all theme cards
  document.querySelectorAll(".theme-option-card").forEach((card) => {
    card.addEventListener("click", () => {
      const selected = card.getAttribute("data-theme-value");
      applyTheme(selected, true);
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const selected = card.getAttribute("data-theme-value");
        applyTheme(selected, true);
      }
    });
  });
}

// ─── Settings Drawer Controller ──────────────────────────────────────────────
function openSettingsDrawer() {
  if (!settingsDrawer || !settingsDrawerBackdrop) return;
  settingsDrawer.classList.add("open");
  settingsDrawer.setAttribute("aria-hidden", "false");
  settingsDrawerBackdrop.classList.add("active");
  settingsDrawerBackdrop.setAttribute("aria-hidden", "false");
  settingsToggleBtn?.classList.add("active");
  settingsToggleBtn?.setAttribute("aria-expanded", "true");
  document.body.style.overflow = "hidden";
  updateSettingsDrawerModeInfo();
}

function closeSettingsDrawer() {
  if (!settingsDrawer || !settingsDrawerBackdrop) return;
  settingsDrawer.classList.remove("open");
  settingsDrawer.setAttribute("aria-hidden", "true");
  settingsDrawerBackdrop.classList.remove("active");
  settingsDrawerBackdrop.setAttribute("aria-hidden", "true");
  settingsToggleBtn?.classList.remove("active");
  settingsToggleBtn?.setAttribute("aria-expanded", "false");
  document.body.style.overflow = "";
}

function updateSettingsDrawerModeInfo() {
  const isAgentic = currentEngineMode === "agentic";
  
  if (drawerModePill) {
    drawerModePill.className = `drawer-mode-pill ${isAgentic ? "mode-agentic" : "mode-zero-shot"}`;
  }
  if (drawerModePillText) {
    drawerModePillText.textContent = isAgentic ? "🧠 Agentic RAG System" : "⚡ Zero-Shot Baseline";
  }
  if (drawerModeExplanation) {
    if (isAgentic) {
      drawerModeExplanation.innerHTML = `
        <strong>Agentic RAG Mode</strong> executes a multi-step iterative pipeline with live tool-calling. It connects <em>batiai/gemma4-e2b:q4</em> with <strong>ChromaDB</strong> vector search, <strong>Kùzu</strong> graph knowledge lookups, a Python 3 subprocess <strong>execution sandbox</strong>, and scientific <strong>web search</strong> (arXiv, PubChem, NIST) to generate and verify calculations.
      `;
    } else {
      drawerModeExplanation.innerHTML = `
        <strong>Zero-Shot Baseline Mode</strong> executes direct prompt inference through <em>batiai/gemma4-e2b:q4</em> with <strong>zero external tools</strong> and <strong>zero RAG retrieval</strong>. This mode assesses raw foundation model weights, native computational chemistry syntax, and ASE/DeePMD code generation capabilities under pure pre-trained recall.
      `;
    }
  }

  if (drawerModeTraits) {
    if (isAgentic) {
      drawerModeTraits.innerHTML = `
        <div class="trait-item">
          <span class="trait-icon">📚</span>
          <div class="trait-text">
            <span class="trait-title">Hybrid Retrieval: Active</span>
            <span class="trait-sub">ChromaDB semantic embeddings + Kùzu property graph</span>
          </div>
        </div>
        <div class="trait-item">
          <span class="trait-icon">🐳</span>
          <div class="trait-text">
            <span class="trait-title">Subprocess Sandbox: Enabled</span>
            <span class="trait-sub">Executes generated script in isolated Python sandbox</span>
          </div>
        </div>
        <div class="trait-item">
          <span class="trait-icon">🌐</span>
          <div class="trait-text">
            <span class="trait-title">Web Search & Re-ranking: Active</span>
            <span class="trait-sub">Live search for literature constants, lattice units & formulas</span>
          </div>
        </div>
      `;
    } else {
      drawerModeTraits.innerHTML = `
        <div class="trait-item">
          <span class="trait-icon">🔒</span>
          <div class="trait-text">
            <span class="trait-title">External Tools: Disabled</span>
            <span class="trait-sub">No Python sandbox or shell subprocess execution</span>
          </div>
        </div>
        <div class="trait-item">
          <span class="trait-icon">📚</span>
          <div class="trait-text">
            <span class="trait-title">Knowledge Graph & RAG: Disabled</span>
            <span class="trait-sub">No ChromaDB vectors or Kùzu graph lookups</span>
          </div>
        </div>
        <div class="trait-item">
          <span class="trait-icon">🎯</span>
          <div class="trait-text">
            <span class="trait-title">Evaluation Objective</span>
            <span class="trait-sub">Measures raw parametric chemistry knowledge vs 100-pt rubric</span>
          </div>
        </div>
      `;
    }
  }

  if (drawerBtnZeroShot) {
    drawerBtnZeroShot.classList.toggle("active", !isAgentic);
  }
  if (drawerBtnAgentic) {
    drawerBtnAgentic.classList.toggle("active", isAgentic);
  }
}

function initSettingsDrawer() {
  settingsToggleBtn?.addEventListener("click", () => {
    if (settingsDrawer?.classList.contains("open")) {
      closeSettingsDrawer();
    } else {
      openSettingsDrawer();
    }
  });

  closeSettingsBtn?.addEventListener("click", closeSettingsDrawer);
  btnCloseSettingsFooter?.addEventListener("click", closeSettingsDrawer);
  settingsDrawerBackdrop?.addEventListener("click", closeSettingsDrawer);

  drawerBtnZeroShot?.addEventListener("click", () => {
    setEngineMode("zero_shot");
    showToast("Switched to Zero-Shot Baseline", "info", 1500);
  });

  drawerBtnAgentic?.addEventListener("click", () => {
    setEngineMode("agentic");
    showToast("Switched to Agentic RAG System", "info", 1500);
  });
}

// ─── Connection Health Check ─────────────────────────────────────────────────
function initConnectionCheck() {
  const badge = document.getElementById("connection-badge");
  const dot = document.getElementById("connection-dot");
  const text = document.getElementById("connection-text");
  if (!badge) return;

  async function checkHealth() {
    try {
      const res = await fetch("/api/model_info", { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        badge.className = "connection-badge online";
        text.textContent = "Online";
      } else {
        badge.className = "connection-badge offline";
        text.textContent = "Error";
      }
    } catch {
      badge.className = "connection-badge offline";
      text.textContent = "Offline";
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);
}

// ─── Scroll to Top ───────────────────────────────────────────────────────────
function initScrollToTop() {
  const btn = document.getElementById("scroll-top-btn");
  if (!btn || !messagesContainer) return;
  messagesContainer.addEventListener("scroll", () => {
    if (messagesContainer.scrollTop > 400) {
      btn.style.display = "flex";
      btn.classList.add("visible");
    } else {
      btn.classList.remove("visible");
      setTimeout(() => { if (!btn.classList.contains("visible")) btn.style.display = "none"; }, 300);
    }
  });
  btn.addEventListener("click", () => {
    messagesContainer.scrollTo({ top: 0, behavior: "smooth" });
  });
}

// ─── Keyboard Shortcuts ──────────────────────────────────────────────────────
function initKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Ctrl+N: New Session
    if ((e.ctrlKey || e.metaKey) && e.key === "n") {
      e.preventDefault();
      startNewSession();
      showToast("New session started", "success", 1500);
    }
    // Ctrl+K: Focus input
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      promptInput?.focus();
    }
    // Escape: Close mobile sidebar or settings drawer
    if (e.key === "Escape") {
      sidebar?.classList.remove("open");
      closeSettingsDrawer();
    }
  });
}

// ─── Draft Recovery ──────────────────────────────────────────────────────────
function initDraftRecovery() {
  if (!promptInput) return;
  const saved = localStorage.getItem("scicompute_draft");
  if (saved && saved.trim()) {
    promptInput.value = saved;
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 180) + "px";
  }
  let debounceTimer = null;
  promptInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      localStorage.setItem("scicompute_draft", promptInput.value);
    }, 500);
  });
}

// ─── Skeleton Loading ────────────────────────────────────────────────────────
function showSkeleton() {
  const item = document.createElement("div");
  item.className = "message-item assistant";
  item.id = "skeleton-loader";
  item.innerHTML = `
    <div class="message-bubble" style="min-width:300px;">
      <div class="skeleton skeleton-line" style="width:90%"></div>
      <div class="skeleton skeleton-line" style="width:75%"></div>
      <div class="skeleton skeleton-line" style="width:60%"></div>
    </div>
  `;
  chatThread.appendChild(item);
  scrollToBottom();
  return item;
}
function removeSkeleton() {
  const el = document.getElementById("skeleton-loader");
  if (el) el.remove();
}

// ─── Animated Score Counter ──────────────────────────────────────────────────
function animateScoreCounter(element, targetScore, duration = 1200) {
  if (!element) return;
  let start = 0;
  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * targetScore);
    element.textContent = current;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Mode Switching Handler
function setEngineMode(mode, persist = true) {
  currentEngineMode = (mode === "agentic" || mode === "agentic_rag") ? "agentic" : "zero_shot";
  if (persist) {
    localStorage.setItem("scicompute_engine_mode", currentEngineMode);
  }

  if (currentEngineMode === "agentic") {
    btnModeZeroShot?.classList.remove("active");
    btnModeAgenticRag?.classList.add("active");

    if (engineStatusPill) {
      engineStatusPill.className = "pill-badge pill-agentic";
      engineStatusPill.title = "Agentic RAG Mode (ChromaDB • Kùzu Graph • Docker Sandbox • Web Search)";
    }
    if (engineStatusText) {
      engineStatusText.textContent = "🧠 Agentic RAG Mode (Tools Active)";
    }
    if (sidebarStatusDot) {
      sidebarStatusDot.style.background = "var(--accent-primary)";
      sidebarStatusDot.style.boxShadow = "0 0 8px var(--accent-glow)";
    }
    if (sidebarModeLabel) {
      sidebarModeLabel.textContent = "Agentic RAG Assistant";
    }
    if (sidebarModeDesc) {
      sidebarModeDesc.textContent = "ChromaDB • Kùzu • Docker • Web";
    }
  } else {
    btnModeAgenticRag?.classList.remove("active");
    btnModeZeroShot?.classList.add("active");

    if (engineStatusPill) {
      engineStatusPill.className = "pill-badge pill-zero-shot";
      engineStatusPill.title = "Zero-Shot Baseline (Direct LLM inference without tools or RAG)";
    }
    if (engineStatusText) {
      engineStatusText.textContent = "⚡ Zero-Shot Mode (No Tools • No RAG)";
    }
    if (sidebarStatusDot) {
      sidebarStatusDot.style.background = "var(--color-info)";
      sidebarStatusDot.style.boxShadow = "0 0 8px rgba(56, 189, 248, 0.4)";
    }
    if (sidebarModeLabel) {
      sidebarModeLabel.textContent = "Zero-Shot Baseline";
    }
    if (sidebarModeDesc) {
      sidebarModeDesc.textContent = "No Tools • No RAG";
    }
  }

  updateSettingsDrawerModeInfo();
}

// Category Display Metadata
const CATEGORY_META = {
  // Flagship RAG Benchmark Categories
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
  },
  // Dynamic Hybrid Evaluator Categories
  syntax_and_safety: {
    label: "1. Syntax Parsing & Security Isolation Audit",
    max: 15
  },
  domain_implementation: {
    label: "2. Scientific Domain Implementation & Constraints",
    max: 30
  },
  runtime_execution_and_contracts: {
    label: "3. Scope Resolution & Runtime Execution",
    max: 25
  },
  scientific_rigor: {
    label: "4. Scientific Rigor (Units, Naming, Docs, Error Handling)",
    max: 15
  },
  artifacts_and_presentation: {
    label: "5. Visualization, Figures & Persistence",
    max: 15
  }
};

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  initThemeToggle();
  initSettingsDrawer();
  initKeyboardShortcuts();
  initConnectionCheck();
  initScrollToTop();
  initDraftRecovery();
  setupTextareaAutoGrow();
  setupEventListeners();
  const savedMode = localStorage.getItem("scicompute_engine_mode");
  setEngineMode(savedMode || currentEngineMode, false);
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
  if (btnModeZeroShot) {
    btnModeZeroShot.addEventListener("click", () => setEngineMode("zero_shot"));
  }
  if (btnModeAgenticRag) {
    btnModeAgenticRag.addEventListener("click", () => setEngineMode("agentic"));
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
      const drawerModel = document.getElementById("drawer-model-name");
      if (drawerModel) {
        drawerModel.textContent = data.model_name;
      }
      const drawerCompute = document.getElementById("drawer-compute-device");
      if (drawerCompute && data.gpu_mode) {
        const isCpu = data.num_gpu === 0 || data.gpu_mode.toLowerCase().includes("cpu");
        drawerCompute.textContent = isCpu ? "CPU-only (NUM_GPU=0)" : "GPU (AMD ROCm / HIP)";
        drawerCompute.className = isCpu ? "spec-card-val" : "spec-card-val highlight-accent";
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

  // Explicitly ensure UI is in Zero-Shot Evaluation Mode
  setEngineMode("zero_shot", false);

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
      <summary class="thinking-summary">
        <div class="thinking-summary-left">
          <span class="thinking-summary-icon">🧠</span>
          <span class="thinking-summary-title">Thinking Process</span>
          <span class="thinking-summary-count"><span class="thinking-word-count">0</span> words</span>
        </div>
        <span class="thinking-summary-chevron">▾</span>
      </summary>
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
            thinkingContent.innerHTML = renderThinkingBody(thinkingAccumulated);
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
              thinkingContent.innerHTML = renderThinkingBody(thinkingAccumulated);
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
      const isHybrid = ev.evaluation_type === "dynamic_hybrid";
      const badgeIcon = isHybrid ? "🧪" : "🎯";
      const domainLabel = ev.domain ? ` • ${ev.domain.replace(/_/g, " ")}` : "";

      item.innerHTML = `
        <div class="eval-history-left">
          <span class="eval-history-title">${badgeIcon} Eval #${ev.id} • ${ev.score}/100</span>
          <span class="eval-history-time">${dateStr}${domainLabel}</span>
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
  const summaryText = ev.summary || report.summary || "Script benchmark completed.";
  const catScores = ev.category_scores || {};
  const codeText = ev.generated_code || "";

  const isHybrid = ev.evaluation_type === "dynamic_hybrid";
  const titleText = isHybrid
    ? `Dynamic Scientific Evaluator (${(ev.domain || "Simulation").replace(/_/g, " ").toUpperCase()})`
    : "Interfacial Water Dynamics Generator";
  const modeLabel = isHybrid ? "Dynamic Hybrid Test Suite" : "Zero-Shot Flagship Benchmark";

  // 1. Header
  let html = `
    <div class="eval-card-header">
      <div class="score-badge-container">
        <div class="score-circle">
          <span class="score-num">${score}</span>
          <span class="score-denom">/ 100</span>
        </div>
        <div class="score-text-details">
          <h3>${escapeHtml(titleText)}</h3>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="grade-pill ${gradeClass}">Grade ${grade}</span>
            <span style="font-size:0.75rem; color:var(--text-muted);">Model: ${escapeHtml(modelName)}</span>
          </div>
        </div>
      </div>
      <div class="message-meta">
        <span class="latency-pill">⚡ ${latency}</span>
        <span>${escapeHtml(modeLabel)}</span>
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
  const activeCategoryKeys = Object.keys(catScores).length > 0
    ? Object.keys(catScores)
    : Object.keys(CATEGORY_META).slice(0, 5);

  // Client-side consistency check: ensure category scores sum to headline score
  const totalCatSum = activeCategoryKeys.reduce((acc, k) => acc + ((catScores[k] && catScores[k].score) || 0), 0);
  const scaleRatio = (totalCatSum > 0 && score < totalCatSum) ? (score / totalCatSum) : 1;

  for (const key of activeCategoryKeys) {
    const meta = CATEGORY_META[key] || { label: key.replace(/_/g, " ").toUpperCase(), max: 25 };
    const catData = catScores[key] || { score: 0, max: meta.max };
    const rawCatScore = catData.score !== undefined ? catData.score : 0;
    const curScore = scaleRatio < 1 ? Math.round(rawCatScore * scaleRatio) : rawCatScore;
    const maxScore = catData.max || meta.max;
    const pct = scaleRatio < 1 
      ? Math.min(100, Math.round((curScore / maxScore) * 100))
      : (catData.pct !== undefined ? catData.pct : Math.round((curScore / maxScore) * 100));

    // Dynamic color gradient based on percentage so failing/degraded categories are visually distinct
    let barGradient = "var(--accent-gradient)";
    if (pct <= 35) {
      barGradient = "linear-gradient(90deg, #ef4444, #f87171)";
    } else if (pct <= 60) {
      barGradient = "linear-gradient(90deg, #f59e0b, #fbbf24)";
    }

    html += `
      <div class="cat-row">
        <div class="cat-meta">
          <span>${meta.label}</span>
          <span><strong>${curScore}</strong> / ${maxScore} pts (${pct}%)</span>
        </div>
        <div class="cat-bar-bg">
          <div class="cat-bar-fill" style="width: ${Math.min(pct, 100)}%; background: ${barGradient};"></div>
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

  // Animate the score counter
  const scoreNumEl = card.querySelector(".score-num");
  if (scoreNumEl && score > 0) {
    animateScoreCounter(scoreNumEl, score, 1200);
  }
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
      const isAgentic = (s.mode === "agentic" || s.agent_used === "AgenticRAG");
      const badgeHtml = isAgentic
        ? `<span class="session-mode-badge badge-agentic">🧠 Agentic</span>`
        : `<span class="session-mode-badge badge-zero-shot">⚡ Zero-Shot</span>`;

      el.innerHTML = `
        <span class="session-title-text" title="${escapeHtml(s.title)}">${escapeHtml(s.title)}</span>
        ${badgeHtml}
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

    // Automatically synchronize engine mode to match the loaded session
    const hasAgenticMessage = messages.some(
      (m) => m.agent_used === "AgenticRAG" || m.agent_used === "agentic"
    );
    setEngineMode(hasAgenticMessage ? "agentic" : "zero_shot", false);

    messages.forEach((msg) => {
      appendMessage(msg.role, msg.content, {
        timestamp: msg.timestamp,
        agent_used: msg.agent_used,
        model: msg.model_name,
        tool_calls_json: msg.tool_calls_json
      });
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
      showToast("Session deleted", "success", 2000);
      if (currentSessionId === sessionId) {
        startNewSession();
      } else {
        await loadSessions();
      }
    }
  } catch (err) {
    console.error("Error deleting session:", err);
    showToast("Failed to delete session", "error", 2500);
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

// Render or update intermediate agentic execution steps in the message bubble
function renderAgentTraceStep(container, stepEvent) {
  if (!container) return;
  container.style.display = "flex";

  const stepKey = stepEvent.step || "step";
  let stepEl = container.querySelector(`[data-step="${stepKey}"]`);
  
  let icon = "⚡";
  if (stepKey.startsWith("rag")) icon = "📚";
  else if (stepKey.startsWith("code_gen")) icon = "📝";
  else if (stepKey.startsWith("sandbox")) icon = "🐳";
  else if (stepKey.startsWith("web")) icon = "🌐";
  else if (stepKey.startsWith("synthesis")) icon = "🧠";

  const statusClass = stepEvent.status || "running";
  const titleText = stepEvent.title || "Agent Execution Step";

  if (!stepEl) {
    stepEl = document.createElement("div");
    stepEl.className = `agent-trace-step ${statusClass}`;
    stepEl.setAttribute("data-step", stepKey);

    stepEl.innerHTML = `
      <div class="agent-trace-icon">${icon}</div>
      <div class="agent-trace-body">
        <div class="agent-trace-title">
          <span>${escapeHtml(titleText)}</span>
          <span class="agent-trace-status ${statusClass}">${escapeHtml(statusClass)}</span>
        </div>
        ${stepEvent.details ? `<div class="agent-trace-details">${escapeHtml(stepEvent.details)}</div>` : ""}
      </div>
    `;
    container.appendChild(stepEl);
  } else {
    stepEl.className = `agent-trace-step ${statusClass}`;
    const statusSpan = stepEl.querySelector(".agent-trace-status");
    if (statusSpan) {
      statusSpan.className = `agent-trace-status ${statusClass}`;
      statusSpan.textContent = statusClass;
    }
    const detailsDiv = stepEl.querySelector(".agent-trace-details");
    if (stepEvent.details) {
      if (detailsDiv) {
        detailsDiv.textContent = stepEvent.details;
      } else {
        const newDetails = document.createElement("div");
        newDetails.className = "agent-trace-details";
        newDetails.textContent = stepEvent.details;
        stepEl.querySelector(".agent-trace-body").appendChild(newDetails);
      }
    }
  }
}

// Chat Submission with Real-Time Token Streaming & Mode Dispatch
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;

  promptInput.value = "";
  promptInput.style.height = "auto";
  localStorage.removeItem("scicompute_draft");
  welcomeHero.style.display = "none";

  appendMessage("user", text);
  scrollToBottom();

  const isAgentic = (currentEngineMode === "agentic");

  // Create streaming assistant message bubble
  const assistantItem = document.createElement("div");
  assistantItem.className = "message-item assistant";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  bubble.innerHTML = `
    <div class="stream-header-bar">
      <span class="live-timer-badge">⏳ ${isAgentic ? "Agentic RAG Pipeline Active..." : "Starting inference..."} 0.0s</span>
      <span class="live-model-pill">${isAgentic ? "🧠 Agentic RAG • ChromaDB + Sandbox + Web" : "batiai/gemma4-e2b:q4 • Zero-Shot Baseline"}</span>
    </div>
    <div class="agent-trace-container" style="${isAgentic ? "" : "display:none;"}"></div>
    <details class="thinking-details" style="display:none;" open>
      <summary class="thinking-summary">
        <div class="thinking-summary-left">
          <span class="thinking-summary-icon">🧠</span>
          <span class="thinking-summary-title">Thinking Process</span>
          <span class="thinking-summary-count"><span class="thinking-word-count">0</span> words</span>
        </div>
        <span class="thinking-summary-chevron">▾</span>
      </summary>
      <div class="thinking-stream-content"></div>
    </details>
    <div class="content-stream-container"></div>
    <span class="streaming-cursor"></span>
  `;

  assistantItem.appendChild(bubble);
  chatThread.appendChild(assistantItem);
  scrollToBottom();

  const timerBadge = bubble.querySelector(".live-timer-badge");
  const agentTraceContainer = bubble.querySelector(".agent-trace-container");
  const thinkingDetails = bubble.querySelector(".thinking-details");
  const thinkingWordCount = bubble.querySelector(".thinking-word-count");
  const thinkingContent = bubble.querySelector(".thinking-stream-content");
  const contentContainer = bubble.querySelector(".content-stream-container");
  const cursor = bubble.querySelector(".streaming-cursor");

  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    timerBadge.textContent = isAgentic
      ? `🧠 Agentic Workflow... ${elapsed}s`
      : `⏳ Streaming... ${elapsed}s`;
  }, 100);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: currentSessionId,
        mode: isAgentic ? "agentic" : "single_shot"
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
          } else if (event.type === "agent_step") {
            renderAgentTraceStep(agentTraceContainer, event);
            scrollToBottom();
          } else if (event.type === "thinking") {
            thinkingDetails.style.display = "block";
            thinkingAccumulated += event.delta;
            thinkingContent.innerHTML = renderThinkingBody(thinkingAccumulated);
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
              thinkingContent.innerHTML = renderThinkingBody(thinkingAccumulated);
              thinkingDetails.open = false; // Collapse thinking process upon completion
            }

            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";
            metaDiv.style.marginTop = "8px";
            metaDiv.innerHTML = `
              <span class="latency-pill">⚡ ${elapsed}s</span>
              <span>batiai/gemma4-e2b:q4</span>
              <span>${isAgentic ? "Agentic RAG Engine" : "Zero-Shot Baseline"}</span>
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

  // If this message has stored tool/agent trace steps, display them at the top of the bubble
  if (role === "assistant" && meta.tool_calls_json) {
    try {
      const steps = typeof meta.tool_calls_json === "string" ? JSON.parse(meta.tool_calls_json) : meta.tool_calls_json;
      if (Array.isArray(steps) && steps.length > 0) {
        const traceContainer = document.createElement("div");
        traceContainer.className = "agent-trace-container";
        traceContainer.style.display = "flex";
        steps.forEach((step) => {
          renderAgentTraceStep(traceContainer, step);
        });
        bubble.appendChild(traceContainer);
      }
    } catch (e) {
      console.warn("Error rendering saved agent steps:", e);
    }
  }

  const contentDiv = document.createElement("div");
  contentDiv.innerHTML = renderMarkdown(content);
  bubble.appendChild(contentDiv);

  item.appendChild(bubble);

  if (role === "assistant") {
    const metaDiv = document.createElement("div");
    metaDiv.className = "message-meta";
    const isAgent = (meta.agent_used === "AgenticRAG" || meta.mode === "agentic");
    const agentLabel = isAgent ? "Agentic RAG Engine" : "Zero-Shot Baseline";
    const latencyPart = meta.latency !== undefined ? `<span class="latency-pill">⚡ ${Number(meta.latency).toFixed(1)}s</span>` : "";
    metaDiv.innerHTML = `
      ${latencyPart}
      <span>${escapeHtml(meta.model || "batiai/gemma4-e2b:q4")}</span>
      <span>${agentLabel}</span>
    `;
    item.appendChild(metaDiv);
  }

  chatThread.appendChild(item);
  hydrateMathFormulas(bubble);
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

// Render formatted text and KaTeX math formulas inside thinking details blocks
function renderThinkingBody(raw) {
  if (!raw) return "";
  let text = raw.trim();

  // 1. Extract and protect display math ($$...$$ or \[...\])
  const displayMaths = [];
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (match, tex) => {
    const rendered = renderKatex(tex, true);
    const token = `__THINK_DMATH_${displayMaths.length}__`;
    displayMaths.push(rendered);
    return token;
  });
  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (match, tex) => {
    const rendered = renderKatex(tex, true);
    const token = `__THINK_DMATH_${displayMaths.length}__`;
    displayMaths.push(rendered);
    return token;
  });

  // 2. Extract and protect inline math (\(...\) or $...$)
  const inlineMaths = [];
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (match, tex) => {
    const rendered = renderKatex(tex, false);
    const token = `__THINK_IMATH_${inlineMaths.length}__`;
    inlineMaths.push(rendered);
    return token;
  });
  text = text.replace(/\$([^\$\n]+?)\$/g, (match, tex) => {
    const trimmed = tex.trim();
    if (!trimmed || /^\d+(?:,\d{3})*(?:\.\d+)?$/.test(trimmed)) return match;
    const rendered = renderKatex(trimmed, false);
    const token = `__THINK_IMATH_${inlineMaths.length}__`;
    inlineMaths.push(rendered);
    return token;
  });

  // 3. Extract and protect code blocks
  const codeBlocks = [];
  text = text.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, (m, lang, code) => {
    const token = `__THINK_CODE_${codeBlocks.length}__`;
    codeBlocks.push(`<pre class="thinking-code-block"><code>${escapeHtml(code.trim())}</code></pre>`);
    return token;
  });

  // 4. Extract and protect inline code
  const inlineCodes = [];
  text = text.replace(/`([^`\n]+)`/g, (m, code) => {
    const token = `__THINK_ICODE_${inlineCodes.length}__`;
    inlineCodes.push(`<code class="thinking-inline-code">${escapeHtml(code)}</code>`);
    return token;
  });

  // 5. Unescape markdown underscores and escape non-math text safely
  text = text.replace(/\\_/g, "_");
  text = escapeHtml(text);

  // 6. Bold: **text**
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // 7. Parse line by line for structured lists, sub-bullets, and paragraphs
  const lines = text.split("\n");
  const processedLines = [];
  let inOl = false;
  let inUl = false;
  let inSubUl = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      if (inSubUl) { processedLines.push("</ul>"); inSubUl = false; }
      continue;
    }

    // Sub-bullet indented (2+ spaces or tab followed by * or -)
    const subUlMatch = line.match(/^(\s{2,}|\t+)[\*\-]\s+(.*)$/);
    if (subUlMatch) {
      if (!inSubUl) {
        processedLines.push('<ul class="thinking-sub-ul">');
        inSubUl = true;
      }
      let content = subUlMatch[2].replace(/\*([^*]+)\*/g, "<em>$1</em>");
      processedLines.push(`<li>${content}</li>`);
      continue;
    } else if (inSubUl) {
      processedLines.push("</ul>");
      inSubUl = false;
    }

    // Numbered item: 1. , 2. etc
    const olMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
    if (olMatch) {
      if (inUl) { processedLines.push("</ul>"); inUl = false; }
      if (!inOl) { processedLines.push('<ol class="thinking-ol">'); inOl = true; }
      let content = olMatch[2].replace(/\*([^*]+)\*/g, "<em>$1</em>");
      processedLines.push(`<li>${content}</li>`);
      continue;
    }

    // Bullet item: * or -
    const ulMatch = trimmed.match(/^[\*\-]\s+(.*)$/);
    if (ulMatch) {
      if (inOl) { processedLines.push("</ol>"); inOl = false; }
      if (!inUl) { processedLines.push('<ul class="thinking-ul">'); inUl = true; }
      let content = ulMatch[1].replace(/\*([^*]+)\*/g, "<em>$1</em>");
      processedLines.push(`<li>${content}</li>`);
      continue;
    }

    // Normal paragraph line
    if (inOl) { processedLines.push("</ol>"); inOl = false; }
    if (inUl) { processedLines.push("</ul>"); inUl = false; }
    let content = trimmed.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    processedLines.push(`<p class="thinking-p">${content}</p>`);
  }

  if (inSubUl) processedLines.push("</ul>");
  if (inUl) processedLines.push("</ul>");
  if (inOl) processedLines.push("</ol>");

  let result = processedLines.join("\n");

  // 8. Restore KaTeX rendered math and code tokens safely without $ pattern corruption
  displayMaths.forEach((m, i) => {
    result = result.replace(new RegExp(`__THINK_DMATH_${i}__`, "g"), () => m);
  });
  inlineMaths.forEach((m, i) => {
    result = result.replace(new RegExp(`__THINK_IMATH_${i}__`, "g"), () => m);
  });
  codeBlocks.forEach((c, i) => {
    result = result.replace(new RegExp(`__THINK_CODE_${i}__`, "g"), () => c);
  });
  inlineCodes.forEach((c, i) => {
    result = result.replace(new RegExp(`__THINK_ICODE_${i}__`, "g"), () => c);
  });

  return result;
}

// Lightweight Markdown & Scientific LaTeX Renderer with KaTeX Studio
function renderMarkdown(raw, isStreaming = false) {
  if (!raw) return "";
  let text = raw;

  // 1. Process and protect details / thinking blocks with KaTeX math rendering
  const detailsBlocks = [];
  text = text.replace(/<details class="thinking-details"([\s\S]*?)>([\s\S]*?)<\/details>/gi, (match, attrs, inner) => {
    const summaryMatch = inner.match(/<summary>([\s\S]*?)<\/summary>/i);
    let countMatch = summaryMatch ? summaryMatch[1].match(/\((\d+)\s*words\)/i) : null;
    const rawBody = inner.replace(/<summary>[\s\S]*?<\/summary>/i, "").trim();
    let wordCount = countMatch ? countMatch[1] : (rawBody ? rawBody.split(/\s+/).filter(Boolean).length : 0);
    const renderedBody = renderThinkingBody(rawBody);
    const summaryHtml = `
      <summary class="thinking-summary">
        <div class="thinking-summary-left">
          <span class="thinking-summary-icon">🧠</span>
          <span class="thinking-summary-title">Thinking Process</span>
          <span class="thinking-summary-count">${wordCount} words</span>
        </div>
        <span class="thinking-summary-chevron">▾</span>
      </summary>
    `;
    const html = `<details class="thinking-details"${attrs}>${summaryHtml}<div class="thinking-stream-content">${renderedBody}</div></details>`;
    const token = `__DETAILS_BLOCK_${detailsBlocks.length}__`;
    detailsBlocks.push(html);
    return token;
  });

  // Handle any other details blocks
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

  // 7. Markdown Headings (h5 through h1)
  text = text.replace(/^##### (.*$)/gim, "<h5>$1</h5>");
  text = text.replace(/^#### (.*$)/gim, "<h4>$1</h4>");
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
      if (p.startsWith("<h1") || p.startsWith("<h2") || p.startsWith("<h3") || p.startsWith("<h4") || p.startsWith("<h5")) return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    })
    .filter(Boolean)
    .join("");

  // 11. Restore tokens in reverse order safely without $ pattern corruption
  displayMaths.forEach((mathHtml, i) => {
    rendered = rendered.replace(new RegExp(`<p>__DISPLAY_MATH_${i}__</p>|__DISPLAY_MATH_${i}__`, "g"), () => mathHtml);
  });

  inlineMaths.forEach((mathHtml, i) => {
    rendered = rendered.replace(new RegExp(`__INLINE_MATH_${i}__`, "g"), () => mathHtml);
  });

  inlineCodes.forEach((codeHtml, i) => {
    rendered = rendered.replace(new RegExp(`__INLINE_CODE_${i}__`, "g"), () => codeHtml);
  });

  codeBlocks.forEach((blockHtml, i) => {
    rendered = rendered.replace(new RegExp(`<p>__CODE_BLOCK_${i}__</p>|__CODE_BLOCK_${i}__`, "g"), () => blockHtml);
  });

  detailsBlocks.forEach((detailsHtml, i) => {
    rendered = rendered.replace(new RegExp(`<p>__DETAILS_BLOCK_${i}__</p>|__DETAILS_BLOCK_${i}__`, "g"), () => detailsHtml);
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
    showToast("Code copied to clipboard", "success", 2000);
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove("copied");
    }, 2000);
  }).catch(() => {
    btn.textContent = "Error";
    showToast("Failed to copy code", "error", 2000);
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
