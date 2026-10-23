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

// Prompt Management & Modal Elements
const addPromptBtn = document.getElementById("add-prompt-btn");
const resetPromptsBtn = document.getElementById("reset-prompts-btn");
const promptModal = document.getElementById("prompt-modal");
const promptModalBackdrop = document.getElementById("prompt-modal-backdrop");
const closePromptModalBtn = document.getElementById("close-prompt-modal-btn");
const cancelPromptBtn = document.getElementById("cancel-prompt-btn");
const promptForm = document.getElementById("prompt-form");
const promptFormId = document.getElementById("prompt-form-id");
const promptFormTitle = document.getElementById("prompt-form-title");
const promptFormDomain = document.getElementById("prompt-form-domain");
const promptFormText = document.getElementById("prompt-form-text");
const promptModalTitle = document.getElementById("prompt-modal-title");
const promptModalSubtitle = document.getElementById("prompt-modal-subtitle");
const savePromptBtnText = document.getElementById("save-prompt-btn-text");

// Confirm Modal Elements
const confirmModal = document.getElementById("confirm-modal");
const confirmModalBackdrop = document.getElementById("confirm-modal-backdrop");
const confirmModalTitle = document.getElementById("confirm-modal-title");
const confirmModalDesc = document.getElementById("confirm-modal-desc");
const confirmModalCancel = document.getElementById("confirm-modal-cancel");
const confirmModalOk = document.getElementById("confirm-modal-ok");
const confirmModalIconBadge = document.getElementById("confirm-modal-icon-badge");
let confirmActionCallback = null;

// Multimodal Image Attachment & Lightbox Elements
let attachedImages = []; // Array of { name, dataUrl }

const attachImageBtn = document.getElementById("attach-image-btn");
const imageUploadInput = document.getElementById("image-upload-input");
const attachedImagesPreview = document.getElementById("attached-images-preview");
const inputContainer = document.getElementById("input-container");
const lightboxModal = document.getElementById("lightbox-modal");
const lightboxBackdrop = document.getElementById("lightbox-backdrop");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCloseBtn = document.getElementById("lightbox-close-btn");

function openLightbox(src) {
  if (!lightboxModal || !lightboxImg) return;
  lightboxImg.src = src;
  lightboxModal.classList.add("active");
  lightboxModal.setAttribute("aria-hidden", "false");
  if (lightboxBackdrop) {
    lightboxBackdrop.classList.add("active");
    lightboxBackdrop.setAttribute("aria-hidden", "false");
  }
}

function closeLightbox() {
  if (!lightboxModal || !lightboxImg) return;
  lightboxModal.classList.remove("active");
  lightboxModal.setAttribute("aria-hidden", "true");
  if (lightboxBackdrop) {
    lightboxBackdrop.classList.remove("active");
    lightboxBackdrop.setAttribute("aria-hidden", "true");
  }
  lightboxImg.src = "";
}

function handleFilesSelected(files) {
  if (!files || files.length === 0) return;
  Array.from(files).forEach((file) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      attachedImages.push({ name: file.name, dataUrl: ev.target.result });
      renderAttachedImagesPreview();
    };
    reader.readAsDataURL(file);
  });
}

function renderAttachedImagesPreview() {
  if (!attachedImagesPreview) return;
  if (attachedImages.length === 0) {
    attachedImagesPreview.style.display = "none";
    attachedImagesPreview.innerHTML = "";
    return;
  }
  attachedImagesPreview.style.display = "flex";
  attachedImagesPreview.innerHTML = attachedImages
    .map((img, idx) => `
      <div class="attached-image-chip">
        <img src="${img.dataUrl}" class="attached-image-thumb" alt="${escapeHtml(img.name)}">
        <span class="attached-image-name">${escapeHtml(img.name)}</span>
        <button type="button" class="attached-image-remove" data-idx="${idx}" aria-label="Remove image">&times;</button>
      </div>
    `)
    .join("");
  attachedImagesPreview.querySelectorAll(".attached-image-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      attachedImages.splice(Number(btn.dataset.idx), 1);
      renderAttachedImagesPreview();
    });
  });
}

attachImageBtn?.addEventListener("click", () => imageUploadInput?.click());
imageUploadInput?.addEventListener("change", (e) => {
  handleFilesSelected(e.target.files);
  imageUploadInput.value = "";
});
promptInput?.addEventListener("paste", (e) => {
  if (e.clipboardData?.files?.length) {
    handleFilesSelected(e.clipboardData.files);
  }
});

if (inputContainer) {
  inputContainer.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    inputContainer.classList.add("drag-over");
  });
  inputContainer.addEventListener("dragleave", (e) => {
    e.preventDefault();
    e.stopPropagation();
    inputContainer.classList.remove("drag-over");
  });
  inputContainer.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    inputContainer.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) {
      handleFilesSelected(e.dataTransfer.files);
    }
  });
}

lightboxCloseBtn?.addEventListener("click", closeLightbox);
lightboxBackdrop?.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && lightboxModal?.classList.contains("active")) {
    closeLightbox();
  }
});

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

// ─── Scroll Management & Auto-Scroll Lock ──────────────────────────────────
let isUserScrolledUp = false;

function scrollToBottom(force = false) {
  if (!messagesContainer) return;
  if (force) {
    isUserScrolledUp = false;
  }
  if (!isUserScrolledUp) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

function initScrollManagement() {
  const topBtn = document.getElementById("scroll-top-btn");
  const bottomBtn = document.getElementById("scroll-bottom-btn");
  if (!messagesContainer) return;

  function handleScroll() {
    const scrollBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
    // If user scrolled up more than 90px from the bottom, lock auto-scroll
    if (scrollBottom > 90) {
      isUserScrolledUp = true;
      if (bottomBtn) {
        bottomBtn.style.display = "flex";
        bottomBtn.classList.add("visible");
      }
    } else {
      isUserScrolledUp = false;
      if (bottomBtn) {
        bottomBtn.classList.remove("visible");
        setTimeout(() => { if (!bottomBtn.classList.contains("visible")) bottomBtn.style.display = "none"; }, 250);
      }
    }

    // Scroll to Top button visibility
    if (topBtn) {
      if (messagesContainer.scrollTop > 350) {
        topBtn.style.display = "flex";
        topBtn.classList.add("visible");
      } else {
        topBtn.classList.remove("visible");
        setTimeout(() => { if (!topBtn.classList.contains("visible")) topBtn.style.display = "none"; }, 250);
      }
    }
  }

  messagesContainer.addEventListener("scroll", handleScroll, { passive: true });

  topBtn?.addEventListener("click", () => {
    messagesContainer.scrollTo({ top: 0, behavior: "smooth" });
  });

  bottomBtn?.addEventListener("click", () => {
    isUserScrolledUp = false;
    messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: "smooth" });
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

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  initThemeToggle();
  initSettingsDrawer();
  initKeyboardShortcuts();
  initConnectionCheck();
  initScrollManagement();
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
  // Legacy evaluation button removed
  if (btnModeZeroShot) {
    btnModeZeroShot.addEventListener("click", () => setEngineMode("zero_shot"));
  }
  if (btnModeAgenticRag) {
    btnModeAgenticRag.addEventListener("click", () => setEngineMode("agentic"));
  }

  // Benchmark prompt management listeners
  if (addPromptBtn) {
    addPromptBtn.addEventListener("click", () => openPromptModal("create"));
  }
  if (resetPromptsBtn) {
    resetPromptsBtn.addEventListener("click", confirmResetPrompts);
  }
  if (closePromptModalBtn) {
    closePromptModalBtn.addEventListener("click", closePromptModal);
  }
  if (cancelPromptBtn) {
    cancelPromptBtn.addEventListener("click", closePromptModal);
  }
  if (promptModalBackdrop) {
    promptModalBackdrop.addEventListener("click", closePromptModal);
  }
  if (promptForm) {
    promptForm.addEventListener("submit", handlePromptFormSubmit);
  }
  if (confirmModalCancel) {
    confirmModalCancel.addEventListener("click", closeConfirmModal);
  }
  if (confirmModalBackdrop) {
    confirmModalBackdrop.addEventListener("click", closeConfirmModal);
  }
  if (confirmModalOk) {
    confirmModalOk.addEventListener("click", handleConfirmModalOk);
  }

  // Global escape key listener to dismiss open modals or drawers
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (promptModal && promptModal.classList.contains("active")) {
        closePromptModal();
      } else if (confirmModal && confirmModal.classList.contains("active")) {
        closeConfirmModal();
      }
    }
  });
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
      if (benchmarkPrompts.length === 0) {
        benchmarkList.innerHTML = `<div style="font-size:0.75rem; color:var(--text-muted); padding:10px 4px; text-align:center;">No prompts saved. Click + to add one.</div>`;
      } else {
        benchmarkPrompts.forEach((bp) => {
          const item = document.createElement("div");
          item.className = "benchmark-item";
          item.setAttribute("data-prompt-id", bp.id);
          item.innerHTML = `
            <div class="benchmark-header">
              <span class="benchmark-domain">${escapeHtml(bp.domain)}</span>
              <div class="benchmark-item-actions">
                <button class="btn-prompt-action edit-prompt-btn" title="Edit prompt: ${escapeHtml(bp.title)}" aria-label="Edit prompt">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                  </svg>
                </button>
                <button class="btn-prompt-action delete-prompt-btn" title="Delete prompt: ${escapeHtml(bp.title)}" aria-label="Delete prompt">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  </svg>
                </button>
              </div>
            </div>
            <div class="benchmark-title">${escapeHtml(bp.title)}</div>
          `;

          item.addEventListener("click", (e) => {
            if (e.target.closest(".btn-prompt-action")) return;
            selectBenchmarkPrompt(bp);
          });

          const editBtn = item.querySelector(".edit-prompt-btn");
          if (editBtn) {
            editBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              openPromptModal("edit", bp);
            });
          }

          const delBtn = item.querySelector(".delete-prompt-btn");
          if (delBtn) {
            delBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              confirmDeletePrompt(bp);
            });
          }

          benchmarkList.appendChild(item);
        });
      }
    }

    // Wire up welcome hero cards
    document.querySelectorAll(".domain-card").forEach((card) => {
      card.onclick = () => {
        const id = card.getAttribute("data-prompt-id");
        const found = benchmarkPrompts.find((p) => p.id === id);
        if (found) selectBenchmarkPrompt(found);
      };
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

// ─── Benchmark Prompt Modal Operations ─────────────────────────────────────────

function openPromptModal(mode, bp = null) {
  if (!promptModal || !promptModalBackdrop) return;
  if (mode === "edit" && bp) {
    if (promptModalTitle) promptModalTitle.textContent = "Edit Benchmark Prompt";
    if (promptModalSubtitle) promptModalSubtitle.textContent = "Modify instructions or domain category";
    if (savePromptBtnText) savePromptBtnText.textContent = "Save Changes";
    if (promptFormId) promptFormId.value = bp.id;
    if (promptFormTitle) promptFormTitle.value = bp.title;
    if (promptFormDomain) promptFormDomain.value = bp.domain;
    if (promptFormText) promptFormText.value = bp.prompt;
  } else {
    if (promptModalTitle) promptModalTitle.textContent = "Create Benchmark Prompt";
    if (promptModalSubtitle) promptModalSubtitle.textContent = "Add a new benchmark prompt for model evaluation";
    if (savePromptBtnText) savePromptBtnText.textContent = "Create Prompt";
    if (promptFormId) promptFormId.value = "";
    if (promptFormTitle) promptFormTitle.value = "";
    if (promptFormDomain) promptFormDomain.value = "Custom";
    if (promptFormText) promptFormText.value = "";
  }
  promptModal.classList.add("active");
  promptModalBackdrop.classList.add("active");
  promptModal.setAttribute("aria-hidden", "false");
  promptModalBackdrop.setAttribute("aria-hidden", "false");
  setTimeout(() => promptFormTitle?.focus(), 60);
}

function closePromptModal() {
  if (!promptModal || !promptModalBackdrop) return;
  promptModal.classList.remove("active");
  promptModalBackdrop.classList.remove("active");
  promptModal.setAttribute("aria-hidden", "true");
  promptModalBackdrop.setAttribute("aria-hidden", "true");
}

async function handlePromptFormSubmit(e) {
  e.preventDefault();
  const id = promptFormId?.value?.trim();
  const title = promptFormTitle?.value?.trim();
  const domain = promptFormDomain?.value?.trim();
  const prompt = promptFormText?.value?.trim();

  if (!title || !domain || !prompt) {
    showToast("Please fill in all prompt fields", "warning", 3000);
    return;
  }

  try {
    if (id) {
      // Update existing prompt
      const res = await fetch(`/api/benchmark_prompts/${encodeURIComponent(id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, domain, prompt })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to update prompt");
      }
      showToast("Prompt updated successfully", "success", 2500);
    } else {
      // Create new prompt
      const res = await fetch("/api/benchmark_prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, domain, prompt })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to create prompt");
      }
      showToast("Prompt created successfully", "success", 2500);
    }
    closePromptModal();
    await loadBenchmarkPrompts();
  } catch (err) {
    console.error("Error saving prompt:", err);
    showToast(err.message || "Error saving prompt", "error", 4000);
  }
}

// ─── Confirmation Modal Helper ────────────────────────────────────────────────

function showConfirmModal({ title, message, okText = "Confirm", isDanger = false, onConfirm }) {
  if (!confirmModal || !confirmModalBackdrop) return;
  if (confirmModalTitle) confirmModalTitle.textContent = title;
  if (confirmModalDesc) confirmModalDesc.textContent = message;
  if (confirmModalOk) {
    confirmModalOk.textContent = okText;
    if (isDanger) {
      confirmModalOk.className = "btn btn-danger";
      if (confirmModalIconBadge) confirmModalIconBadge.className = "modal-icon-badge danger";
    } else {
      confirmModalOk.className = "btn btn-primary";
      if (confirmModalIconBadge) confirmModalIconBadge.className = "modal-icon-badge";
    }
  }
  confirmActionCallback = onConfirm;
  confirmModal.classList.add("active");
  confirmModalBackdrop.classList.add("active");
  confirmModal.setAttribute("aria-hidden", "false");
  confirmModalBackdrop.setAttribute("aria-hidden", "false");
}

function closeConfirmModal() {
  if (!confirmModal || !confirmModalBackdrop) return;
  confirmModal.classList.remove("active");
  confirmModalBackdrop.classList.remove("active");
  confirmModal.setAttribute("aria-hidden", "true");
  confirmModalBackdrop.setAttribute("aria-hidden", "true");
  confirmActionCallback = null;
}

function handleConfirmModalOk() {
  if (typeof confirmActionCallback === "function") {
    confirmActionCallback();
  }
  closeConfirmModal();
}

function confirmDeletePrompt(bp) {
  showConfirmModal({
    title: "Delete Prompt?",
    message: `Are you sure you want to delete "${bp.title}" (${bp.domain})? This benchmark prompt will be permanently removed.`,
    okText: "Delete Prompt",
    isDanger: true,
    onConfirm: async () => {
      try {
        const res = await fetch(`/api/benchmark_prompts/${encodeURIComponent(bp.id)}`, {
          method: "DELETE"
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Failed to delete prompt");
        }
        showToast("Prompt deleted", "success", 2000);
        await loadBenchmarkPrompts();
      } catch (err) {
        console.error("Error deleting prompt:", err);
        showToast(err.message || "Failed to delete prompt", "error", 3000);
      }
    }
  });
}

function confirmResetPrompts() {
  showConfirmModal({
    title: "Reset to Default Prompts?",
    message: "This will revert the prompt list back to the default benchmark suite (IrO2 interfacial analysis, DFT cutoffs, LAMMPS DeePMD, dataset layout, and surface kinetics). Custom or edited prompts will be reset.",
    okText: "Reset Defaults",
    isDanger: false,
    onConfirm: async () => {
      try {
        const res = await fetch("/api/benchmark_prompts/reset", { method: "POST" });
        if (!res.ok) throw new Error("Failed to reset prompts");
        showToast("Benchmark prompts reset to defaults", "info", 2500);
        await loadBenchmarkPrompts();
      } catch (err) {
        console.error("Error resetting prompts:", err);
        showToast("Failed to reset prompts", "error", 3000);
      }
    }
  });
}

// =============================================================================
// Gemini Code Evaluation Engine
// =============================================================================

function extractCodeFromMessage(text) {
  if (!text) return "";
  // Check for fenced python code blocks first
  const pyMatch = text.match(/```(?:python|py)\s*?\n([\s\S]*?)```/i);
  if (pyMatch && pyMatch[1].trim()) return pyMatch[1].trim();

  // Check for any fenced block that contains code tokens
  const allBlocks = Array.from(text.matchAll(/```[a-zA-Z0-9_-]*\s*?\n([\s\S]*?)```/g));
  for (const match of allBlocks) {
    const body = match[1].trim();
    if (body && (body.includes("import ") || body.includes("def ") || body.includes("class ") || body.includes("plt.") || body.includes("np."))) {
      return body;
    }
  }

  // Any non-empty code block
  const anyBlock = text.match(/```[a-zA-Z0-9_-]*\s*?\n([\s\S]*?)```/);
  if (anyBlock && anyBlock[1].trim()) return anyBlock[1].trim();

  return "";
}

function extractDockerOutputFromMessage(text) {
  if (!text) return "";
  const match = text.match(/### 🐳 Secure Docker Sandbox Execution[\s\S]*?(?:---|\Z)/i);
  if (match) return match[0].trim();
  const outMatch = text.match(/\*\*Container Output\*\*:\s*```[a-zA-Z]*\s*([\s\S]*?)```/i);
  if (outMatch) return outMatch[1].trim();
  return "";
}

function getLastUserPrompt() {
  const userMessages = document.querySelectorAll(".message-item.user .message-bubble");
  if (userMessages.length > 0) {
    return userMessages[userMessages.length - 1].textContent.trim();
  }
  return "";
}

function createGeminiReportCardElement(evalData, codeText = "", userPrompt = "") {
  const card = document.createElement("div");
  card.className = "gemini-eval-card";

  const score = evalData.score !== undefined ? Math.round(evalData.score) : 0;
  const grade = evalData.grade || (score >= 80 ? "A" : (score >= 60 ? "B" : (score >= 40 ? "C" : (score >= 20 ? "D" : "F"))));
  const verdict = evalData.verdict || (score < 50 ? "FAIL" : (score < 80 ? "BORDERLINE" : "PASS"));
  const modelName = evalData.model || evalData.evaluator_model || "gemini-2.5-flash";
  const latency = evalData.latency_seconds ? `${Number(evalData.latency_seconds).toFixed(1)}s` : "";

  const scoreClass = score >= 80 ? "score-pass" : (score >= 50 ? "score-borderline" : "score-fail");
  const verdictClass = score >= 80 ? "verdict-pass" : (score >= 50 ? "verdict-borderline" : "verdict-fail");

  const catScores = evalData.category_scores || {};
  const execScore = catScores.execution_integrity !== undefined ? catScores.execution_integrity : (score < 25 ? Math.min(score, 5) : Math.round(score * 0.25));
  const sciScore = catScores.scientific_accuracy !== undefined ? catScores.scientific_accuracy : (score < 25 ? Math.min(score, 5) : Math.round(score * 0.35));
  const taskScore = catScores.task_compliance !== undefined ? catScores.task_compliance : (score < 25 ? Math.min(score, 5) : Math.round(score * 0.25));
  const qualScore = catScores.code_quality !== undefined ? catScores.code_quality : (score < 25 ? Math.min(score, 5) : Math.round(score * 0.15));

  const criticalIssues = evalData.critical_issues || [];
  const strengths = evalData.strengths || [];
  const recommendations = evalData.recommendations || [];

  let html = `
    <div class="gemini-card-header">
      <div class="gemini-card-title-group">
        <div class="gemini-eval-badge-icon" aria-hidden="true">⚖️</div>
        <div class="gemini-eval-title-text">
          <h4>Gemini Code Audit</h4>
          <div class="gemini-eval-meta">
            <span>Model: <strong>${escapeHtml(modelName)}</strong></span>
            ${latency ? `<span>• ⚡ ${latency}</span>` : ""}
            <span>• Strict 100-pt Rubric</span>
          </div>
        </div>
      </div>
      <div class="gemini-score-pill-container">
        <div class="gemini-score-badge ${scoreClass}">
          <span class="score-num">${score}</span>
          <span class="score-denom">/ 100</span>
        </div>
        <span class="gemini-verdict-tag ${verdictClass}">${escapeHtml(verdict)} • Grade ${escapeHtml(grade)}</span>
      </div>
    </div>

    <div class="gemini-summary-box">
      <p><strong>Static code review</strong> • Execution: ${escapeHtml(evalData.execution_status || "Untested")}. Supplied logs are unverified.</p>
      ${evalData.score_cap != null ? `<p>Blocking defect: score capped at ${evalData.score_cap}/100 (category subtotal: ${evalData.raw_score}).</p>` : ""}
      ${evalData.persistence_error ? `<p>${escapeHtml(evalData.persistence_error)}</p>` : ""}
      <strong>Executive Critique:</strong> ${escapeHtml(evalData.summary || "Evaluation completed.")}
    </div>

    <div class="gemini-category-grid">
      <div class="gemini-cat-card">
        <span class="gemini-cat-label">Execution</span>
        <span class="gemini-cat-score">${execScore} <span style="font-size:0.75rem; color:var(--text-muted);">/ 25</span></span>
        <div class="gemini-cat-bar">
          <div class="gemini-cat-fill" style="width:${Math.min(100, (execScore / 25) * 100)}%; background:${execScore < 10 ? '#ef4444' : (execScore < 18 ? '#f59e0b' : '#10b981')};"></div>
        </div>
      </div>
      <div class="gemini-cat-card">
        <span class="gemini-cat-label">Scientific Rigor</span>
        <span class="gemini-cat-score">${sciScore} <span style="font-size:0.75rem; color:var(--text-muted);">/ 35</span></span>
        <div class="gemini-cat-bar">
          <div class="gemini-cat-fill" style="width:${Math.min(100, (sciScore / 35) * 100)}%; background:${sciScore < 15 ? '#ef4444' : (sciScore < 25 ? '#f59e0b' : '#10b981')};"></div>
        </div>
      </div>
      <div class="gemini-cat-card">
        <span class="gemini-cat-label">Task Compliance</span>
        <span class="gemini-cat-score">${taskScore} <span style="font-size:0.75rem; color:var(--text-muted);">/ 25</span></span>
        <div class="gemini-cat-bar">
          <div class="gemini-cat-fill" style="width:${Math.min(100, (taskScore / 25) * 100)}%; background:${taskScore < 10 ? '#ef4444' : (taskScore < 18 ? '#f59e0b' : '#10b981')};"></div>
        </div>
      </div>
      <div class="gemini-cat-card">
        <span class="gemini-cat-label">Code Quality</span>
        <span class="gemini-cat-score">${qualScore} <span style="font-size:0.75rem; color:var(--text-muted);">/ 15</span></span>
        <div class="gemini-cat-bar">
          <div class="gemini-cat-fill" style="width:${Math.min(100, (qualScore / 15) * 100)}%; background:${qualScore < 6 ? '#ef4444' : (qualScore < 11 ? '#f59e0b' : '#10b981')};"></div>
        </div>
      </div>
    </div>
  `;

  if (criticalIssues.length > 0) {
    html += `
      <div class="gemini-issues-section">
        <div class="gemini-section-header header-danger">❌ Critical Defects & Deductions (${criticalIssues.length})</div>
        <ul class="gemini-detail-list gemini-issues-list">
          ${criticalIssues.map(issue => `<li><span>•</span><span>${escapeHtml(issue)}</span></li>`).join("")}
        </ul>
      </div>
    `;
  }

  if (strengths.length > 0) {
    html += `
      <div class="gemini-strengths-section">
        <div class="gemini-section-header header-success">✔️ Strengths Identified in Review (${strengths.length})</div>
        <ul class="gemini-detail-list gemini-strengths-list">
          ${strengths.map(s => `<li><span>•</span><span>${escapeHtml(s)}</span></li>`).join("")}
        </ul>
      </div>
    `;
  }

  if (recommendations.length > 0) {
    html += `
      <div class="gemini-recs-section">
        <div class="gemini-section-header header-info">💡 Actionable Engineering Fixes (${recommendations.length})</div>
        <ul class="gemini-detail-list gemini-recs-list">
          ${recommendations.map(r => `<li><span>•</span><span>${escapeHtml(r)}</span></li>`).join("")}
        </ul>
      </div>
    `;
  }

  html += `
    <div class="gemini-card-footer">
      <span>${evalData.evaluation_id ? `Audit Run #${evalData.evaluation_id}` : 'Instant Assessment'} • Evidence-based static review</span>
      <span>${evalData.execution_status ? `Execution: ${escapeHtml(evalData.execution_status)}` : 'Grounded on Task'}</span>
    </div>
  `;

  card.innerHTML = html;
  return card;
}

function attachGeminiEvaluateButton(bubble, content, promptText = "", sessionId = null) {
  if (!bubble) return;
  if (bubble.querySelector(".gemini-eval-action-container")) return;

  const extractedCode = extractCodeFromMessage(content);
  if (!extractedCode) return;

  const dockerOut = extractDockerOutputFromMessage(content);

  const container = document.createElement("div");
  container.className = "gemini-eval-action-container";

  container.innerHTML = `
    <div class="gemini-eval-btn-wrapper">
      <button class="btn-gemini-eval" type="button" title="Evaluate script with Gemini (Strict 100-pt rubric)">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true">
          <path d="M12 3v18M18 6l-6-3-6 3M18 18l-6 3-6-3M6 10l-4 2 4 2M18 10l4 2-4 2" />
        </svg>
        <span>Evaluate Script (Gemini)</span>
      </button>
      <span class="gemini-eval-hint">Strict 100-pt audit • Static review</span>
    </div>
    <div class="gemini-eval-result-host" style="display:none;"></div>
  `;

  const btn = container.querySelector(".btn-gemini-eval");
  const resultHost = container.querySelector(".gemini-eval-result-host");

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.classList.add("is-loading");
    const btnSpan = btn.querySelector("span");
    btnSpan.textContent = "Evaluating with Gemini...";

    try {
      const taskPrompt = promptText || getLastUserPrompt();
      const res = await fetch("/api/evaluate/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: content,
          prompt: taskPrompt,
          execution_output: dockerOut,
          session_id: sessionId || currentSessionId
        })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Evaluation failed." }));
        throw new Error(err.detail || `Server returned ${res.status}`);
      }

      const evalData = await res.json();
      resultHost.innerHTML = "";
      resultHost.appendChild(createGeminiReportCardElement(evalData, extractedCode, taskPrompt));
      resultHost.style.display = "block";

      btn.classList.remove("is-loading");
      btn.disabled = false;
      btnSpan.textContent = "🔄 Re-evaluate with Gemini";

      await loadEvaluationHistory();
      scrollToBottom(false);
    } catch (err) {
      btn.classList.remove("is-loading");
      btn.disabled = false;
      btnSpan.textContent = "⚠️ Retry Gemini Evaluation";
      showToast(`Gemini evaluation error: ${err.message}`, "error", 4500);
    }
  });

  const metaDiv = bubble.querySelector(".message-meta");
  if (metaDiv) {
    bubble.insertBefore(container, metaDiv);
  } else {
    bubble.appendChild(container);
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
      evalHistoryList.innerHTML = `<div style="font-size:0.75rem; color:var(--text-muted); padding:6px 0;">No code evaluations yet</div>`;
      return;
    }

    evals.forEach((ev) => {
      const item = document.createElement("div");
      item.className = "eval-history-item";

      const dateStr = ev.created_at ? formatShortDate(ev.created_at) : "Recent";
      const score = Math.round(ev.score !== undefined ? ev.score : 0);
      const grade = ev.grade || (score >= 80 ? "A" : (score >= 60 ? "B" : (score >= 40 ? "C" : (score >= 20 ? "D" : "F"))));
      const gradeClass = `grade-${grade}`;
      const verdict = ev.verdict ? ` • ${ev.verdict}` : "";

      item.innerHTML = `
        <div class="eval-history-left">
          <span class="eval-history-title">⚖️ Audit #${ev.id} • ${score}/100</span>
          <span class="eval-history-time">${dateStr}${verdict}</span>
        </div>
        <span class="eval-history-badge ${gradeClass}">${grade}</span>
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

// Render the Gemini audit report card in thread
function renderEvaluationReport(ev) {
  const container = document.createElement("div");
  container.className = "message-item assistant";

  const card = createGeminiReportCardElement(ev, ev.generated_code, ev.prompt);
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
        tool_calls_json: msg.tool_calls_json,
        images_json: msg.images_json
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
  if (!text && attachedImages.length === 0) return;
  const imagesToSend = attachedImages.map((i) => i.dataUrl);

  promptInput.value = "";
  promptInput.style.height = "auto";
  localStorage.removeItem("scicompute_draft");
  welcomeHero.style.display = "none";

  appendMessage("user", text, { images: imagesToSend });
  attachedImages = [];
  renderAttachedImagesPreview();
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
  scrollToBottom(true);

  const timerBadge = bubble.querySelector(".live-timer-badge");
  const agentTraceContainer = bubble.querySelector(".agent-trace-container");
  const thinkingDetails = bubble.querySelector(".thinking-details");
  const thinkingWordCount = bubble.querySelector(".thinking-word-count");
  const thinkingContent = bubble.querySelector(".thinking-stream-content");
  const contentContainer = bubble.querySelector(".content-stream-container");
  const cursor = bubble.querySelector(".streaming-cursor");

  contentContainer.innerHTML = getPrefillPlaceholderHtml(isAgentic);
  let hasReceivedFirstToken = false;

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
        mode: isAgentic ? "agentic" : "single_shot",
        ...(imagesToSend.length > 0 ? { images: imagesToSend } : {})
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
            if (!hasReceivedFirstToken) {
              hasReceivedFirstToken = true;
              contentContainer.innerHTML = "";
            }
            renderAgentTraceStep(agentTraceContainer, event);
            scrollToBottom(false);
          } else if (event.type === "thinking") {
            if (!hasReceivedFirstToken) {
              hasReceivedFirstToken = true;
              contentContainer.innerHTML = "";
            }
            thinkingDetails.style.display = "block";
            thinkingAccumulated += event.delta;
            thinkingContent.innerHTML = renderThinkingBody(thinkingAccumulated);
            const words = thinkingAccumulated.split(/\s+/).filter(Boolean).length;
            thinkingWordCount.textContent = words;
            scrollToBottom(false);
          } else if (event.type === "content") {
            if (!hasReceivedFirstToken) {
              hasReceivedFirstToken = true;
              contentContainer.innerHTML = "";
            }
            contentAccumulated += event.delta;
            const isCodeUnclosed = ((contentAccumulated.match(/```/g) || []).length % 2 === 1);
            if (cursor) cursor.style.display = isCodeUnclosed ? "none" : "inline-block";
            contentContainer.innerHTML = renderMarkdown(contentAccumulated, true);
            scrollToBottom(false);
          } else if (event.type === "artifacts" && Array.isArray(event.artifacts)) {
            const gallery = document.createElement("div");
            gallery.className = "message-images-grid";
            event.artifacts.forEach((fileName) => {
              if (/\.(png|jpg|jpeg|svg|webp)$/i.test(fileName)) {
                const wrap = document.createElement("div");
                wrap.className = "message-image-item";
                const imgSrc = `/public/${encodeURIComponent(fileName)}?t=${Date.now()}`;
                wrap.innerHTML = `<img src="${imgSrc}" alt="${escapeHtml(fileName)}" title="${escapeHtml(fileName)}">`;
                wrap.addEventListener("click", () => openLightbox(imgSrc));
                gallery.appendChild(wrap);
              }
            });
            if (gallery.children.length > 0) {
              bubble.insertBefore(gallery, cursor);
              scrollToBottom(false);
            }
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

            attachGeminiEvaluateButton(bubble, contentAccumulated, text, currentSessionId);

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
            scrollToBottom(false);
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

  // If this message has attached images (either live array or stored images_json), render image gallery
  let imgsToRender = [];
  if (meta.images && Array.isArray(meta.images)) {
    imgsToRender = meta.images;
  } else if (meta.images_json) {
    try {
      const parsed = typeof meta.images_json === "string" ? JSON.parse(meta.images_json) : meta.images_json;
      if (Array.isArray(parsed)) imgsToRender = parsed;
    } catch (e) {
      console.warn("Error parsing images_json:", e);
    }
  }

  if (imgsToRender.length > 0) {
    const gallery = document.createElement("div");
    gallery.className = "message-images-grid";
    imgsToRender.forEach((imgSrc) => {
      const wrap = document.createElement("div");
      wrap.className = "message-image-item";
      wrap.innerHTML = `<img src="${imgSrc}" alt="Attached image" loading="lazy">`;
      wrap.addEventListener("click", () => openLightbox(imgSrc));
      gallery.appendChild(wrap);
    });
    bubble.appendChild(gallery);
  }

  // Attach lightbox click handlers to all image items or markdown images
  bubble.querySelectorAll(".message-image-item img, .markdown-body img").forEach((img) => {
    img.style.cursor = "pointer";
    img.addEventListener("click", () => openLightbox(img.src));
  });

  if (role === "assistant") {
    attachGeminiEvaluateButton(bubble, content, meta.prompt || getLastUserPrompt(), currentSessionId);
  }

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

// Helper to generate an informative prefill status placeholder while model ingests prompt tokens
function getPrefillPlaceholderHtml(isAgentic = false) {
  const deviceBadge = document.getElementById("drawer-compute-device")?.textContent?.trim() || "";
  const isCpu = deviceBadge.toLowerCase().includes("cpu");
  const deviceText = isCpu ? "CPU Mode" : "GPU Mode";
  const desc = isCpu
    ? "Ingesting prompt context on CPU. In CPU-only mode, ingesting a large context prompt (4,000+ tokens) takes 1–2 minutes before the first token streams."
    : (isAgentic ? "Analyzing query, retrieving knowledge from ChromaDB/Kùzu, and assembling context..." : "Ingesting prompt context into AMD Radeon GPU memory...");

  return `
    <div class="prefill-status-container" id="prefill-status-indicator">
      <div class="prefill-spinner"></div>
      <div class="prefill-text">
        <div class="prefill-title">⚡ ${deviceText} • Ingesting Context...</div>
        <div class="prefill-desc">${desc}</div>
      </div>
    </div>
  `;
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
