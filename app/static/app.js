/**
 * English-to-Spanish MarianMT NMT Microservice Frontend Application
 * Handles live dual-pane translation, debouncing, keyboard shortcuts,
 * dynamic batching telemetry, toast notices, and service health probes.
 */

(() => {
  // DOM Elements
  const sourceText = document.getElementById('sourceText');
  const targetText = document.getElementById('targetText');
  const targetPlaceholder = document.getElementById('targetPlaceholder');
  const targetLoading = document.getElementById('targetLoading');
  const targetError = document.getElementById('targetError');
  const errorMessage = document.getElementById('errorMessage');
  const errorRetryBtn = document.getElementById('errorRetryBtn');

  const charCounter = document.getElementById('charCounter');
  const clearBtn = document.getElementById('clearBtn');
  const copyBtn = document.getElementById('copyBtn');
  const copyIcon = document.getElementById('copyIcon');
  const checkIcon = document.getElementById('checkIcon');
  const copyBtnText = document.getElementById('copyBtnText');
  const translateBtn = document.getElementById('translateBtn');
  const beamSizeSelect = document.getElementById('beamSizeSelect');
  const swapDirectionBtn = document.getElementById('swapDirectionBtn');

  const latencyBadge = document.getElementById('latencyBadge');
  const tokenPill = document.getElementById('tokenPill');
  const modelPill = document.getElementById('modelPill');

  const healthDot = document.getElementById('healthDot');
  const healthText = document.getElementById('healthText');
  const footerHealthDot = document.getElementById('footerHealthDot');
  const footerHealthText = document.getElementById('footerHealthText');
  const footerUptime = document.getElementById('footerUptime');

  const toast = document.getElementById('toast');
  const toastText = document.getElementById('toastText');

  // Configuration & State
  const MAX_CHARS = 1000;
  const DEBOUNCE_DELAY_MS = 400;
  const HEALTH_POLL_INTERVAL_MS = 5000;

  let debounceTimer = null;
  let toastTimer = null;
  let currentAbortController = null;
  let lastTranslatedText = '';
  let lastTranslatedBeam = null;

  // Initialize
  function init() {
    setupPlatformShortcuts();
    setupEventListeners();
    updateCharCount();
    checkHealth();
    setInterval(checkHealth, HEALTH_POLL_INTERVAL_MS);
    sourceText.focus();
  }

  function setupPlatformShortcuts() {
    const isMac = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
    const shortcutKbd = translateBtn ? translateBtn.querySelector('kbd') : null;
    if (shortcutKbd && isMac) {
      shortcutKbd.textContent = '⌘+↵';
    }
  }

  function setupEventListeners() {
    // Input handling with debounce
    sourceText.addEventListener('input', handleSourceInput);

    // Keyboard shortcut: Ctrl+Enter or Cmd+Enter
    sourceText.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        clearTimeout(debounceTimer);
        triggerTranslation(true);
      }
    });

    // Clear button
    clearBtn.addEventListener('click', handleClear);

    // Copy button
    copyBtn.addEventListener('click', handleCopy);

    // Translate button
    translateBtn.addEventListener('click', () => {
      clearTimeout(debounceTimer);
      triggerTranslation(true);
    });

    // Beam size change
    beamSizeSelect.addEventListener('change', () => {
      const text = sourceText.value.trim();
      const currentBeam = parseInt(beamSizeSelect.value, 10);
      if (text && (text !== lastTranslatedText || currentBeam !== lastTranslatedBeam)) {
        triggerTranslation(true);
      }
    });

    // Swap Direction button (informative notification)
    if (swapDirectionBtn) {
      swapDirectionBtn.addEventListener('click', () => {
        showToast('Direction fixed: MarianMT (opus-mt-en-es) is a unidirectional English → Spanish model.');
      });
    }

    // Sample text pills
    document.querySelectorAll('.sample-pill').forEach((pill) => {
      pill.addEventListener('click', () => {
        const text = pill.getAttribute('data-text') || pill.textContent.trim();
        sourceText.value = text;
        updateCharCount();
        sourceText.focus();
        triggerTranslation(true);
      });
    });

    // Error retry button
    if (errorRetryBtn) {
      errorRetryBtn.addEventListener('click', () => {
        triggerTranslation(true);
      });
    }
  }

  function showToast(message, durationMs = 3000) {
    if (!toast || !toastText) return;
    toastText.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove('show');
    }, durationMs);
  }

  function handleSourceInput() {
    updateCharCount();

    const text = sourceText.value.trim();
    if (!text) {
      if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
      }
      clearTimeout(debounceTimer);
      resetTarget();
      return;
    }

    if (text.length > MAX_CHARS) {
      showError(`Input exceeds maximum allowed length of ${MAX_CHARS} characters.`);
      return;
    }

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      triggerTranslation(false);
    }, DEBOUNCE_DELAY_MS);
  }

  function updateCharCount() {
    const len = sourceText.value.length;
    charCounter.textContent = `${len} / ${MAX_CHARS}`;

    if (len >= MAX_CHARS) {
      charCounter.classList.add('text-red-400');
      charCounter.classList.remove('text-zinc-500', 'text-amber-400');
    } else if (len > MAX_CHARS * 0.9) {
      charCounter.classList.add('text-amber-400');
      charCounter.classList.remove('text-zinc-500', 'text-red-400');
    } else {
      charCounter.classList.add('text-zinc-500');
      charCounter.classList.remove('text-amber-400', 'text-red-400');
    }

    clearBtn.disabled = len === 0;
  }

  function handleClear() {
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
    }
    clearTimeout(debounceTimer);
    sourceText.value = '';
    updateCharCount();
    resetTarget();
    sourceText.focus();
  }

  function resetTarget() {
    lastTranslatedText = '';
    lastTranslatedBeam = null;
    targetText.textContent = '';
    targetText.classList.add('hidden');
    targetLoading.classList.add('hidden');
    targetError.classList.add('hidden');
    targetPlaceholder.classList.remove('hidden');

    copyBtn.disabled = true;
    latencyBadge.textContent = '-- ms';
    tokenPill.textContent = '0 tokens';
  }

  async function triggerTranslation(forced = false) {
    const rawText = sourceText.value;
    const text = rawText.trim();
    const beamSize = parseInt(beamSizeSelect.value, 10) || 4;

    if (!text) {
      resetTarget();
      return;
    }

    if (text.length > MAX_CHARS) {
      showError(`Text length exceeds ${MAX_CHARS} characters.`);
      return;
    }

    // Skip redundant translations if already translated with same beam
    if (!forced && text === lastTranslatedText && beamSize === lastTranslatedBeam) {
      return;
    }

    // Cancel in-flight request
    if (currentAbortController) {
      currentAbortController.abort();
    }
    currentAbortController = new AbortController();

    // UI state: loading
    setLoadingState(true);

    const startTime = performance.now();

    try {
      const response = await fetch('/v1/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          beam_size: beamSize,
        }),
        signal: currentAbortController.signal,
      });

      if (!response.ok) {
        let errMessage = `HTTP error ${response.status}`;
        try {
          const errData = await response.json();
          if (Array.isArray(errData.detail) && errData.detail.length > 0) {
            // Cleanly format validation errors without raw JSON brackets
            errMessage = errData.detail
              .map((d) => (d.msg ? d.msg.replace(/^Value error,\s*/i, '') : JSON.stringify(d)))
              .join('; ');
          } else if (typeof errData.detail === 'string') {
            errMessage = errData.detail;
          } else if (errData.detail) {
            errMessage = JSON.stringify(errData.detail);
          }
        } catch (_) {
          // Response was not JSON
        }
        throw new Error(errMessage);
      }

      const data = await response.json();
      const clientElapsedMs = performance.now() - startTime;

      // Stale check: discard response if input was modified or cleared while in-flight
      if (sourceText.value.trim() !== text) {
        return;
      }

      if (data.translations && data.translations.length > 0) {
        const resultItem = data.translations[0];
        displayTranslation(resultItem.target, data.latency_ms || clientElapsedMs, data.model);
        lastTranslatedText = text;
        lastTranslatedBeam = beamSize;
      } else {
        throw new Error('No translation returned by engine.');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // Ignored aborted fetch
        return;
      }
      showError(err.message || 'Translation failed.');
    } finally {
      setLoadingState(false);
      currentAbortController = null;
    }
  }

  function setLoadingState(isLoading) {
    if (isLoading) {
      targetPlaceholder.classList.add('hidden');
      targetError.classList.add('hidden');
      targetLoading.classList.remove('hidden');
      translateBtn.disabled = true;
    } else {
      targetLoading.classList.add('hidden');
      translateBtn.disabled = false;
    }
  }

  function displayTranslation(translatedText, latencyMs, modelName) {
    targetPlaceholder.classList.add('hidden');
    targetError.classList.add('hidden');
    targetLoading.classList.add('hidden');

    targetText.textContent = translatedText;
    targetText.classList.remove('hidden');

    // Latency timing badge
    const roundedLatency = Math.round(latencyMs * 10) / 10;
    latencyBadge.textContent = `${roundedLatency} ms`;

    // Estimate generated tokens with correct grammatical singular/plural
    const words = translatedText.trim().split(/\s+/).filter(Boolean).length;
    const estimatedTokens = Math.max(1, Math.round(words * 1.3));
    tokenPill.textContent = `${estimatedTokens} ${estimatedTokens === 1 ? 'token' : 'tokens'}`;

    if (modelName && modelPill) {
      modelPill.textContent = modelName.replace('Helsinki-NLP/', '');
    }

    copyBtn.disabled = false;
  }

  function showError(msg) {
    targetLoading.classList.add('hidden');
    targetText.classList.add('hidden');
    targetPlaceholder.classList.add('hidden');
    targetError.classList.remove('hidden');

    errorMessage.textContent = msg;
    copyBtn.disabled = true;
    latencyBadge.textContent = '-- ms';
    tokenPill.textContent = '0 tokens';
  }

  async function handleCopy() {
    const textToCopy = targetText.textContent;
    if (!textToCopy) return;

    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(textToCopy);
      } else {
        // Fallback for older browsers
        const tempTextarea = document.createElement('textarea');
        tempTextarea.value = textToCopy;
        tempTextarea.style.position = 'fixed';
        tempTextarea.style.opacity = '0';
        document.body.appendChild(tempTextarea);
        tempTextarea.select();
        document.execCommand('copy');
        document.body.removeChild(tempTextarea);
      }

      // Visual feedback
      copyIcon.classList.add('hidden');
      checkIcon.classList.remove('hidden');
      copyBtnText.textContent = 'Copied!';
      copyBtn.classList.add('border-emerald-500/50', 'text-emerald-400');

      setTimeout(() => {
        copyIcon.classList.remove('hidden');
        checkIcon.classList.add('hidden');
        copyBtnText.textContent = 'Copy';
        copyBtn.classList.remove('border-emerald-500/50', 'text-emerald-400');
      }, 2000);
    } catch (err) {
      console.error('Failed to copy translation:', err);
    }
  }

  // Health and Readiness Polling
  async function checkHealth() {
    try {
      const [healthResp, readyResp] = await Promise.allSettled([
        fetch('/healthz', { cache: 'no-store' }),
        fetch('/readyz', { cache: 'no-store' }),
      ]);

      let isHealthy = false;
      let isReady = false;
      let uptimeSec = 0;
      let readyDetails = '';

      if (healthResp.status === 'fulfilled' && healthResp.value.ok) {
        isHealthy = true;
        const hData = await healthResp.value.json();
        uptimeSec = Math.floor(hData.uptime_seconds || 0);
      }

      if (readyResp.status === 'fulfilled') {
        if (readyResp.value.ok) {
          isReady = true;
        } else {
          try {
            const rData = await readyResp.value.json();
            readyDetails = rData.details || 'Model initializing';
          } catch (_) {
            readyDetails = 'Model initializing';
          }
        }
      }

      updateHealthUI(isHealthy, isReady, uptimeSec, readyDetails);
    } catch (_) {
      updateHealthUI(false, false, 0, 'Connection lost');
    }
  }

  function updateHealthUI(isHealthy, isReady, uptimeSec, details) {
    healthDot.classList.remove('ready', 'degraded', 'offline');

    if (isHealthy && isReady) {
      healthDot.classList.add('ready');
      healthText.textContent = 'Ready';
      healthText.className = 'text-xs font-medium text-emerald-400';
      if (footerHealthDot) footerHealthDot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-500';
      if (footerHealthText) footerHealthText.textContent = 'Active • CTranslate2 INT8 Ready';
    } else if (isHealthy && !isReady) {
      healthDot.classList.add('degraded');
      healthText.textContent = 'Initializing';
      healthText.className = 'text-xs font-medium text-amber-400';
      if (footerHealthDot) footerHealthDot.className = 'w-1.5 h-1.5 rounded-full bg-amber-500';
      if (footerHealthText) footerHealthText.textContent = `Degraded • ${details || 'Model loading'}`;
    } else {
      healthDot.classList.add('offline');
      healthText.textContent = 'Offline';
      healthText.className = 'text-xs font-medium text-red-400';
      if (footerHealthDot) footerHealthDot.className = 'w-1.5 h-1.5 rounded-full bg-red-500';
      if (footerHealthText) footerHealthText.textContent = 'Service Unreachable';
    }

    if (uptimeSec >= 0 && footerUptime) {
      if (uptimeSec < 60) {
        footerUptime.textContent = `Uptime: ${uptimeSec}s`;
      } else {
        const mins = Math.floor(uptimeSec / 60);
        const secs = uptimeSec % 60;
        footerUptime.textContent = `Uptime: ${mins}m ${secs}s`;
      }
    }
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
