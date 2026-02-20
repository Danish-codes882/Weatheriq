/**
 * WeatherIQ — Frontend Application
 * ─────────────────────────────────────────────────────────────
 * Architecture: Module pattern, no frameworks, no bundler required
 * Canvas: Particle system with 3 weather themes
 * Theme: Full CSS custom property driven, smooth transitions
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════
   CONFIG
═══════════════════════════════════════════════════════════════ */

const CONFIG = {
  API_BASE: window.location.origin,
  TIMEOUT_MS: 30000,
  GALLERY_INTERVAL_MS: 5200,
  TOAST_DURATION_MS: 3600,
  QUICK_CITIES: ['London', 'Tokyo', 'Dubai', 'New York', 'Sydney', 'Paris', 'Mumbai', 'Istanbul'],
  LOADER_MESSAGES: [
    'Fetching live conditions…',
    'Running ML analysis…',
    'Generating forecast…',
    'Building theme…',
    'Loading imagery…',
  ],
};

/* ═══════════════════════════════════════════════════════════════
   SVG ICONS LIBRARY
═══════════════════════════════════════════════════════════════ */

const ICONS = {
  sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
  snowflake: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><line x1="12" y1="2" x2="12" y2="22"/><path d="M17 7l-5-5-5 5"/><path d="M17 17l-5 5-5-5"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M7 7l-5 5 5 5"/><path d="M17 7l5 5-5 5"/></svg>`,
  thermometer: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z"/></svg>`,
  cloud: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M18 10h-1.26A8 8 0 109 20h9a5 5 0 000-10z"/></svg>`,
  tshirt: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.38 3.46L16 2l-4 4-4-4-4.38 1.46a1 1 0 00-.62.94v2.2a1 1 0 00.57.9L6 8.5V20a2 2 0 002 2h8a2 2 0 002-2V8.5l2.43-1.1a1 1 0 00.57-.9v-2.2a1 1 0 00-.62-.94z"/></svg>`,
  jacket: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l4 4 4-4 1 18H3L4 2l4 4 4-4z"/><line x1="8" y1="6" x2="8" y2="22"/><line x1="16" y1="6" x2="16" y2="22"/></svg>`,
  shorts: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 3h16l-2 9H6L4 3z"/><path d="M6 12l2 9M18 12l-2 9"/><line x1="8" y1="3" x2="8" y2="12"/><line x1="16" y1="3" x2="16" y2="12"/></svg>`,
  coat: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L8 6l-5 2v13a1 1 0 001 1h4V11l4 2 4-2v11h4a1 1 0 001-1V8l-5-2-4-4z"/></svg>`,
  trendUp: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>`,
  trendDown: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>`,
  trendStable: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
  checkCircle: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="9 12 11 14 15 10"/></svg>`,
  alertCircle: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  cpu: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>`,
};

/* ═══════════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════════ */

const State = {
  loading: false,
  dimmed: false,
  currentTheme: 'normal',
  chart: null,
  gallery: { index: 0, images: [], timer: null },
  particles: { raf: null, list: [] },
};

/* ═══════════════════════════════════════════════════════════════
   UTILITIES
═══════════════════════════════════════════════════════════════ */

const qs = (sel, ctx = document) => ctx.querySelector(sel);
const qsa = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function safe(str) {
  const d = document.createElement('div');
  d.textContent = String(str ?? '');
  return d.innerHTML;
}

function setText(id, val) {
  const el = qs(id);
  if (el) el.textContent = String(val ?? '—');
}

function setHtml(id, html) {
  const el = qs(id);
  if (el) el.innerHTML = html;
}

function round(n, d = 1) {
  return typeof n === 'number' ? +n.toFixed(d) : n;
}

/* ═══════════════════════════════════════════════════════════════
   CANVAS PARTICLE SYSTEM
═══════════════════════════════════════════════════════════════ */

const Particles = (() => {
  let canvas, ctx, W, H;
  const particles = [];
  let raf = null;
  let currentType = 'normal';
  let accentRgb = [0, 212, 168];

  const THEMES = {
    hot:    { type: 'ember',  count: 50, color: [255, 107, 43] },
    cold:   { type: 'snow',   count: 65, color: [41, 170, 255] },
    normal: { type: 'drift',  count: 30, color: [0, 212, 168] },
  };

  function hexToRgb(hex) {
    const n = parseInt(hex.replace('#', ''), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function rng(a, b) { return Math.random() * (b - a) + a; }

  function createParticle(scatter = false) {
    const t = THEMES[currentType] || THEMES.normal;
    const [r, g, b] = t.color;

    if (t.type === 'ember') {
      return {
        x: rng(0, W),
        y: scatter ? rng(0, H) : rng(H, H * 1.25),
        vx: rng(-0.35, 0.35),
        vy: rng(-1.4, -0.45),
        r: rng(1.5, 4.5),
        alpha: rng(0.06, 0.22),
        maxAlpha: rng(0.1, 0.28),
        wobble: rng(0, Math.PI * 2),
        wobbleSpeed: rng(0.025, 0.07),
        rgb: [r, g, b],
        type: 'ember',
      };
    }
    if (t.type === 'snow') {
      return {
        x: rng(0, W),
        y: scatter ? rng(-H * 0.2, H) : rng(-30, -5),
        vx: rng(-0.5, 0.5),
        vy: rng(0.5, 1.9),
        r: rng(1.5, 4),
        alpha: rng(0.35, 0.8),
        drift: rng(0, Math.PI * 2),
        driftSpeed: rng(0.015, 0.04),
        rgb: [r, g, b],
        type: 'snow',
      };
    }
    // drift / orbs
    return {
      x: rng(0, W),
      y: scatter ? rng(0, H) : rng(H * 1.1, H * 1.4),
      vx: rng(-0.12, 0.12),
      vy: rng(-0.35, -0.1),
      r: rng(40, 130),
      alpha: rng(0.015, 0.06),
      rgb: [r, g, b],
      type: 'drift',
    };
  }

  function updateParticle(p) {
    if (p.type === 'ember') {
      p.wobble += p.wobbleSpeed;
      p.x += p.vx + Math.sin(p.wobble) * 0.4;
      p.y += p.vy;
      if (p.y < -20) { p.y = rng(H * 1.05, H * 1.2); p.x = rng(0, W); }
    } else if (p.type === 'snow') {
      p.drift += p.driftSpeed;
      p.x += p.vx + Math.sin(p.drift) * 0.5;
      p.y += p.vy;
      if (p.y > H + 15) { p.y = rng(-30, -5); p.x = rng(0, W); }
    } else {
      p.x += p.vx;
      p.y += p.vy;
      if (p.y < -p.r * 2) { p.y = rng(H * 1.05, H * 1.2); p.x = rng(0, W); }
    }
  }

  function drawParticle(p) {
    ctx.save();
    ctx.globalAlpha = p.alpha;
    const [r, g, b] = p.rgb;

    if (p.type === 'ember' || p.type === 'snow') {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fill();
    } else {
      const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r);
      grd.addColorStop(0, `rgba(${r},${g},${b},0.7)`);
      grd.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();
    }

    ctx.restore();
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      drawParticle(p);
      updateParticle(p);
    }
    raf = requestAnimationFrame(loop);
  }

  function setTheme(theme) {
    currentType = theme;
    if (raf) cancelAnimationFrame(raf);
    particles.length = 0;

    const t = THEMES[theme] || THEMES.normal;
    for (let i = 0; i < t.count; i++) {
      particles.push(createParticle(true));
    }

    if (canvas) loop();
  }

  function init() {
    canvas = qs('#particle-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    function resize() {
      W = canvas.width = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener('resize', resize, { passive: true });
    document.body.classList.add('particles-ready');
    setTheme('normal');
  }

  return { init, setTheme };
})();

/* ═══════════════════════════════════════════════════════════════
   THEME MANAGER
═══════════════════════════════════════════════════════════════ */

const Theme = (() => {
  const map = {
    HOT:    { cls: 'theme-hot',    particle: 'hot',    accent: '#FF6B2B' },
    COLD:   { cls: 'theme-cold',   particle: 'cold',   accent: '#29AAFF' },
    NORMAL: { cls: 'theme-normal', particle: 'normal', accent: '#00D4A8' },
  };

  function apply(classification) {
    const key = (classification || 'NORMAL').toUpperCase();
    const t = map[key] || map.NORMAL;
    State.currentTheme = key.toLowerCase();

    // Swap body class
    document.body.classList.remove('theme-hot', 'theme-cold', 'theme-normal');
    document.body.classList.add(t.cls);

    // Particles
    Particles.setTheme(t.particle);
  }

  return { apply };
})();

/* ═══════════════════════════════════════════════════════════════
   LOADER
═══════════════════════════════════════════════════════════════ */

const Loader = (() => {
  let el, labelEl, interval;

  function init() {
    el = qs('#loader');
    labelEl = el ? qs('#loader-label', el) : null;
  }

  function show() {
    if (!el) return;
    el.classList.add('active');
    let i = 0;
    if (labelEl) labelEl.textContent = CONFIG.LOADER_MESSAGES[0];

    interval = setInterval(() => {
      i = (i + 1) % CONFIG.LOADER_MESSAGES.length;
      if (labelEl) {
        labelEl.style.opacity = '0';
        setTimeout(() => {
          if (labelEl) {
            labelEl.textContent = CONFIG.LOADER_MESSAGES[i];
            labelEl.style.opacity = '1';
            labelEl.style.transition = 'opacity 0.2s ease';
          }
        }, 150);
      }
    }, 950);
  }

  function hide() {
    clearInterval(interval);
    if (el) el.classList.remove('active');
  }

  return { init, show, hide };
})();

/* ═══════════════════════════════════════════════════════════════
   TOAST
═══════════════════════════════════════════════════════════════ */

const Toast = (() => {
  let container;

  function init() {
    container = qs('#toast-container');
  }

  function show(message, type = 'ok') {
    if (!container) return;

    const icon = type === 'ok' ? ICONS.checkCircle : ICONS.alertCircle;
    const t = document.createElement('div');
    t.className = `toast toast--${type}`;
    t.innerHTML = `<span class="toast__icon">${icon}</span><span>${safe(message)}</span>`;
    container.appendChild(t);

    setTimeout(() => {
      t.classList.add('toast--out');
      setTimeout(() => t.remove(), 320);
    }, CONFIG.TOAST_DURATION_MS);
  }

  return { init, show };
})();

/* ═══════════════════════════════════════════════════════════════
   CHART
═══════════════════════════════════════════════════════════════ */

const TempChart = (() => {
  function render(forecast) {
    const canvas = qs('#temp-chart');
    if (!canvas || !forecast) return;

    if (State.chart) { State.chart.destroy(); State.chart = null; }

    const labels = forecast.hourly_labels || [];
    const data = forecast.hourly_temperatures || [];
    if (!data.length) return;

    const root = getComputedStyle(document.documentElement);
    const accent = root.getPropertyValue('--accent').trim() || '#00D4A8';
    const accent2 = root.getPropertyValue('--accent-2').trim() || '#00B4D8';

    const ctx = canvas.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.offsetHeight || 200);
    grad.addColorStop(0, accent + '30');
    grad.addColorStop(0.7, accent + '05');
    grad.addColorStop(1, accent + '00');

    State.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          data,
          borderColor: accent,
          backgroundColor: grad,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: accent,
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
          fill: true,
          tension: 0.45,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1200, easing: 'easeOutQuart' },
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(13,16,23,0.95)',
            borderColor: accent,
            borderWidth: 1,
            titleColor: 'rgba(240,244,255,0.5)',
            bodyColor: '#F0F4FF',
            padding: 12,
            cornerRadius: 10,
            titleFont: { family: 'IBM Plex Mono', size: 10 },
            bodyFont: { family: 'IBM Plex Mono', size: 13 },
            callbacks: {
              title: i => i[0]?.label || '',
              label: i => ` ${i.raw}°C`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            border: { display: false },
            ticks: {
              color: 'rgba(240,244,255,0.28)',
              font: { family: 'IBM Plex Mono', size: 9 },
              maxTicksLimit: 8,
              maxRotation: 0,
            },
          },
          y: {
            min: Math.floor(Math.min(...data) - 2),
            max: Math.ceil(Math.max(...data) + 2),
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            border: { display: false },
            ticks: {
              color: 'rgba(240,244,255,0.28)',
              font: { family: 'IBM Plex Mono', size: 9 },
              callback: v => `${v}°`,
            },
          },
        },
      },
    });
  }

  return { render };
})();

/* ═══════════════════════════════════════════════════════════════
   GALLERY
═══════════════════════════════════════════════════════════════ */

const Gallery = (() => {
  let track, dotsEl, images = [], idx = 0, timer = null;

  function init(imgs) {
    images = imgs || [];
    idx = 0;
    track = qs('#gallery-track');
    dotsEl = qs('#gallery-dots');
    if (!track) return;

    clearInterval(timer);
    renderSlides();
    renderDots();
    bindControls();
    if (images.length > 1) autoAdvance();
  }

  function renderSlides() {
    if (!images.length) {
      track.innerHTML = `<div class="gallery__empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" width="36" height="36"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg><p>No images available</p></div>`;
      return;
    }

    track.innerHTML = '';
    images.forEach((img, i) => {
      const slide = document.createElement('div');
      slide.className = 'gallery__slide';
      slide.style.transform = `translateX(${(i - 0) * 100}%)`;

      const im = document.createElement('img');
      im.src = img.url;
      im.alt = img.alt_text || `City image ${i + 1}`;
      im.loading = 'lazy';
      im.onerror = () => { im.style.opacity = '0'; };

      const cap = document.createElement('div');
      cap.className = 'gallery__slide-caption';
      cap.innerHTML = `<span>${safe(img.alt_text || '')}</span>`;

      slide.appendChild(im);
      slide.appendChild(cap);
      track.appendChild(slide);
    });

    updateSlides();
  }

  function renderDots() {
    if (!dotsEl) return;
    dotsEl.innerHTML = '';
    images.forEach((_, i) => {
      const d = document.createElement('button');
      d.className = `gallery__dot${i === 0 ? ' is-active' : ''}`;
      d.setAttribute('aria-label', `Go to image ${i + 1}`);
      d.addEventListener('click', () => { goTo(i); resetAuto(); });
      dotsEl.appendChild(d);
    });
  }

  function updateSlides() {
    qsa('.gallery__slide', track).forEach((slide, i) => {
      slide.style.transform = `translateX(${(i - idx) * 100}%)`;
    });
    qsa('.gallery__dot', dotsEl || document).forEach((d, i) => {
      d.classList.toggle('is-active', i === idx);
    });
  }

  function goTo(i) {
    if (!images.length) return;
    idx = ((i % images.length) + images.length) % images.length;
    updateSlides();
  }

  function bindControls() {
    qs('#gal-prev')?.addEventListener('click', () => { goTo(idx - 1); resetAuto(); });
    qs('#gal-next')?.addEventListener('click', () => { goTo(idx + 1); resetAuto(); });

    const viewport = qs('#gallery-track');
    if (!viewport) return;
    let startX = 0;
    viewport.addEventListener('touchstart', e => { startX = e.changedTouches[0].screenX; }, { passive: true });
    viewport.addEventListener('touchend', e => {
      const dx = startX - e.changedTouches[0].screenX;
      if (Math.abs(dx) > 48) { goTo(idx + (dx > 0 ? 1 : -1)); resetAuto(); }
    }, { passive: true });
  }

  function autoAdvance() {
    timer = setInterval(() => goTo(idx + 1), CONFIG.GALLERY_INTERVAL_MS);
  }

  function resetAuto() {
    clearInterval(timer);
    if (images.length > 1) autoAdvance();
  }

  return { init };
})();

/* ═══════════════════════════════════════════════════════════════
   RENDER HELPERS
═══════════════════════════════════════════════════════════════ */

function renderProbBars(containerId, probs = {}, topKey = null) {
  const container = qs(containerId);
  if (!container) return;

  container.innerHTML = '';
  const entries = Object.entries(probs).sort(([, a], [, b]) => b - a);
  const maxVal = Math.max(...Object.values(probs), 0.01);

  entries.forEach(([label, val], i) => {
    const pct = Math.round((val || 0) * 100);
    const isTop = label === topKey || val === maxVal;

    const bar = document.createElement('div');
    bar.className = 'prob-bar';
    bar.setAttribute('role', 'listitem');
    bar.innerHTML = `
      <div class="prob-bar__row">
        <span class="prob-bar__name">${safe(label)}</span>
        <span class="prob-bar__pct">${pct}%</span>
      </div>
      <div class="prob-bar__track">
        <div class="prob-bar__fill${isTop ? ' is-top' : ''}"
          style="width:${pct}%; --bar-delay:${0.45 + i * 0.07}s"></div>
      </div>
    `;
    container.appendChild(bar);
  });
}

function renderMetrics(metrics) {
  const strip = qs('#metrics-strip');
  if (!strip || !metrics) return;

  const cells = [
    { label: 'Forecaster',   type: 'LinearRegression',    score: metrics.forecaster?.r2_score,  pred: '24h Temp' },
    { label: 'Clothing AI',  type: 'LogisticRegression',  score: metrics.advisor?.accuracy,     pred: '5 Classes' },
    { label: 'Classifier',   type: 'Multi-class LR',      score: metrics.classifier?.accuracy,  pred: 'HOT/COLD/NRM' },
    { label: 'Cluster Eng',  type: 'KMeans (n=4)',        score: null,                          pred: '4 Patterns' },
  ];

  strip.innerHTML = cells.map(c => {
    const pct = c.score !== null && c.score !== undefined ? Math.round(c.score * 100) : null;
    return `
      <div class="metric-cell">
        <div class="metric-cell__label">${safe(c.label)}</div>
        <div class="metric-cell__type">${safe(c.type)}</div>
        <div class="metric-cell__value">${pct !== null ? pct + '%' : 'On'}</div>
        ${pct !== null ? `<div class="metric-cell__bar"><div class="metric-cell__bar-fill" style="width:${pct}%"></div></div>` : ''}
      </div>
    `;
  }).join('');
}

/* ═══════════════════════════════════════════════════════════════
   MAIN RENDER
═══════════════════════════════════════════════════════════════ */

function renderResults(data) {
  const { weather, ml, theme, images } = data;
  if (!weather) { Toast.show('No weather data returned.', 'err'); return; }

  // 1. Apply theme first (drives colors + particles)
  const cls = ml?.classification?.classification || 'NORMAL';
  Theme.apply(cls);

  // 2. Show results section
  const section = qs('#results-section');
  if (section) {
    section.hidden = false;
    section.removeAttribute('hidden');
    setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
  }

  // 3. Render weather data
  renderWeather(weather, cls);

  // 4. Render ML
  renderClassification(ml);
  renderClothing(ml?.clothing);
  renderForecast(ml?.forecast);
  renderMetrics(ml?.training_metrics);

  // 5. Gallery
  Gallery.init(images || []);

  // 6. Update desktop instrument panel
  const conf = ml?.composite_confidence;
  setText('#inst-condition', cls);
  setText('#inst-conf', conf ? Math.round(conf * 100) + '%' : '—');
  setText('#inst-trend', ml?.forecast?.trend === 'rising' ? 'Rising' : ml?.forecast?.trend === 'falling' ? 'Falling' : 'Stable');
  setText('#inst-clothing', ml?.clothing?.recommendation || '—');
}

function renderWeather(w, cls) {
  setText('#city-name', w.city || '—');
  setText('#country-name', w.country || '');
  setText('#main-temp', round(w.temperature, 0));
  setText('#weather-desc', w.description || '—');
  setText('#weather-feels', `Feels like ${round(w.feels_like, 0)}°C`);

  // Classification tag
  const tag = qs('#weather-class-tag');
  if (tag) {
    tag.className = `weather-tag weather-tag--${(cls || 'NORMAL').toLowerCase()}`;
  }
  setText('#weather-class-label', cls || '—');

  // Stats
  const setVal = (id, val, unit = '') => {
    const el = qs(id);
    if (!el) return;
    const valEl = el.querySelector('.stat-item__value');
    if (valEl) valEl.innerHTML = `${safe(val ?? '—')}${unit ? `<small>${safe(unit)}</small>` : ''}`;
  };

  setVal('#stat-humidity', round(w.humidity, 0), '%');
  setVal('#stat-wind', round(w.wind_speed, 0), 'km/h');
  setVal('#stat-pressure', w.pressure, 'hPa');
  setVal('#stat-visibility', round(w.visibility, 1), 'km');
  setVal('#stat-uv', w.uv_index ?? '—');
  setVal('#stat-cloud', round(w.cloud_cover, 0), '%');

  // Source
  setHtml('#source-label', safe(w.is_fallback ? 'Synthetic fallback data' : w.source || 'Live data'));
}

function renderClassification(ml) {
  if (!ml?.classification) return;
  const { classification, confidence, class_probabilities } = ml.classification;

  setText('#cls-value', classification || '—');
  setText('#cls-pattern', ml?.pattern?.pattern || '—');

  // Icon
  const iconMap = { HOT: ICONS.sun, COLD: ICONS.snowflake, NORMAL: ICONS.thermometer };
  setHtml('#cls-icon-box', iconMap[classification] || ICONS.thermometer);

  // Confidence
  const conf = ml.composite_confidence || confidence || 0;
  const pct = Math.round(conf * 100);
  const fill = qs('#conf-fill');
  const label = qs('#conf-label');
  if (fill) setTimeout(() => { fill.style.width = `${pct}%`; }, 150);
  if (label) label.textContent = `${pct}%`;
  const track = qs('#conf-track');
  if (track) { track.setAttribute('aria-valuenow', pct); track.setAttribute('aria-label', `${pct}% confidence`); }

  renderProbBars('#cls-probs', class_probabilities || {}, classification);
}

function renderClothing(clothing) {
  if (!clothing) return;

  setText('#cloth-name', clothing.recommendation || '—');
  setText('#cloth-reason', clothing.reasoning || '');
  setText('#cloth-conf', `${Math.round((clothing.confidence || 0) * 100)}% confidence`);

  const iconMap = {
    'Summer Wear': ICONS.shorts,
    'T-Shirt':     ICONS.tshirt,
    'Hoodie':      ICONS.jacket,
    'Light Jacket':ICONS.jacket,
    'Heavy Jacket':ICONS.coat,
  };
  setHtml('#cloth-icon-box', iconMap[clothing.recommendation] || ICONS.tshirt);

  renderProbBars('#cloth-probs', clothing.class_probabilities || {}, clothing.recommendation);
}

function renderForecast(forecast) {
  if (!forecast) return;

  const trending = forecast.trend;
  const trendIcons = { rising: ICONS.trendUp, falling: ICONS.trendDown };
  const trendLabels = { rising: 'Rising', falling: 'Falling', stable: 'Stable' };

  setHtml('#fc-trend-icon', trendIcons[trending] || ICONS.trendStable);
  setText('#fc-trend-label', trendLabels[trending] || 'Stable');
  setText('#fc-delta', `${forecast.temperature_delta > 0 ? '+' : ''}${round(forecast.temperature_delta, 1)}°`);
  setText('#fc-min', `${round(forecast.min_forecast, 1)}°`);
  setText('#fc-max', `${round(forecast.max_forecast, 1)}°`);
  setText('#fc-conf', forecast.confidence ? `${Math.round(forecast.confidence * 100)}%` : '—');

  TempChart.render(forecast);
}

/* ═══════════════════════════════════════════════════════════════
   SEARCH
═══════════════════════════════════════════════════════════════ */

const Search = (() => {
  let form, input, btn, errEl;

  function init() {
    form = qs('#search-form');
    input = qs('#city-input');
    btn = qs('#search-btn');
    errEl = qs('#search-error');

    form?.addEventListener('submit', e => { e.preventDefault(); run(); });

    input?.addEventListener('input', () => {
      errEl?.classList.remove('is-visible');
    });

    // Keyboard shortcut
    document.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        input?.focus();
        input?.select();
      }
      if (e.key === 'Escape' && document.activeElement === input) {
        input?.blur();
      }
    });
  }

  function run() {
    const city = (input?.value || '').trim();
    if (!validate(city)) return;
    showError(null);
    fetchWeather(city);
  }

  function validate(city) {
    if (!city) { showError('Please enter a city name.'); return false; }
    if (city.length < 2) { showError('City name is too short.'); return false; }
    if (city.length > 64) { showError('City name is too long.'); return false; }
    if (/[<>"';&\\|]/.test(city)) { showError('City name contains invalid characters.'); return false; }
    return true;
  }

  function showError(msg) {
    if (!errEl) return;
    if (!msg) { errEl.classList.remove('is-visible'); return; }
    qs('#search-error-text', errEl).textContent = msg;
    errEl.classList.add('is-visible');
  }

  function setLoading(v) {
    State.loading = v;
    if (btn) btn.classList.toggle('is-loading', v);
    if (input) input.disabled = v;
  }

  async function fetchWeather(city) {
    if (State.loading) return;
    setLoading(true);
    Loader.show();

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), CONFIG.TIMEOUT_MS);

      const res = await fetch(`${CONFIG.API_BASE}/api/weather`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.error || `Server error ${res.status}`);
      }

      renderResults(data);
      Toast.show(`${data.city || city} loaded successfully`, 'ok');

    } catch (err) {
      const msg = err.name === 'AbortError'
        ? 'Request timed out. Please try again.'
        : (err.message || 'Failed to fetch weather data.');

      showError(msg);
      Toast.show(msg, 'err');
      console.error('[WeatherIQ] Fetch error:', err);

    } finally {
      setLoading(false);
      Loader.hide();
    }
  }

  function searchCity(city) {
    if (input) input.value = city;
    run();
  }

  return { init, searchCity };
})();

/* ═══════════════════════════════════════════════════════════════
   QUICK CITIES
═══════════════════════════════════════════════════════════════ */

function buildQuickCities() {
  const list = qs('#quick-list');
  if (!list) return;

  CONFIG.QUICK_CITIES.forEach(city => {
    const btn = document.createElement('button');
    btn.className = 'quick-btn';
    btn.textContent = city;
    btn.setAttribute('type', 'button');
    btn.setAttribute('aria-label', `Search weather for ${city}`);
    btn.addEventListener('click', () => Search.searchCity(city));
    list.appendChild(btn);
  });
}

/* ═══════════════════════════════════════════════════════════════
   DIM MODE
═══════════════════════════════════════════════════════════════ */

function initDimMode() {
  const btn = qs('#btn-dark');
  if (!btn) return;

  if (localStorage.getItem('wiq-dim') === '1') {
    document.body.classList.add('dimmed');
    State.dimmed = true;
  }

  btn.addEventListener('click', () => {
    State.dimmed = !State.dimmed;
    document.body.classList.toggle('dimmed', State.dimmed);
    localStorage.setItem('wiq-dim', State.dimmed ? '1' : '0');
  });
}

/* ═══════════════════════════════════════════════════════════════
   INIT
═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Core systems
  Particles.init();
  Loader.init();
  Toast.init();

  // Default theme
  document.body.classList.add('theme-normal');

  // UI modules
  Search.init();
  buildQuickCities();
  initDimMode();

  // Keyboard shortcut hint
  if (navigator.platform?.toLowerCase().includes('mac')) {
    const hint = qs('.hero__hint');
    if (hint) hint.innerHTML = hint.innerHTML.replace('Ctrl+K', '⌘K');
  }

  // Scroll: tighten nav border on scroll
  const navInner = qs('.nav__inner');
  let scrolled = false;
  window.addEventListener('scroll', () => {
    const now = window.scrollY > 60;
    if (now !== scrolled) {
      scrolled = now;
      if (navInner) {
        navInner.style.borderBottomColor = now
          ? 'rgba(255,255,255,0.12)'
          : 'rgba(255,255,255,0.055)';
      }
    }
  }, { passive: true });

  console.info(
    '%c WeatherIQ %c v1.0 ready ',
    'background:#080B12;color:#00D4A8;font-family:monospace;padding:3px 6px;',
    'background:#080B12;color:#666;font-family:monospace;padding:3px 6px;'
  );
});
