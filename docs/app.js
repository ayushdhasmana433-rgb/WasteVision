/* WasteVision in the browser.
 * The EfficientNetV2-B0 backbone runs in ONNX Runtime Web (model/backbone.onnx, input 1x224x224x3 in 0..255).
 * The classification head (global average pooling, Dense 128 + ReLU, Dense = number of classes in site.json) and Grad-CAM are computed here
 * from model/head.bin, so the class scores and the heat map match the Keras model in app/ exactly.
 * Calibration, stream probabilities, the Special Care rule and the "not sure" rule follow app/wv_core.py.
 */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const E = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (s, kw) => s.replace(/\{(\w+)\}/g, (m, k) => (k in kw ? kw[k] : m));
  const pct = (x) => (x >= 0.1 ? (x * 100).toFixed(0) : (x * 100).toFixed(1)) + '%';
  const pct1 = (x) => (x * 100).toFixed(1) + '%';
  // Stream key from its name in site.json: Wet -> wet, Special Care -> special, Sanitary -> sanitary (same rule as wv_core.stream_key).
  const streamKey = (name) => name.split(' ')[0].toLowerCase();
  const SIZE = 224, FEAT = 1280, HID = 128, CELLS = 49;

  // ------------------------------------------------------------ icons (same as the Gradio app)
  const I = (p) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
  const ICON = {
    wet: I('<path d="M5 19c0-8 5-13 14-14 0 9-5 14-12 14"/><path d="M5 19c3-4 6-6 9-8"/>'),
    dry: I('<path d="M7 10l3-5h4l3 5"/><path d="M17 10l2 4-3 5H8l-3-5 2-4"/><path d="M10 15h4"/>'),
    special: I('<path d="M12 3.5l9 15.5H3z"/><path d="M12 10v4"/><path d="M12 16.8h.01"/>'),
    sanitary: I('<path d="M8.5 3.5h7l2.5 4.5v11a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 6 19V8z"/><path d="M6 8h12"/><path d="M10 13.5h4"/>'),
    unsure: I('<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.5"/><path d="M12 17h.01"/>'),
    photo: I('<rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-8 8"/>'),
  };
  const GLYPH_INNER = {
    wet: '<path d="M19 34c0-6 3.5-9.5 10-10 0 6.5-3.5 10-8.5 10M19 34c2-2.6 4-4 6-5.2" fill="none" stroke-width="2.2" stroke-linecap="round"/>',
    dry: '<path d="M21 29.5l2-3.3h3l2 3.3M28 29.5l1.3 2.6-2 3.3h-6.6l-2-3.3 1.3-2.6" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    special: '<path d="M24 24.5l6 10.5H18z" fill="none" stroke-width="2.2" stroke-linejoin="round"/><path d="M24 28.5v2.6" stroke-width="2.2" stroke-linecap="round"/>',
    sanitary: '<path d="M20.5 24h7l2 3.5v7a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1v-7z" fill="none" stroke-width="2.2" stroke-linejoin="round"/><path d="M18.5 27.5h11" stroke-width="2.2" stroke-linecap="round"/>',
    unsure: '<path d="M21.5 27.5a2.6 2.6 0 1 1 3.6 2.4c-.7.3-1.1.8-1.1 1.6" fill="none" stroke-width="2.2" stroke-linecap="round"/><circle cx="24" cy="35" r="1.3"/>',
  };
  const glyph = (k) => '<svg class="wv-glyph" viewBox="0 0 48 48" aria-hidden="true">'
    + '<g class="wv-lidg"><rect x="9" y="9" width="30" height="5" rx="2.5"/><rect x="19.5" y="5.5" width="9" height="3.5" rx="1.75"/></g>'
    + '<path d="M12 18h24l-2.2 21.4a3 3 0 0 1-3 2.6H17.2a3 3 0 0 1-3-2.6z" fill="none" stroke-width="3.2" stroke-linejoin="round"/>'
    + (GLYPH_INNER[k] || GLYPH_INNER.unsure) + '</svg>';

  // ------------------------------------------------------------ state
  let NC = 0;   // number of classes, set from site.json
  let SITE, CFG, TX, lang = 'en', res = null, session = null, W = null, busy = false, stream = null, reqId = 0;
  let modelReady, loadFraction = 0;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
  };
  const t = () => TX.T[lang];
  const show = (sel, on) => { const el = $(sel); if (el) el.hidden = !on; };   // tolerate an older cached page

  // ------------------------------------------------------------ rendering (ported from app/app.py)
  const card = (kind, band, body) => `<section class="wv-res wv-${kind}" role="status" aria-live="polite">${band}<div class="wv-resbody">${body}</div></section>`;
  const band = (kind, eyebrow, title, sub, conf) => `<div class="wv-band">${glyph(kind)}<div class="wv-bandtext"><div class="wv-eyebrow">${E(eyebrow)}</div>`
    + `<div class="wv-verdict">${E(title)}</div><div class="wv-sub">${E(sub)}</div>${conf ? `<span class="wv-conf">${E(conf)}</span>` : ''}</div></div>`;

  function renderEmpty() {
    const rows = CFG.streams.map(streamKey).map((k) => `<li class="wv-leg wv-leg-${k}"><span class="wv-dot" aria-hidden="true">${ICON[k]}</span>`
      + `<span><b>${E(TX.BINS[lang][k][0])}</b> <span class="wv-legsub">· ${E(TX.BINS[lang][k][1])}</span><small>${E(TX.LEGEND[lang][k])}</small></span></li>`).join('');
    return `<section class="wv-res wv-empty" role="status" aria-live="polite"><div class="wv-emptyhead">${ICON.photo}`
      + `<div><div class="wv-h">${E(t().empty_title)}</div><div class="wv-muted">${E(t().empty_body)}</div></div></div><ul class="wv-legend">${rows}</ul></section>`;
  }
  function renderError(key) {
    return `<section class="wv-res wv-error" role="alert"><div class="wv-emptyhead">${ICON.unsure}<div><div class="wv-h">${E(t().eyebrow_unsure)}</div><div>${E(t()[key])}</div></div></div></section>`;
  }
  function renderBusy(key) {
    const bar = key === 'loading' ? `<div class="progress" aria-hidden="true"><span style="width:${Math.round(loadFraction * 100)}%"></span></div>` : '';
    return `<section class="wv-res wv-busy" role="status" aria-live="polite"><div class="wv-emptyhead"><span class="spinner" aria-hidden="true"></span><div><div class="wv-h">${E(t()[key])}</div></div></div>${bar}</section>`;
  }
  function renderVerdict(r) {
    if (!r) return renderEmpty();
    const T = t(), names = TX.CLASS_NAMES[lang], key = r.verdictKey, material = r.topClass;
    let [binName, binSub] = TX.BINS[lang][key];
    let caution = '', conf = '', kind, eyebrow, materialLine;
    if (r.mode === 'unsure') {
      kind = 'unsure'; eyebrow = T.eyebrow_unsure;
      binName = T.unsure_title; binSub = fmt(T.best_guess, { b: TX.BINS[lang][key][0] });
      materialLine = E(fmt(T.might_be, { m: names[material] }));
      conf = pct(r.streamProb);
    } else {
      kind = key;
      eyebrow = T['eyebrow_' + key] || T.eyebrow_sure;   // eyebrow_special, eyebrow_sanitary; wet and dry use "Put it in"
      if (r.mode === 'precaution') {
        caution = `<div class="wv-caution" role="note">${ICON.special}<p><b>${E(T.eyebrow_precaution)}.</b> `
          + `${E(fmt(T.precaution_body, { m: names[r.topClass], pct: pct(r.hazardProb), h: names[r.hazardClass] }))}</p></div>`;
        conf = pct(r.hazardProb);
      } else {
        conf = `${r.streamProb >= 0.9 ? T.very_likely : T.likely} · ${pct(r.streamProb)}`;
      }
      materialLine = E(fmt(r.classSure ? T.looks_like : T.might_be, { m: names[material] }));
    }
    let tip = r.mode === 'unsure' ? T.unsure_body : TX.TIPS[lang][material];
    if (r.mode === 'precaution') tip = TX.TIPS[lang][r.hazardClass];
    return card(kind, band(kind, eyebrow, binName, binSub, conf), `${caution}<p class="wv-material">${materialLine}</p><p class="wv-tip">${E(tip)}</p>`);
  }
  function renderDetails(r) {
    const names = TX.CLASS_NAMES[lang];
    const row = (label, v, cls) => `<li class="wv-bar"><span class="wv-barlabel">${E(label)}</span><span class="wv-track" aria-hidden="true">`
      + `<span class="wv-fill ${cls || ''}" style="width:${Math.max(v * 100, 1).toFixed(1)}%"></span></span><span class="wv-pct">${pct1(v)}</span></li>`;
    const streams = CFG.streams.map((s, i) => row(TX.BINS[lang][streamKey(s)][0], r.streamProbs[i], streamKey(s))).join('');
    const mats = r.ranked.slice(0, 3).map(([c, v]) => row(names[c], v)).join('');
    return `<h3 class="mini">${E(t().bins_heading)}</h3><ul class="wv-bars">${streams}</ul><h3 class="mini">${E(t().materials_heading)}</h3><ul class="wv-bars">${mats}</ul>`;
  }
  function renderAbout() {
    const T = t(), m = CFG.metrics, te = m.test, real = m.test_real || {}, india = m.dwsd_test || {}, ci = m.dwsd_ci || {};
    const d = CFG.data || {}, totals = d.totals || {}, sources = Object.keys(d.by_source || {}).filter(Boolean);
    const fmtCi = (c) => (c ? `${(c[0] * 100).toFixed(1)}–${(c[1] * 100).toFixed(1)}%` : '–');
    let extra = '';
    if (real.n) extra += fmt(T.about_extra_real, { n_real: real.n });
    if (india.n) extra += fmt(T.about_extra_india, { n_india: india.n });
    const tr = (key, value, kw = {}) => `<tr><td>${E(fmt(T[key][0], { n: CFG.classes.length }))}</td><td class="num">${value}</td><td>${E(fmt(T[key][1], kw))}</td></tr>`;
    const rows = [tr('m_stream', pct1(te.stream_acc)), tr('m_class', pct1(te.class_acc))];
    if (real.n) rows.push(tr('m_real', pct1(real.stream_acc)));
    if (te['sc_recall@0.20'] != null) {
      rows.push(tr('m_sc', pct1(te['sc_recall@0.20']), { argmax: pct1(te.sc_recall_argmax || 0) }));
      rows.push(tr('m_fa', pct1(te['sc_false_alarm@0.20'])));
    }
    if (india.n) rows.push(tr('m_india', pct1(india.stream_acc), { ci: fmtCi(ci.stream_acc) }));
    const th = T.about_th.map((h) => `<th scope="col">${E(h)}</th>`).join('');
    const how = ['how_safety', 'how_unsure', 'how_why'].map((k) => `<li>${fmt(T[k], { sc_t: pct(CFG.special_care_threshold), t_stream: pct(CFG.not_sure_stream_threshold) })}</li>`).join('');
    const limits = (CFG.sanitary_supported === false ? [T.limit_no_sanitary] : []).concat(T.limits).map((x) => `<li>${E(x)}</li>`).join('');
    const modelLine = fmt(T.about_model, { arch: E(CFG.architecture), params: CFG.params_m.toFixed(1), n_train: totals.train || '–', n_src: sources.length, src: E(sources.join(', ')) });
    const names = TX.CLASS_NAMES[lang];
    const gal = CFG.gallery.map((g) => {
      const place = g.source === 'DWSD' ? 'India · ' : '';
      const cap = g.kind === 'correct' ? `✓ ${place}${names[g.true] || g.true}` : `✗ ${place}${names[g.true] || g.true} → ${names[g.pred] || g.pred}`;
      return `<li><figure><img src="gallery/${encodeURI(g.file)}" alt="${E(cap)}" loading="lazy" width="224" height="224"><figcaption>${E(cap)}</figcaption></figure></li>`;
    }).join('');
    return `<section class="card"><h2>${E(T.about_h_perf)}</h2><p>${E(fmt(T.about_intro, { n_test: te.n, extra }))}</p>`
      + `<table class="table"><thead><tr>${th}</tr></thead><tbody>${rows.join('')}</tbody></table></section>`
      + `<div class="about2"><section class="card"><h2>${E(T.about_h_how)}</h2><ul>${how}</ul></section>`
      + `<section class="card"><h2>${E(T.about_h_limits)}</h2><ul>${limits}</ul></section></div>`
      + `<p class="attrib">${modelLine}</p><h2>${E(T.gallery_h)}</h2><ul class="gallery">${gal}</ul>`;
  }

  function applyText() {
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-t]').forEach((el) => {
      const k = el.dataset.t;
      el.textContent = k === 'footer'
        ? fmt(CFG.sanitary_supported === false ? t().footer_unsupported : t().footer, { n: CFG.classes.length })
        : t()[k];
    });
    document.querySelectorAll('.lang button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.lang === lang)));
    $('.pickers').setAttribute('aria-label', t().photo_label);
    $('#cam').setAttribute('aria-label', t().cam_label);
    $('#attrib').textContent = fmt(t().attrib, { arch: CFG.architecture });
    $('#about').innerHTML = renderAbout();
    if (!busy) $('#verdict').innerHTML = renderVerdict(res);
    if (res) $('#details').innerHTML = renderDetails(res);
    renderExamples();
  }
  function renderExamples() {
    const names = TX.CLASS_NAMES[lang];
    $('#examples').innerHTML = CFG.examples.map((e, i) => `<li><button type="button" data-ex="${i}" aria-label="${E(names[e.label] || e.label)}">`
      + `<img src="examples/${encodeURI(e.file)}" alt="" loading="lazy" width="56" height="56"></button></li>`).join('');
  }

  // ------------------------------------------------------------ model
  async function fetchWithProgress(url, onFrac) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    const total = Number(r.headers.get('content-length')) || 0;
    if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
    const reader = r.body.getReader(), parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length; onFrac(Math.min(1, got / total));
    }
    const out = new Uint8Array(got);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }
  async function loadModel() {
    ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const [onnx, head] = await Promise.all([
      fetchWithProgress('model/backbone.onnx', (f) => {
        loadFraction = f;
        const bar = document.querySelector('.progress span');
        if (bar) bar.style.width = `${Math.round(f * 100)}%`;
      }),
      fetch('model/head.bin').then((r) => { if (!r.ok) throw new Error('head.bin'); return r.arrayBuffer(); }),
    ]);
    const raw = new Float32Array(head);
    let o = 0;
    const take = (n) => { const a = raw.subarray(o, o + n); o += n; return a; };
    if (raw.length !== FEAT * HID + HID + HID * NC + NC) throw new Error(`head.bin has ${raw.length} values, expected ${FEAT * HID + HID + HID * NC + NC} for ${NC} classes`);
    W = { W1: take(FEAT * HID), b1: take(HID), W2: take(HID * NC), b2: take(NC) };
    session = await ort.InferenceSession.create(onnx, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(SIZE * SIZE * 3), [1, SIZE, SIZE, 3]) });
  }

  // Head and Grad-CAM. score = max logit (as in wv_core._forward_with_cam).
  // d score / d fmap[i,j,k] = g_k / 49 with g = W1 · (W2[:,c] ⊙ 1[h > 0]); weights = spatial mean of that gradient.
  function head(fm) {
    const { W1, b1, W2, b2 } = W;
    const pooled = new Float32Array(FEAT);
    for (let p = 0; p < CELLS; p++) for (let k = 0; k < FEAT; k++) pooled[k] += fm[p * FEAT + k];
    for (let k = 0; k < FEAT; k++) pooled[k] /= CELLS;
    const h = Float32Array.from(b1);
    for (let k = 0; k < FEAT; k++) { const v = pooled[k]; if (v !== 0) for (let j = 0; j < HID; j++) h[j] += v * W1[k * HID + j]; }
    const z = Float32Array.from(b2);
    for (let j = 0; j < HID; j++) { const a = h[j] > 0 ? h[j] : 0; if (a) for (let c = 0; c < NC; c++) z[c] += a * W2[j * NC + c]; }
    let c = 0;
    for (let i = 1; i < NC; i++) if (z[i] > z[c]) c = i;
    const m = new Float32Array(HID);
    for (let j = 0; j < HID; j++) m[j] = h[j] > 0 ? W2[j * NC + c] : 0;
    const g = new Float32Array(FEAT);
    for (let k = 0; k < FEAT; k++) { let s = 0; for (let j = 0; j < HID; j++) s += W1[k * HID + j] * m[j]; g[k] = s / CELLS; }
    const cam = new Float32Array(CELLS);
    let mx = 0;
    for (let p = 0; p < CELLS; p++) {
      let s = 0;
      for (let k = 0; k < FEAT; k++) s += fm[p * FEAT + k] * g[k];
      cam[p] = s > 0 ? s : 0; if (cam[p] > mx) mx = cam[p];
    }
    for (let p = 0; p < CELLS; p++) cam[p] /= (mx + 1e-8);
    return { z, cam };
  }

  function decide(z) {
    const T = CFG.temperature, classes = CFG.classes, streams = CFG.streams;
    const zz = Array.from(z, (v) => v / T), zmax = Math.max(...zz);
    const e = zz.map((v) => Math.exp(v - zmax)), sum = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / sum);
    const sp = streams.map(() => 0);
    classes.forEach((cls, i) => { sp[streams.indexOf(CFG.stream_of[cls])] += p[i]; });
    const top = p.indexOf(Math.max(...p));
    const sIdx = sp.indexOf(Math.max(...sp)), scIdx = streams.indexOf('Special Care');
    const scClasses = classes.map((c, i) => i).filter((i) => CFG.stream_of[classes[i]] === 'Special Care');
    const hazard = scClasses.reduce((a, b) => (p[b] > p[a] ? b : a));
    let mode, verdict;
    if (sp[scIdx] >= CFG.special_care_threshold && sIdx !== scIdx) { mode = 'precaution'; verdict = 'Special Care'; }
    else if (sp[sIdx] < CFG.not_sure_stream_threshold) { mode = 'unsure'; verdict = streams[sIdx]; }
    else { mode = 'sure'; verdict = streams[sIdx]; }
    const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
    return {
      mode, verdict, verdictKey: streamKey(verdict), topClass: classes[top], topProb: p[top], classSure: p[top] >= CFG.not_sure_class_threshold,
      streamProbs: sp, streamProb: sp[sIdx], hazardClass: classes[hazard], hazardProb: sp[scIdx], ranked: order.map((i) => [classes[i], p[i]]), probs: p,
    };
  }

  // ------------------------------------------------------------ images
  function loadImage(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob), img = new Image();
      img.onload = () => resolve({ img, url });
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode')); };
      img.src = url;
    });
  }
  const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  // Pad to a square exactly like tf.image.resize_with_pad: fit inside, centre, zero (black) padding.
  function padBox(w, h, size) {
    const ratio = Math.max(w / size, h / size), rw = Math.floor(w / ratio), rh = Math.floor(h / ratio);
    return { x: Math.floor((size - rw) / 2), y: Math.floor((size - rh) / 2), w: rw, h: rh };
  }
  function drawPadded(src, w, h, size) {
    const c = canvas(size, size), g = c.getContext('2d');
    g.fillStyle = '#000'; g.fillRect(0, 0, size, size);
    const b = padBox(w, h, size);
    g.fillStyle = '#fff'; g.fillRect(b.x, b.y, b.w, b.h);       // transparent photos are flattened onto white
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.drawImage(src, b.x, b.y, b.w, b.h);
    return c;
  }
  // Same steps as wv_core.model_input: max side 320, JPEG q93, resize_with_pad to 224, round and clip.
  async function modelInput(img) {
    const w0 = img.naturalWidth, h0 = img.naturalHeight, s = Math.min(1, 320 / Math.max(w0, h0));
    const w1 = Math.max(1, Math.round(w0 * s)), h1 = Math.max(1, Math.round(h0 * s));
    const c1 = canvas(w1, h1), g1 = c1.getContext('2d');
    g1.fillStyle = '#fff'; g1.fillRect(0, 0, w1, h1);
    g1.imageSmoothingEnabled = true; g1.imageSmoothingQuality = 'high';
    g1.drawImage(img, 0, 0, w1, h1);
    const jpeg = await new Promise((r) => c1.toBlob(r, 'image/jpeg', 0.93));
    const { img: img2, url } = await loadImage(jpeg);
    const c2 = drawPadded(img2, w1, h1, SIZE);
    URL.revokeObjectURL(url);
    const px = c2.getContext('2d').getImageData(0, 0, SIZE, SIZE).data, x = new Float32Array(SIZE * SIZE * 3);
    for (let i = 0, j = 0; i < px.length; i += 4) { x[j++] = px[i]; x[j++] = px[i + 1]; x[j++] = px[i + 2]; }
    return x;
  }
  // Spotlight like wv_core.spotlight: bright where the model looked, dimmed elsewhere, warm tint on the hot spots.
  function drawSpotlight(img, cam) {
    const S = 448, base = drawPadded(img, img.naturalWidth, img.naturalHeight, S).getContext('2d').getImageData(0, 0, S, S);
    const small = canvas(7, 7), sg = small.getContext('2d'), sd = sg.createImageData(7, 7);
    for (let p = 0; p < CELLS; p++) { const v = Math.round(cam[p] * 255); sd.data.set([v, v, v, 255], p * 4); }
    sg.putImageData(sd, 0, 0);
    const big = canvas(S, S), bg = big.getContext('2d');
    bg.imageSmoothingEnabled = true; bg.imageSmoothingQuality = 'high';
    bg.drawImage(small, 0, 0, S, S);
    const heat = bg.getImageData(0, 0, S, S).data, out = base.data, warm = [1.0, 0.72, 0.28];
    for (let i = 0; i < out.length; i += 4) {
      const hh = heat[i] / 255, k = 0.28 + 0.72 * hh, w = hh * hh * 0.22;
      for (let c = 0; c < 3; c++) out[i + c] = Math.max(0, Math.min(255, (out[i + c] / 255 * k + warm[c] * w) * 255));
    }
    $('#cam').getContext('2d').putImageData(base, 0, 0);
  }

  // ------------------------------------------------------------ flow
  async function analyse(blob) {
    if (!blob) return;
    const my = ++reqId;                       // a newer photo always replaces an older one still in progress
    stopCamera();
    busy = true; res = null;
    $('#why').hidden = true;
    let loaded;
    try { loaded = await loadImage(blob); } catch (e) { if (my === reqId) { busy = false; $('#verdict').innerHTML = renderError('error_read'); } return; }
    if (my !== reqId) { URL.revokeObjectURL(loaded.url); return; }
    const { img, url } = loaded, prev = $('#preview');
    if (prev.dataset.url) URL.revokeObjectURL(prev.dataset.url);
    prev.src = url; prev.dataset.url = url; prev.hidden = false; $('#drop').hidden = true;
    try {
      if (!session) { $('#verdict').innerHTML = renderBusy('loading'); await modelReady; }
      if (my !== reqId) return;
      $('#verdict').innerHTML = renderBusy('analysing');
      await new Promise((r) => requestAnimationFrame(() => r()));
      const x = await modelInput(img);
      if (my !== reqId) return;
      const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [1, SIZE, SIZE, 3]) });
      if (my !== reqId) return;
      const { z, cam } = head(out[session.outputNames[0]].data);
      res = decide(z);
      window.__wv_last = { logits: Array.from(z), cam: Array.from(cam), ...res };
      busy = false;
      $('#verdict').innerHTML = renderVerdict(res);
      $('#details').innerHTML = renderDetails(res);
      drawSpotlight(img, cam);
      $('#why').hidden = false;
      if (window.innerWidth < 840) $('#result').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    } catch (e) {
      console.error(e);
      if (my !== reqId) return;
      busy = false;
      $('#verdict').innerHTML = renderError(session ? 'error_model' : 'error_load');
    }
  }

  // Camera inside the page. Handing off to the phone's camera app can make Android unload this tab
  // (the model takes a lot of memory), and the photo is lost when the page reloads. A live camera
  // view in the page avoids leaving it. The camera-app input stays as a fallback.
  const canLiveCamera = () => !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  function renderCameraFallback() {
    return `<section class="wv-res wv-error" role="alert"><div class="wv-emptyhead">${ICON.unsure}<div><div class="wv-h">${E(t().eyebrow_unsure)}</div>`
      + `<div>${E(t().error_camera)}</div><label class="btn primary fallbackcam"><span>${E(t().camera_app)}</span>`
      + `<input class="pick sr" type="file" accept="image/*" capture="environment"></label></div></div></section>`;
  }
  function showCameraFallback() {
    $('#verdict').innerHTML = renderCameraFallback();
    $('#result').scrollIntoView({ behavior: 'auto', block: 'start' });
  }
  async function startCamera() {
    if (!canLiveCamera()) { showCameraFallback(); return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false });
    } catch (e) { stream = null; showCameraFallback(); return; }
    const v = $('#video');
    v.srcObject = stream; v.hidden = false; $('#preview').hidden = true; $('#drop').hidden = true;
    $('#well').classList.add('live');
    await v.play().catch(() => {});
    show('#camBtn', false); show('#camctl', true); show('#shotBtn', true); show('#stopBtn', true);
    $('#well').scrollIntoView({ behavior: 'auto', block: 'center' });
    if ($('#shotBtn')) $('#shotBtn').focus({ preventScroll: true });
  }
  function stopCamera() {
    if (stream) { stream.getTracks().forEach((tr) => tr.stop()); stream = null; }
    const v = $('#video'); v.srcObject = null; v.hidden = true;
    $('#well').classList.remove('live');
    show('#camBtn', true); show('#camctl', false);
    if (!$('#camctl')) { show('#shotBtn', false); show('#stopBtn', false); }
    if ($('#preview').hidden) $('#drop').hidden = false;
  }
  function capture() {
    const v = $('#video');
    if (!v.videoWidth) return;
    const c = canvas(v.videoWidth, v.videoHeight);
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob((b) => analyse(b), 'image/jpeg', 0.92);
  }
  // Read a picked file into memory at once, so clearing the input or the browser releasing
  // its temporary camera file cannot invalidate it later.
  async function fromInput(el) {
    const f = el.files && el.files[0];
    if (!f) return;
    let blob = f;
    try { blob = new Blob([await f.arrayBuffer()], { type: f.type || 'image/jpeg' }); } catch (e) { /* keep the File */ }
    el.value = '';
    selectTab('sort');
    analyse(blob);
  }

  function selectTab(id) {
    ['sort', 'about'].forEach((k) => {
      const on = k === id;
      $(`#tab-${k}`).setAttribute('aria-selected', String(on));
      $(`#tab-${k}`).tabIndex = on ? 0 : -1;
      $(`#panel-${k}`).hidden = !on;
    });
  }

  function wire() {
    document.addEventListener('change', (e) => {
      const el = e.target;
      if (el.classList && el.classList.contains('pick') && el.files && el.files[0]) fromInput(el);
    });
    const pickCam = $('#pickCam');
    if (pickCam) pickCam.addEventListener('click', (e) => {
      if (canLiveCamera()) { e.preventDefault(); startCamera(); }   // otherwise the label opens the camera app
    });
    const well = $('#well');
    ['dragenter', 'dragover'].forEach((ev) => well.addEventListener(ev, (e) => { e.preventDefault(); well.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) => well.addEventListener(ev, () => well.classList.remove('drag')));
    well.addEventListener('drop', (e) => { e.preventDefault(); const f = [...(e.dataTransfer.files || [])].find((x) => x.type.startsWith('image/')); if (f) analyse(f); });
    document.addEventListener('paste', (e) => {
      const it = [...((e.clipboardData && e.clipboardData.items) || [])].find((x) => x.type.startsWith('image/'));
      if (it) { selectTab('sort'); analyse(it.getAsFile()); }
    });
    $('#examples').addEventListener('click', async (e) => {
      const b = e.target.closest('button[data-ex]');
      if (!b) return;
      const ex = CFG.examples[Number(b.dataset.ex)];
      try { analyse(await (await fetch(`examples/${encodeURI(ex.file)}`)).blob()); } catch (err) { $('#verdict').innerHTML = renderError('error_read'); }
    });
    $('#camBtn').addEventListener('click', startCamera);
    $('#stopBtn').addEventListener('click', stopCamera);
    $('#shotBtn').addEventListener('click', capture);
    document.querySelectorAll('.lang button').forEach((b) => b.addEventListener('click', () => { lang = b.dataset.lang; store.set('wv-lang', lang); applyText(); }));
    $('#tab-sort').addEventListener('click', () => selectTab('sort'));
    $('#tab-about').addEventListener('click', () => selectTab('about'));
    $('.tabs').addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = $('#tab-sort').getAttribute('aria-selected') === 'true' ? 'about' : 'sort';
      selectTab(next); $(`#tab-${next}`).focus();
    });
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const theme = () => document.documentElement.classList.toggle('dark', mq.matches);
    if (mq.addEventListener) mq.addEventListener('change', theme);
  }

  async function init() {
    SITE = await (await fetch('model/site.json?v=4')).json();
    CFG = SITE.config; TX = SITE.text; NC = CFG.classes.length;
    const saved = store.get('wv-lang');
    if (saved === 'hi' || saved === 'en') lang = saved;
    else if ((navigator.language || '').toLowerCase().startsWith('hi')) lang = 'hi';
    wire();
    applyText();
    modelReady = loadModel();
    modelReady.then(() => { window.__wv_ready = true; }).catch((e) => {
      console.error(e);
      if (!res) $('#verdict').innerHTML = renderError('error_load');
    });
  }
  init();
})();
