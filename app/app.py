"""WasteVision: which bin does it go in?

Run:  python app.py                 (opens http://127.0.0.1:7860 in your browser)
      python app.py --lan           (also reachable from a phone on the same Wi-Fi)
      python app.py --share         (temporary public link through Gradio's tunnel)

On Hugging Face Spaces (SPACE_ID is set) it serves on 0.0.0.0:7860 without opening a browser.
The model is loaded once into this process and warmed up before the page opens, so there is no
server or network between the interface and the model.
"""
import argparse
import base64
import html
import json
import os
import re
import socket
import sys
import traceback

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import warnings
warnings.filterwarnings("ignore", message=".*removed in Gradio 6.0.*")
HERE = os.path.dirname(os.path.abspath(__file__))

import gradio as gr

from wv_core import WasteModel, FeedbackStore
from wv_text import LANGS, CLASS_NAMES, BINS, LEGEND, TIPS, T

E = html.escape
STREAM_KEY = {"Wet": "wet", "Dry": "dry", "Special Care": "special"}
# The public online demo does not collect photos; feedback is only offered when the app runs locally.
SHOW_FEEDBACK = not os.environ.get("SPACE_ID")

# ---------------------------------------------------------------- icons: 24px grid, 1.75 stroke, labelled by adjacent text
_I = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>'
ICON = {
    "wet": _I.format('<path d="M5 19c0-8 5-13 14-14 0 9-5 14-12 14"/><path d="M5 19c3-4 6-6 9-8"/>'),
    "dry": _I.format('<path d="M7 10l3-5h4l3 5"/><path d="M17 10l2 4-3 5H8l-3-5 2-4"/><path d="M10 15h4"/>'),
    "special": _I.format('<path d="M12 3.5l9 15.5H3z"/><path d="M12 10v4"/><path d="M12 16.8h.01"/>'),
    "unsure": _I.format('<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.5"/><path d="M12 17h.01"/>'),
    "photo": _I.format('<rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-8 8"/>'),
}
# The verdict glyph: a bin whose lid lifts when the answer lands (the one animation in the app).
BIN_GLYPH = ('<svg class="wv-glyph" viewBox="0 0 48 48" aria-hidden="true">'
             '<g class="wv-lidg"><rect x="9" y="9" width="30" height="5" rx="2.5"/><rect x="19.5" y="5.5" width="9" height="3.5" rx="1.75"/></g>'
             '<path d="M12 18h24l-2.2 21.4a3 3 0 0 1-3 2.6H17.2a3 3 0 0 1-3-2.6z" fill="none" stroke-width="3.2" stroke-linejoin="round"/>'
             '{inner}</svg>')
GLYPH_INNER = {
    "wet": '<path d="M19 34c0-6 3.5-9.5 10-10 0 6.5-3.5 10-8.5 10M19 34c2-2.6 4-4 6-5.2" fill="none" stroke-width="2.2" stroke-linecap="round"/>',
    "dry": '<path d="M21 29.5l2-3.3h3l2 3.3M28 29.5l1.3 2.6-2 3.3h-6.6l-2-3.3 1.3-2.6" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    "special": '<path d="M24 24.5l6 10.5H18z" fill="none" stroke-width="2.2" stroke-linejoin="round"/><path d="M24 28.5v2.6" stroke-width="2.2" stroke-linecap="round"/>',
    "unsure": '<path d="M21.5 27.5a2.6 2.6 0 1 1 3.6 2.4c-.7.3-1.1.8-1.1 1.6" fill="none" stroke-width="2.2" stroke-linecap="round"/><circle cx="24" cy="35" r="1.3"/>',
}


def themed_svg(path):
    """Inline an SVG and bind its ink/paper fills to CSS variables so it follows light and dark mode."""
    s = open(path, encoding="utf-8").read()
    s = s.replace('fill="#151915"', 'style="fill:var(--wv-ink)"').replace('fill="#ffffff"', 'style="fill:var(--wv-raised)"')
    s = s.replace('stroke="#ffffff"', 'style="stroke:var(--wv-raised)"')
    s = re.sub(r'<title>.*?</title>', '', s)
    return s.replace('<svg ', '<svg class="wv-lockup" ', 1)


def pct(x):
    return f"{x * 100:.0f}%" if x >= 0.1 else f"{x * 100:.1f}%"


def pct1(x):
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------- rendering
def card(kind, band, body, live=True):
    attrs = ' role="status" aria-live="polite"' if live else ''
    return f'<section class="wv-res wv-{kind}"{attrs}>{band}<div class="wv-resbody">{body}</div></section>'


def band(kind, eyebrow, title, sub, conf=""):
    glyph = BIN_GLYPH.format(inner=GLYPH_INNER.get(kind, GLYPH_INNER["unsure"]))
    conf_html = f'<span class="wv-conf">{E(conf)}</span>' if conf else ""
    return (f'<div class="wv-band">{glyph}<div class="wv-bandtext"><div class="wv-eyebrow">{E(eyebrow)}</div>'
            f'<div class="wv-verdict">{E(title)}</div><div class="wv-sub">{E(sub)}</div>{conf_html}</div></div>')


def render_empty(lang):
    t = T[lang]
    rows = "".join(f'<li class="wv-leg wv-leg-{k}"><span class="wv-dot" aria-hidden="true">{ICON[k]}</span>'
                   f'<span><b>{E(BINS[lang][k][0])}</b> <span class="wv-legsub">· {E(BINS[lang][k][1])}</span>'
                   f'<small>{E(LEGEND[lang][k])}</small></span></li>'
                   for k in ("wet", "dry", "special"))
    return (f'<section class="wv-res wv-empty" role="status" aria-live="polite"><div class="wv-emptyhead">{ICON["photo"]}'
            f'<div><div class="wv-h">{E(t["empty_title"])}</div><div class="wv-muted">{E(t["empty_body"])}</div></div></div>'
            f'<ul class="wv-legend">{rows}</ul></section>')


def render_error(lang, key):
    t = T[lang]
    return (f'<section class="wv-res wv-error" role="alert"><div class="wv-emptyhead">{ICON["unsure"]}'
            f'<div><div class="wv-h">{E(t["eyebrow_unsure"])}</div><div>{E(t[key])}</div></div></div></section>')


def render_verdict(res, lang):
    if res is None:
        return render_empty(lang)
    t, names = T[lang], CLASS_NAMES[lang]
    corrected = res.get("corrected")
    if corrected == "not_waste":
        return card("unsure", band("unsure", t["eyebrow_corrected"], names["not_waste"], ""),
                    f'<p class="wv-tip">{E(t["not_waste_body"])}</p>')
    key = res["verdict_key"]
    material = corrected or res["top_class"]
    bin_name, bin_sub = BINS[lang][key]
    caution, conf = "", ""
    if corrected:
        kind, eyebrow = key, t["eyebrow_corrected"]
        material_line = E(names[material])
    elif res["mode"] == "unsure":
        kind, eyebrow = "unsure", t["eyebrow_unsure"]
        bin_name, bin_sub = t["unsure_title"], t["best_guess"].format(b=BINS[lang][key][0])
        material_line = E(t["might_be"].format(m=names[material]))
        conf = pct(res["stream_prob"])
    else:
        kind = key
        eyebrow = t["eyebrow_special"] if key == "special" else t["eyebrow_sure"]
        if res["mode"] == "precaution":
            caution = (f'<div class="wv-caution" role="note">{ICON["special"]}<p><b>{E(t["eyebrow_precaution"])}.</b> '
                       f'{E(t["precaution_body"].format(m=names[res["top_class"]], pct=pct(res["hazard_prob"]), h=names[res["hazard_class"]]))}</p></div>')
            conf = pct(res["hazard_prob"])
        else:
            word = t["very_likely"] if res["stream_prob"] >= 0.9 else t["likely"]
            conf = f'{word} · {pct(res["stream_prob"])}'
        phrase = t["looks_like"] if res["class_sure"] else t["might_be"]
        material_line = E(phrase.format(m=names[material]))
    tip_key = material if material in TIPS[lang] else res["top_class"]
    tip = t["unsure_body"] if (res["mode"] == "unsure" and not corrected) else TIPS[lang][tip_key]
    if res["mode"] == "precaution" and not corrected:
        tip = TIPS[lang][res["hazard_class"]]
    body = f'{caution}<p class="wv-material">{material_line}</p><p class="wv-tip">{E(tip)}</p>'
    return card(kind, band(kind, eyebrow, bin_name, bin_sub, conf), body)


def render_details(res, lang):
    if res is None:
        return ""
    t, names = T[lang], CLASS_NAMES[lang]

    def row(label, v, cls=""):
        return (f'<li class="wv-bar"><span class="wv-barlabel">{E(label)}</span>'
                f'<span class="wv-track" aria-hidden="true"><span class="wv-fill {cls}" style="width:{max(v * 100, 1):.1f}%"></span></span>'
                f'<span class="wv-pct">{pct1(v)}</span></li>')
    streams = "".join(row(BINS[lang][STREAM_KEY[s]][0], v, STREAM_KEY[s]) for s, v in res["stream_probs"].items())
    mats = "".join(row(names[c], v) for c, v in res["ranked"][:3])
    return (f'<div class="wv-details"><h3 class="wv-mini">{E(t["bins_heading"])}</h3><ul class="wv-bars">{streams}</ul>'
            f'<h3 class="wv-mini">{E(t["materials_heading"])}</h3><ul class="wv-bars">{mats}</ul></div>')


def render_why_head(lang, visible=True):
    t = T[lang]
    return (f'<h2 class="wv-h2">{E(t["why"])}</h2><p class="wv-attrib wv-camnote">{E(t["cam_note"])}</p>'
            if visible else "")


def render_attrib(lang, model):
    return f'<p class="wv-attrib">{E(T[lang]["attrib"].format(v=model.version, arch=model.arch))}</p>'


def render_bar_title(lang):
    return f'<span class="wv-tag">{E(T[lang]["tagline"])}</span>'


def render_about(lang, model, lockup):
    t = T[lang]
    m = model.cfg["metrics"]
    te = m["test"]
    real = m.get("test_real") or {}
    india, ci = m.get("dwsd_test") or {}, m.get("dwsd_ci") or {}
    d = model.cfg.get("data", {})
    totals = d.get("totals", {})
    sources = [s for s in d.get("by_source", {}) if s]
    fmt_ci = lambda c: f"{c[0] * 100:.1f}–{c[1] * 100:.1f}%" if c else "–"
    params = sum(int(w.shape.num_elements()) for w in model.model.weights) / 1e6
    extra = ""
    if real.get("n"):
        extra += t["about_extra_real"].format(n_real=real["n"])
    if india.get("n"):
        extra += t["about_extra_india"].format(n_india=india["n"])

    def tr(key, value, **kw):
        lab, mean = t[key]
        return f'<tr><td>{E(lab)}</td><td class="num">{value}</td><td>{E(mean.format(**kw))}</td></tr>'
    rows = [tr("m_stream", pct1(te["stream_acc"])), tr("m_class", pct1(te["class_acc"]))]
    if real.get("n"):
        rows.append(tr("m_real", pct1(real["stream_acc"])))
    if "sc_recall@0.20" in te:
        rows.append(tr("m_sc", pct1(te["sc_recall@0.20"]), argmax=pct1(te.get("sc_recall_argmax", 0))))
        rows.append(tr("m_fa", pct1(te["sc_false_alarm@0.20"])))
    if india.get("n"):
        rows.append(tr("m_india", pct1(india["stream_acc"]), ci=fmt_ci(ci.get("stream_acc"))))
    th = "".join(f"<th scope=\"col\">{E(h)}</th>" for h in t["about_th"])
    how = "".join(f"<li>{t[k].format(sc_t=pct(model.sc_threshold), t_stream=pct(model.t_stream))}</li>"
                  for k in ("how_safety", "how_unsure", "how_why"))
    limits = "".join(f"<li>{E(x)}</li>" for x in t["limits"])
    model_line = t["about_model"].format(arch=E(model.arch), params=f"{params:.1f}", n_train=totals.get("train", "–"),
                                         n_src=len(sources), src=E(", ".join(sources)), v=E(model.version))
    return (f'<div class="wv-about"><section class="wv-card wv-pad"><h2 class="wv-h2">{E(t["about_h_perf"])}</h2>'
            f'<p>{E(t["about_intro"].format(n_test=te["n"], extra=extra))}</p>'
            f'<table class="wv-table"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
            f'<div class="wv-about2"><section class="wv-card wv-pad"><h2 class="wv-h2">{E(t["about_h_how"])}</h2><ul>{how}</ul></section>'
            f'<section class="wv-card wv-pad"><h2 class="wv-h2">{E(t["about_h_limits"])}</h2><ul>{limits}</ul></section></div>'
            f'<p class="wv-attrib">{model_line}</p><h2 class="wv-h2 wv-galh">{E(t["gallery_h"])}</h2></div>')


def gallery_items(lang, export_dir):
    path = os.path.join(export_dir, "gallery.json")
    if not os.path.exists(path):
        return []
    names = CLASS_NAMES[lang]
    items = []
    for g in json.load(open(path, encoding="utf-8")):
        f = os.path.join(export_dir, "gallery", g["file"])
        if not os.path.exists(f):
            continue
        place = "India · " if g.get("source") == "DWSD" else ""
        if g.get("kind") == "correct":
            cap = f"✓ {place}{names.get(g['true'], g['true'])}"
        else:
            cap = f"✗ {place}{names.get(g['true'], g['true'])} → {names.get(g['pred'], g['pred'])}"
        items.append((f, cap))
    return items


def example_paths(export_dir):
    path = os.path.join(export_dir, "examples.json")
    if not os.path.exists(path):
        return []
    out = []
    for e in json.load(open(path, encoding="utf-8")):
        f = os.path.join(export_dir, "examples", e["file"])
        if os.path.exists(f):
            out.append([f])
    return out


def fix_choices(res, lang):
    names = CLASS_NAMES[lang]
    order = [c for c, _ in res["ranked"]] if res else list(CLASS_NAMES["en"])[:-1]
    return [(names[c], c) for c in order] + [(names["not_waste"], "not_waste")]


# ---------------------------------------------------------------- theme (values from the WasteVision design system tokens)
def make_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.gray,
        font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Noto Sans Devanagari", "sans-serif"],
        font_mono=["ui-monospace", "SF Mono", "Cascadia Mono", "monospace"],
        text_size=gr.themes.Size(xxs="12px", xs="13px", sm="14px", md="15px", lg="17px", xl="22px", xxl="28px"),
        radius_size=gr.themes.Size(xxs="4px", xs="6px", sm="8px", md="8px", lg="16px", xl="16px", xxl="16px"),
        spacing_size=gr.themes.Size(xxs="2px", xs="4px", sm="6px", md="8px", lg="12px", xl="16px", xxl="24px"),
    ).set(
        body_background_fill="#f2f3ef", body_background_fill_dark="#121512",
        background_fill_primary="#ffffff", background_fill_primary_dark="#1b1f1b",
        background_fill_secondary="#e6e8e2", background_fill_secondary_dark="#0c0e0c",
        block_background_fill="#ffffff", block_background_fill_dark="#1b1f1b",
        block_border_width="0px", block_border_color="#d8dbd3", block_border_color_dark="#2a302a",
        border_color_primary="#d8dbd3", border_color_primary_dark="#2a302a",
        body_text_color="#151915", body_text_color_dark="#eef0eb",
        body_text_color_subdued="#566052", body_text_color_subdued_dark="#a3ab9f",
        block_label_text_color="#566052", block_label_text_color_dark="#a3ab9f",
        block_label_background_fill="#ffffff", block_label_background_fill_dark="#1b1f1b",
        block_title_text_color="#566052", block_title_text_color_dark="#a3ab9f",
        button_primary_background_fill="#151915", button_primary_background_fill_dark="#eef0eb",
        button_primary_background_fill_hover="#2d332d", button_primary_background_fill_hover_dark="#ffffff",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#121512",
        button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#1b1f1b",
        button_secondary_background_fill_hover="#e6e8e2", button_secondary_background_fill_hover_dark="#262b26",
        button_secondary_text_color="#151915", button_secondary_text_color_dark="#eef0eb",
        button_secondary_border_color="#7f867b", button_secondary_border_color_dark="#7b8478",
        button_large_radius="8px", button_small_radius="8px",
        input_background_fill="#ffffff", input_background_fill_dark="#1b1f1b",
        input_border_color="#7f867b", input_border_color_dark="#7b8478",
        color_accent="#1b4f99", color_accent_soft="#e2ebf7", color_accent_soft_dark="#172233",
        checkbox_label_background_fill_selected="#151915", checkbox_label_background_fill_selected_dark="#eef0eb",
        checkbox_label_text_color_selected="#ffffff", checkbox_label_text_color_selected_dark="#121512",
        checkbox_label_border_color="#7f867b", checkbox_label_border_color_dark="#7b8478",
        shadow_drop="none", shadow_drop_lg="none", block_shadow="none",
    )


def load_css():
    css = open(os.path.join(HERE, "assets", "wv.css"), encoding="utf-8").read()
    faces = []
    for fam, file, weight in [("Anek Latin", "AnekLatin-Display.woff2", "500 800"),
                              ("Anek Devanagari", "AnekDevanagari-Bold.woff2", "700"),
                              ("Noto Sans Devanagari", "NotoSansDevanagari-Text.woff2", "400 600")]:
        p = os.path.join(HERE, "assets", "fonts", file)
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            stretch = "font-stretch:100% 125%;" if fam == "Anek Latin" else ""
            faces.append(f'@font-face{{font-family:"{fam}";font-weight:{weight};{stretch}font-style:normal;font-display:swap;'
                         f'src:url(data:font/woff2;base64,{b64}) format("woff2");}}')
    return css.replace("/*FONTS*/", "\n".join(faces))


SCROLL_JS = """() => { if (window.innerWidth < 840) { const r = document.getElementById('wv-result');
  if (r) r.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'}); } }"""
LANG_JS = """(choice) => { document.documentElement.lang = choice === 'हिंदी' ? 'hi' : 'en'; }"""

# Phones: hand a photo from the native camera or gallery picker to the image input.
# Page-frame rules live in <head> because Gradio scopes the css= string under
# ".gradio-container .contain", which breaks selectors on the container itself
# inside @media blocks.
PAGE_CSS = """<style>
html, body { max-width: 100%; }
@media (max-width: 720px) {
  .gradio-container { padding: 0 12px !important; }
  .gradio-container > main.app { padding: 8px 0 16px !important; }
}
</style>"""

PICK_JS = """<script>
(function () {
  const sel = '#wv-photo input[type=file]';
  const wait = ms => new Promise(r => setTimeout(r, ms));
  async function uploadInput() {
    for (let i = 0; i < 60; i++) {
      const el = document.querySelector(sel);
      if (el) return el;
      const btns = [...document.querySelectorAll('#wv-photo button')];
      const label = b => ((b.getAttribute('aria-label') || '') + ' ' + (b.title || '')).toLowerCase();
      const clear = btns.find(b => /clear|remove/.test(label(b)));
      const upload = btns.find(b => /upload/.test(label(b)));
      if (clear) clear.click(); else if (upload) upload.click();
      await wait(60);
    }
    return null;
  }
  document.addEventListener('change', async (e) => {
    const src = e.target;
    if (!src || !src.classList || !src.classList.contains('wv-pick')) return;
    const picked = src.files && src.files[0];
    if (!picked) return;
    // Copy the photo into memory first: a camera photo can be a temporary file the browser releases.
    let file = picked;
    try { file = new File([await picked.arrayBuffer()], picked.name || 'photo.jpg', { type: picked.type || 'image/jpeg' }); } catch (err) {}
    src.value = '';
    const target = await uploadInput();
    if (target) {
      const dt = new DataTransfer();
      dt.items.add(file);
      target.files = dt.files;
      target.dispatchEvent(new Event('change', { bubbles: true }));
    }
    src.value = '';
  }, true);
})();
</script>"""

_CAM = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>'
_GAL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-8 8"/></svg>'


def render_pickers(lang):
    t = T[lang]
    return (f'<div class="wv-pickers" role="group" aria-label="{E(t["photo_label"])}">'
            f'<label class="wv-pickbtn wv-pickcam">{_CAM}<span>{E(t["take_photo"])}</span>'
            f'<input class="wv-pick" type="file" accept="image/*" capture="environment"></label>'
            f'<label class="wv-pickbtn">{_GAL}<span>{E(t["choose_photo"])}</span>'
            f'<input class="wv-pick" type="file" accept="image/*"></label></div>')


# ---------------------------------------------------------------- app
def build(model, store, export_dir):
    lockup = themed_svg(os.path.join(HERE, "assets", "wv-lockup.svg"))
    examples = example_paths(export_dir)
    lang0 = "en"
    t0 = T[lang0]
    head = ('<meta name="theme-color" content="#f2f3ef" media="(prefers-color-scheme: light)">'
            '<meta name="theme-color" content="#121512" media="(prefers-color-scheme: dark)">'
            '<meta name="description" content="WasteVision: photograph one waste item and see whether it goes in the Wet, Dry or Special Care stream.">'
            + PAGE_CSS + PICK_JS)

    with gr.Blocks(theme=make_theme(), css=load_css(), title="WasteVision", analytics_enabled=False, head=head,
                   fill_width=False) as demo:
        st_res = gr.State(None)
        st_lang = gr.State(lang0)

        with gr.Row(elem_id="wv-bar", equal_height=True):
            brand = gr.HTML(f'<div class="wv-brand">{lockup}<h1 class="wv-sr">WasteVision</h1>{render_bar_title(lang0)}</div>',
                            elem_id="wv-brandcell")
            lang = gr.Radio(list(LANGS), value="English", show_label=False, container=False, elem_id="wv-lang",
                            label="Language / भाषा")

        with gr.Tabs(elem_id="wv-tabs"):
            with gr.Tab(t0["tab_sort"], id="sort") as tab_sort:
                with gr.Row(equal_height=False, elem_id="wv-main"):
                    with gr.Column(scale=5, min_width=300, elem_classes="wv-card wv-capture"):
                        pickers = gr.HTML(render_pickers(lang0), elem_id="wv-pickers")
                        image = gr.Image(sources=["upload", "webcam", "clipboard"], type="pil", label=t0["photo_label"],
                                         height=300, show_download_button=False, show_share_button=False,
                                         show_fullscreen_button=False,
                                         webcam_options=gr.WebcamOptions(
                                             mirror=False,
                                             constraints={"video": {"facingMode": {"ideal": "environment"},
                                                                    "width": {"ideal": 1280}, "height": {"ideal": 960}}}),
                                         elem_id="wv-photo")
                        hint = gr.HTML(f'<p class="wv-hint">{E(t0["hint"])}</p>')
                        ex_head = gr.HTML(f'<h2 class="wv-mini wv-exh">{E(t0["examples"])}</h2>', visible=bool(examples))
                        if examples:
                            gr.Examples(examples=examples, inputs=[image], label=None, examples_per_page=12,
                                        elem_id="wv-examples")
                    with gr.Column(scale=6, min_width=300, elem_classes="wv-card wv-resultcol", elem_id="wv-result"):
                        verdict = gr.HTML(render_empty(lang0), elem_id="wv-verdict")
                        why_head = gr.HTML("", elem_id="wv-whyhead")
                        with gr.Row(equal_height=False, elem_id="wv-why", visible=False) as why_row:
                            cam = gr.Image(label=t0["cam_label"], interactive=False, height=176, show_download_button=False,
                                           show_share_button=False, show_fullscreen_button=False, elem_id="wv-cam",
                                           min_width=160, scale=2)
                            with gr.Column(min_width=220, scale=3, elem_id="wv-detailcell"):
                                details = gr.HTML("")
                        with gr.Row(visible=False, elem_id="wv-fb", equal_height=True) as fb_group:
                            fb_q = gr.HTML(f'<p class="wv-q">{E(t0["feedback_q"])}</p>', elem_id="wv-fbq")
                            yes_btn = gr.Button(t0["yes"], variant="secondary", elem_classes="wv-btn", min_width=88, scale=0)
                            no_btn = gr.Button(t0["no"], variant="secondary", elem_classes="wv-btn", min_width=120, scale=0)
                        with gr.Group(visible=False, elem_id="wv-fixgroup") as fix_group:
                            fix = gr.Radio(fix_choices(None, lang0), label=t0["fix_label"], elem_id="wv-fix")
                            save_btn = gr.Button(t0["save_fix"], variant="primary", elem_classes="wv-btn")
                            privacy = gr.HTML(f'<p class="wv-attrib">{E(t0["feedback_privacy"])}</p>')
                        msg = gr.HTML("", elem_id="wv-msg")
                        attrib = gr.HTML(render_attrib(lang0, model))

            with gr.Tab(t0["tab_about"], id="about") as tab_about:
                about = gr.HTML(render_about(lang0, model, lockup))
                gal = gr.Gallery(gallery_items(lang0, export_dir), columns=4, height="auto", allow_preview=True,
                                 show_label=False, object_fit="cover", show_share_button=False, elem_id="wv-gallery")

        footer = gr.HTML(f'<footer class="wv-footer">{E(t0["footer"])}</footer>')

        # ---------------- events
        def on_image(img, lg):
            hide = gr.update(visible=False)
            if img is None:
                return (render_empty(lg), "", gr.update(visible=False), gr.update(value=None), "", hide, hide,
                        gr.update(value=None), "", None)
            try:
                res = model.predict(img)
            except Exception:
                traceback.print_exc()
                return (render_error(lg, "error_model"), "", gr.update(visible=False), gr.update(value=None), "",
                        hide, hide, gr.update(value=None), "", None)
            spot = model.spotlight(res["image"], res["cam"])
            return (render_verdict(res, lg), render_why_head(lg), gr.update(visible=True), gr.update(value=spot),
                    render_details(res, lg), gr.update(visible=SHOW_FEEDBACK), hide,
                    gr.update(choices=fix_choices(res, lg), value=None), "", res)

        image.change(on_image, [image, st_lang],
                     [verdict, why_head, why_row, cam, details, fb_group, fix_group, fix, msg, st_res],
                     show_progress="minimal").then(None, None, None, js=SCROLL_JS)

        def on_yes(res, lg):
            if res is None:
                return gr.update(), ""
            n = store.save(res["image"], res, res["top_class"], model.version)
            return gr.update(visible=False), f'<p class="wv-msg" role="status">{E(T[lg]["saved_yes"].format(n=n))}</p>'

        yes_btn.click(on_yes, [st_res, st_lang], [fb_group, msg])
        no_btn.click(lambda: (gr.update(visible=True), ""), None, [fix_group, msg])

        def on_save(res, choice, lg):
            if res is None:
                return gr.update(), gr.update(), gr.update(), "", res
            if not choice:
                return gr.update(), gr.update(), gr.update(), f'<p class="wv-msg wv-err" role="alert">{E(T[lg]["pick_first"])}</p>', res
            n = store.save(res["image"], res, choice, model.version)
            res = dict(res, corrected=choice)
            if choice != "not_waste":
                res["verdict_key"] = STREAM_KEY[model.stream_of[choice]]
            bin_name = BINS[lg][res["verdict_key"]][0] if choice != "not_waste" else CLASS_NAMES[lg]["not_waste"]
            text = T[lg]["saved_fix"].format(b=bin_name, n=n)
            return (render_verdict(res, lg), gr.update(visible=False), gr.update(visible=False),
                    f'<p class="wv-msg" role="status">{E(text)}</p>', res)

        save_btn.click(on_save, [st_res, fix, st_lang], [verdict, fix_group, fb_group, msg, st_res])

        def on_lang(choice, res):
            lg = LANGS.get(choice, "en")
            t = T[lg]
            return (lg, f'<div class="wv-brand">{lockup}<h1 class="wv-sr">WasteVision</h1>{render_bar_title(lg)}</div>', render_pickers(lg),
                    gr.update(label=t["tab_sort"]), gr.update(label=t["tab_about"]),
                    gr.update(label=t["photo_label"]), f'<p class="wv-hint">{E(t["hint"])}</p>',
                    f'<h2 class="wv-mini wv-exh">{E(t["examples"])}</h2>', render_verdict(res, lg),
                    render_why_head(lg, res is not None), gr.update(label=t["cam_label"]), render_details(res, lg),
                    f'<p class="wv-q">{E(t["feedback_q"])}</p>', gr.update(value=t["yes"]), gr.update(value=t["no"]),
                    gr.update(label=t["fix_label"], choices=fix_choices(res, lg)), gr.update(value=t["save_fix"]),
                    f'<p class="wv-attrib">{E(t["feedback_privacy"])}</p>', "", render_attrib(lg, model),
                    render_about(lg, model, lockup), gr.update(value=gallery_items(lg, export_dir)),
                    f'<footer class="wv-footer">{E(t["footer"])}</footer>')

        lang.change(on_lang, [lang, st_res],
                    [st_lang, brand, pickers, tab_sort, tab_about, image, hint, ex_head, verdict, why_head, cam, details,
                     fb_q, yes_btn, no_btn, fix, save_btn, privacy, msg, attrib, about, gal, footer],
                    show_progress="hidden")
        lang.change(None, [lang], None, js=LANG_JS)
    return demo


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def main():
    on_space = bool(os.environ.get("SPACE_ID"))
    ap = argparse.ArgumentParser(description="WasteVision demo app")
    ap.add_argument("--export", default=os.path.join(HERE, "export"), help="folder with config.json and the model")
    ap.add_argument("--feedback", default=os.path.join(HERE, "feedback"), help="where corrections are saved")
    ap.add_argument("--port", type=int, default=7860 if on_space else None, help="port (default: first free from 7860)")
    ap.add_argument("--lan", action="store_true", default=on_space, help="also serve on this computer's Wi-Fi address")
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--no-browser", action="store_true", default=on_space)
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.export, "config.json")):
        sys.exit(f"Model folder not found: {a.export}\nPut the 'export' folder from Google Drive next to app.py.")
    print("Loading the WasteVision model ...", flush=True)
    model = WasteModel(a.export)
    print(f"Model {model.version} ({model.arch}) ready in {model.load_seconds:.1f}s", flush=True)
    store = FeedbackStore(a.feedback)
    demo = build(model, store, a.export)
    if a.lan and not on_space and (ip := lan_ip()):
        print(f"On a phone on the same Wi-Fi, open: http://{ip}:{a.port or 7860}", flush=True)
    demo.queue(default_concurrency_limit=1, max_size=32).launch(
        server_name="0.0.0.0" if a.lan else "127.0.0.1", server_port=a.port, share=a.share,
        inbrowser=not a.no_browser, show_api=False, ssr_mode=False, allowed_paths=[a.export],
        favicon_path=os.path.join(HERE, "assets", "favicon.png"), max_file_size="25mb", show_error=True)


if __name__ == "__main__":
    main()
