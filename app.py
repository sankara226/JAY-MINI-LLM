"""Chat with Jay: a Gradio web page for the tiny language model.

Run locally:   python app.py
Run in Colab:  from app import launch; launch(weights_dir="/content/drive/MyDrive/tiny_llm", share=True)

"""
import html
import inspect
import os
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from jay.chat import (
    BLOCK_SIZE, MODEL_FILES, available_versions, decode, decode_partial, load_jay,
    peek_inside, prepare_prompt, stream_reply_ids, turn_ids,
)

HERE = Path(__file__).parent
WEIGHTS_DIR = Path(os.environ.get("JAY_WEIGHTS_DIR", HERE / "weights"))
KARPATHY_VIDEO = "https://www.youtube.com/watch?v=7xTGNNLPyMI"


# ---------------------------------------------------------------- weights

def download_weights_from_hub(weights_dir):
    """On Hugging Face Spaces: fetch the weights from the model repo named in JAY_MODEL_REPO."""
    repo_id = os.environ.get("JAY_MODEL_REPO")
    if not repo_id:
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    hf_hub_download(repo_id, "tokenizer.json", local_dir=weights_dir)
    for filename in MODEL_FILES.values():
        try:
            hf_hub_download(repo_id, filename, local_dir=weights_dir)
        except EntryNotFoundError:
            pass  # that version was not uploaded; the dropdown simply won't offer it


_loaded = {}  # version -> Jay, so each model is loaded only once


def get_jay(weights_dir, version):
    key = (str(weights_dir), version)
    if key not in _loaded:
        _loaded[key] = load_jay(weights_dir, MODEL_FILES[version])
    return _loaded[key]


# ---------------------------------------------------------------- helpers

def message_text(content):
    """Gradio can store a message as a string or as a list of parts: keep only the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, (list, tuple)):
        return "".join(message_text(part) for part in content)
    return ""


def history_to_pairs(history):
    """Gradio messages [{"role": ..., "content": ...}] -> [(user_msg, jay_reply), ...]"""
    pairs, pending_user = [], None
    for msg in history:
        text = message_text(msg.get("content"))
        if msg.get("role") == "user":
            pending_user = text
        elif msg.get("role") == "assistant" and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs


def memory_bar(used=0, forgotten_turns=0, remembering=False):
    """A thin bar: how much of Jay's 256-token context window the last prompt + reply filled."""
    used = min(used, BLOCK_SIZE)
    if not remembering:
        note = "Jay only sees your latest message. Turn on “Remember the conversation” to replay the chat."
    elif forgotten_turns:
        s = "s" if forgotten_turns > 1 else ""
        note = f"Jay forgot the {forgotten_turns} oldest exchange{s} to make room."
    else:
        note = "When the chat gets too long, Jay forgets the oldest exchanges first."
    return f"""
<div class="memory">
  <div class="memory-row"><span>Memory</span><span class="num">{used} of {BLOCK_SIZE} tokens</span></div>
  <div class="memory-track" role="progressbar" aria-label="Memory used" aria-valuemin="0"
       aria-valuemax="{BLOCK_SIZE}" aria-valuenow="{used}"><div style="width:{used / BLOCK_SIZE * 100:.1f}%"></div></div>
  <p>{note}</p>
</div>"""


PEEK_HINT = '<p class="hint">Send Jay a message first, then look inside its last reply.</p>'


def top_tokens_html(top):
    rows = "".join(
        f'<li><code>{html.escape(repr(tok))}</code>'
        f'<span class="bar"><span style="width:{p * 100:.1f}%"></span></span>'
        f'<span class="num">{p * 100:.1f}%</span></li>'
        for tok, p in top
    )
    return (f'<div class="top5"><p>Jay\'s top 5 guesses for the first token of its reply, '
            f"before any randomness:</p><ol>{rows}</ol></div>")


def heatmap_figure(tokens, weights, layer, head):
    """Transparent background + a see-through blue scale, so it reads in light and dark mode."""
    blue = LinearSegmentedColormap.from_list("jay", [(0.15, 0.39, 0.92, 0.0), (0.15, 0.39, 0.92, 1.0)])
    grey = "#8E8E8E"
    labels = [t.replace("\n", "\\n") for t in tokens]
    size = max(4.0, min(7.0, 0.2 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.imshow(weights, cmap=blue, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=9, color=grey, family="monospace")
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=grey, family="monospace")
    ax.set_title(f"Layer {layer}, head {head}", color=grey, fontsize=10, loc="left")
    ax.set_xlabel("looks back at this token", color=grey, fontsize=9)
    ax.set_ylabel("this token", color=grey, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color(grey)
        spine.set_alpha(0.3)
    ax.tick_params(colors=grey, length=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- page

EMPTY_CHAT = f"""## Chat with Jay

A tiny language model built from scratch. It tells short stories for kids, adds small numbers step by step, and says “I don't know” to most other questions. Built by following Andrej Karpathy's [Deep Dive into LLMs like ChatGPT]({KARPATHY_VIDEO})."""

NOTE = ('<p class="note">Jay is a learning project. It makes mistakes and knows almost nothing '
        'about the world.</p>')

ABOUT = f"""
<div class="about">
  <p><b>Jay</b> is a 12.3M-parameter GPT trained from scratch on children's stories
  (TinyStories), then taught to chat, to say “I don't know”, and to practise sums.</p>
  <p><b>v1</b> first chat model · <b>v2</b> says “I don't know” · <b>v3</b> after practising sums</p>
  <p><a href="{KARPATHY_VIDEO}" target="_blank" rel="noopener">Watch the video that inspired it</a></p>
</div>"""

EXAMPLES = ["Hi!", "Who are you?", "Tell me a story about a dog.", "What is 4 + 3?",
            "What is the capital of Japan?"]


def build_demo(weights_dir=WEIGHTS_DIR):
    weights_dir = Path(weights_dir)
    download_weights_from_hub(weights_dir)
    versions = available_versions(weights_dir)
    if not versions:
        raise FileNotFoundError(
            f"No chat_model*.pt in {weights_dir}. Copy tokenizer.json and chat_model_v*.pt there "
            "from Google Drive (MyDrive/tiny_llm), or set JAY_WEIGHTS_DIR / JAY_MODEL_REPO.")

    def respond(message, history, temperature, max_new_tokens, version, remember):
        jay = get_jay(weights_dir, version)
        # Jay was trained almost only on single questions: earlier turns often confuse it,
        # so the conversation is only replayed when "Remember the conversation" is on.
        pairs = history_to_pairs(history) if remember else []
        prompt_ids, kept = prepare_prompt(jay.tokenizer, pairs, message)
        forgotten = len(pairs) - len(kept)

        def bar_now(reply_len):
            return memory_bar(len(prompt_ids) + reply_len, forgotten, remember)

        reply_ids = []
        yield "", bar_now(0), None
        for token_id in stream_reply_ids(jay, prompt_ids, temperature, int(max_new_tokens)):
            reply_ids.append(token_id)
            yield decode_partial(jay.tokenizer, reply_ids), bar_now(len(reply_ids)), None
        last = {"version": version, "prompt": prompt_ids, "reply": reply_ids}
        final = decode(jay.tokenizer, reply_ids).strip() or "…"
        yield final, bar_now(len(reply_ids)), last

    def peek(last, layer, head):
        if not last:
            return PEEK_HINT, None
        jay = get_jay(weights_dir, last["version"])
        top, tokens, weights = peek_inside(jay, last["prompt"], last["reply"], int(layer) - 1, int(head) - 1,
                                          max_heatmap_tokens=24)
        return top_tokens_html(top), heatmap_figure(tokens, weights, int(layer), int(head))

    with gr.Blocks(title="Chat with Jay", fill_height=True) as demo:
        last_reply = gr.State(None)

        # ---- Sidebar: everything secondary to the conversation
        with gr.Sidebar(width=280, open=True, elem_id="sidebar"):
            new_chat = gr.Button("New chat", variant="secondary", elem_id="new-chat")
            gr.HTML('<h2 class="side-title">Settings</h2>', padding=False, container=False)
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature",
                                    info="Low = predictable, high = surprising")
            max_tokens = gr.Slider(20, 200, value=150, step=10, label="Max new tokens",
                                   info="The longest reply Jay may write")
            remember = gr.Checkbox(False, label="Remember the conversation",
                                   info="Jay was trained on single questions, so earlier messages "
                                        "often confuse it.")
            meter = gr.HTML(memory_bar(), padding=False, container=False)
            with gr.Accordion("Peek inside", open=False, elem_id="peek"):
                layer = gr.Slider(1, 6, value=1, step=1, label="Layer")
                head = gr.Slider(1, 6, value=1, step=1, label="Head")
                peek_btn = gr.Button("Look at the last reply", variant="secondary", size="sm")
                top5 = gr.HTML(PEEK_HINT, padding=False)
                heatmap = gr.Plot(show_label=False)
            with gr.Accordion("About Jay", open=False):
                gr.HTML(ABOUT, padding=False)

        # ---- Main column: model picker, conversation, composer, note
        with gr.Column(elem_id="main"):
            version = gr.Dropdown([(f"Jay {v}", v) for v in versions], value=versions[0],
                                  show_label=False, container=False, elem_id="model-picker",
                                  min_width=120, scale=0)
            chat = gr.ChatInterface(
                respond,
                chatbot=gr.Chatbot(placeholder=EMPTY_CHAT, show_label=False, elem_id="jay-chat",
                                   layout="bubble", scale=1),
                textbox=gr.Textbox(placeholder="Message Jay", show_label=False, submit_btn=True,
                                   stop_btn=True, autofocus=True, elem_id="composer", lines=1,
                                   max_lines=8),
                additional_inputs=[temperature, max_tokens, version, remember],
                additional_outputs=[meter, last_reply],
                examples=[[e] for e in EXAMPLES],
                run_examples_on_click=True,
                fill_height=True,
            )
            gr.HTML(NOTE, padding=False, container=False)

        new_chat.click(lambda: ([], [], memory_bar(), None, PEEK_HINT, None), None,
                       [chat.chatbot, chat.chatbot_state, meter, last_reply, top5, heatmap])
        peek_btn.click(peek, [last_reply, layer, head], [top5, heatmap])
        for slider in (layer, head):
            slider.release(peek, [last_reply, layer, head], [top5, heatmap])

    return demo


# ---------------------------------------------------------------- theme

# Colour tokens: (light, dark). The same values are in static/jay.css.
COLORS = {
    "bg": ("#FFFFFF", "#212121"), "sidebar": ("#F9F9F9", "#181818"),
    "surface": ("#F4F4F4", "#303030"), "border": ("#E5E5E5", "#3A3A3A"),
    "text": ("#0D0D0D", "#ECECEC"), "text2": ("#5D5D5D", "#B4B4B4"),
    "strong": ("#0D0D0D", "#FFFFFF"), "on_strong": ("#FFFFFF", "#0D0D0D"),
    "focus": ("#2563EB", "#60A5FA"),
}
SYSTEM_FONT = ["ui-sans-serif", "-apple-system", "system-ui", "Segoe UI", "Roboto", "Helvetica", "Arial",
               "sans-serif"]
SYSTEM_MONO = ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"]


def make_theme():
    """Gradio's own settings, mapped onto the colour tokens for light and dark mode."""
    theme = gr.themes.Base(font=SYSTEM_FONT, font_mono=SYSTEM_MONO, radius_size=gr.themes.sizes.radius_lg)
    mapping = {
        "body_background_fill": "bg", "background_fill_primary": "bg", "background_fill_secondary": "sidebar",
        "block_background_fill": "bg", "panel_background_fill": "sidebar",
        "body_text_color": "text", "body_text_color_subdued": "text2", "block_label_text_color": "text2",
        "block_title_text_color": "text", "block_info_text_color": "text2", "accordion_text_color": "text",
        "border_color_primary": "border", "block_border_color": "border",
        "input_background_fill": "surface", "input_border_color": "border",
        "input_border_color_focus": "focus", "color_accent": "focus", "slider_color": "strong",
        "checkbox_background_color_selected": "strong", "checkbox_border_color_selected": "strong",
        "button_primary_background_fill": "strong", "button_primary_background_fill_hover": "text2",
        "button_primary_text_color": "on_strong", "button_primary_border_color": "strong",
        "button_secondary_background_fill": "bg", "button_secondary_background_fill_hover": "surface",
        "button_secondary_text_color": "text", "button_secondary_border_color": "border",
        "link_text_color": "text", "link_text_color_hover": "text2", "code_background_fill": "surface",
    }
    settable = inspect.signature(theme.set).parameters  # not every setting has a "_dark" twin
    values = {}
    for setting, token in mapping.items():
        light, dark = COLORS[token]
        values[setting] = light
        if f"{setting}_dark" in settable:
            values[f"{setting}_dark"] = dark
    values.update(shadow_drop="none", shadow_drop_lg="none", block_shadow="none", block_shadow_dark="none")
    return theme.set(**values)


# Runs once in the browser: on phones, start with the sidebar closed.
PAGE_JS = """() => {
  if (window.innerWidth < 768) {
    setTimeout(() => document.querySelector('.sidebar.open .toggle-button')?.click(), 300);
  }
}"""


def launch(weights_dir=WEIGHTS_DIR, **kwargs):
    """Build and start the app with Jay's styling. kwargs go to demo.launch (e.g. share=True)."""
    demo = build_demo(weights_dir)
    demo.launch(theme=make_theme(), css_paths=[HERE / "static" / "jay.css"], js=PAGE_JS, **kwargs)
    return demo


if __name__ == "__main__":
    launch()
