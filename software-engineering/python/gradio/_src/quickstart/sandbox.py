import gradio as gr
from datetime import datetime
import uuid, json, os

SESSIONS_FILE = "sessions.json"   # ←超簡易保存

def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        return json.load(open(SESSIONS_FILE))
    return {}

def save_sessions(sessions):
    json.dump(sessions, open(SESSIONS_FILE,"w"), ensure_ascii=False, indent=2)

sessions = load_sessions()
current_id  = list(sessions)[0] if sessions else None

def bot_fn(message, history):
    # ここに LLM 呼び出しを書く
    return "（ダミー回答）"   # ChatInterface 用

def new_chat():
    sid = datetime.now().strftime("%m%d-%H%M-")+uuid.uuid4().hex[:4]
    sessions[sid] = []
    save_sessions(sessions)
    return gr.update(choices=list(sessions), value=sid), []

def choose_chat(sid):
    return sid, sessions.get(sid, [])

def save_turn(sid, history):
    sessions[sid] = history
    save_sessions(sessions)

with gr.Blocks(theme=gr.themes.Default()) as demo:
    # ---------- 左サイドバー ----------
    with gr.Sidebar():
        gr.Markdown("### 会話履歴")
        sid_select = gr.Radio(list(sessions), label="", value=current_id)
        new_btn    = gr.Button("＋ 新規チャット")

    # ---------- 中央 ChatGPT 風 ----------
    chat = gr.ChatInterface(bot_fn, title="どこから始めますか？")

    # --------- イベント結線 ----------
    new_btn.click(new_chat, outputs=[sid_select, chat])
    sid_select.change(choose_chat, inputs=sid_select, outputs=[sid_select, chat])
    new_btn.click(new_chat, outputs=[sid_select, chat])

    sid_select.change(
        fn=choose_chat,
        inputs=sid_select,
        outputs=[sid_select, chat],
    )

    # チャット履歴が更新されるたびにセーブ
    chat.chatbot.change(
        fn=save_turn,
        inputs=[sid_select, chat.chatbot],
        outputs=[],
    )

demo.launch()
