import gradio as gr

def echo(message, history):
    return message

demo = gr.ChatInterface(fn=echo, type="messages", examples=["hello", "hola", "merhaba"], title="Echo Bot")
demo.launch()
