import uuid

import gradio as gr

from app.main import build_orchestrator

# Initialize the orchestrator with memory store for HF Spaces compatibility (read-only filesystem)
orchestrator = build_orchestrator(use_memory_store=True)

with gr.Blocks() as demo:
    gr.Markdown("# Mental Health Support Chatbot")
    gr.Markdown("A RAG-based Mental Health Support Chatbot to help you with your thoughts and feelings.")
    
    session_state = gr.State(lambda: str(uuid.uuid4()))
    
    chatbot = gr.Chatbot(value=[])
    msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
    clear = gr.ClearButton([msg, chatbot])

    def user_input(user_message, history):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_response(history, session_id):
        if not history:
            return history, session_id
        if not session_id or callable(session_id):
            session_id = str(uuid.uuid4())
        user_message = history[-1]["content"][0]["text"]
        bot_reply = orchestrator.process_message(user_message, session_id=session_id)
        history.append({"role": "assistant", "content": bot_reply})
        return history, session_id

    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, session_state], [chatbot, session_state]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
