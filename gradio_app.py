import uuid
import gradio as gr

from app.main import build_orchestrator

# Initialize the orchestrator with memory store for HF Spaces compatibility (read-only filesystem)
orchestrator = build_orchestrator(use_memory_store=True)

custom_css = """
.gradio-container { max-width: 100% !important; padding: 1.5rem !important; }
.clear-btn-container button { width: 100% !important; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=custom_css) as demo:
    gr.Markdown("# Mental Health Support Chatbot")
    gr.Markdown("A RAG-based Mental Health Support Chatbot to help you with your thoughts and feelings.")
    
    session_state = gr.State(lambda: str(uuid.uuid4()))
    
    # We use type="messages" to support dictionary-based chat history
    chatbot = gr.Chatbot(value=[])
    
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(placeholder="Type your message here...", show_label=False, container=False)
        with gr.Column(scale=4):
            audio_in = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                show_label=False, 
                container=False,
                interactive=True
            )
            
    with gr.Row(elem_classes="clear-btn-container"):
        clear = gr.ClearButton([msg, chatbot, audio_in], variant="secondary", size="md")

    # --- Text Input Handlers ---
    def user_text_input(user_message, history):
        if not user_message.strip():
            return "", history
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_text_response(history, session_id):
        if not history:
            return history, session_id
        if not session_id or callable(session_id):
            session_id = str(uuid.uuid4())
            
        # Safely extract the text from the dictionary
        user_message = history[-1]["content"]
        
        try:
            bot_reply = orchestrator.process_message(user_message=user_message, session_id=session_id)
            history.append({"role": "assistant", "content": bot_reply})
        except Exception as e:
            history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
            
        return history, session_id

    # --- Audio Input Handlers ---
    def user_audio_input(audio_path, history):
        if not audio_path:
            return audio_path, history
        if history is None:
            history = []
        # Immediately display a placeholder in the UI so the user knows it's working
        return audio_path, history + [{"role": "user", "content": "🎤 [Voice Message Processing...]"}]

    def bot_audio_response(audio_path, history, session_id):
        if not audio_path:
            return None, history, session_id
        if not session_id or callable(session_id):
            session_id = str(uuid.uuid4())
            
        try:
            bot_reply, transcribed_text = orchestrator.process_message(audio_file_path=audio_path, session_id=session_id)
            # Update the last user message to show it was sent, then append bot response
            history[-1]["content"] = f"🎤 {transcribed_text}"
            history.append({"role": "assistant", "content": bot_reply})
        except Exception as e:
            history[-1]["content"] = "🎤 *[Voice note failed to transcribe]*"
            history.append({"role": "assistant", "content": f"⚠️ Audio Error: {str(e)}"})
            
        # Return None for the audio component to clear it after processing
        return gr.update(value=None), history, session_id

    # --- Event Wiring ---
    # Text Submission Chain
    msg.submit(
        user_text_input, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot], 
        queue=False
    ).then(
        bot_text_response, 
        inputs=[chatbot, session_state], 
        outputs=[chatbot, session_state]
    )

    # Audio Recording Chain (Triggers when recording stops)
    audio_in.stop_recording(
        user_audio_input, 
        inputs=[audio_in, chatbot], 
        outputs=[audio_in, chatbot], 
        queue=False
    ).then(
        bot_audio_response, 
        inputs=[audio_in, chatbot, session_state], 
        outputs=[audio_in, chatbot, session_state]
    )

if __name__ == "__main__":
    demo.launch()