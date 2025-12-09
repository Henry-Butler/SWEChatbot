# app.py
import streamlit as st
import torch
import os
import threading
import time
from flask import Flask, request, jsonify
import requests
from Chatbot import Tokenizer, ModelFactory, CFG, mhaModel, Chatbot, ModelBuilder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tok = Tokenizer(model_prefix=CFG["tokenizer_prefix"], vocab_size=CFG["vocab_size"])
tok.load()

# Load model
vocab_size = tok.processor.GetPieceSize()
factory = ModelFactory()
model = factory.getModel(
    model_type="mha",
    vocab_size=vocab_size,
    max_seq_len=CFG["seq_len"] - 1,
    d_model=CFG["d_model"],
    n_layers=CFG["n_layers"],
    n_heads=CFG["n_heads"],
    d_ff=CFG["d_ff"],
    dropout=CFG["dropout"]
)
# Load trained weights
model_path = os.path.join(os.getcwd(), "final_model.pt")
model.load_weights(model_path, map_location=device)
model.to(device)
model.eval()

# Setup chatbot wrapper
chatbot = Chatbot(model, tok)

# --- Flask API (runs in background thread) ---
flask_app = Flask(__name__)


@flask_app.route('/generate', methods=['POST'])
def api_generate():
    # Accept JSON or form
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': '`prompt` is required'}), 400

    strategy = (data.get('strategy') or 'greedy').lower()
    try:
        max_new_tokens = int(data.get('max_new_tokens', 50))
    except Exception:
        max_new_tokens = 50
    try:
        k = int(data.get('k', 50))
    except Exception:
        k = 50
    try:
        p = float(data.get('p', 0.9))
    except Exception:
        p = 0.9
    try:
        temperature = float(data.get('temperature', 1.0))
    except Exception:
        temperature = 1.0

    try:
        if strategy == 'greedy':
            out = chatbot.greedy(prompt, max_new_tokens=max_new_tokens)
        elif strategy in ('top_k', 'top-k'):
            out = chatbot.top_k(prompt, max_new_tokens=max_new_tokens, k=k, temperature=temperature)
        elif strategy in ('top_p', 'top-p'):
            out = chatbot.top_p(prompt, max_new_tokens=max_new_tokens, p=p, temperature=temperature)
        else:
            return jsonify({'error': "Unknown strategy. Use 'greedy', 'top_k', or 'top_p'."}), 400
    except Exception as e:
        return jsonify({'error': f'Generation failed: {e}'}), 500

    return jsonify({'generated': out})


@flask_app.route('/health', methods=['GET'])
def api_health():
    # Basic health check: model present and callable
    ok = chatbot is not None
    return jsonify({'ok': ok})


def run_flask():
    # Disable reloader to avoid spawning multiple threads
    flask_app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)

# Start Flask server once per Streamlit session
if 'flask_thread_started' not in st.session_state:
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    # small sleep to give server a moment to start (optional)
    time.sleep(0.1)
    st.session_state['flask_thread_started'] = True
    st.info('Flask API started at http://127.0.0.1:8000 (endpoints: /generate, /health)')

# Streamlit UI
st.title("📜 Custom GPT-2 Chatbot (MHA)")
st.write("Generate text using a GPT-2 style model.")

prompt = st.text_area("Enter prompt:", height=150)

generation_type = st.radio(
    "Generation method:",
    ("Greedy", "Top-K", "Top-P")
)

max_tokens = st.slider("Max new tokens:", 10, 127, 50)
k_val = st.slider("Top-K value:", 10, 100, 50)
p_val = st.slider("Top-P (nucleus) probability:", 0.1, 1.0, 0.9)
temp = st.slider("Temperature:", 0.1, 2.0, 1.0)
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        # Prepare payload for API
        strategy_map = {
            "Greedy": "greedy",
            "Top-K": "top_k",
            "Top-P": "top_p",
        }
        payload = {
            "prompt": prompt,
            "strategy": strategy_map.get(generation_type, "greedy"),
            "max_new_tokens": int(max_tokens),
            "k": int(k_val),
            "p": float(p_val),
            "temperature": float(temp),
        }

        with st.spinner("Generating via API..."):
            try:
                resp = requests.post("http://127.0.0.1:8000/generate", json=payload, timeout=120)
            except Exception as e:
                st.error(f"Failed to call API: {e}")
                resp = None

        if resp is None:
            pass
        else:
            if resp.status_code != 200:
                # Try to show JSON error message if present
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                st.error(f"API error ({resp.status_code}): {err}")
            else:
                try:
                    out_json = resp.json()
                    output = out_json.get("generated", "")
                except Exception as e:
                    st.error(f"Invalid response from API: {e}")
                    output = ""

                st.success("Generated text:")
                st.text_area("Output:", output, height=300)