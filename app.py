# Force correct huggingface-hub version at runtime (optional safety)
import os
os.system("pip install huggingface-hub==0.26.2 --no-cache-dir")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Load model (using a small CPU-friendly model for testing)
model_name = "gpt2"  # you can change to your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt..."),
    outputs=gr.Textbox(label="Generated Text"),
    title="My Personal AI"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)


