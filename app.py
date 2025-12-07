import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Note: Gradio is not needed here anymore.

# --- 1. Model Loading ---

# Use Streamlit's caching decorator for resource-heavy objects like models
@st.cache_resource
def load_model_and_tokenizer():
    # Load model (using a small CPU-friendly model for testing)
    model_name = "gpt2"
    
    # Inform the user what's happening
    st.info(f"Loading model: {model_name}. This happens once.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Using 'cpu' to avoid resource errors unless you have a paid GPU plan
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Load resources (will only run once due to the cache decorator)
tokenizer, model = load_model_and_tokenizer()

def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Ensure the model is moved to the appropriate device if needed, 
    # though 'gpt2' usually runs fine on CPU for small requests.
    
    with torch.no_grad():
        # Set max_new_tokens for generation limit
        outputs = model.generate(**inputs, max_new_tokens=max_length)
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 2. Streamlit Interface ---

st.title("🤖 My Personal AI Demo")
st.markdown("Enter a prompt and hit the button to generate text using the **GPT-2** model.")

# Create the input area
prompt_input = st.text_area(
    "Enter your prompt:", 
    placeholder="What are the three best ways to learn Python?",
    height=100
)

# Create the generate button
if st.button("Generate Text"):
    if prompt_input:
        # Show a spinner while the model is running
        with st.spinner('Generating response...'):
            generated_text = generate_text(prompt_input)
            
            # Display the result
            st.subheader("Generated Response:")
            st.success(generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")

# The Gradio launch block (if __name__ == "__main__": iface.launch(...)) is removed. 
# Streamlit Cloud runs the script directly.
