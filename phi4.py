import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Define generation config
GENERATION_CONFIG = {
    "max_length": 8192,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
    "repetition_penalty": 1.2,  # Reduce repetition
    "do_sample": True  # Enable sampling
}

# Define system instruction for ML/Data Science expertise (NO CONTEXT VERSION)
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in machine learning, deep learning, and data science. You provide helpful, accurate, and educational responses to questions about these topics.

When answering a query:
1. Provide clear explanations with appropriate technical detail for the complexity of the question.
2. When explaining concepts, include practical examples to illustrate how they work.
3. If relevant, mention advantages, limitations, and common use cases.
4. For complex topics, break down your explanation into understandable components.
5. Maintain a professional and educational tone throughout your responses.

If asked about topics unrelated to machine learning, deep learning, or data science, you can still answer, but state that this is outside your primary area of expertise.

Your goal is to be a helpful and informative resource for people learning about or working in machine learning, deep learning, and data science.
"""

def format_prompt_with_system_instruction(question):
    """Format prompt with system instruction for Phi-4 (no context version)"""
    return f"<|system|>\n{SYSTEM_INSTRUCTION}\n<|user|>\n{question}\n<|assistant|>"

def main():
    # Print startup message
    print("\n===== Phi-4 ML/Data Science Expert (NO CONTEXT) =====")
    print("Loading model, please wait...\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "microsoft/Phi-4-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimizations for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Move model to device if not using device_map="auto"
    if device == "cpu":
        model = model.to(device)
    
    print("\nML/Data Science Expert Model loaded successfully!")
    print("This version uses NO CONTEXT CHUNKS - model will use its own knowledge.")
    print("Type your machine learning and data science questions below.")
    print("Type 'exit', 'quit', or 'q' to end the program.\n")
    
    # Interactive loop
    while True:
        # Get user question
        question = input("Question: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        # Format the prompt with system instruction (no context)
        prompt = format_prompt_with_system_instruction(question)
        
        # Tokenize the prompt with proper attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Generate answer
        print("Generating answer...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **GENERATION_CONFIG,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the assistant's reply
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("<|assistant|>")[-1].strip()

        # Print the response
        print("\nAnswer:")
        print(assistant_response)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()