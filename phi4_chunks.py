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

# Define system instruction for ML/Data Science expertise
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in machine learning, deep learning, and data science. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.
When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a data science expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.
Remember:
- You are an expert in machine learning, deep learning, and data science, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside the chunks.
- If asked about topics unrelated to data science or machine learning, politely redirect the conversation back to related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.
Your goal is to provide accurate, helpful information about machine learning, deep learning, and data science based solely on the content of the text chunks you receive with each query.
"""

# Sample ML/Data Science context chunks
ML_CHUNKS = [
    """Convolutional Neural Networks (CNNs) are a specialized type of neural network for processing structured grid data such as images. CNNs use convolutional layers that apply sliding filters across the input, effectively learning spatial hierarchies of features. A typical CNN architecture consists of convolutional layers, pooling layers for downsampling, and fully connected layers for classification. Key components include convolutional filters that detect features like edges and textures, activation functions like ReLU, and pooling operations that reduce dimensionality while preserving important information.""",
    
    """Recurrent Neural Networks (RNNs) are neural networks designed to work with sequential data by maintaining an internal state (memory). Unlike feedforward networks, RNNs have connections that feed back into the network, allowing them to use their internal state to process sequences of inputs. However, basic RNNs suffer from vanishing/exploding gradient problems, making them difficult to train on long sequences. LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) architectures were developed to address these limitations by incorporating gating mechanisms that better control information flow through the network.""",
    
    """Random Forests are an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests correct for decision trees' habit of overfitting to their training set by training each tree on a random subset of the data (bagging) and features. This approach creates diverse trees whose predictions have low correlation, making the ensemble more robust and accurate than individual trees. Random forests are known for their high accuracy, ability to handle large datasets with higher dimensionality, and resistance to overfitting.""",
    
    """Gradient Boosting is a machine learning technique that produces a strong predictive model by combining multiple weaker models, typically decision trees. Unlike random forests, which train trees independently, gradient boosting builds trees sequentially, with each new tree correcting errors made by the previous ones. The algorithm minimizes a loss function by adding models that follow the negative gradient of the loss. Popular implementations include XGBoost, LightGBM, and CatBoost, which offer various optimizations for speed and performance. Gradient boosting typically provides higher accuracy than random forests but may be more prone to overfitting without proper regularization.""",
    
    """Support Vector Machines (SVMs) are supervised learning models used for classification and regression tasks. The algorithm works by finding the hyperplane that best separates data points of different classes while maximizing the margin between the closest points (support vectors) from each class. SVMs can perform linear and non-linear classification by using the kernel trick, which implicitly maps inputs into high-dimensional feature spaces. Common kernels include linear, polynomial, and radial basis function (RBF). SVMs are effective in high-dimensional spaces, memory efficient, and versatile through different kernel functions.""",
    
    """Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. PCA works by identifying the principal components (eigenvectors) of the data's covariance matrix, which represent the directions of maximum variance. By projecting the data onto these principal components, PCA achieves dimensionality reduction while minimizing information loss. PCA is commonly used for data visualization, noise reduction, and as a preprocessing step for machine learning algorithms to mitigate the curse of dimensionality."""
]

def format_prompt_with_system_instruction(question, context=None):
    """Format prompt with system instruction for Phi-4"""
    # For Phi-4, we'll integrate system instruction into the prompt
    if context:
        return f"<|system|>\n{SYSTEM_INSTRUCTION}\n<|user|>\nContext: {context}\n\nQuestion: {question}\n<|assistant|>"
    else:
        return f"<|system|>\n{SYSTEM_INSTRUCTION}\n<|user|>\n{question}\n<|assistant|>"

def main():
    # Print startup message
    print("\n===== Phi-4 ML/Data Science Expert (WITH CONTEXT) =====")
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
    print("This version uses PROVIDED CONTEXT CHUNKS for answers.")
    print("Type your machine learning and data science questions below.")
    print("Type 'exit', 'quit', or 'q' to end the program.\n")
    
    # Join the chunks with a separator
    context = "\n---\n".join(ML_CHUNKS)
    
    # Interactive loop
    while True:
        # Get user question
        question = input("Question: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        # Format the prompt with system instruction and context
        prompt = format_prompt_with_system_instruction(question, context)
        
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