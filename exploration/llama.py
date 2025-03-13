import requests
import logging
import sys

# Ollama API endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_llama')

# Define generation config with improved parameters
GENERATION_CONFIG = {
    "max_length": 256,
    "temperature": 0.1, 
    "top_p": 0.9,
    "repeat_penalty": 1.3,
    "stream": False
}

def format_prompt(system_instruction, question):
    """Format the prompt for Ollama."""
    prompt = f"""
{system_instruction}

My question is: {question}
"""
    return prompt

def query_ollama(question, model_name=OLLAMA_MODEL):
    """
    Query the Ollama model directly without any RAG or context retrieval.
    """
    try:
        # Use simplified system prompt
        system_prompt = """
        You are an AI assistant specialized in machine learning, deep learning, and data science. You provide helpful, accurate, and educational responses to questions about these topics.

        When answering a query:
        1. Provide clear explanations with appropriate technical detail for the complexity of the question.
        2. When explaining concepts, include practical examples to illustrate how they work.
        3. If relevant, mention advantages, limitations, and common use cases.
        4. Break down your explanation into understandable components.
        5. Maintain a professional and educational tone throughout your responses.
        6. Keep your answers concise and to the point.
        7. If you don't know the answer, say so.
        8. Do not include unnecessary information or repetitive explanations.
        9. Format your response clearly and directly address the question.
        """
        
        # Format the prompt
        prompt = format_prompt(system_prompt, question)
        
        # Prepare the request payload for Ollama
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": GENERATION_CONFIG["stream"],
            "temperature": GENERATION_CONFIG["temperature"],
            "top_p": GENERATION_CONFIG["top_p"],
            "repeat_penalty": GENERATION_CONFIG["repeat_penalty"],
        }
        
        logger.info(f"Sending request to Ollama API with model: {model_name}")
        
        # Call Ollama API
        response = requests.post(OLLAMA_URL, json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            assistant_response = result.get("response", "")
            logger.info("Successfully generated response")
            return assistant_response
        else:
            error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Sorry, I encountered an error while processing your question: {error_msg}"
    
    except Exception as e:
        error_msg = f"Error in query_ollama: {str(e)}"
        logger.exception(error_msg)
        return f"Sorry, I encountered an error while processing your question: {error_msg}"

def main():
    """Main function to run an interactive Ollama query system without RAG."""
    try:
        print("\n===== Simple Ollama Question Answering System =====")
        print("This script queries Ollama's Llama 3.1 model directly without RAG.")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("-" * 60)
        
        # Default model name for Ollama
        model_name = OLLAMA_MODEL
        
        # Allow command-line override of model
        if len(sys.argv) > 1:
            model_name = sys.argv[1]
            print(f"Using model: {model_name}")
        
        while True:
            # Get user question
            question = input("\nPlease enter your question: ").strip()
            
            # Check if user wants to quit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Simple Ollama System. Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                print("Please enter a valid question.")
                continue
            
            print(f"\nProcessing question: {question}")
            
            # Query Ollama directly
            response = query_ollama(
                question=question,
                model_name=model_name
            )
            
            print(f"\n--- Response ---")
            print(response)
            print("-" * 60)
        
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())