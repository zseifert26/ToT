import ollama
import re
from datasets import load_dataset

# Service to interact with Ollama 
class OllamaService:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name

    # Generate language model response based on input prompt
    def generate_response(self, prompt):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
        ) 
        return response['message']['content']
    
# Model to handle baseline eval
class BaselineModel:
    def __init__(self, model_name="llama3.1"):
        self.ai_service = OllamaService(model_name)

    # Get answer from model for a given question
    def get_answer(self, question):
        prompt = f"Question: {question}. Provide the correct answer."
        response = self.ai_service.generate_response(prompt)
        return response

    # Extract numeric answer from model response
    def extract_answer(self, response):
        prompt = f"Extract the final answer number from the text: '{response}'. Return only the number."
        response = self.call_llm(prompt) 
        cleaned_response = re.sub(r'[^0-9]', '', response)  
        return cleaned_response

    def call_llm(self, prompt):
        return self.ai_service.generate_response(prompt)

if __name__ == "__main__":
    ds = load_dataset("567-labs/gsm8k", split="test[:100]") # Load test dataset
    baseline_score = 0

    print("\nRunning Baseline Evaluation...")
    baseline_model = BaselineModel("llama3.1")

    # Iterate over test set
    for idx, row in enumerate(ds):
        question = row['question']
        actual_answer = str(row['answer'])
        cleaned_actual_answer = re.sub(r'[^0-9]', '', actual_answer)

        predicted_answer = baseline_model.get_answer(question)
        cleaned_predicted_answer = baseline_model.extract_answer(predicted_answer)

        if cleaned_actual_answer == cleaned_predicted_answer:
            baseline_score += 1

    print(f"Baseline Final score: {baseline_score}/100")