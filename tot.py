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

# Single thought node in the tree
class ThoughtNode:
    def __init__(self, thought, parent=None, children=None):
        self.thought = thought
        self.parent = parent
        self.children = children or []

# Manages the Tree of Thoughts process for exploring possible solutions
class TreeOfThought: 
    def __init__(self, root_prompt, max_iterations=3, breadth_limit=3):
        self.root = ThoughtNode(root_prompt)
        self.max_iterations = max_iterations
        self.breadth_limit = breadth_limit
        self.ai_service = OllamaService()
        self.current_thoughts = [self.root]
        self.original_question = root_prompt

    def call_llm(self, prompt):
        response = self.ai_service.generate_response(prompt)
        return response

    # Explores possible thoughts (solutions) in breadth-first manner
    def explore_bfs(self, thought_nodes):
        new_thought_nodes = []
        
        for thought_node in thought_nodes:
            candidate_thoughts = []
            
            for _ in range(self.breadth_limit):  
                history = self.get_thought_history(thought_node)
                prompt = (f"Problem: '{self.original_question}'.\n"
                        f"Previous attempts: '{history}'.\n"
                        "Please provide an improved or corrected solution.")
                response = self.call_llm(prompt)
                if response:
                    candidate_thoughts.append(ThoughtNode(f"{response}", parent=thought_node))

            thought_node.children.extend(candidate_thoughts)
            new_thought_nodes.extend(candidate_thoughts)

        return new_thought_nodes
    
    # Retrives thought history from root node
    def get_thought_history(self, thought_node):
        history = []
        node = thought_node
        while node:
            history.append(node.thought)
            node = node.parent  
        return " -> ".join(reversed(history))

    # Extracts numerical answer 
    def extract_answer(self, thought_node):
        prompt = f"Extract the final answer number from the text: '{thought_node.thought}'. Return only the number."
        response = self.call_llm(prompt)
        cleaned_response = re.sub(r'[^0-9]', '', response)
    
        return cleaned_response

    # Main loop for generating thoughts and exploring possible solutions
    def run(self):
        iteration = 0
        answerset = set()  

        while self.current_thoughts and iteration < self.max_iterations:
            print(f"Iteration {iteration + 1}:")
            self.current_thoughts = self.explore_bfs(self.current_thoughts)
            for thought_node in self.current_thoughts:
                #print(f"Explored Thought: {thought_node.thought}")
                response = self.extract_answer(thought_node)
                answerset.add(response.strip())  
            
            iteration += 1

        return answerset

# Evaluate the model on the GSM8K dataset
if __name__ == "__main__":
    ds = load_dataset("567-labs/gsm8k", split="test[:100]") 
    tot_score = 0
    baseline_score = 0

    for idx, row in enumerate(ds):
        question = row['question']
        actual_answer = str(row['answer'])  
        cleaned_actual_answer = re.sub(r'[^0-9]', '', actual_answer)

        tot = TreeOfThought(question, max_iterations=3, breadth_limit=3)
        possible_answers = tot.run() 

        if actual_answer in possible_answers:
            tot_score += 1

    print(f"Final score: {tot_score}/100")
