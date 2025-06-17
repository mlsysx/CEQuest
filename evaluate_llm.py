import json
import re
import numpy as np
import time
from CustomOpenAILLM import CustomOpenAILLM

# Load the JSON dataset
json_path = 'CEQuest_v0.1.json'
with open(json_path, 'r') as f:
    dataset = json.load(f)
questions = dataset['questions']

# Initialize the LLM (update api_key, model_name, base_url as needed)
llm = CustomOpenAILLM(api_key='api_key', model_name='llama3.3', base_url="http://localhost:11434/v1", debug=False)

# Function to extract the answer from LLM output
def extract_answer(output, question_type):
    if question_type == 'TRUE/FALSE':
        # For TRUE/FALSE questions, look for True/False in the output
        output = str(output).strip().upper()
        if 'TRUE' in output:
            return "True"
        elif 'FALSE' in output:
            return "False"
        return None
    else:
        # For multiple-choice questions, look for A-E
        match = re.search(r'\b([A-E])\b', str(output).strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

num_runs = 5
accuracies = []
times = []

for run in range(num_runs):
    print(f"\n--- Run {run+1} ---")
    correct = 0
    total = 0
    start_time = time.time()
    for idx, question in enumerate(questions):
        # Prepare prompt based on question type
        if question['Question Type'] == 'TRUE/FALSE':
            prompt = f"{question['Question']}\nPlease answer with only TRUE or FALSE."
        else:
            # Multiple-choice question
            options = []
            for i, opt in enumerate(['A', 'B', 'C', 'D', 'E']):
                if i < len(question['Options']):
                    options.append(f"{opt}. {question['Options'][i]}")
            context_str = '\n'.join(options)
            prompt = f"{question['Question']}\n{context_str}\nPlease answer with only the letter of the correct option (A, B, C, D, or E)."

        # Get LLM prediction
        try:
            output = llm.generate(prompt)
        except Exception as e:
            print(f"Error on question {idx}: {e}")
            output = ''
        
        pred = extract_answer(output, question['Question Type'])
        gt = str(question['Correct Answer']).strip()
        if question['Question Type'] == 'TRUE/FALSE':
            gt = gt.capitalize()  # Ensure True/False format matches
        
        total += 1
        is_correct = pred == gt
        if is_correct:
            correct += 1
        print(f"Q{idx+1} ({question['Question Type']}): Predicted={pred}, GT={gt}, Output={output}, Correct={is_correct}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Run {run+1} Accuracy: {correct / total if total > 0 else 0:.2%} ({correct}/{total})")
    print(f"Run {run+1} Evaluation Time: {elapsed:.2f} seconds")
    accuracies.append(correct / total if total > 0 else 0)
    times.append(elapsed)

accuracies = np.array(accuracies)
times = np.array(times)
print(f"\nAverage Accuracy over {num_runs} runs: {accuracies.mean():.2%}")
print(f"Standard Deviation: {accuracies.std():.2%}")
print(f"Average Evaluation Time: {times.mean():.2f} seconds")
print(f"Evaluation Time Std: {times.std():.2f} seconds")
