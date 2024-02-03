import os
import sys
import csv
import openai
import prompts

from pprint import pprint
from langchain.chat_models import ChatOpenAI 
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
)

# ------------------------ Paths and Environment Setup ------------------------
current = os.getcwd()
parent = os.path.dirname(current)
sys.path.append(parent)

# Ensure the OpenAI API key is set
os.environ['OPENAI_API_KEY'] = 'sk-...'
openai.api_key = os.environ['OPENAI_API_KEY']
print(openai.api_key)

# ----------------------- BFI Constants -------------------------
BFI_abc = prompts.BFI_abc
BFI_prompt = prompts.BFI_prompt
creative_init_prompt = prompts.creative_init_prompt
analytic_init_prompt = prompts.analytic_init_prompt
# Define the BFI scale scoring
BFI_scale = {
    "Extraversion": [1, 6, 11, 16, 21, 26, 31, 36],
    "Agreeableness": [2, 7, 12, 17, 22, 27, 32, 37, 42],
    "Conscientiousness": [3, 8, 13, 18, 23, 28, 33, 38, 43],
    "Neuroticism": [4, 9, 14, 19, 24, 29, 34, 39],
    "Openness": [5, 10, 15, 20, 25, 30, 35, 40, 41, 44]
}
reverse_scored = [6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34, 35, 41]
r_score_mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}

# ------------------------ Function Definitions ------------------------

#Initilize GPT models
def create_llm(init_prompt):
    # LLM
    llm = ChatOpenAI(temperature=0.7, model = "gpt-3.5-turbo")
    
    # Prompt 
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(init_prompt),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    
    return conversation

#preprocess the data to calculate BFI scores
def preprocess_data(response_text, characteristics_a, BFI_scale):
    
    # Extract scores from the response text
    scores = [int(item.split(') ')[1]) for item in response_text.split('\n') if item]
    print(f'scores: {scores}')
    print(f'len(scores): {len(scores)}')
    print(f'type(scores):{type(scores)}')
    
    # Safety check to ensure scores list is the expected length
    #return empty dictionary so that "valid_response" is False in run_multiple_llm and 
    #response will run again
    if len(scores) != 44:
        return scores, {}
    
    # Save scores of the statements per category
    BFI_output = {}
    for trait, items in BFI_scale.items():
        trait_scores = {}
        for item in items:
            # Safety check to ensure item is a valid index
            if item <= len(characteristics_a) and item <= len(scores):
                statement = characteristics_a[item - 1]
                trait_scores[statement] = scores[item - 1]
        BFI_output[f"{trait} Personality"] = trait_scores
        
    return scores, BFI_output

# Compute the BFI score
def calculate_BFI(scores, BFI_scale, reverse_scored, r_score_mapping):
    # Compute the BFI score
    BFI_scores = {}
    for trait, items in BFI_scale.items():
        total = 0
        for item in items:
            score = scores[item - 1]
            # Calculate the R scores
            if item in reverse_scored:
                score = r_score_mapping[score]
            total += score
        BFI_scores[trait] = total

    return BFI_scores

# Run LLMs multiple times
def run_multiple_llm(instance, BFI_prompt, BFI_scale, runs):
    """
    Run the LLM multiple times and return the results.

    Args:
    - instance : the instance created by instantiate_models
    - initialization_prompt (str): The initialization prompt for the LLM.
    - BFI_prompt (str): The BFI prompt for the LLM.
    - BFI_scale (dict): The BFI scale to classify the different statements.
    - runs (int): The number of times to run the LLM.
    - response_format (str): The expected format of the response.

    Returns:
    - list: A list of results from each run.
    """
    results_list = []

    for _ in range(runs):
        valid_response = False
        while not valid_response:
            # Initialize LLM
            print(f'instance: \n {instance}')

            # Generate a response using the instance
            response = instance({"question": BFI_prompt})
            print(f'response: \n {response}')
            
            # Preprocess the data
            scores, BFI_output = preprocess_data(response["text"], BFI_abc, BFI_scale)
            
            # Check if the response matches the desired format (the length of all the statements)
            if len(scores) == 44:
                valid_response = True
            else:
                print("-" * 50)
                print("Response format did not match. Retrying...")
                print("-" * 50)
        
        # Calculate the BFI scores
        result = calculate_BFI(scores, BFI_scale, reverse_scored, r_score_mapping)
        
        # Append the result to the list
        results_list.append(result)

    return results_list

def solo_writing_task(model_instance, question):
    """
    Conducts a conversation using a single language model instance.

    Parameters:
    model_instance (function): The function representing the language model instance.
    question (str): The initial question to start the conversation.

    Returns:
    list: A list of tuples containing the message.
    """
    conversation_history = []
    
    
    # Model produces a message
    model_question = f"{question}"
    
    response = model_instance({
        "question": model_question,
        "chat_history": "\n".join(text for _, text in conversation_history)
    })
    print(f"Model response: {response['text']}")
    conversation_history.append(("model", response["text"]))
        
    return conversation_history


def print_conversation_history(conversation_history):
    """
    Prints the conversation history in a specified format.

    Parameters:
    conversation_history (list): The conversation history to print.
    """
    for idx, (personality, message) in enumerate(conversation_history, 1):
        print(80 * "-")
        print(f"{personality.capitalize()} LLM Message {idx}: {message}\n")
        

def save_BFI(data, group, experimental_condition, output_directory, filename='BFI_data.csv'):
    """
    Save the provided data to a CSV file in the specified format, automatically assigning a Subject_ID.

    Parameters:
    - data (dict): The data to be saved, where keys are the column names and values are the data for each column.
    - group (str): The group of the subject ('Analytical' or 'Creative').
    - experimental_condition (int): The experimental condition (0 or 1).
    - output_directory (str): The directory where the CSV file will be saved.
    - filename (str, optional): The name of the output CSV file. Defaults to 'BFI_data.csv'.
    """
    # Full path to the file
    full_path = os.path.join(output_directory, filename)

    # Determine the next Subject_ID
    subject_id = 1
    if os.path.isfile(full_path):
        with open(full_path, 'r') as f:
            reader = csv.reader(f)
            # Skip the header
            next(reader, None)
            # Count the rows that are not empty
            subject_id = sum(1 for row in reader if row) + 1

    with open(full_path, 'a', newline='') as csvfile:
        fieldnames = ['Subject_ID', 'Group', 'Experimental_condition', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty to write headers
        file_is_empty = os.stat(full_path).st_size == 0 if os.path.exists(full_path) else True
        if file_is_empty:
            writer.writeheader()

        # Prepare the row to be written
        row = {
            'Subject_ID': subject_id,
            'Group': group,
            'Experimental_condition': experimental_condition
        }
        row.update(data)

        writer.writerow(row)

    print(f"Data saved to {full_path}")

def save_solo_writing(messages, filename="Childhood.csv", path="output_path", group="Analytical", experimental_condition=0):
    """
    Save or append the messages to a CSV file in the specified path.

    If the file already exists, new rows will be added with the correct subject numbering.
    If not, a new file will be created starting with '1'.

    Parameters:
    - messages (list): The messages to be saved.
    - filename (str): The name of the output CSV file, default is 'Childhood.csv'.
    - path (str): The path where the CSV file will be saved, default is 'output_path'.
    - group (str): The group type ('Analytical' or 'Creative'), default is 'Analytical'.
    - experimental_condition (int): The experimental condition (0 or 1), default is 0.
    """
    # Ensure the output directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)
    file_exists = os.path.isfile(full_path)

    # Determine the next subject number if the file already exists
    next_subject_number = 1
    if file_exists:
        with open(full_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_rows = list(reader)
            if existing_rows:
                # Subtract 1 for the header and start with the next number
                next_subject_number = len(existing_rows)

    with open(full_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Subject_ID', 'Group', 'Experimental_condition', 'Story']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        for message in messages:
            writer.writerow({
                'Subject_ID': next_subject_number, 
                'Group': group,
                'Experimental_condition': experimental_condition,
                'Story': message.replace('"', '""')  # Escape double quotes for CSV format
            })
            next_subject_number += 1


def instantiate_model(init_prompt):
    return create_llm(init_prompt)

def get_bfi_results(instance, prompt, scale, runs):
    return run_multiple_llm(instance, prompt, scale, runs)

def validate_bfi_results(results):
    for trait, score in results[0].items():
        if not isinstance(score, int):
            raise ValueError(f"Invalid score for {trait}: {score}")
            
def is_valid_word_count(story, min_words=500, max_words=900):
    """Checks if the story word count is within range."""
    word_count = len(story.split())
    return min_words <= word_count <= max_words

def save_bfi_results(results, group, condition, directory, filename):
    save_BFI(results, group=group, experimental_condition=condition, output_directory=directory, filename=filename)

def create_init_prompt_with_bfi(init_prompt, bfi_results):
    bfi_scores_text = ', '.join(f"{trait}: {score}" for trait, score in bfi_results.items())
    return f"{init_prompt} Your BFI scores are {bfi_scores_text}"

def perform_writing_task(instance, question):
    conversation_history = solo_writing_task(instance, question)
    print_conversation_history(conversation_history)
    return conversation_history

def save_stories_to_csv(stories, filename, path, group, condition):
    save_solo_writing(stories, filename=filename, path=path, group=group, experimental_condition=condition)

def main(subject_count, persona, init_prompt):
    for i in range(subject_count):                              
        valid_response = False
        while not valid_response:
            try:
                #1. Instantiate the model
                instance = instantiate_model(init_prompt)
                runs = 1
                
                #2. Get initial BFI results
                results = get_bfi_results(instance, BFI_prompt, BFI_scale, runs)
                
                # Check if the BFI results are in the correct format 
                validate_bfi_results(results)
                valid_response = True
            except ValueError as e:
                print(f"Caught ValueError: {e}. Retrying...")
            except IndexError as I:
                print(f"Caught IndexError: {I}. Retrying...")

        print("\n analytic BFI Results: \n")
        pprint(results)
        
        #3. Save initial BFI results to 'output/BFI_data_init.csv' 
        save_bfi_results(results[0], persona, 0, "output", 'Control/CONTROL_BFI_data_init.csv')
        
        #4. Individual writing task. 
        question = "Please share a personal story below in 800 words. Do not explicitly mention your personality traits in the story."    
        story_valid = False
        
        #Checking for story length. Story length is good if: 500 words > x > 900 words
        while not story_valid:
            conversation_history = perform_writing_task(instance, question)
            stories = [message for _, message in conversation_history]
            print(stories)
            story_length = len(stories[0].split())
            print(f"Story length: {story_length} words")  # Print the length of the story

            if is_valid_word_count(stories[0]):
                story_valid = True
            else:
                print("Story does not meet word count requirements. Retrying...")
        
        #5. Save stories in CSV files
        save_stories_to_csv(stories, "Control/CONTROL_Childhood.csv", "output", persona, 0)
        
        #6. Run BFI again to test consistency through writing
        results = get_bfi_results(instance, BFI_prompt, BFI_scale, runs)
        valid_response = False
        while not valid_response:
            try:
                validate_bfi_results(results)
                valid_response = True
            except ValueError as e:
                print(f"Caught ValueError: {e}. Retrying...")

        print("\n analytic BFI Results: \n")
        pprint(results)
        
        #7. save BFI results after writing
        save_bfi_results(results[0], persona, 0, "output", 'Control/CONTROL_BFI_data_writing.csv')

if __name__ == "__main__":
    subject_count = 10                                    # Adjust this to change the number of subjects 
    main(subject_count,'Analytic', analytic_init_prompt)
    #main(subject_count,'Creative', creative_init_prompt)
    
        