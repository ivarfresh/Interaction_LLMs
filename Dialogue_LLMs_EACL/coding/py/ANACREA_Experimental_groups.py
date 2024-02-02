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
           # instance = create_llm(initialization_prompt)
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

def converse(analytic_instance, creative_instance, question, turns=1):
    """
    Conducts a conversation between two language model instances, alternating turns.

    Parameters:
    analytic_instance (function): The function representing the analytic language model instance.
    creative_instance (function): The function representing the creative language model instance.
    question (str): The initial question to start the conversation.
    turns (int): The number of turns each language model will take in the conversation.

    Returns:
    list: A list of tuples containing the personality type and the corresponding message.
    """
    conversation_history = []
    
    for turn in range(turns):
        # Analytic LLM produces a message
        analytic_question = f"{question}." if not conversation_history else \
            f"{question}. Last response to question is \"{conversation_history[-1][1]}\". Collaborate to solve \"{question}\"."
        
        response = analytic_instance({
            "question": analytic_question,
            "chat_history": "\n".join(text for _, text in conversation_history)
        })
        #print(f"Analytic LLM response: {response['text']}")
        conversation_history.append(("analytic", response["text"]))
        
        # Creative LLM produces a message
        creative_question = f"{question}. Last response to question is \"{response['text']}\". Collaborate to solve \"{question}\"."
        
        response = creative_instance({
            "question": creative_question,
            "chat_history": "\n".join(text for _, text in conversation_history)
        })
        #print(f"Creative LLM response: {response['text']}")
        conversation_history.append(("creative", response["text"]))
        
    return conversation_history

def print_conversation_history(conversation_history):
    """
    Prints the conversation history in a specified format.

    Parameters:
    conversation_history (list): The conversation history to print.
    """
    print()
    print()
    print()
    print(30 * "-", "CONVERSATION HISTORY", 30 * "-")
    for idx, (personality, message) in enumerate(conversation_history, 1):
        print(80 * "-")
        print(f"{personality.capitalize()} LLM Message {idx}: {message}\n")
        
        
def extract_messages_by_type(conversation_history, message_type):
    """
    Extract messages from the conversation history by type.

    Parameters:
    - conversation_history (list): The conversation history.
    - message_type (str): The type of message to extract ('analytic' or 'creative').

    Returns:
    - list: A list of extracted messages.
    """
    return [message for personality, message in conversation_history if personality == message_type]

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

def save_messages_to_csv(messages, group, experimental_condition, filename, path):
    """
    Save or append the messages to a CSV file in the specified path.

    If the file already exists, new rows will be added with the correct subject numbering.
    If not, a new file will be created starting with 'Subject 1'.

    Parameters:
    - messages (list): The messages to be saved.
    - group (str): The group type ('Analytic' or 'Creative').
    - experimental_condition (int): The experimental condition (0 or 1).
    - filename (str): The name of the output CSV file.
    - path (str): The path where the CSV file will be saved.
    """
    full_path = os.path.join(path, filename)
    file_exists = os.path.isfile(full_path)
    next_subject_number = 1

    # Determine the next subject number if the file already exists
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
                'Story': message  # Use the message content for the 'Story' column
            })
            next_subject_number += 1
            
    print(f"Data saved to {full_path}")

def dir_path():
    """
    Gets the path to the output directory relative to the current working directory.

    This function assumes that the current working directory is the desired base directory
    (i.e., 'coding/scripts/py'). It then appends 'output' to this path, checks if the
    resulting 'output' directory exists, and creates it if it does not.

    Returns:
        str: The absolute path to the 'output' directory.
    """
    # Get the working directory: Make sure working directory is: coding/scripts/py 
    script_dir = os.getcwd()
    print(f"Current working directory: {script_dir}")
    
    # Set the output directory relative to the script location
    output_path = os.path.join(script_dir, 'output')
    
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    return output_path


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
    
def main(subject_count):
    for i in range(subject_count): 
        valid_response = False
        while not valid_response:
            try:
                #1. Instantiate the models
                analytic_instance = create_llm(analytic_init_prompt)
                creative_instance = create_llm(creative_init_prompt)
                
                #2. Get the initial BFI results
                runs = 1
                analytic_results = run_multiple_llm(analytic_instance, BFI_prompt, BFI_scale, runs)
                creative_results = run_multiple_llm(creative_instance, BFI_prompt, BFI_scale, runs)
               
                # Check if the BFI results are in the correct format 
                validate_bfi_results(analytic_results)
                validate_bfi_results(creative_results)
                valid_response = True
            except ValueError as e:
                print(f"Caught ValueError: {e}. Retrying...")
            except IndexError as I:
                print(f"Caught ValueError: {I}. Retrying...")

        print("\n analytic BFI Results: \n")
        pprint(analytic_results)
        
        print("\n Creative BFI Results: \n")
        pprint(creative_results)
        
        #3. Save initial BFI results to 'output/BFI_data_init.csv' 
        save_bfi_results(analytic_results[0], 'Analytic', 1, "output/ANACREA", 'ANACREA_analytic_BFI_data_init.csv')
        save_bfi_results(creative_results[0], 'Creative', 1, "output/ANACREA", 'ANACREA_creative_BFI_data_init.csv')
                
        #4. Collaborative writing task with length check
        question = "Please share a personal story below in 800 words. Do not explicitly mention your personality traits in the story."
        
        story_valid = False
        while not story_valid:
            conversation_history = converse(analytic_instance, creative_instance, question, turns=1)
            print_conversation_history(conversation_history)
        
            analytic_messages = extract_messages_by_type(conversation_history, 'analytic')
            creative_messages = extract_messages_by_type(conversation_history, 'creative')
        
            # Calculate lengths of both stories
            analytic_story_length = len(analytic_messages[0].split()) if analytic_messages else 0
            creative_story_length = len(creative_messages[0].split()) if creative_messages else 0
        
            print(f"Analytic Story length: {analytic_story_length} words")
            print(f"Creative Story length: {creative_story_length} words")
        
            # Check if both stories meet the word count requirements
            if is_valid_word_count(analytic_messages[0]) and is_valid_word_count(creative_messages[0]):
                story_valid = True
            else:
                print("One or both stories do not meet word count requirements. Retrying...")

        #5.Save messages to CSV in the specified output path
        #Define output path
        output_path = dir_path()
        print(output_path)
        
        # Define Groups
        analytic_group = 'Analytic'
        creative_group = 'Creative'
        
        # Define filenames
        analytic_filename = 'ANACREA/ANACREA_analytic_childhood'
        creative_filename = 'ANACREA/ANACREA_creative_childhood'
        
        # Save collaborative stories
        save_messages_to_csv(analytic_messages, analytic_group, 1, f"{analytic_filename}.csv", output_path)
        save_messages_to_csv(creative_messages, creative_group, 1, f"{creative_filename}.csv", output_path)      
        
        #6. Run BFI again to test consistency through writing
        analytic_results = run_multiple_llm(analytic_instance, BFI_prompt, BFI_scale, runs)
        creative_results = run_multiple_llm(creative_instance, BFI_prompt, BFI_scale, runs) 
        valid_response = False
        while not valid_response:
            try:
                validate_bfi_results(analytic_results)
                validate_bfi_results(creative_results)
                
                valid_response = True
            except ValueError as e:
                print(f"Caught ValueError: {e}. Retrying...")
            except IndexError as I:
                print(f"Caught ValueError: {I}. Retrying...")
        
        print("\n analytic BFI Results: \n")
        pprint(analytic_results)
        
        print("\n creative BFI Results: \n")
        pprint(creative_results)
        
        #7. save BFI results after writing
        save_bfi_results(analytic_results[0], 'Analytic', 1, "output/ANACREA", 'ANACREA_analytic_BFI_data_writing.csv')
        save_bfi_results(creative_results[0], 'Creative', 1, "output/ANACREA", 'ANACREA_creative_BFI_data_writing.csv')
        
# Main execution
if __name__ == "__main__":
    subject_count = 1
    main(subject_count)
    
    

   