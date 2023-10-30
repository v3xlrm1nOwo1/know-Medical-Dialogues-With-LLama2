from datasets import Dataset, load_dataset, DatasetDict
from config import *



# download dataset
dataset = load_dataset(DATASET_ID)


def format_instruction(medical_condition: str, treatment_options: str):
	return f'''### Instruction:
            For describe the treatment options the following conversation.

            ### Explaining medical conditions:
            {medical_condition.strip()}

            ### Describe the treatment options:
            {treatment_options.strip()}
            '''.strip()


def generate_instruction_dataset(data_point):

    return {
        'medical_condition': data_point['instruction'],
        'treatment_options': data_point['output'],
        'text': format_instruction(medical_condition=data_point['instruction'], treatment_options=data_point['output'])
    }


def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=RANDOM_SEED)
        .map(generate_instruction_dataset).remove_columns(['input',])
    )


def split_dataset(dataset=dataset):
    dataset = process_dataset(dataset)
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)
    
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    # gather everyone if you want to have a single DatasetDict
    know_medical_dialogue_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})
    
    return know_medical_dialogue_dataset

