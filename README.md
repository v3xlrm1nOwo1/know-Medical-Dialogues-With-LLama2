



# llama2 for know Medical Dialogues:



## `ABOUT PROJECT:`

I created this project for I learning how finetuning LLM and understanding and generating medically-informed dialogue.  aiming to provide medical information or insights, especially for scenarios with limited access to healthcare resources.



## `MODEL USED:`
For this project I used <a href='https://huggingface.co/NousResearch/Llama-2-7b-hf'>LLama-2-7b-hf</a> model from <a href='https://huggingface.co/NousResearch2'>NousResearch2</a>, Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.


## `REQUIREMENTS:`

```zsh
!pip install pytorch
!pip install -q -U bitsandbytes
!pip install transformers==4.31
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q datasets
!pip install evaluate
!pip install -qqq trl==0.7.1
```


## `DATASET AND PREPROCESS THE 'Dialog Know-Medical-Dialogue' DATASET:`


### `DESCRIPTION ABOUT DATASET:`

The <a href='https://huggingface.co/datasets/knowrohit07/know_medical_dialogue_v2'>knowrohit07/know_medical_dialogues_v2</a> dataset is a collection of conversational exchanges between patients and doctors on various medical topics. It aims to capture the intricacies, uncertainties, and questions posed by individuals regarding their health and the medical guidance provided in response.

You need to convert the 'Dialog Know-Medical-Dialogue' (prompt-response) pairs into explicit instructions for the LLM. Prepend an instruction to the start of the dialog with `For describe the treatment options the following conversation.` and to the start of the describe the treatment options with `Describe the treatment options` as follows:

### `TRAINING PROMPT 'DIALOGUE':`

```py
def format_instruction(medical_condition: str, treatment_options: str):
	return f'''### Instruction:
            For describe the treatment options the following conversation.

            ### Explaining medical conditions:
            {medical_condition.strip()}

            ### Describe the treatment options:
            {treatment_options.strip()}
            '''.strip()
```



## `PARAMETER EFFICIENT FINETUNING:`

I used `QLoRA` for parameter efficient finetuning

### `QLoRA EXPLAINED:`

QLORA: An efficient fine-tuning technique that uses low-rank adapters injected into each layer of the LLM, greatly reducing the number of trainable parameters and GPU memory requirement.

#### QLoRA Explained:
<img src='assets\QLoRA Explained.gif' width=100% height='400px'/>

#### MERGING QLoRA ADAPTER With BASE MODEL:
<img src='assets\Merging LoRA ADAPTER With BASE MODEL.png' width=100% height='400px'/>


## `NOTE`

I did not have the resources, such as the Internet, electricity, device, etc., to train the model well and choose the appropriate learning rate, so there were no results.


> To contribute to the project, please contribute directly. I am happy to do so, and if you have any comments, advice, job opportunities, or want me to contribute to a project, please contact me <a href='mailto:V3xlrm1nOwo1@gmail.com' target='blank'>V3xlrm1nOwo1@gmail.com</a>