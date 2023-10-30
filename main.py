import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from trl import SFTTrainer

from config import *
from process_data import split_dataset


# ==============================utils============================== #
def print_trainable_parameters(model):
    '''
    Prints the number of trainable parameters in the model.
    '''

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():

        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


# get model text generate
def get_model_generate(index: int):

    medical_condition = test_dataset['instruction'][index]
    treatment_options = test_dataset['output'][index]

    prompt = f'''
    ### Instruction:
    For describe the treatment options the following conversation.

    ### Explaining medical conditions:
    {medical_condition}

    ### Describe the treatment options:
    '''

    input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()

    outputs = trained_model.generate(
        input_ids=input_ids, 
        max_new_tokens=500, 
        )
    
    output= tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

    dash_line = '-'.join('' for x in range(100))
    result = f''''{dash_line}\nINPUT PROMPT:\n{prompt}{dash_line}\nBASELINE HUMAN DESCRIBE THE TREATMENT OPTIONS:\n{treatment_options}\n{dash_line}\nTRAINED MODEL GENERATED TEXT :\n{output}'''
    
    return result


# save model
def model_save():
    peft_model_path=PEFT_MODEL_DIR
    trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)


# push to pub
def push_to_hub():
    trained_model = AutoPeftModelForCausalLM.from_pretrained(
        PEFT_MODEL_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    # Merge LoRA and base model
    merged_model = trained_model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained('merged_model', safe_serialization=True)
    tokenizer.save_pretrained('merged_model')

    # push merged model to the hub
    # merged_model.push_to_hub("user/repo")
    # tokenizer.push_to_hub("user/repo")


# ==============================Process and Split Dataset============================== #
know_medical_dialogue_dataset = split_dataset()
train_dataset = know_medical_dialogue_dataset['train']
validation_dataset = know_medical_dialogue_dataset['validation']
test_dataset = know_medical_dialogue_dataset['test']


# ==============================Tokenizer============================== #
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


# ==============================Model============================== #
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# get model from hugginface
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, quantization_config=bnb_config, device_map='auto')

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print(model)

# lora config
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], #specific to Llama models.
    lora_dropout=0.1,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)


# =====================Traning Arguments===================== #
training_arguments = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    logging_steps=1,
    learning_rate=LEARNING_RATE,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=EPOCHS,
    evaluation_strategy='steps',
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy='epoch',
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to='tensorboard',
    save_safetensors=True,
    lr_scheduler_type='cosine',
    seed=RANDOM_SEED,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=lora_config,
    dataset_text_field='text',
    max_seq_length=MAX_SEQ_LEANGTH,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

# ==============================Save Model==============================
model_save()

# ==============================Inference============================== #
model.config.use_cache = True
model.eval()

# load base LLM model and tokenizer
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    PEFT_MODEL_DIR,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_DIR)

index = int(input('Enter index: '))
result = get_model_generate(index=index)

# ==============================Merge Trained LoRA Adapter With BASE MODEL and Push Model to Hub============================== #

push_to_hub()

