import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from pathlib import Path
import pathlib
from git import Repo


import json
import os
import torch.nn as nn
from transformers import AutoConfig

from huggingface_hub import login, logout

#login()  # displays a widget in a notebook, a prompt in terminal otherwise

json_file = json.load(open("Data.json"))
if(json_file[0]["huggingface_access_token"] != ""):
    login(json_file[0]["huggingface_access_token"]) # non-blocking login

data = load_dataset("json", data_files=json_file[0]["data"]) if json_file[0]['PreProcessedData?'] == True else load_dataset("json", json_file[0]["ProccessedData_outDIR"])
valid_data = load_dataset("json", data_files=json_file[0]["eval_data"])
print("data loaded")
model = AutoModelForCausalLM.from_pretrained(json_file[0]["model"])
tokenizer = AutoTokenizer.from_pretrained(json_file[0]["model_tokenizer"])
print("model loaded")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

MICRO_BATCH_SIZE = json_file[0]["MICRO_BATCH_SIZE"]  # Generally 4 for cpu, 8 for gpu, higher batch = higher ram.
#BATCH_SIZE = 192
GRADIENT_ACCUMULATION_STEPS = json_file[0]["GRADIENT_ACCUMULATION_STEPS"] #BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = json_file[0]["EPOCHS"]  # Heavily varies depending on how u want to train it.
LEARNING_RATE = json_file[0]["LEARNING_RATE"]  # Should be between 2e-5 <-> 5e-5 could be more could be less.
CUTOFF_LEN = json_file[0]["CUTOFF_LEN"]  # 400 is a decent length.
MAX_STEP = json_file[0]["MAX_STEP"] # Easier to use than Epochs.

if not json_file[0]['PreProcessedData?']:
    def generate_prompt(data_point):
        if data_point["instruction"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""

    #valid_data = valid_data.map(lambda data_point: {"prompt": tokenizer(generate_prompt(data_point))})
    #data = data.map(lambda data_point: {"prompt": tokenizer(generate_prompt(data_point))})

    print("data conversion step 1 done")
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )

    valid_data = valid_data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )
    

    print("Saving data, this step might take a while, dont worry")
    json_data = []
    for i in range(len(data["train"])):
        json_data.append(data['train'][i])
        if(i % 1000 == 0):
            print(i)
            
    import json
    with open(json_file[0]["ProccessedData_outDIR"], 'w') as f:
        json.dump(json_data, f)
    

print("data conversion step 2 done \n ")
print("Training starting!")
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True if json_file[0]["CPU_MODE"] is False else False,#False For NO GPU, OTHERWISE TRUE IF YOU HAVE A GPU.
        logging_steps=1,
        output_dir=json_file[0]["out_dir"],
        save_total_limit=10,
        max_steps=MAX_STEP,
        auto_find_batch_size=True if json_file[0]["MICRO_BATCH_SIZE"] == 0 else False,
        per_device_eval_batch_size=1,  # Set batch size for evaluation
        eval_accumulation_steps=1,
        evaluation_strategy="steps",
        load_best_model_at_end=json_file[0]["load_best_model_at_end"],  # Save best model at the end of training
        save_steps=json_file[0]["save_steps"],
        eval_steps=json_file[0]["eval_steps"],
        no_cuda=False if json_file[0]["CPU_MODE"] is False else True,##CPU = TRUE/GPU = FALSE
        #tpu_num_cores=6,
)


trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=valid_data["train"],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("training begun.")


trainer.train(resume_from_checkpoint=json_file[0]["Load_Checkpoint"])

trainer.save_model(json_file[0]["out_dir"])
tokenizer.save_pretrained(json_file[0]["out_dir"])
print("training done and saved, check foler " + json_file[0]["out_dir"])
if(json_file[0]["huggingface_access_token"] != ""):
    model.push_to_hub("Dampish/ELIAI_1B", use_auth_token=True)

    logout() # logout completely


print("Done, thx for using my code.")
