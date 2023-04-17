print("Loading model, this may take a minute.")
import torch
import transformers
print("<-+      >")
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
print("<--+     >")
import json
import os
print("<---+    >")
import torch.nn as nn
from transformers import AutoConfig, GenerationConfig
print("<----+   >")
json_da = json.load(open("Settings.json"))

model = AutoModelForCausalLM.from_pretrained((os.getcwd() + "/" + json_da[0]["out_dir"] + "/"))
tokenizer = AutoTokenizer.from_pretrained(json_da[0]["model_tokenizer"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = 0 
print("<------+ >")

print("<-------+>")
params = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.15,
    "typical_p": 1,
    "max_length": 100,
}
print("Loaded " + str(model)[:25] + "\n\n\n")
print("Model Is Loaded And Ready.")
while True:
    input_str = input("Enter prompt and parameters (if any), separated by '||': ")
    input_list = input_str.split("||")
    prompt = input_list[0].strip()

    if len(input_list) > 1:
        params_str = input_list[1].strip()
        params_list = params_str.split()
        for param in params_list:
            key_val = param.split("=")
            if key_val[0] in params:
                params[key_val[0]] = float(key_val[1])
            else:
                print(f"Invalid parameter name: {key_val[0]}")

        print("Updated parameters:")
        print(params)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = inputs.to("cpu")
    print("Generating...")

    generation_config = GenerationConfig(
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        repetition_penalty=params["repetition_penalty"],
        typical_p=params["typical_p"],
    )

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        output_scores=True,
        max_length=params["max_length"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    for s in generation_output:
        print(tokenizer.decode(s))
