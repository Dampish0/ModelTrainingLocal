# ModelTrainingLocal

License:
cc-by-nc-4.0, Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/

For training on online jupyter notebooks: https://github.com/Dampish0/ModelTrainingLocal

HELLO, HERE IS THE CODE FOR TRAINING ON LOCAL PC.

Here is training data, 2 DATASET DONT GET THEM CONFUSED!!
https://huggingface.co/datasets/Dampish/QuickTrain/tree/main

If you use my code, give credit.

So ill make it short
You can change variables in the json file here called "Data.json".

![image1](/Image21.png)
"huggingface_access_token" is the token for it to write to huggingface, if left blank it will not push to huggingface.co otherwise fill.
"model" is the actual model from huggingface.co or a local model of choice, its better to use huggingface though.
"model_tokenizer" is the base model you are using, the one you are fine tuning, you need the orginal tokenizer to avoid issues, this is simple.
"Data" is the actual training data.
"valid_data" is the validation data.
"out_dir" is the directory where you where you want the output and/or what its going to be named. Will automatically make on if it doesnt exist.


the only parameters u need to worry about are gradient_accumulation_steps
and learning rate
epoch is basically for how long you want to train it
its easier to limit training with max step than using epoch
cutoff_len is not important, it only changes how long the instruction is before it cuts it off
some models have a max limit of 1024 tokens

![image2](/image31.png)

GRADIENT_ACCUMULATION_STEPS, you want this to be between 4 and 32, apparently the more u have the better training u get.

Batch_size and micro_batch_size is exactly the same in this case. This number basically dictates how fast the step training goes, it also uses alot more vram when you increase it, the more the better if you ask me. Generally 4 for cpu, 8 for gpu, higher batch = higher ram. This script automatically utilizes maximal amount of micro_batch_size as it can, generally you dont need to touch this number, could lead to crashes and pain trying to maximize the gpu usage. IF 0 THEN AUTOMATICAL micro_batch_size OTHERWISE IT WILL TAKE YOUR NUMBER!! 

LEARNING_RATE Should be between 2e-5 <-> 5e-5 could be more could be less.

CUTOFF_LEN is the length of string before it cuts it off.

MAX_STEP is Easier to use than Epochs, basically gives a max amount of steps and ignores epoch number, for example if you give it 1000 max steps that could equal to 7.61 epochs, this number is completely random.

I rather you DO NOT use the last variables as they are hard to work with, unless you know what you are doing. Only play with CPU_MODE.
![image2](/image41.png)
Im too lazy to explain them so good luck.

Thanks for using my code
