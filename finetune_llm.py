from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
import peft
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import numpy as np
from transformers.integrations import TensorBoardCallback
import tqdm
from tqdm import tqdm
import argparse



def finetuning_opt(args) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)  # Context-manager 

    # Load pre-trained tokenizer and model
    model_name = "facebook/opt-125m"  # or any other LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

    # Prepare your dataset
    with open("coco_train_10k.json","r") as file:
        data = json.load(file)
    captions = list(data.values())
    captions = [caption.replace('\n', '') for caption in captions]

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="longest", truncation=True,max_length = 70)
    
    def prepare_dataset(examples):
        examples['labels'] = examples['input_ids']
        return examples
    
    caption_dataset = Dataset.from_dict({"text": captions})
    tokenized_captions = caption_dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=10000,  # Increase if your system has more RAM available
        num_proc = 16 # Adjust based on your CPU cores
    )    

    prepared_dataset = tokenized_captions.map(
        prepare_dataset, 
        batched=True, 
        batch_size=10000,  # Increase if your system has more RAM available
        num_proc= 16  # Adjust based on your CPU cores
    )    

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Adjust target modules based on model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    if args.lora_ckpt is not None:
        model = get_peft_model(model, lora_config  )
        model.print_trainable_parameters()
        try :
            checkpoint_name = f"{args.lora_ckpt}/adapter_model.bin"
            adapters_weights = torch.load(checkpoint_name)

        except:
            print("error")
            # checkpoint_name = f"{args.lora_ckpt}/adapter_model.safetensors"
            # print(checkpoint_name)
            # adapters_weights = torch.load(checkpoint_name)
        # print(f"peft weights loaded from {checkpoint_name}")
        # set_peft_model_state_dict(model, adapters_weights)
        
    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./finetune_coco_val2017_2k_rank_8",
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=2500,
        save_steps=120,
        save_total_limit=2,
        learning_rate=1e-3,
        seed = 42,
        dataloader_pin_memory=True,
        fp16 = True,  # Enable mixed precision training
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=50,  # Log every 500 steps
        dataloader_num_workers=16
    )

    # Initialize Trainer with a custom loss function
    # original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Keep a copy of the original model for regularization

    class RegularizedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # print(inputs)
            outputs = model(**inputs)

            loss = outputs.loss
            # reg_loss = regularization_loss(model, original_model)
            total_loss = loss # + reg_loss
            # print("total_loss :", total_loss.keys())
            return (total_loss,outputs) if return_outputs else total_loss

    trainer = RegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        callbacks=[TensorBoardCallback()],
        tokenizer=tokenizer
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained("./finetune_coco_val2017_2k_rank_8")
    tokenizer.save_pretrained("./finetune_coco_val2017_2k_rank_8")

if __name__ == "__main__":
    seed = 42 #42  #some time it was 123 or 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        default="finetune_coco_val2017_2k"
    )
    args = parser.parse_args()
    finetuning_opt(args)       