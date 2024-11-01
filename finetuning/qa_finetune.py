import argparse
# import bitsandbytes as bnb

import os
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import Dataset
from functools import partial
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']
WANDB_KEY = os.environ['WANDB_KEY']

PROJ_DIR = 'cocoruta-7b-qa'

TRAINING_OUTPUT_DIR = f"/workspace/data/output/felipeoes/results/{PROJ_DIR}/llama2-ba-legislation-qa-peft"
PEFT_MODEL_PATH = f"/workspace/data/output/felipeoes/results/{PROJ_DIR}/peft_model_llama2-ba-legislation-qa"
MERGED_MODEL_PATH = f"/workspace/data/output/felipeoes/results/{PROJ_DIR}/llama2-ba-legislation-qa-peft-merged"

# nnot using quantization for now. Maybe after fine-tunning, quantize the model for inference
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Monitor training with wandb
wandb.login(key=WANDB_KEY)


def load_model(model_name, bnb_config=None):
    n_gpus = torch.cuda.device_count()
    # max_memory = f'{40960}MB'
    # each gpu has 80 gb
    max_memory = f'{76 * 1024}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        # device_map={"": 0},
        # max_memory={i: max_memory for i in range(n_gpus)},
        token=HUGGINGFACE_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=HUGGINGFACE_TOKEN)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    return model, tokenizer


DATASET_PATH = 'full_filtered_qa_blue_amazon_legislation_35k.csv'
df = pd.read_csv(DATASET_PATH)
dataset = Dataset.from_pandas(df)
print(f"Dataset: {dataset}")


def create_prompt_formats(sample: dict):
    """
    Format various fields of the sample ('question', 'answer')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    SYSTEM_START_TOKEN = "<<SYS>>"
    SYSTEM_END_TOKEN = "<</SYS>>"
    INSTRUCTION_START_TOKEN = "[INST]"
    INSTRUCTION_END_TOKEN = "[/INST]"
    SYSTEM_MESSAGE = """Você é um respondedor de perguntas, especializado em responder perguntas sobre a legislação da Amazônia Azul. Você deve responder a pergunta abaixo, fornecendo uma resposta completa e detalhada. Suas respostas não devem incluir conteúdo prejudicial, como discurso de ódio, linguagem ofensiva ou conteúdo sexualmente explícito.
    
Caso a pergunta não faça sentido, você deve explicar o por quê ao invés de responder algo que não seja coerente. Se você não souber a resposta, diga que não sabe e não compartilhe informações falsas."""

    SYSTEM_PROMPT = f"<s>{INSTRUCTION_START_TOKEN} {SYSTEM_START_TOKEN}\n{SYSTEM_MESSAGE}\n{SYSTEM_END_TOKEN}"
    INSTRUCTION_KEY = "### Pergunta:"
    # INPUT_KEY = "Input:" # later try  with context, if necessary
    RESPONSE_KEY = "### Resposta:"
    END_KEY = "### Fim"

    instruction = f"\n{INSTRUCTION_KEY}\n{sample['question']} {INSTRUCTION_END_TOKEN}"
    # input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['answer']}"
    end = f"{END_KEY}"

    parts = [part for part in [instruction, response, end] if part]

    formatted_prompt = SYSTEM_PROMPT + "\n\n".join(parts) + " </s>"

    sample["text"] = formatted_prompt
    return sample


def get_max_length(model: "torch.nn.Module"):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: Dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        # remove_columns=["instruction", "context", "response", "text", "category"],
        remove_columns=dataset.column_names + ["text"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(
        sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=32,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model: "torch.nn.Module", use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


if __name__ == "__main__":

    SEED = 202310
    set_seed(SEED)

    # model_name = "meta-llama/Llama-2-7b-hf"
    # model will be loaded in Docker. No need to download from HuggingFace
    model_name = "./meta-llama_Llama-2-7b-hf"
    model, tokenizer = load_model(model_name)

    # Preprocess dataset
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, SEED, dataset)


    def train(model, tokenizer, dataset_train, dataset_validation, output_dir: str, peft_model_path: str):
        # Apply preprocessing to the model to prepare it by
        # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
        model.gradient_checkpointing_enable()

        # 2 - Using the prepare_model_for_kbit_training method from PEFT
        model = prepare_model_for_kbit_training(model)

        # Get lora module names
        modules = find_all_linear_names(model)

        # Create PEFT config for these modules and wrap the model to PEFT
        peft_config = create_peft_config(modules)
        model = get_peft_model(model, peft_config)

        # Print information about the percentage of trainable parameters
        print_trainable_parameters(model)

        # define early stopping callback
        # stop training if eval loss doesn't improve for 3 epochs
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

        # Training parameters
        trainer = Trainer(
            model=model,
            train_dataset=dataset_train,
            eval_dataset=dataset_validation,
            args=TrainingArguments(
                # per_device_train_batch_size=1,
                # auto_find_batch_size=True,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                # gradient_accumulation_steps=4,
                warmup_steps=2,
                learning_rate=2e-5,
                # using 16-bit training (bfloat16). Using bf16 instead of fp16 since it's faster and this GPU supports it
                bf16=True,
                # using these (save_total_limit and save_strategy) to save space. Only saving the last 5 checkpoints
                save_total_limit=5,
                # logging_steps=100,
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                # eval_steps=100,
                save_strategy="epoch",
                num_train_epochs=15,
                output_dir=output_dir,
                # will load the best model from the checkpoint. Since save_total_limit is 3, it will load the best model from the last 3 checkpoints
                load_best_model_at_end=True,
                # optim="paged_adamw_8bit", # using default (adamw_torch) since not using quantization
                report_to="wandb",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                seed=SEED,
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[early_stopping],
        )

        # re-enable for inference to speed up predictions for similar inputs
        model.config.use_cache = False

        # SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
        # Verifying the datatypes before training

        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items():
            total += v
        for k, v in dtypes.items():
            print(k, v, v/total)

        do_train = True

        # Launch training
        print("Training...")

        output_dir_exists = os.path.exists(
            output_dir) and len(os.listdir(output_dir)) > 0
        # get the last checkpoint (its ordered)
        checkpoint_dir = os.path.join(
            output_dir, sorted(os.listdir(output_dir))[-1]) if output_dir_exists else None
        print(
            f"Output dir exists: {output_dir_exists}. Resuming training from checkpoint: {checkpoint_dir}")

        if do_train:
            train_result = trainer.train(
                resume_from_checkpoint=checkpoint_dir)
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # eval metrics
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            trainer.save_state()
            print(metrics)

        # Saving model
        print("Saving last checkpoint of the model...")
        os.makedirs(peft_model_path, exist_ok=True)
        trainer.model.save_pretrained(peft_model_path)

        best_ckpt_path = trainer.state.best_model_checkpoint
        if best_ckpt_path is None:
            print("No best checkpoint found. Using last checkpoint.")
            best_ckpt_path = checkpoint_dir
        else:
            # save to txt file
            with open(os.path.join(peft_model_path, "best_ckpt_path.txt"), "w") as f:
                f.write(best_ckpt_path)

        # Free memory for merging weights
        del model
        del trainer
        torch.cuda.empty_cache()


    # separate dataset into train, validation and test (80, 10, 10)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=SEED)
    dataset_train = dataset["train"]
    dataset_test = dataset["test"]
    dataset_test_val = dataset_test.train_test_split(
        test_size=0.5, shuffle=True, seed=SEED)
    dataset_test = dataset_test_val["train"]
    dataset_validation = dataset_test_val["test"]

    print(f'Dataset train size: {len(dataset_train)}')
    print(f'Dataset validation size: {len(dataset_validation)}')
    print(f'Dataset test size: {len(dataset_test)}')

    # create folders if they don't exist
    os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

    # save train, test and validation datasets for evaluation later
    dataset_train.to_csv(
        f"/workspace/data/output/felipeoes/results/{PROJ_DIR}/dataset_train.csv")
    dataset_test.to_csv(
        f"/workspace/data/output/felipeoes/results/{PROJ_DIR}/dataset_test.csv")
    dataset_validation.to_csv(
        f"/workspace/data/output/felipeoes/results/{PROJ_DIR}/dataset_validation.csv")

    train(model, tokenizer, dataset_train, dataset_validation,
        TRAINING_OUTPUT_DIR, PEFT_MODEL_PATH)

    print("Training finished. Merging weights...")

    # merge weights for new model
    model = AutoPeftModelForCausalLM.from_pretrained(
        PEFT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("Weights merged. Fine-tuning finished.")