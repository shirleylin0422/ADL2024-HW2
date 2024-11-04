import argparse
import math
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from nltk.tokenize import sent_tokenize
from tw_rouge import get_rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=30,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--log_data_checkpoints",
        type=int,
        default=5,
        help="Number of data points to generate metric curve",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Test only",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Test only",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.") 
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    
    args = parser.parse_args()

    return args




def main():
    args = parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    ####################
    def preprocess_function(examples):
      
        model_inputs = tokenizer(
            examples["maintext"],
            max_length=args.max_seq_length,
            truncation=True,
        )
        labels = tokenizer(
            examples["title"], max_length=args.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    evaluation_scores = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": []
    }
    
    def compute_metrics(eval_pred):
        print("compute_metrics----------------------------------------------")
        
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
        # Compute ROUGE scores
        # result = rouge_score.compute(
        #     predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        # )
        # Compute ROUGE scores
        result = get_rouge(decoded_preds, decoded_labels)
        # logger.info("ROUGE Result: %s", result)
        # Extract the median scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        result = {key: value["f"] * 100 for key, value in result.items()}
        evaluation_scores["rouge-1"].append(result["rouge-1"])
        evaluation_scores["rouge-2"].append(result["rouge-2"])
        evaluation_scores["rouge-l"].append(result["rouge-l"])
        # print(result)
        logger.info("ROUGE Result f1 score: %s", result)
        return {k: round(v, 4) for k, v in result.items()}
    ####################
    
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        # extension = args.train_file.split(".")[-1]
        extension = "json"
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        # extension = args.validation_file.split(".")[-1]
        extension = "json"

    raw_datasets = load_dataset(extension, data_files=data_files)
    # train_dataset = raw_datasets["train"]

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        raw_datasets["train"].column_names
    )
    # for test
    if args.testing:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(100))
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(100))

    num_update_steps_per_epoch = math.ceil(len(tokenized_datasets["train"])) / args.gradient_accumulation_steps
    train_steps = args.num_train_epochs * num_update_steps_per_epoch
    logging_steps = train_steps//args.log_data_checkpoints
    print("========================================================================")
    print("train_steps:", train_steps)
    print("logging_steps:", logging_steps)
    args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        logging_dir=args.output_dir+'/logs',
        logging_strategy="epoch", 
    )   

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=======================================on gpu=======================================")
        model.to(device)

    for param in model.parameters():
        param.data = param.data.contiguous()    
    print("=======================================Training=======================================")
    trainer.train()
    print("=======================================Evaling=======================================")
    trainer.evaluate()
    trainer.save_model(args.output_dir)

    epochs = range(1, len(evaluation_scores["rouge-1"]) + 1)
    plt.plot(epochs, evaluation_scores["rouge-1"], label="ROUGE-1")
    plt.plot(epochs, evaluation_scores["rouge-2"], label="ROUGE-2")
    plt.plot(epochs, evaluation_scores["rouge-l"], label="ROUGE-L")
    plt.xlabel("Epoch")
    plt.ylabel("ROUGE Score")
    plt.title("ROUGE Scores over Epochs")
    plt.legend()
    plt.savefig("rouge_scores.png")


if __name__ == "__main__":
    main()