import argparse
import json
from tqdm import tqdm
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction for Summarize task")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="A jsonl file containing the testing data."
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="A jsonl file"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="model generation algorithm parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=0, help="model generation algorithm parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="model generation algorithm parameter 0~1"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="model generation algorithm parameter 0~1"
    )
    parser.add_argument(
        "--do_sample", type=bool, default=False, help="model generation algorithm parameter True/False"
    )
    
    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if torch.cuda.is_available():
        print("=======================================on gpu=======================================")
        device = torch.device("cuda")
        model.to(device)

    predictions = []
    with open(args.input_file, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)  
        infile.seek(0)  

        progress_bar = tqdm(total=total_lines, desc="Processing Summarize prediction", unit="sample")

        for line in infile:
            data = json.loads(line)
            maintext = data["maintext"]

            inputs = tokenizer(
                maintext, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(device)

            output_ids = model.generate(
                **inputs, 
                max_length=300, 
                num_beams=args.num_beams, 
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=args.do_sample
            )

            title = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            prediction = {"title": title, "id": data["id"]}
            predictions.append(prediction)

            progress_bar.update(1)


    with open(args.output_file, "w", encoding="cp950") as outfile:
        for prediction in predictions:
            json.dump(prediction, outfile, ensure_ascii=True)
            outfile.write("\n")

if __name__ == "__main__":

    main()