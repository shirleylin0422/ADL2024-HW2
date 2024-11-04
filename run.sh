# ./run.sh C:/Users/Shirley/Desktop/ADL_hw2/data/public_test.jsonl C:/Users/Shirley/Desktop/ADL_hw2/data/r_output_public.jsonl

INPUT_FILE="$1"
OUTPUT_FILE="$2"
MODEL_PATH="./model"


python predict.py \
  --model_name_or_path $MODEL_PATH \
  --num_beams 5 \
  --input_file $INPUT_FILE \
  --output_file $OUTPUT_FILE