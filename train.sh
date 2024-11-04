python train.py \
  --model_name_or_path google/mt5-small \
  --train_file ./data/train.jsonl \
  --validation_file ./data/valid.jsonl \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_train_epochs 10 \
  --weight_decay 0.01 \
  --max_seq_length 1024 \
  --max_target_length 128 \
  --output_dir ./model