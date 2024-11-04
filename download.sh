# Download package punkt
python -c "import nltk; nltk.download('punkt')"

# Download model
mkdir -p ./model
gdown --fuzzy https://drive.google.com/file/d/1HWAFVFV_wqn8dTPUInO8k7BsElzFlBa-/view?usp=sharing -O ./model/config.json
gdown --fuzzy https://drive.google.com/file/d/17OakR_SUEZDWHNl2XOyKRYcY91rdN0vC/view?usp=sharing -O ./model/model.safetensors
gdown --fuzzy https://drive.google.com/file/d/1RnLAzda1lwKdPLyP0qWdF51mxXnFxsDJ/view?usp=sharing -O ./model/special_tokens_map.json
gdown --fuzzy https://drive.google.com/file/d/1_6Y07krRRpm5wJgx71_f9yUjtbTMy49r/view?usp=sharing -O ./model/spiece.model
gdown --fuzzy https://drive.google.com/file/d/1IbYXIS_XKWXH5sQvAa6utZL__HZdet6Y/view?usp=sharing -O ./model/tokenizer_config.json
gdown --fuzzy https://drive.google.com/file/d/1y78NR5E1N_8LnSz10rFroKOLhMn3yItf/view?usp=sharing -O ./model/tokenizer.json

# Download data
mkdir -p ./data
gdown --fuzzy https://drive.google.com/file/d/1OSLs6QssjJRMsE5XkkWJ8JeXKNvJ3d06/view?usp=sharing -O ./data/train.jsonl
gdown --fuzzy https://drive.google.com/file/d/1BFS9oob2tdUbXot8SmM9dRNEpMK80iFL/view?usp=sharing -O ./data/valid.jsonl