conda activate mb
# baseline
python main.py --model_name roberta-base --dataset sst2 --r 8 --lora_alpha 16 --target_modules query value --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&
python main.py --model_name roberta-base --dataset trec --r 8 --lora_alpha 16 --target_modules query value --learning_rate 1e-3 --epochs 1 --batch_size 256 --weight_decay 1e-2 &&

# ablation 1: target_modules
python main.py --model_name roberta-base --dataset sst2 --r 8 --lora_alpha 16 --target_modules query value key attention.output.dense --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&
python main.py --model_name roberta-base --dataset trec --r 8 --lora_alpha 16 --target_modules query value key attention.output.dense --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&

# ablation 2: r
python main.py --model_name roberta-base --dataset sst2 --r 4 --lora_alpha 16 --target_modules query value --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&
python main.py --model_name roberta-base --dataset trec --r 4 --lora_alpha 16 --target_modules query value --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&
python main.py --model_name roberta-base --dataset sst2 --r 16 --lora_alpha 16 --target_modules query value --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&
python main.py --model_name roberta-base --dataset trec --r 16 --lora_alpha 16 --target_modules query value --learning_rate 1e-3 --epochs 25 --batch_size 256 --weight_decay 1e-2 &&