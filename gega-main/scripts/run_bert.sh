TYPE=$1
LAMBDA=$2
SEED=$3

NAME=${TYPE}_lambda${LAMBDA}
#NAME=mao_lambda0.1
python run.py --do_train \
--data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name ${NAME} \
--save_path ${NAME} \
--train_file train_annotated.json \
--dev_file dev.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--lr_transformer 5e-5 \
--max_grad_norm 1.0 \
--evi_thresh 0.2 \
--evi_lambda ${LAMBDA} \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed ${SEED} \
--num_class 97


python run.py --do_train --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --save_path mao_lambda0.1 --train_file train_annotated.json --dev_file dev.json --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 5e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 30.0 --seed 66 --num_class 97
python run.py --do_train --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --save_path re_lambda0.1 --train_file train_annotated.json --dev_file dev.json --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 5e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 30.0 --seed 66 --num_class 97




#python run.py --do_train --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --save_path mao_lambda0.1 --train_file train_annotated.json --dev_file dev.json --train_batch_size 2 --test_batch_size 4 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 5e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 1.0 --seed 66 --num_class 97
#python run.py --do_train --data_dir data --transformer_type bert --model_name_or_path bert-base-cased --display_name redocred_lambda0.1 --save_path redocred_lambda0.1 --train_file train_annotated.json --dev_file dev.json --train_batch_size 2 --test_batch_size 4 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 5e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 1.0 --seed 66 --num_class 97
