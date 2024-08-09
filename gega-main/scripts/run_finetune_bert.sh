TYPE=$1
LOAD_DIR=$2
LAMBDA=$3
SEED=$4

NAME=${TYPE}_lambda${LAMBDA}

python run.py --do_train \
--data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--train_file train_annotated.json \
--dev_file dev.json \
--save_path ${NAME} \
--load_path ${LOAD_DIR} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--lr_transformer 1e-6 \
--lr_added 3e-6 \
--max_grad_norm 2.0 \
--evi_thresh 0.2 \
--evi_lambda ${LAMBDA} \
--warmup_ratio 0.06 \
--num_train_epochs 10.0 \
--seed ${SEED} \
--num_class 97


#python run.py --do_train --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --train_file train_annotated.json --dev_file dev.json --save_path mao_lambda0.1 --load_path mao_lambda0.1/2024-05-05_15_03_52 --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 1e-6 --lr_added 3e-6 --max_grad_norm 2.0 --evi_thresh 0.2 --evi_lambda 0.05 --warmup_ratio 0.06 --num_train_epochs 10.0 --seed 66 --num_class 97
#python run.py --do_train --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --train_file train_annotated.json --dev_file dev.json --save_path re_lambda0.1 --load_path re_lambda0.1/2024-05-07_18_47_06 --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 1e-6 --lr_added 3e-6 --max_grad_norm 2.0 --evi_thresh 0.2 --evi_lambda 0.05 --warmup_ratio 0.06 --num_train_epochs 10.0 --seed 66 --num_class 97

