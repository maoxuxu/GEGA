TYPE=$1
TEACHER_DIR=$2
LAMBDA=$3
SEED=$4

NAME=${TYPE}_lambda${LAMBDA}

python run.py --do_train \
--data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--train_file train_distant.json \
--dev_file dev.json \
--teacher_sig_path ${TEACHER_DIR} \
--save_path ${NAME} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 2 \
--evaluation_steps 5000 \
--num_labels 4 \
--lr_transformer 3e-5 \
--max_grad_norm 5.0 \
--evi_thresh 0.2 \
--attn_lambda ${LAMBDA} \
--warmup_ratio 0.06 \
--num_train_epochs 2.0 \
--seed ${SEED} \
--num_class 97


#python run.py --do_train --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --train_file train_distant.json --dev_file dev.json --teacher_sig_path mao_lambda0.1/2024-05-05_00_26_00 --save_path mao_lambda0.1 --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 2 --evaluation_steps 5000 --num_labels 4 --lr_transformer 3e-5 --max_grad_norm 5.0 --evi_thresh 0.2 --attn_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 2.0 --seed 66 --num_class 97
#python run.py --do_train --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --train_file train_distant.json --dev_file dev.json --teacher_sig_path re_lambda0.1/2024-05-07_13_41_59 --save_path re_lambda0.1 --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 2 --evaluation_steps 5000 --num_labels 4 --lr_transformer 3e-5 --max_grad_norm 5.0 --evi_thresh 0.2 --attn_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 2.0 --seed 66 --num_class 97
