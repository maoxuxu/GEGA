NAME=$1
LOAD_DIR=$2


python run.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--load_path ${LOAD_DIR} \
--eval_mode single \
--test_file train_distant.json \
--test_batch_size 4 \
--evi_thresh 0.2 \
--num_labels 4 \
--num_class 97 \
--save_attn \


#python run.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --load_path mao_lambda0.1/2024-05-05_00_26_00 --eval_mode single --test_file train_distant.json --test_batch_size 4 --evi_thresh 0.2 --num_labels 4 --num_class 97 --save_attn
#python run.py --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --load_path re_lambda0.1/2024-05-07_13_41_59 --eval_mode single --test_file train_distant.json --test_batch_size 4 --evi_thresh 0.2 --num_labels 4 --num_class 97 --save_attn
