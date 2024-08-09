NAME=$1
MODEL_DIR=$2
SPLIT=$3

python run.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--load_path ${MODEL_DIR} \
--eval_mode single \
--test_file ${SPLIT}.json \
--test_batch_size 8 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97

#python run.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --load_path mao_lambda0.1/2024-05-06_09_49_30 --eval_mode single --test_file dev.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97
#python run.py --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --load_path re_lambda0.1/2024-05-08_11_15_53 --eval_mode single --test_file dev.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97

#python run.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --load_path mao_lambda0.1/2024-05-06_09_49_30 --eval_mode single --test_file test.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97
#python run.py --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --load_path re_lambda0.1/2024-05-08_11_15_53 --eval_mode single --test_file test.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97

python run.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name  ${NAME} \
--load_path ${MODEL_DIR} \
--eval_mode fushion \
--test_file ${SPLIT}.json \
--test_batch_size 8 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97

#python run.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --load_path mao_lambda0.1/2024-05-06_09_49_30 --eval_mode fushion --test_file dev.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97
#python run.py --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --load_path re_lambda0.1/2024-05-08_11_15_53 --eval_mode fushion --test_file dev.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97

#python run.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --load_path mao_lambda0.1/2024-05-06_09_49_30 --eval_mode fushion --test_file test.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97
#python run.py --data_dir dataset/redocred --transformer_type bert --model_name_or_path bert-base-cased --display_name re_lambda0.1 --load_path re_lambda0.1/2024-05-08_11_15_53 --eval_mode fushion --test_file test.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97




python run.py --data_dir dataset/revisitdocred --transformer_type bert --model_name_or_path bert-base-cased --display_name mao_lambda0.1 --load_path mao_lambda0.1/2024-05-21_00_13_55 --eval_mode single --test_file test.json --test_batch_size 4 --num_labels 2 --evi_thresh 0.2 --num_class 97
