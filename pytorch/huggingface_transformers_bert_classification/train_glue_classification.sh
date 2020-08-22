export TASK_NAME=myown

python -W ignore ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path ./pretrain_model/Chinese-BERT-wwm/ \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_eval \
    --data_dir ./data_set/patent/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=32   \
    --per_gpu_predict_batch_size=32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --output_dir ./output/