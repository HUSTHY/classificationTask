python -W ignore train_TextBert.py \
        --batch_size 16 \
        --model_path ./pretrain_model/Chinese-BERT-wwm \
        --requires_grad true\
        --data_file_path data_set/patent \
        --max_sentence_length 400