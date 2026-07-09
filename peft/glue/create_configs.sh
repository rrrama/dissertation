for model in "roberta-base" "roberta-large"
do
    for rank in 1 2 4 8 16 32 64 128 256
    do
        python create_json_config.py --model $model --rank $rank
    done
done