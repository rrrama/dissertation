model="roberta-large"
rank=8
for task in "mrpc"
do
    for lr_lorta in  "1E-02" "1E-03" "1E-04"
    do
        for lr_head in "1E-02" "1E-03" "1E-04"
        do
            CUDA_VISIBLE_DEVICES=0 python run_glue.py configs/sweep/${model}/${task}/r=${rank}/lr_${lr_lorta}_lrc_${lr_head}.json
        done
    done
done