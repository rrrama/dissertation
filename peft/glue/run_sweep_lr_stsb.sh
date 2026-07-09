model="roberta-base"
task="stsb"
rank=4
for lr_lorta in  "1E-02" "5E-02" "5E-03" "1E-03" "1E-04"
do
    for lr_head in "1E-02" "5E-02" "5E-03" "1E-03" "1E-04"
    do
        for task in "stsb" 
        do
            CUDA_VISIBLE_DEVICES=0 python run_glue.py configs/sweep/${model}/${task}/r=${rank}/lr_${lr_lorta}_lrc_${lr_head}.json
        done
    done
done