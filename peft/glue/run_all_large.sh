model="roberta-large"
for rank in 4 8 16 32 64 128 256
do
    for task in "mrpc" "mnli" "qnli" "rte" "sst2" "stsb" 
    do
        CUDA_VISIBLE_DEVICES=0 python run_glue.py configs/${model}_${task}_r=${rank}.json
    done
done