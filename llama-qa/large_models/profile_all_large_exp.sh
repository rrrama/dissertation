for mode in lorta lora
do
    for rank in 4
    do
        for bz in 1
        do
            MODEL=meta-llama/Llama-2-7b-hf RANK=$rank TASK=Copa MODE=$mode EPOCH=0.2 BS=$bz LR=1e-3 DEVICE=0 bash measure.sh
        done
    done
done
