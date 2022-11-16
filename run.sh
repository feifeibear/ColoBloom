export GPU_NUM=2

torchrun --standalone --nproc_per_node=${GPU_NUM} ./bloom-inference-scripts/bloom-accelerate-inference.py \
        --name '/data2/users/lczht/bloom-560m'  \
        --batch_size 1 \
        --benchmark 2>&1 | tee bloom-int8-accelerate-inference_bs=1.txt


        # --dtype int8 \
