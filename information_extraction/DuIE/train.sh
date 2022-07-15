set -eux

export BATCH_SIZE=4
export LR=2e-5
export EPOCH=1

unset CUDA_VISIBLE_DEVICES
#python -m paddle.distributed.launch --gpus "0" run_duie.py \
#                            --device gpu \
#                            --seed 42 \
#                            --do_train \
#                            --data_path conf \
#                            --max_seq_length 128 \
#                            --batch_size $BATCH_SIZE \
#                            --num_train_epochs $EPOCH \
#                            --learning_rate $LR \
#                            --warmup_ratio 0.06 \
#                            --output_dir checkpoints

python run_duie.py \
      --device cpu \
      --seed 42 \
      --do_train \
      --data_path /Users/geng/data/DuIE2.0/ \
      --max_seq_length 128 \
      --batch_size $BATCH_SIZE \
      --num_train_epochs $EPOCH \
      --learning_rate $LR \
      --warmup_ratio 0.06 \
      --output_dir checkpoints
