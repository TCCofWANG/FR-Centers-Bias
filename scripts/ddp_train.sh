export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=8
gpus=0
arr=(${CUDA_VISIBLE_DEVICES//,/ })
for var in ${arr[@]}
do
    let gpus+=1
done
torchrun  --nproc_per_node=${gpus} \
          train.py  --num_workers ${OMP_NUM_THREADS} \
                    --ckpt_save_freq 1 \
                    --batch_size 512 \
                    --use_16bit \
                    --lr 0.1 \
                    --lr_milestones 8,16,22 \
                    --end_epoch 25 \
                    --gamma 0.1 \
                    --prefix centers_bias \
                    --head_type CentersBiasFace \
                    --scale 64.0 \
                    --margin 0.4 \
                    --m_m 0.2 \
                    --model iresnet18 \
                    $@
