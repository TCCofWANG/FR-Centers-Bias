export CUDA_VISIBLE_DEVICES="0"
python train.py --num_workers 8 \
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