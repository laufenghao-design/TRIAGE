export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHCODEC_NUM_THREADS=1
export DECORD_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 12326 \
downstream_phase/run_phase_training.py \
--suffix "" \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model  surgformer_HTA_KCA_ToMe \
--pretrained_path ./models/TimeSformer_divST_8x32_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path "./datasets/Cholec80/"  \
--eval_data_path "./datasets/Cholec80/"  \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 24 \
--sampling_rate 4 \
--eval \
--finetune "./weight/Surgformer/Cholec80/Surgformer_HTA_KCA_16_4/mp_rank_00_model_states.pt" \
--data_set Cholec80 \
--frames_dir frames_256 \
--data_fps 1fps \
--output_dir "./results/" \
--log_dir "./results/" \
--num_workers 2 \
--dist_eval \
--no_auto_resume \
--enable_ToMe \
--config "./config.yaml"
# --enable_deepspeed \
