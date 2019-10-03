cp ./experiments/painting/inceptionresnetv2/snapshot/* ./experiments/painting/inceptionresnetv2_adapt_phase4/snapshot
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --folder ./experiments/painting/inceptionresnetv2_adapt_phase4 --resume 4
