cp ./experiments/painting/inceptionv4/snapshot/* ./experiments/painting/inceptionv4_adapt_phase4/snapshot
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --folder ./experiments/painting/inceptionv4_adapt_phase4 --resume 4
