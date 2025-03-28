###########################################################
###########################################################
### Market

###########################################################
## ResNet18
python train.py --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last

python train.py --gpu_ids 0  --name use_rn18_market_2 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --rge 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_market_2 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last

python train.py --gpu_ids 0  --name use_rn18_market_3 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --erasing_p 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_market_3 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last

python train.py --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --ours 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last

###########################################################
## EfficientNet
# Baseline
python train.py --gpu_ids 0 --name eff_market_1 --use_efficient --batchsize 32  --data_dir data/Market/pytorch --total_epoch 60
python test.py --gpu_ids 0 --name eff_market_1 --use_efficient  --batchsize 32  --test_dir data/Market/pytorch --which_epoch last

# Random Grayscale
python train.py --gpu_ids 0 --name eff_market_2 --use_efficient --batchsize 32  --data_dir data/Market/pytorch --rge 0.5 --total_epoch 60
python test.py --gpu_ids 0 --name eff_market_2 --use_efficient  --batchsize 32  --test_dir data/Market/pytorch --which_epoch last 

# Random Erasing
python train.py --gpu_ids 0 --name eff_market_3 --use_efficient --batchsize 32  --data_dir data/Market/pytorch --erasing_p 0.5 --total_epoch 60
python test.py --gpu_ids 0 --name eff_market_3 --use_efficient  --batchsize 32  --test_dir data/Market/pytorch --which_epoch last 

# Background Patch Shuffle
python train.py --gpu_ids 0 --name eff_market_4 --use_efficient --batchsize 32  --data_dir data/Market/pytorch --ours 0.5 --total_epoch 60
python test.py --gpu_ids 0 --name eff_market_4 --use_efficient  --batchsize 32  --test_dir data/Market/pytorch --which_epoch last 


###########################################################
###########################################################
### DukeMTMC-reID

###########################################################
## ResNet18
python train.py --gpu_ids 0  --name use_rn18_duke_1 --use_rn18 --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_duke_1 --use_rn18 --batchsize 32  --test_dir data/DukeMTMC-reID/pytorch --which_epoch last

python train.py --gpu_ids 0  --name use_rn18_duke_2 --use_rn18 --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --rge 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_duke_2 --use_rn18 --batchsize 32  --test_dir data/DukeMTMC-reID/pytorch --which_epoch last

python train.py --gpu_ids 0  --name use_rn18_duke_3 --use_rn18 --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --erasing_p 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_duke_3 --use_rn18 --batchsize 32  --test_dir data/DukeMTMC-reID/pytorch --which_epoch last

python train.py --gpu_ids 0  --name use_rn18_duke_4 --use_rn18 --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --ours 0.5 --rpn 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_duke_4 --use_rn18 --batchsize 32  --test_dir data/DukeMTMC-reID/pytorch --which_epoch last


###########################################################
## EfficientNet
# Baseline
python train.py --gpu_ids 0 --name eff_duke_1 --use_efficient --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --total_epoch 60
python test.py --gpu_ids 0 --name eff_duke_1 --use_efficient  --batchsize 32 --test_dir data/DukeMTMC-reID/pytorch --which_epoch last 

# Random Grayscale
python train.py --gpu_ids 0 --name eff_duke_2 --use_efficient --batchsize 32 --data_dir data/DukeMTMC-reID/pytorch --rge 0.5 --total_epoch 60
python test.py --gpu_ids 0 --name eff_duke_2 --use_efficient  --batchsize 32 --test_dir data/DukeMTMC-reID/pytorch --which_epoch last 

# Random Erasing
python train.py --gpu_ids 0 --name eff_duke_3 --use_efficient --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --erasing_p 0.5 --total_epoch 60
python test.py --gpu_ids 0 --name eff_duke_3 --use_efficient  --batchsize 32 --test_dir data/DukeMTMC-reID/pytorch --which_epoch last 

# Background Patch Shuffle
python train.py --gpu_ids 0 --name eff_duke_4 --use_efficient --batchsize 32  --data_dir data/DukeMTMC-reID/pytorch --ours 0.5 --rpn 0.5 --total_epoch 60
python test.py --gpu_ids 0 --name eff_duke_4 --use_efficient  --batchsize 32 --test_dir data/DukeMTMC-reID/pytorch --which_epoch last 

