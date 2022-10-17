# ADL-HW1 (r11922189)

## Environment (sample code)

<span style="color:red">**Skip this if environment is setted up.**</span>

```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing (sample code)

<span style="color:red">**Skip this if environment is setted up.**</span>

```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```

## Download pretrained models
```shell
# This include pretrained models and embedding files
bash download.sh
```

## Test pretrained models
```shell
# Test intent classification
bash ./intent_cls.sh data/intent/test.json pred_intent.csv

# Test slot tagging
bash ./slot_tag.sh data/slot/test.json pred_slot.csv
```

## Train models

<span style="color:red">**Change --device argument with your cuda device. (cpu/cuda/cuda:0/cuda:1)**</span>
<span style="color:red">**or "CUDA_VISIBLE_DEVICES=0 python xxx.py"**</span>

```shell
# This reproduce my best result in intent classification
# Finish in 5 minutes, validation accuracy is about 0.94
python train_intent.py --lr 1e-3 --num_epoch 100 --dropout 0.5 --batch_size 384 --num_layers 2 --hidden_size 512 --device cuda

# This reproduce my best result in slot tagging
# Finish in 5 minutes, validation accuracy is about 0.8
python train_slot.py --lr 1e-3 --num_epoch 100 --dropout 0.5 --batch_size 128 --num_layers 2 --hidden_size 512 --device cuda
```

## Test your own model

<span style="color:red">**Change --device argument with your cuda device. (cpu/cuda/cuda:0/cuda:1)**</span>
<span style="color:red">**or "CUDA_VISIBLE_DEVICES=0 python xxx.py"**</span>

```shell
# intent classification
python test_intent.py --dropout 0.5 --batch_size 384 --num_layers 2 --hidden_size 512 --device cuda --test_file data/intent/test.json --ckpt_path ckpt/intent/model.pt --pred_file pred_intent.csv 

# slot tagging
python test_slot.py --dropout 0.5 --batch_size 128 --num_layers 2 --hidden_size 512 --device cuda --test_file data/slot/test.json --ckpt_path ckpt/slot/model.pt --pred_file pred_slot.csv 
```