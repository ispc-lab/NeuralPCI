#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py  \
--dataset NL_Drive \
--dataset_path ./data/NL-Drive/test/ \
--scenes_list ./data/NL-Drive/test/scene_list/scene_list.txt \
--num_points 8192 \
--interval 4 \
--iters 1000 \
--lr 0.001 \
--layer_width 512 \
--act_fn LeakyReLU \
> .log_NL_Drive_release_test 2>&1
