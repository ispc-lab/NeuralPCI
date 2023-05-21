#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py  \
--dataset DHB \
--dataset_path ./data/DHB-dataset \
--scenes_list ./data/DHB_scene_list_test.txt \
--num_points 1024 \
--interval 4 \
--iters 1000 \
--lr 0.001 \
--layer_width 512 \
--act_fn LeakyReLU \
> .log_DHB_release_test 2>&1
