# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# For training the model, 550 667 985


SEEDS=(171 354)
N=10
K=5
mode=inter
dataset=FewNERD

for seed in ${SEEDS[@]}
    do
    python3 main.py \
        --gpu_device=0 \
        --seed=${seed} \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=10-k_100_2_32_3_max_loss_2_5_BIEOS \
        --train_mode=span \
        --inner_steps=2 \
        --inner_size=32 \
        --max_ft_steps=3 \
        --lambda_max_loss=2 \
        --inner_lambda_max_loss=5 \
        --tagging_scheme=BIEOS \
        --viterbi=hard \
        --concat_types=None \
        --ignore_eval_test

    python3 main.py \
        --seed=${seed} \
        --gpu_device=0 \
        --lr_inner=1e-4 \
        --lr_meta=1e-4 \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --inner_similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=10-k_100_type_2_32_3_10_10_BIEOS \
        --train_mode=type \
        --inner_steps=2 \
        --inner_size=32 \
        --max_ft_steps=3 \
        --concat_types=None \
        --lambda_max_loss=2.0 \
        --use_knn

    cp Results/${dataset}/${mode}/KNN/model-${N}-${K}/Bert/bert-base-uncased-innerSteps_2-innerSize_32-lrInner_0.0001-lrMeta_0.0001-maxSteps_5001-seed_${seed}/name_10-k_100_type_2_32_3_10_10_BIEOS/en_type_pytorch_model.bin Results/${dataset}/${mode}/KNN/model-${N}-${K}/Bert/bert-base-uncased-innerSteps_2-innerSize_32-lrInner_3e-05-lrMeta_3e-05-maxSteps_5001-seed_${seed}/name_10-k_100_2_32_3_max_loss_2_5_BIEOS

    python3 main.py \
        --gpu_device=0 \
        --seed=${seed} \
        --N=${N} \
        --K=${K} \
        --mode=${mode} \
        --similar_k=10 \
        --name=10-k_100_2_32_3_max_loss_2_5_BIEOS \
        --concat_types=None \
        --test_only \
        --eval_mode=two-stage \
        --inner_steps=2 \
        --inner_size=32 \
        --max_ft_steps=3 \
        --max_type_ft_steps=3 \
        --lambda_max_loss=2.0 \
        --inner_lambda_max_loss=5.0 \
        --inner_similar_k=10 \
        --viterbi=hard \
        --tagging_scheme=BIEOS \
        --use_knn
done
