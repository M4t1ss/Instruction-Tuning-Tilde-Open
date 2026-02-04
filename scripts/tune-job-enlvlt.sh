#!/bin/bash

#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=32:00:00
#PBS -P gag51407
#PBS -m abe
#PBS -o output.log
#PBS -e error.log
#PBS -M email@gmail.com
#PBS -v USE_SSH=2

source /etc/profile.d/modules.sh

module load cuda/12.8/12.8.1

export CUDA_HOME=/apps/cuda/12.8.1
export MKL_SERVICE_FORCE_INTEL=GNU
export MKL_THREADING_LAYER=1



python finetune.py \
	--model-path "TildeAI/TildeOpen-30b" \
	--batch-size 4 \
	--eval-batch-size 32 \
	--eval-accumulation-steps 5 \
	--gradient-accumulation-steps 16 \
	--eval-steps 100 \
	--log-steps 10 \
	--save-steps 100 \
	--data-paths \
		"data/multilinugal-dolly-15k/English.json" \
		"data/multilinugal-dolly-15k/Latvian.json" \
		"data/multilinugal-dolly-15k/Lithuanian.json" \
		"data/multilingual-alpaca-52k/English.json" \
		"data/multilingual-alpaca-52k/Latvian.json" \
		"data/multilingual-alpaca-52k/Lithuanian.json" \
	--output-name "tildeopen-30b-lora-adapter-enlvlt-2" \
	--use-lora \
	--lora-r 8 \
	--lora-alpha 32 \
	--lora-dropout 0.05 \
	--lora-target-modules "q_proj,v_proj" \
	--load-in-4bit \
	--save-model \
	--train-split 0.97 \
	--test-split 0.01 \
	--seed 347155 \
	--epochs 1

