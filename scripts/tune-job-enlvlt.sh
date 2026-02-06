#!/bin/bash

#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=100:00:00
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
	--eval-batch-size 16 \
	--eval-accumulation-steps 5 \
	--gradient-accumulation-steps 16 \
	--eval-steps 100 \
	--log-steps 10 \
	--save-steps 100 \
	--data-paths \
		"/home/aad13940yw/experiments/tilde-open-it/instruction-tuning-gemma-2b/data/multilinugal-dolly-15k/English.json" \
		"/home/aad13940yw/experiments/tilde-open-it/instruction-tuning-gemma-2b/data/multilinugal-dolly-15k/Latvian.json" \
		"/home/aad13940yw/experiments/tilde-open-it/instruction-tuning-gemma-2b/data/multilingual-alpaca-52k/English.json" \
		"/home/aad13940yw/experiments/tilde-open-it/instruction-tuning-gemma-2b/data/multilingual-alpaca-52k/Latvian.json" \
	--hf-datasets \
		"zhengr/ultrachat_200k" \
		"utter-project/EuroBlocks-SFT-Synthetic-1124" \
		"martinsu/latvian-wikipedia-qa-gemma3" \
	--output-name "tildeopen-30b-lora-adapter-enlv-2" \
	--use-lora \
	--lora-r 8 \
	--lora-alpha 32 \
	--lora-dropout 0.05 \
	--lora-target-modules "q_proj,v_proj" \
	--load-in-4bit \
	--save-model \
	--train-split 0.99 \
	--test-split 0.004 \
	--seed 347155 \
	--epochs 1

