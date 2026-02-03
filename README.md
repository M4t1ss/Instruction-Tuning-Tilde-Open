# Instruction-tuning Tilde-Open 30b
Fine-tuning the [Tilde-Open 30b](https://huggingface.co/TildeAI/TildeOpen-30b) model on Alpaca to follow instructions.

**Blog post**: https://m-ali.org/posts/instruction-tuning-tutorial/


What was used to fine-tune the model:
```
finetune.py \
    --model-path "TildeAI/TildeOpen-30b" --batch-size 2 --eval-accumulation-steps 5 \
    --data-paths "alpaca-data.json" "instruction-data.json" \
    --output-name "gemma-2b-lora-adapter" \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-target-modules "q_proj,v_proj" \
    --load-in-8bit \
    --save-model --epochs 2
```
