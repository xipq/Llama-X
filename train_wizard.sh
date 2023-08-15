source deactivate
conda activate llamax

deepspeed train_wizard.py \
    --model_name_or_path $HOME/llama/7B_converted \
    --data_path data/wizard_subset.json \
    --output_dir $HOME/llama/wizard_subset \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed src/configs/deepspeed_config.json \
    --fp16 True

python $HOME/gpuserver-hacker/src/train.py --gpu_devices 0 1 2 3 4 5 6 7