python shoe.py\
    --save_dir ./SAES_dev\
    --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
    --layers 0\
    --architectures standard\
    --components embed

python shoe.py\
    --save_dir ./SAEs_dev\
    --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
    --layers 0 1 2 3 4 5\
    --architectures standard\
    --components mlp attn resid

# python shoe.py\
#     --save_dir ./shoe_dev\
#     --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
#     --layers 0 1 2 3 4 5\
#     --architectures batch_top_k\
#     --components resid mlp attn