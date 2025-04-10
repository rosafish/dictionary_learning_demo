# python shoe.py\
#     --save_dir ./SAES_dev\
#     --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
#     --layers 0\
#     --architectures standard\
#     --components embed

# python shoe.py\
#     --save_dir ./SAEs_dev\
#     --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
#     --layers 4\
#     --architectures standard\
#     --components resid

python shoe.py\
    --save_dir ./sfc_full\
    --model_name EleutherAI/pythia-70m-deduped\
    --layers 4\
    --architectures standard\
    --components resid