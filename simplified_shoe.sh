# python simplified_shoe.py\
#     --save_dir ./ctx_len_128\
#     --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
#     --layers 4\
#     --architectures standard\
#     --components resid

# python simplified_shoe.py\
#     --save_dir ./simplified_k60\
#     --model_name EleutherAI/pythia-70m-deduped\
#     --layers 4\
#     --architectures batch_top_k\
#     --components resid

# python simplified_shoe.py\
#     --save_dir ./resid_5\
#     --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
#     --layers 5\
#     --architectures batch_top_k\
#     --components resid


python simplified_shoe.py\
    --save_dir ./eval_on_shoe\
    --model_name /data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head\
    --layers 5\
    --architectures batch_top_k\
    --components resid\
    --do_eval

    # --do_train