from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any
import torch as t
import itertools

from dictionary_learning.trainers.standard import StandardTrainer, StandardTrainerAprilUpdate
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.dictionary import AutoEncoder, JumpReluAutoEncoder


class TrainerType(Enum):
    STANDARD = "standard"
    BATCH_TOP_K = "batch_top_k"
    JUMP_RELU = "jump_relu"

@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype

@dataclass
class SparsityPenalties:
    standard: list[float]

num_tokens = 50_000_000 
print(f"NOTE: Training on {num_tokens} tokens")

eval_num_inputs = 200
random_seeds = [0]
dictionary_widths = [2**14]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8
RESAMPLE_STEPS = 5000

learning_rates = [3e-4]

wandb_project = "pythia-70m-sweep"

LLM_CONFIG = {
    "/data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_lr0.0005_epochs30_batch8_warmup100_pythia_cls_head": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
        # llm_batch_size=64, context_length=128, sae_batch_size=2048, dtype=t.float32
    ),
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
}

SPARSITY_PENALTIES = SparsityPenalties(
    # standard=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    standard=[0.1],
)

# TARGET_L0s = [20, 40, 80, 160, 320, 640]
# TARGET_L0s = [60]
TARGET_L0s = [60, 90, 120, 150]

@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]
    component: str

@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None

@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001

@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold

def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    component: str,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
    resample_steps=RESAMPLE_STEPS,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction)
    # decay_start = None
    resample_steps = None


    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
        "component": component,
        # "resample_steps": resample_steps,
    }

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    # if TrainerType.JUMP_RELU.value in architectures:
    #     for seed, dict_size, learning_rate, target_l0 in itertools.product(
    #         seeds, dict_sizes, learning_rates, TARGET_L0s
    #     ):
    #         config = JumpReluTrainerConfig(
    #             **base_config,
    #             trainer=JumpReluTrainer,
    #             dict_class=JumpReluAutoEncoder,
    #             sparsity_warmup_steps=sparsity_warmup_steps,
    #             lr=learning_rate,
    #             dict_size=dict_size,
    #             seed=seed,
    #             target_l0=target_l0,
    #             wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
    #         )
    #         trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))
 
    return trainer_configs
