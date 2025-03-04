from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any
import torch as t
import itertools

from dictionary_learning.trainers.standard import StandardTrainer, StandardTrainerAprilUpdate
from dictionary_learning.dictionary import AutoEncoder

class TrainerType(Enum):
    STANDARD = "standard"

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

learning_rates = [3e-4]

wandb_project = "pythia-70m-shoe-simple-two-level-standard-sae-sparsity2e-2"

LLM_CONFIG = {
    "/data/rosa/work_in_progress/compositional_interpretability/outputs/shoe_simple_two_level_pythia_classification_head_pythia_base": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
}

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[0.02],
)

TARGET_L0s = [20, 40, 80, 160, 320, 640]


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

@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None

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
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction)

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
 
    return trainer_configs
