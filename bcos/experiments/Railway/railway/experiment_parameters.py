import math  # noqa

from torch import nn

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
    RailwayClassificationPresetTrain,
    RailwayClassificationPresetTest

)
from bcos.experiments.utils import configs_cli, update_config
from bcos.modules import DetachableGNLayerNorm2d, norms
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

from bcos.data.presets import (
    RailwayClassificationPresetTest,
    RailwayClassificationPresetTrain,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.modules.losses import BinaryCrossEntropyLoss
from bcos.optim import LRSchedulerFactory, OptimizerFactory
import math

__all__ = ["CONFIGS"]

NUM_CLASSES = 2

DEFAULT_NUM_EPOCHS = 25
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_CROP_SIZE = 224  # Adjust based on your specific requirements

DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_steps=10_000,
    interval="step",
    warmup_decay=0.01,
)

LONG_WARM_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_steps=50_000,
    interval="step",
    warmup_decay=0.01,
)

# Define ViT architectures similar to your ImageNet setup
SIMPLE_VIT_ARCHS = [
    "simple_vit_ti_patch16_224",
    "simple_vit_s_patch16_224",
    "simple_vit_b_patch16_224",
    "simple_vit_l_patch16_224",
]

# Default transformations for train and test
DEFAULT_TRAIN_TRANSFORM = RailwayClassificationPresetTrain()
DEFAULT_TEST_TRANSFORM = RailwayClassificationPresetTest()

# Create default configurations
DEFAULTS = dict(
    data=dict(
        train_transform=DEFAULT_TRAIN_TRANSFORM,
        test_transform=DEFAULT_TEST_TRANSFORM,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=4,
        num_classes=NUM_CLASSES,
    ),
    model=dict(
        is_bcos=True,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=nn.LayerNorm,  # Adjust as per your model requirements
            act_layer=nn.GELU,
            channels=3,  # Adjust channels based on your input data
        ),
        bcos_args=dict(
            b=2,
        ),
    ),
    criterion=BinaryCrossEntropyLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=OptimizerFactory(name="Adam", lr=DEFAULT_LR),
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
)

# Generate configurations for each ViT architecture
CONFIGS = dict()
for name in SIMPLE_VIT_ARCHS:
    CONFIGS[name] = update_config(DEFAULTS, dict(model=dict(name=name)))

# Optionally, create configs with different seeds
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[420, 1337]))

# Command-line interface for accessing configurations
if __name__ == "__main__":
    configs_cli(CONFIGS)
    print(CONFIGS.keys())
