from typing import Literal, Optional
from .registry import register_dataset
from darnax.datasets.classification.mnist import Mnist
from darnax.datasets.classification.cifar10_features import Cifar10FeaturesSmall
import logging

logger = logging.getLogger(__name__)


@register_dataset("entangled_mnist")
def build_entangled_mnist(batch_size: int, num_images_per_class: int):
    return Mnist(
        batch_size,
        num_images_per_class=num_images_per_class,
        linear_projection=100,
        flatten=True,
        label_mode="c-rescale",
        x_transform="sign",
    )


@register_dataset("general_mnist")
def build_general_mnist(
    batch_size: int = 64,
    linear_projection: Optional[int] = None,
    *,
    num_images_per_class: Optional[int] = None,
    label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
    x_transform: Literal["sign", "tanh", "identity"] = "sign",
):
    return Mnist(
        batch_size,
        linear_projection,
        num_images_per_class,
        label_mode,
        x_transform,
    )


@register_dataset("cifar_features_s")
def build_general_mnist(
    batch_size: int = 64,
    linear_projection: Optional[int] = None,
    *,
    num_images_per_class: Optional[int] = None,
    label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
    x_transform: Literal["sign", "tanh", "identity"] = "sign",
):
    return Cifar10FeaturesSmall(
        batch_size,
        linear_projection,
        num_images_per_class,
        label_mode,
        x_transform,
    )
