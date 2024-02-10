from typing import Any, Tuple, Union
import math
import itertools

from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from PIL import ImageFont
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import Callback
from slot_attention.segmentation_metrics import adjusted_rand_index

import wandb


FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 7)
CMAPSPEC = ListedColormap(
    ["black", "red", "green", "blue", "yellow", "cyan", "magenta"]
    + list(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                zip(
                    [get_cmap("tab20b")(i) for i in range(i, 20, 4)],
                    [get_cmap("tab20c")(i) for i in range(i, 20, 4)],
                )
            )
            for i in range(4)
        )
    )
    + [get_cmap("Set3")(i) for i in range(12)]
    + ["white"],
    name="SemSegMap",
)


@torch.no_grad()
def cmap_tensor(t):
    t_hw = t.cpu().numpy()
    o_hwc = CMAPSPEC(t_hw)[..., :3]  # drop alpha
    o = torch.from_numpy(o_hwc).transpose(-1, -2).transpose(-2, -3).to(t.device)
    return o


def conv_transpose_out_shape(
    in_size, stride, padding, kernel_size, out_padding, dilation=1
):
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + out_padding
        + 1
    )


def assert_shape(
    actual: Union[torch.Size, Tuple[int, ...]],
    expected: Tuple[int, ...],
    message: str = "",
):
    assert (
        actual == expected
    ), f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def rescale(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def slightly_off_center_crop(image: torch.Tensor) -> torch.Tensor:
    crop = ((29, 221), (64, 256))  # Get center crop. (height, width)
    # `image` has shape [channels, height, width]
    return image[:, crop[0][0] : crop[0][1], crop[1][0] : crop[1][1]]


def slightly_off_center_mask_crop(mask: torch.Tensor) -> torch.Tensor:
    # `mask` has shape [max_num_entities, height, width, channels]
    crop = ((29, 221), (64, 256))  # Get center crop. (height, width)
    return mask[:, crop[0][0] : crop[0][1], crop[1][0] : crop[1][1], :]


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):
    def log_images(self, trainer, pl_module, stage):
        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images(stage=stage)
                trainer.logger.experiment.log(
                    {stage + "/images": [wandb.Image(images)]}, commit=False
                )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.log_images(trainer, pl_module, stage="validation")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.log_images(trainer, pl_module, stage="train")


def to_rgb_from_tensor(x: torch.Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = torch.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-1)
    return channels, masks


def linear(in_features, out_features, bias=True, weight_init="xavier", gain=1.0):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m


def warm_and_decay_lr_scheduler(
    step: int, warmup_steps_pct, decay_steps_pct, total_steps, gamma
):
    warmup_steps = warmup_steps_pct * total_steps
    decay_steps = decay_steps_pct * total_steps
    assert step < total_steps
    if step < warmup_steps:
        factor = step / warmup_steps
    else:
        factor = 1
    factor *= gamma ** (step / decay_steps)
    return factor


def cosine_anneal(step: int, start_value, final_value, start_step, final_step):
    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value


def linear_warmup(step, start_value, final_value, start_step, final_step):
    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value


def visualize(image, recon_orig, gen, attns, N=8):
    _, _, H, W = image.shape
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon_orig = recon_orig[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    gen = gen[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)

    return torch.cat((image, recon_orig, gen, attns), dim=1).view(-1, 3, H, W)


def compute_ari(prediction, mask, batch_size, height, width, max_num_entities):
    # Ground-truth segmentation masks are always returned in the canonical
    # [batch_size, max_num_entities, height, width, channels] format. To use these
    # as an input for `segmentation_metrics.adjusted_rand_index`, we need them in
    # the [batch_size, n_points, n_true_groups] format,
    # where n_true_groups == max_num_entities. We implement this reshape below.
    # Note that 'oh' denotes 'one-hot'.
    desired_shape = [batch_size, height * width, max_num_entities]
    true_groups_oh = torch.permute(mask, [0, 2, 3, 4, 1])
    # `true_groups_oh` has shape [batch_size, height, width, channels, max_num_entries]
    true_groups_oh = torch.reshape(true_groups_oh, desired_shape)

    # prediction = tf.random_uniform(desired_shape[:-1],
    #                                         minval=0, maxval=max_num_entities,
    #                                         dtype=tf.int32)
    # prediction_oh = F.one_hot(prediction, max_num_entities)

    ari = adjusted_rand_index(true_groups_oh[..., 1:], prediction)
    return ari


def flatten_all_but_last(tensor, n_dims=1):
    shape = list(tensor.shape)
    batch_dims = shape[:-n_dims]
    flat_tensor = torch.reshape(tensor, [np.prod(batch_dims)] + shape[-n_dims:])

    def unflatten(other_tensor):
        other_shape = list(other_tensor.shape)
        return torch.reshape(other_tensor, batch_dims + other_shape[1:])

    return flat_tensor, unflatten


def sa_segment(masks, threshold, return_cmap=True):
    # `masks` has shape [batch_size, num_entries, channels, height, width].
    # Ensure that each pixel belongs to exclusively the slot with the
    # greatest mask value for that pixel. Mask values are in the range
    # [0, 1] and sum to 1. Add 1 to `segmentation` so the values/colors
    # line up with `segmentation_thresholded`
    segmentation = masks.to(torch.float).argmax(1).squeeze(1) + 1
    segmentation = segmentation.to(torch.uint8)
    cmap_segmentation = cmap_tensor(segmentation)

    pred_masks = masks.clone()
    # For each pixel, if the mask value for that pixel is less than the
    # threshold across all slots, then it belongs to an imaginary slot. This
    # imaginary slot is concatenated to the masks as the first mask with any
    # pixels that belong to it set to 1. So, the argmax will always select the
    # imaginary slot for a pixel if that pixel is in the imaginary slot. This
    # has the effect such that any pixel that is masked by multiple slots will
    # be grouped together into one mask and removed from its previous mask.
    less_than_threshold = pred_masks < threshold
    thresholded = torch.all(less_than_threshold, dim=1, keepdim=True).to(torch.float)
    pred_masks = torch.cat([thresholded, pred_masks], dim=1)
    segmentation_thresholded = pred_masks.to(torch.float).argmax(1).squeeze(1)
    cmap_segmentation_thresholded = cmap_tensor(
        segmentation_thresholded.to(torch.uint8)
    )

    # `cmap_segmentation` and `cmap_segmentation_thresholded` have shape
    # [batch_size, channels=3, height, width].
    if return_cmap:
        return (
            segmentation,
            segmentation_thresholded,
            cmap_segmentation,
            cmap_segmentation_thresholded,
        )
    return segmentation, segmentation_thresholded


def PolyArea(x, y):
    """Implementation of Shoelace formula for polygon area"""
    # From https://stackoverflow.com/a/30408825
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_largest_objects(objects, metric="area"):
    # Implementation partly from https://stackoverflow.com/a/60955825
    # `objects` has shape [batch_size, num_objects, height, width]
    largest_objects = []
    for batch_idx in range(objects.shape[0]):
        distances = []
        for object_idx in range(objects.shape[1]):
            ob = objects[batch_idx, object_idx].to(torch.bool)
            if not torch.any(ob):
                distances.append(0)
                continue

            # Find a convex hull in O(N log N)
            points = np.indices((128, 128)).transpose((1, 2, 0))[ob]
            try:
                hull = ConvexHull(points)
            except QhullError:
                distances.append(0)
                continue
            # Extract the points forming the hull
            hullpoints = points[hull.vertices, :]
            if metric == "distance":
                # Naive way of finding the best pair in O(H^2) time if H is number
                # of points on hull
                hdist = cdist(hullpoints, hullpoints, metric="euclidean")
                # Get the farthest apart points
                bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

                point1 = hullpoints[bestpair[0]]
                point2 = hullpoints[bestpair[1]]
                score = np.linalg.norm(point2 - point1)
            elif metric == "area":
                x = hullpoints[:, 0]
                y = hullpoints[:, 1]
                score = PolyArea(x, y)
            distances.append(score)

        largest_objects.append(np.argmax(np.array(distances)))
    return np.array(largest_objects)
