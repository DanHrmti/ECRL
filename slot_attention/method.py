from argparse import Namespace
from typing import Union
from functools import partial
import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision import transforms

from slot_attention.slot_attention_model import SlotAttentionModel
from slot_attention.slate_model import SLATE
from slot_attention.utils import (
    to_rgb_from_tensor,
    warm_and_decay_lr_scheduler,
    cosine_anneal,
    linear_warmup,
    visualize,
    compute_ari,
    sa_segment,
    rescale,
    get_largest_objects,
    cmap_tensor,
)


class SlotAttentionMethod(pl.LightningModule):
    def __init__(
        self,
        model: Union[SlotAttentionModel, SLATE],
        datamodule: pl.LightningDataModule,
        params: Namespace,
    ):
        if type(params) is dict:
            params = Namespace(**params)
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.save_hyperparameters(params)
        self.valid_loss = []

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def step(self, batch):
        mask = None
        if self.params.model_type == "slate":
            self.tau = cosine_anneal(
                self.trainer.global_step,
                self.params.tau_start,
                self.params.tau_final,
                0,
                self.params.tau_steps,
            )
            loss = self.model.loss_function(batch, self.tau, self.params.hard)
        elif self.params.model_type == "sa":
            separation_tau = None
            if self.params.use_separation_loss:
                if self.params.separation_tau:
                    separation_tau = self.params.separation_tau
                else:
                    separation_tau = self.params.separation_tau_max_val - cosine_anneal(
                        self.trainer.global_step,
                        self.params.separation_tau_max_val,
                        0,
                        self.params.separation_tau_start,
                        self.params.separation_tau_end,
                    )
            area_tau = None
            if self.params.use_area_loss:
                if self.params.area_tau:
                    area_tau = self.params.area_tau
                else:
                    area_tau = self.params.area_tau_max_val - cosine_anneal(
                        self.trainer.global_step,
                        self.params.area_tau_max_val,
                        0,
                        self.params.area_tau_start,
                        self.params.area_tau_end,
                    )
            loss, mask = self.model.loss_function(
                batch, separation_tau=separation_tau, area_tau=area_tau
            )

        return loss, mask

    def training_step(self, batch, batch_idx):
        loss_dict, _ = self.step(batch)
        logs = {"train/" + key: val.item() for key, val in loss_dict.items()}
        self.log_dict(logs, sync_dist=True)
        return loss_dict["loss"]

    def sample_images(self, stage="validation"):
        dl = (
            self.datamodule.val_dataloader()
            if stage == "validation"
            else self.datamodule.train_dataloader()
        )
        if stage == 'validation':
            perm = torch.randperm(self.params.val_batch_size)
        else:
            perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))
        if type(batch) == list:
            batch = batch[0][idx]
        else:
            batch = batch[idx]

        if self.params.accelerator:
            batch = batch.to(self.device)
        if self.params.model_type == "sa":
            recon_combined, recons, masks, slots = self.model.forward(batch)
            # `masks` has shape [batch_size, num_entries, channels, height, width].
            threshold = getattr(self.params, "sa_segmentation_threshold", 0.5)
            _, _, cmap_segmentation, cmap_segmentation_thresholded = sa_segment(
                masks, threshold
            )

            # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
            out = torch.cat(
                [
                    to_rgb_from_tensor(batch.unsqueeze(1)),  # original images
                    to_rgb_from_tensor(recon_combined.unsqueeze(1)),  # reconstructions
                    cmap_segmentation.unsqueeze(1),
                    cmap_segmentation_thresholded.unsqueeze(1),
                    to_rgb_from_tensor(recons * masks + (1 - masks)),  # each slot
                ],
                dim=1,
            )

            batch_size, num_slots, C, H, W = recons.shape
            images = vutils.make_grid(
                out.view(batch_size * out.shape[1], C, H, W).cpu(),
                normalize=False,
                nrow=out.shape[1],
            )
        elif self.params.model_type == "slate":
            recon, _, _, attns = self.model(batch, self.tau, True)
            gen_img = self.model.reconstruct_autoregressive(batch)
            vis_recon = visualize(batch, recon, gen_img, attns, N=32)
            images = vutils.make_grid(
                vis_recon, nrow=self.params.num_slots + 3, pad_value=0.2
            )[:, 2:-2, 2:-2]

        return images

    def validation_step(self, batch, batch_idx):
        if type(batch) == list and self.model.supports_masks:
            loss, predicted_mask = self.step(batch[0])
            predicted_mask = torch.permute(predicted_mask, [0, 3, 4, 2, 1])
            # `predicted_mask` has shape [batch_size, height, width, channels, num_entries]
            predicted_mask = torch.squeeze(predicted_mask)
            batch_size, height, width, num_entries = predicted_mask.shape
            predicted_mask = torch.reshape(
                predicted_mask, [batch_size, height * width, num_entries]
            )
            # `predicted_mask` has shape [batch_size, height * width, num_entries]
            # Scale from [0, 1] to [0, 255] to match the true mask.
            predicted_mask = (predicted_mask * 255).type(torch.int)
            ari = compute_ari(
                predicted_mask,
                batch[1],
                len(batch[0]),
                self.params.resolution[0],
                self.params.resolution[1],
                self.datamodule.max_num_entries,
            )
            loss["ari"] = ari
        else:
            if type(batch) == list:
                batch = batch[0]
            loss, _ = self.step(batch)
        self.valid_loss.append(loss)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.valid_loss
        # print(outputs)
        logs = {
            "validation/" + key: torch.stack([x[key] for x in outputs]).float().mean()
            for key in outputs[0].keys()
        }
        self.log_dict(logs, sync_dist=True)
        self.valid_loss.clear()

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-774265729
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        if self.params.model_type == "slate":
            optimizer = optim.Adam(
                [
                    {
                        "params": (
                            x[1]
                            for x in self.model.named_parameters()
                            if "dvae" in x[0]
                        ),
                        "lr": self.params.lr_dvae,
                    },
                    {
                        "params": (
                            x[1]
                            for x in self.model.named_parameters()
                            if "dvae" not in x[0]
                        ),
                        "lr": self.params.lr_main,
                    },
                ],
                weight_decay=self.params.weight_decay,
            )
        elif self.params.model_type == "sa":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.params.lr_main,
                weight_decay=self.params.weight_decay,
            )
        elif self.params.model_type == "gnm":
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.params.lr_main,
                weight_decay=self.params.weight_decay,
            )

        total_steps = self.num_training_steps()

        scheduler_lambda = None
        if self.params.scheduler == "warmup_and_decay":
            warmup_steps_pct = self.params.warmup_steps_pct
            decay_steps_pct = self.params.decay_steps_pct
            scheduler_lambda = partial(
                warm_and_decay_lr_scheduler,
                warmup_steps_pct=warmup_steps_pct,
                decay_steps_pct=decay_steps_pct,
                total_steps=total_steps,
                gamma=self.params.scheduler_gamma,
            )
        elif self.params.scheduler == "warmup":
            scheduler_lambda = partial(
                linear_warmup,
                start_value=0.0,
                final_value=1.0,
                start_step=0,
                final_step=self.params.lr_warmup_steps,
            )

        if scheduler_lambda is not None:
            if self.params.model_type == "slate":
                lr_lambda = [lambda o: 1, scheduler_lambda]
            elif self.params.model_type in ["sa", "gnm"]:
                lr_lambda = scheduler_lambda
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=lr_lambda
            )

            if self.params.model_type == "slate" and hasattr(self.params, "patience"):
                reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.params.patience,
                )
                return (
                    [optimizer],
                    [
                        {"scheduler": scheduler, "interval": "step",},
                        {
                            "scheduler": reduce_on_plateau,
                            "interval": "epoch",
                            "monitor": "validation/loss",
                        },
                    ],
                )

            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step",}],
            )
        return optimizer

    def predict(
        self,
        image,
        do_transforms=False,
        debug=False,
        return_pil=False,
        return_slots=False,
        background_detection="spread_out",
        background_metric="area",
    ):
        """
        `background_detection` options:
            - "spread_out" detects the background as pixels that appear in multiple
            slot masks.
            - "concentrated" assumes the background has been assigned
            to one slot. The slot with the largest distance between two points
            in its mask is assumed to be the background when using
            "concentrated."
            - When using "both," pixels detected using "spread_out"
            and the largest object detected using "concentrated" will be the
            background. The largest object detected using "concentrated" can be
            the background detected by "spread_out."
        `background_metric` is used when `background_detection` is set to "both" or
            "concentrated" to determine which object is largest.
            - "area" will find the object with the largest area
            - "distance" will find the object with the greatest distance between
                two points in that object.
        `return_slots` returns only the Slot Attention slots if using Slot
            Attention.

        """
        assert background_detection in ["spread_out", "concentrated", "both"]
        if return_pil:
            assert debug or (
                len(image.shape) == 3 or image.shape[0] == 1
            ), "Only one image can be passed when using `return_pil` and `debug=False`."

        if do_transforms:
            if getattr(self, "predict_transforms", True):
                current_transforms = []
                if type(image) is not torch.Tensor:
                    current_transforms.append(transforms.ToTensor())
                if self.params.model_type == "sa":
                    current_transforms.append(transforms.Lambda(rescale))
                current_transforms.append(
                    transforms.Resize(
                        self.params.resolution,
                        interpolation=transforms.InterpolationMode.NEAREST,
                    )
                )
                self.predict_transforms = transforms.Compose(current_transforms)
            image = self.predict_transforms(image)

        if len(image.shape) == 3:
            # Add the batch_size dimension (set to 1) if input is a single image.
            image = image.unsqueeze(0)

        if self.params.model_type == "sa":
            recon_combined, recons, masks, slots = self.forward(image)
            if return_slots:
                return slots.view(image.shape[0], -1, *slots.shape[1:])
            threshold = getattr(self.params, "sa_segmentation_threshold", 0.5)
            (
                segmentation,
                segmentation_thresholded,
                cmap_segmentation,
                cmap_segmentation_thresholded,
            ) = sa_segment(masks, threshold)
            # `cmap_segmentation` and `cmap_segmentation_thresholded` have shape
            # [batch_size, channels=3, height, width].
            if background_detection in ["concentrated", "both"]:
                if background_detection == "both":
                    # `segmentation_thresholded` has pixels that are masked by
                    # many slots set to 0 already.
                    objects = F.one_hot(segmentation_thresholded.to(torch.int64))
                else:
                    objects = F.one_hot(segmentation.to(torch.int64))
                # `objects` has shape [batch_size, height, width, num_objects]
                objects = objects.permute([0, 3, 1, 2])
                # `objects` has shape [batch_size, num_objects, height, width]
                largest_object_idx = get_largest_objects(
                    objects.cpu(), metric=background_metric
                )
                # `largest_object_idx` has shape [batch_size]
                largest_object = objects[
                    range(len(largest_object_idx)), largest_object_idx
                ]
                # `largest_object` has shape [batch_size, num_objects=1, height, width]
                largest_object = largest_object.squeeze(1).to(torch.bool)

                segmentation_background = (
                    segmentation_thresholded.clone()
                    if background_detection == "both"
                    else segmentation.clone()
                )
                # Set the largest object to be index 0, the background.
                segmentation_background[largest_object] = 0
                # Recompute the colors now that `largest_object` is the background.
                cmap_segmentation_background = cmap_tensor(segmentation_background)
            elif background_detection == "spread_out":
                segmentation_background = segmentation_thresholded
                cmap_segmentation_background = cmap_segmentation_thresholded
            if debug:
                out = torch.cat(
                    [
                        to_rgb_from_tensor(image.unsqueeze(1)),  # original images
                        to_rgb_from_tensor(
                            recon_combined.unsqueeze(1)
                        ),  # reconstructions
                        cmap_segmentation.unsqueeze(1),
                        cmap_segmentation_background.unsqueeze(1),
                        to_rgb_from_tensor(recons * masks + (1 - masks)),  # each slot
                    ],
                    dim=1,
                )
                batch_size, num_slots, C, H, W = recons.shape
                images = vutils.make_grid(
                    out.view(batch_size * out.shape[1], C, H, W).cpu(),
                    normalize=False,
                    nrow=out.shape[1],
                )
                to_return = images
                if return_pil:
                    to_return = transforms.functional.to_pil_image(to_return.squeeze())
                return to_return
            else:
                to_return = segmentation_background
                if return_pil:
                    to_return = transforms.functional.to_pil_image(
                        cmap_segmentation_background.squeeze()
                    )
                return to_return

        else:
            raise ValueError(
                "The predict function is only implemented for "
                + 'Slot Attention (params.model_type == "sa").'
            )
