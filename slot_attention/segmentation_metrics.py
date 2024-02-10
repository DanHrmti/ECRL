# Based on https://github.com/deepmind/multi_object_datasets/blob/9c40290c5e385d3efbccb34fb776b57d44721cba/segmentation_metrics.py
import torch
import torch.nn.functional as F

"""Implementation of the adjusted Rand index."""


def adjusted_rand_index(true_mask, pred_mask, name="ari_score"):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this function can ignore points in an image
    corresponding to the background (i.e. not to an object).
    Args:
        true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
            The true cluster assignment encoded as one-hot.
        pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
            The predicted cluster assignment encoded as categorical probabilities.
            This function works on the argmax over axis 2.
            name: str. Name of this operation (defaults to "ari_score").
    Returns:
        ARI scores as a tf.float32 `Tensor` of shape [batch_size].
    Raises:
        ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
            We've chosen not to handle the special cases that can occur when you have
            one cluster per datapoint (which would be unusual).
    References:
        Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
            https://link.springer.com/article/10.1007/BF01908075
        Wikipedia
            https://en.wikipedia.org/wiki/Rand_index
        Scikit Learn
            http://scikit-learn.org/stable/modules/generated/\
            sklearn.metrics.adjusted_rand_score.html
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
        # This rules out the n_true_groups == n_pred_groups == n_points
        # corner case, and also n_true_groups == n_pred_groups == 0, since
        # that would imply n_points == 0 too.
        # The sklearn implementation has a corner-case branch which does
        # handle this. We chose not to support these cases to avoid counting
        # distinct clusters just to check if we have one cluster per datapoint.
        raise ValueError(
            "adjusted_rand_index requires n_groups < n_points. We don't handle "
            "the special cases that can occur when you have one cluster "
            "per datapoint."
        )

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    # We convert true and predicted clusters to one-hot ('oh') representations.
    true_mask_oh = true_mask.type(torch.float32)  # already one-hot
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).type(torch.float32)

    n_points = torch.sum(true_mask_oh, axis=[1, 2]).type(torch.float32)

    nij = torch.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, axis=1)
    b = torch.sum(nij, axis=2)

    rindex = torch.sum(nij * (nij - 1), axis=[1, 2])
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2

    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator
    # If a divide by 0 occurs, set the ARI value to 1.
    zeros_in_denominator = torch.argwhere(denominator == 0).flatten()
    if zeros_in_denominator.nelement() > 0:
        ari[zeros_in_denominator] = 1

    # The case where n_true_groups == n_pred_groups == 1 needs to be
    # special-cased (to return 1) as the above formula gives a divide-by-zero.
    # This might not work when true_mask has values that do not sum to one:
    both_single_cluster = torch.logical_and(
        _all_equal(true_group_ids), _all_equal(pred_group_ids)
    )
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def _all_equal(values):
    """Whether values are all equal along the final axis."""
    return torch.all(torch.eq(values, values[..., :1]), axis=-1)
