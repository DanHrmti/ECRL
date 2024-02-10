"""
Loss functions implementations used in the optimization of DLP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# functions
def batch_pairwise_kl(mu_x, logvar_x, mu_y, logvar_y, reverse_kl=False):
    """
    Calculate batch-wise KL-divergence
    mu_x, logvar_x: [batch_size, n_x, points_dim]
    mu_y, logvar_y: [batch_size, n_y, points_dim]
    kl = -0.5 * Σ_points_dim (1 + logvar_x - logvar_y - exp(logvar_x)/exp(logvar_y)
                    - ((mu_x - mu_y) ** 2)/exp(logvar_y))
    """
    if reverse_kl:
        mu_a, logvar_a = mu_y, logvar_y
        mu_b, logvar_b = mu_x, logvar_x
    else:
        mu_a, logvar_a = mu_x, logvar_x
        mu_b, logvar_b = mu_y, logvar_y
    bs, n_a, points_dim = mu_a.size()
    _, n_b, _ = mu_b.size()
    logvar_aa = logvar_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    logvar_bb = logvar_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_aa = mu_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_bb = mu_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    p_kl = -0.5 * (1 + logvar_aa - logvar_bb - logvar_aa.exp() / logvar_bb.exp()
                   - ((mu_aa - mu_bb) ** 2) / logvar_bb.exp()).sum(-1)  # [batch_size, n_x, n_y]
    return p_kl


def batch_pairwise_kl_beta_dist(alpha_post, beta_post, alpha_prior, beta_prior, eps=1e-5):
    """
    Compute kl divergence of Beta variable

    :param alpha_post, beta_post [batch_size, n_x, 1]
    :param alpha_prior,  beta_prior  [batch_size, n_y, 1]
    :return: kl divergence, (B, ...)
    """
    bs, n_a, points_dim = alpha_post.size()
    _, n_b, _ = alpha_prior.size()

    alpha_post = alpha_post.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    beta_post = beta_post.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    alpha_prior = alpha_prior.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    beta_prior = beta_prior.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]

    log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post, beta_post)
    alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post + eps)
    beta = (beta_post - beta_prior) * torch.digamma(beta_post + eps)
    alpha_beta = (alpha_prior - alpha_post + beta_prior - beta_post) * torch.digamma(alpha_post + beta_post + eps)
    kl = (log_bettas + alpha + beta + alpha_beta).sum(-1)  # [batch_size, n_x, n_y]

    return kl


def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x, device=x.device)
    diag_ind_y = torch.arange(0, num_points_y, device=y.device)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum', balance=0.5):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :param balance: balancing coefficient between posterior and prior
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    if balance == 0.5:
        kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
            logvar_o)).sum(1)
    else:
        # detach post
        mu_post = mu.detach()
        logvar_post = logvar.detach()
        mu_prior = mu_o
        logvar_prior = logvar_o
        kl_a = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / torch.exp(logvar_prior) - (
                mu_post - mu_prior).pow(2) / torch.exp(logvar_prior)).sum(1)
        # detach prior
        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_o.detach()
        logvar_prior = logvar_o.detach()
        kl_b = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / torch.exp(logvar_prior) - (
                mu_post - mu_prior).pow(2) / torch.exp(logvar_prior)).sum(1)
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def batch_pairwise_kl_bern(post_prob, prior_prob, reverse_kl=False, eps=1e-15):
    """
    Calculate batch-wise Bernoulli KL-divergence
    post_prob: [batch_size, n_post, 1], in [0,1]
    prior_prob: [batch_size, n_prior, points_dim], in [0,1]
    """
    if reverse_kl:
        a = prior_prob
        b = post_prob
    else:
        a = post_prob
        b = prior_prob
    bs, n_a, points_dim = a.size()
    _, n_b, _ = b.size()
    aa = a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    bb = b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    p_kl = aa * (torch.log(aa + eps) - torch.log(bb + eps)) + (1 - aa) * (
            torch.log(1 - aa + eps) - torch.log(1 - bb + eps))  # [batch_size, n_x, n_y]
    p_kl = p_kl.sum(-1)
    return p_kl


def calc_kl_bern(post_prob, prior_prob, eps=1e-15, reduce='none'):
    """
    Compute kl divergence of Bernoulli variable
    :param post_prob [batch_size, 1], in [0,1]
    :param prior_prob [batch_size, 1], in [0,1]
    :return: kl divergence, (B, ...)
    """
    kl = post_prob * (torch.log(post_prob + eps) - torch.log(prior_prob + eps)) + (1 - post_prob) * (
            torch.log(1 - post_prob + eps) - torch.log(1 - prior_prob + eps))
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


def log_beta_function(alpha, beta, eps=1e-5):
    """
    B(alpha, beta) = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    logB = loggamma(alpha) + loggamma(beta) - loggamaa(alpha + beta)
    """
    # return torch.special.gammaln(alpha) + torch.special.gammaln(beta) - torch.special.gammaln(alpha + beta)
    return torch.lgamma(alpha + eps) + torch.lgamma(beta + eps) - torch.lgamma(alpha + beta + eps)


def calc_kl_beta_dist(alpha_post, beta_post, alpha_prior, beta_prior, reduce='none', eps=1e-5, balance=0.5):
    """
    Compute kl divergence of Beta variable
    https://en.wikipedia.org/wiki/Beta_distribution
    :param alpha_post, beta_post [batch_size, 1]
    :param alpha_prior,  beta_prior  [batch_size, 1]
    :param balance kl balance between posterior and prior
    :return: kl divergence, (B, ...)
    """
    if balance == 0.5:
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post, beta_post)
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior - alpha_post + beta_prior - beta_post) * torch.digamma(alpha_post + beta_post + eps)
        kl = log_bettas + alpha + beta + alpha_beta
    else:
        # detach post
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post.detach(),
                                                                                    beta_post.detach())
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post.detach() + eps)
        beta = (beta_post.detach() - beta_prior) * torch.digamma(beta_post.detach() + eps)
        alpha_beta = (alpha_prior - alpha_post.detach() + beta_prior - beta_post.detach()) * torch.digamma(
            alpha_post.detach() + beta_post.detach() + eps)
        kl_a = log_bettas + alpha + beta + alpha_beta

        # detach prior
        log_bettas = log_beta_function(alpha_prior.detach(), beta_prior.detach()) - log_beta_function(alpha_post,
                                                                                                      beta_post)
        alpha = (alpha_post - alpha_prior.detach()) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior.detach()) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior.detach() - alpha_post + beta_prior.detach() - beta_post) * torch.digamma(
            alpha_post + beta_post + eps)
        kl_b = log_bettas + alpha + beta + alpha_beta
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


# classes
class ChamferLossKL(nn.Module):
    """
    Calculates the KL-divergence between two sets of (R.V.) particle coordinates.
    """

    def __init__(self, use_reverse_kl=False):
        super(ChamferLossKL, self).__init__()
        self.use_reverse_kl = use_reverse_kl

    def forward(self, mu_preds, logvar_preds, mu_gts, logvar_gts, posterior_mask=None):
        """
        mu_preds, logvar_preds: [bs, n_x, feat_dim]
        mu_gts, logvar_gts: [bs, n_y, feat_dim]
        posterior_mask: [bs, n_x]
        """
        p_kl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=False)
        # [bs, n_x, n_y]
        if self.use_reverse_kl:
            p_rkl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=True)
            p_kl = 0.5 * (p_kl + p_rkl.transpose(2, 1))
        mins, _ = torch.min(p_kl, 1)  # [bs, n_y]
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(p_kl, 2)  # [bs, n_x]
        if posterior_mask is not None:
            mins = mins * posterior_mask
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2


class ChamferLossKL2(nn.Module):
    """
    Calculates the KL-divergence between two sets of (R.V.) particle coordinates.
    """

    def __init__(self, use_reverse_kl=False):
        super(ChamferLossKL2, self).__init__()

    def forward(self, mu_preds, logvar_preds, mu_gts, logvar_gts):
        """
        mu_preds, logvar_preds: [bs, n_x, feat_dim]
        mu_gts, logvar_gts: [bs, n_y, feat_dim]
        posterior_mask: [bs, n_x]
        """
        batch_size, n_x, feat_dim = mu_preds.shape
        _, n_y, _ = mu_gts.shape
        mu_dist = batch_pairwise_dist(mu_preds, mu_gts)
        batch_idx = torch.arange(mu_preds.shape[0], device=mu_preds.device, dtype=torch.long)[:, None]
        # [bs, n_x, n_y]
        # --- n_y --- #
        _, argmins = torch.min(mu_dist, 1)  # [bs, n_y]
        # argmins: [bs, n_y]
        logvar_post = logvar_preds[batch_idx, argmins].view(-1, feat_dim)  # [bs, n_y, feat] -> [bs * n_y, feat]
        mu_post = mu_preds[batch_idx, argmins].view(-1, feat_dim)  # [bs, n_y, feat] -> [bs * n_y, feat]
        logvar_prior = logvar_gts.view(-1, feat_dim)  # [bs, n_y, feat] -> [bs * n_y, feat]
        mu_prior = mu_gts.view(-1, feat_dim)  # [bs, n_y, feat] -> [bs * n_y, feat]
        loss_1 = calc_kl(logvar_post, mu_post, logvar_o=logvar_prior, mu_o=mu_prior, reduce='none')  # [bs * n_y,]
        loss_1 = loss_1.view(batch_size, -1).sum(-1)  # [batch_size, ]
        # loss_1 = torch.sum(mins, 1)
        # --- n_x --- #
        _, argmins = torch.min(mu_dist, 2)  # [bs, n_x]
        # argmins: [bs, n_x]
        logvar_post = logvar_preds.view(-1, feat_dim)  # [bs, n_x, feat] -> [bs * n_x, feat]
        mu_post = mu_preds.view(-1, feat_dim)  # [bs, n_x, feat] -> [bs * n_x, feat]
        logvar_prior = logvar_gts[batch_idx, argmins].view(-1, feat_dim)  # [bs, n_x, feat] -> [bs * n_x, feat]
        mu_prior = mu_gts[batch_idx, argmins].view(-1, feat_dim)  # [bs, n_x, feat] -> [bs * n_x, feat]
        # mins, _ = torch.min(p_kl, 2)  # [bs, n_x]
        loss_2 = calc_kl(logvar_post, mu_post, logvar_o=logvar_prior, mu_o=mu_prior, reduce='none')  # [bs * n_x,]
        loss_2 = loss_2.view(batch_size, -1).sum(-1)  # [batch_size, ]
        # loss_2 = torch.sum(mins, 1)
        return 0.5 * (loss_1 + loss_2)


class ChamferLossKLBernoulli(nn.Module):
    """
    Calculates the Bernoulli KL-divergence between two sets of (R.V.) particle coordinates.
    """

    def __init__(self, use_reverse_kl=False, eps=1e-15):
        super(ChamferLossKLBernoulli, self).__init__()
        self.eps = eps
        self.use_reverse_kl = use_reverse_kl

    def forward(self, post_prob, prior_prob):
        p_kl = batch_pairwise_kl_bern(post_prob, prior_prob, reverse_kl=False, eps=self.eps)
        if self.use_reverse_kl:
            p_rkl = batch_pairwise_kl_bern(post_prob, prior_prob, reverse_kl=True, eps=self.eps)
            p_kl = 0.5 * (p_kl + p_rkl.transpose(2, 1))
        mins, _ = torch.min(p_kl, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(p_kl, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.eval()
        self.vggnet.requires_grad_(False)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids=(2, 7, 12, 21, 30), accumulate_mode='sum', device=torch.device("cpu"),
                 normalize=True, use_loss_scale=False, vgg_coeff=0.12151):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids).to(device)
        self.layer_ids = layer_ids
        self.accumulate_mode = accumulate_mode
        self.device = device
        self.use_normalization = normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.use_loss_scale = use_loss_scale
        self.vgg_coeff = vgg_coeff

    def forward(self, I1, I2, reduction='sum', only_image=False):
        b_sz = I1.size(0)
        num_ch = I1.size(1)

        # if self.use_normalization:
        #     I1, I2 = self.normalize(I1), self.normalize(I2)
        if self.accumulate_mode == 'sum':
            loss = ((I1 - I2) ** 2).view(b_sz, -1).sum(1)
            # if normalized, effectively: (1 / (std ** 2)) * (I_1 - I_2) ** 2
        elif self.accumulate_mode == 'ch_mean':
            loss = ((I1 - I2) ** 2).view(b_sz, I1.shape[1], -1).mean(1).sum(-1)
        else:
            loss = ((I1 - I2) ** 2).view(b_sz, -1).mean(1)

        if self.use_normalization:
            I1, I2 = self.normalize(I1), self.normalize(I2)

        if num_ch == 1:
            I1 = I1.repeat(1, 3, 1, 1)
            I2 = I2.repeat(1, 3, 1, 1)

        # original
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)
        # print([f.shape for f in f1])
        # in: 3x128x128: [torch.Size([32, 64, 128, 128]), torch.Size([32, 128, 64, 64]), torch.Size([32, 256, 32, 32]),
        # torch.Size([32, 512, 16, 16]), torch.Size([32, 512, 4, 4])]

        # with normalization and concat
        # f1 = self.vgg(self.normalize(I1))
        # f2 = self.vgg(self.normalize(I2))

        if not only_image:
            for i in range(len(self.layer_ids)):
                if self.accumulate_mode == 'sum':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).sum(1)
                elif self.accumulate_mode == 'ch_mean':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, f1[i].shape[1], -1).mean(1).sum(-1)
                else:
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).mean(1)
                # loss = loss + self.vgg_coeff * layer_loss
                c = self.vgg_coeff if self.use_normalization else 1.0
                loss = loss + c * layer_loss
                # loss = loss + layer_loss

        if self.use_loss_scale:
            # by using `sum` for the features, and using scaling instead of `mean` we maintain the weight
            # of each dimension contribution to the loss
            max_dim = max([np.product(f.shape[1:]) for f in f1])
            # scale = 1 / np.product(I1.shape[1:])
            scale = 1 / max_dim
            # print(f'scale: {I1.shape[1:]}: {scale}')
            loss = scale * loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_dimensions(self, device=torch.device("cpu")):
        dims = []
        dummy_input = torch.zeros(1, 3, 128, 128).to(device)
        dims.append(dummy_input.view(1, -1).size(1))
        f = self.vgg(dummy_input)
        for i in range(len(self.layer_ids)):
            dims.append(f[i].view(1, -1).size(1))
        return dims


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        # self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device, dtype=torch.long)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device, dtype=torch.long)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


def calc_chamfer_kl(mu_post, logvar_post, mu_prior, logvar_prior,
                    mu_depth_post, logvar_depth_post, mu_depth_prior, logvar_depth_prior,
                    mu_scale_post, logvar_scale_post, mu_scale_prior, logvar_scale_prior,
                    mu_features_post, logvar_features_post, mu_features_prior, logvar_features_prior,
                    mu_bg_post, logvar_bg_post, mu_bg_prior, logvar_bg_prior,
                    obj_on_a_post, obj_on_b_post, obj_on_a_prior, obj_on_b_prior, reduce='none'):
    # mu_post, mu_depth_post, mu_scale_post, mu_features_post: [bs, n_kp_a, dim]
    # logvar_post, logvar_depth_post, logvar_scale_post, logvar_features_post: [bs, n_kp_a, dim]
    # mu_bg_post, logvar_bg_post: [bs, dim]
    # obj_on_a_post, obj_on_b_post: [bs, n_kp_a, 1]
    # prior: similar, but n_kp_b instead of n_kp_a

    # calc batch pairwise kls
    loss_kl_kp = batch_pairwise_kl(mu_post, logvar_post, mu_prior, logvar_prior)  # [bs, n_kp_a, n_kp_b]
    loss_kl_depth = batch_pairwise_kl(mu_depth_post, logvar_depth_post, mu_depth_prior, logvar_depth_prior)
    loss_kl_scale = batch_pairwise_kl(mu_scale_post, logvar_scale_post, mu_scale_prior, logvar_scale_prior)
    loss_kl_features = batch_pairwise_kl(mu_features_post, logvar_features_post, mu_features_prior,
                                         logvar_features_prior)
    loss_kl_bg = calc_kl(mu=mu_bg_post, logvar=logvar_bg_post, mu_o=mu_bg_prior, logvar_o=logvar_bg_prior,
                         reduce='none')
    # [bs, ]
    loss_kl_obj_on = batch_pairwise_kl_beta_dist(obj_on_a_post, obj_on_b_post, obj_on_a_prior, obj_on_b_prior)
    # # [bs, n_kp_a, n_kp_b]
    loss_kl_all = loss_kl_kp + loss_kl_depth + loss_kl_scale + loss_kl_features + loss_kl_obj_on  # [bs, n_kp_a, n_kp_b]
    mins, _ = torch.min(loss_kl_all, 1)  # [bs, n_kp_b]
    loss_1 = torch.sum(mins, 1)  # [bs, ]
    mins, _ = torch.min(loss_kl_all, 2)  # [bs, n_kp_a]
    loss_2 = torch.sum(mins, 1)  # [bs, ]
    loss_kl = 0.5 * (loss_1 + loss_2) + loss_kl_bg  # [bs, ]
    if reduce == 'mean':
        return loss_kl.mean()
    elif reduce == 'sum':
        return loss_kl.sum()
    else:
        return loss_kl


if __name__ == '__main__':
    # image_size = 64
    # # image_size = 128
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # loss_func = VGGDistance(accumulate_mode='sum', use_loss_scale=False, normalize=True).to(device)
    # print(loss_func)
    # src = torch.rand(32, 3, image_size, image_size, device=device)
    # target = torch.rand(32, 3, image_size, image_size, device=device)
    # loss = loss_func(src, target)
    # print(f'loss: {loss.mean()}, {loss.shape}')
    #
    n_kp_a = 15
    n_kp_b = 30
    batch_size = 32
    feat_dim = 10

    # mu_a = torch.rand(batch_size, n_kp_a, 2) * 2 - 1
    # logvar_a = torch.log((0.1 * torch.rand_like(mu_a)) ** 2)
    # mu_b = torch.rand(batch_size, n_kp_b, 2) * 2 - 1
    # logvar_b = torch.log((0.1 * torch.ones_like(mu_b)) ** 2)
    # loss_func = ChamferLossKL2()
    # loss = loss_func(mu_a, logvar_a, mu_b, logvar_b)
    # print(f'loss: {loss.shape}, mean: {loss.mean()}')

    # beta_post_a = 0.2 * torch.ones(batch_size, n_kp_a, 1)
    # beta_post_b = 0.2 * torch.ones(batch_size, n_kp_a, 1)
    # beta_prior_a = 0.1 * torch.ones(batch_size, n_kp_b, 1)
    # beta_prior_b = 0.1 * torch.ones(batch_size, n_kp_b, 1)
    #
    # mu_a = torch.rand(batch_size, n_kp_a, 2) * 2 - 1
    # logvar_a = torch.log((0.1 * torch.rand_like(mu_a)) ** 2)
    # mu_b = torch.rand(batch_size, n_kp_b, 2) * 2 - 1
    # logvar_b = torch.log((0.1 * torch.ones_like(mu_b)) ** 2)
    #
    # mu_scale_a = torch.rand(batch_size, n_kp_a, 2) * 2 - 1
    # logvar_scale_a = torch.log((0.1 * torch.rand_like(mu_a)) ** 2)
    # mu_scale_b = torch.rand(batch_size, n_kp_b, 2) * 2 - 1
    # logvar_scale_b = torch.log((0.1 * torch.ones_like(mu_b)) ** 2)
    #
    # mu_depth_a = torch.rand(batch_size, n_kp_a, 1) * 2 - 1
    # logvar_depth_a = torch.log((0.1 * torch.rand_like(mu_depth_a)) ** 2)
    # mu_depth_b = torch.rand(batch_size, n_kp_b, 1) * 2 - 1
    # logvar_depth_b = torch.log((0.1 * torch.ones_like(mu_depth_b)) ** 2)
    #
    # mu_feat_a = torch.rand(batch_size, n_kp_a, feat_dim) * 2 - 1
    # logvar_feat_a = torch.log((0.1 * torch.rand_like(mu_feat_a)) ** 2)
    # mu_feat_b = torch.rand(batch_size, n_kp_b, feat_dim) * 2 - 1
    # logvar_feat_b = torch.log((0.1 * torch.ones_like(mu_feat_b)) ** 2)
    #
    # mu_bg_a = torch.rand(batch_size, feat_dim) * 2 - 1
    # logvar_bg_a = torch.log((0.1 * torch.rand_like(mu_bg_a)) ** 2)
    # mu_bg_b = torch.rand(batch_size, feat_dim) * 2 - 1
    # logvar_bg_b = torch.log((0.1 * torch.ones_like(mu_bg_b)) ** 2)
    #
    # # beta_loss = batch_pairwise_kl_beta_dist(beta_post_a, beta_post_b, beta_prior_a, beta_prior_b)
    # # print(f'beta_loss: {beta_loss.shape}, min_a: {beta_loss.min(2)[0].sum(-1).mean()}')
    #
    # ch_kl_loss = calc_chamfer_kl(mu_a, logvar_a, mu_b, logvar_b,
    #                              mu_depth_a, logvar_depth_a, mu_depth_b, logvar_depth_b,
    #                              mu_scale_a, logvar_scale_a, mu_scale_b, logvar_scale_b,
    #                              mu_feat_a, logvar_feat_a, mu_feat_b, logvar_feat_b,
    #                              mu_bg_a, logvar_bg_a, mu_bg_b, logvar_bg_b,
    #                              beta_post_a, beta_post_b, beta_prior_a, beta_prior_b, reduce='none')
    # print(f'ch_kl_loss: {ch_kl_loss.shape}, ch_kl_loss_mean: {ch_kl_loss.mean()}')

    alpha_post = -1 * torch.ones(10)
    beta_post = -1 * torch.ones(10)
    alpha_prior = -0.5 * torch.ones(10)
    beta_prior = -0.5 * torch.ones(10)
    res_beta = calc_kl_beta_dist(alpha_post, beta_post, alpha_prior, beta_prior)
    print(f'res_beta: {res_beta}')
