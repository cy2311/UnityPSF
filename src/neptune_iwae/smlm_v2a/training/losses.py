from typing import Tuple

import torch
import torch.nn.functional as F
from torch import distributions
from ..utils.helpers import to_tensor


class GMMLoss:
    """Loss as described in Speiser et al., Nature Methods 18(2021): 1082-1090"""
    ch_attr = ['gmm', 'bkg']
    
    def __init__(
        self,
        *,
        xyoffset: Tuple[int, int],
        ch_weight: Tuple[float, float],
        forward_safety: bool = True,
        gmm_target_chunk: int = 16,
        gmm_component_chunk: int = 4096,
        gmm_backend: str = "manual_chunked",
    ):
        """
        Args:
            xyoffset:       (offsetx, offsety) the offsets of the image w.r.t the camera chip
            ch_weight:      The weight for loss_gmm, loss_bkg. Put it to 0 to disable the corresponding channel
            forward_safety: check safety of forward argument 
            gmm_target_chunk: Chunk size for computing GMM log_prob along target dimension (n_max).
                              Smaller values reduce peak GPU memory, and make runtime scale with the
                              number of *active* targets in low-density frames (since empty chunks are skipped).
            gmm_component_chunk: Chunk size over mixture components (H*W). This avoids
                                  torch.distributions.MixtureSameFamily materializing
                                  target x batch x H*W x 4 tensors.
        """
        self.offset_x = xyoffset[0]
        self.offset_y = xyoffset[1]
        self._ch_weight = ch_weight
        self.forward_safety = forward_safety
        self.gmm_target_chunk = int(gmm_target_chunk) if gmm_target_chunk is not None else 0
        self.gmm_component_chunk = int(gmm_component_chunk) if gmm_component_chunk is not None else 0
        self.gmm_backend = str(gmm_backend or "manual_chunked").lower()

    def __call__(self, output:torch.Tensor, pxyz_tar:torch.Tensor, mask:torch.BoolTensor, bkg_tar:torch.Tensor) -> torch.Tensor:
        """
        Compute the Loss_loc term via the Gaussian Micture Model
        Args:
            output:         (N, C=10, H, W) the nn output
            pxyz_tar:       (N, M, C=4) the target phot and xyz processed from the Groundtruth (phot = GT.phot / GT.phot.max()) 
            mask:           (N, M) the activation mask of the ground truth data
            bkg_tar:        (N, H, W) the bkg frame processed from the Groundtruth (bkg = GT.bkg / GT.bkg.max()) 
        Returns:
            torch.Tensor:   (N, 3) the [[loss_count, loss_loc, loss_bkg],...] per batch
        """
        return self.forward(output, pxyz_tar, mask, bkg_tar)
        
    def _compute_loss_gmm(self, p:torch.Tensor, detect_tar:torch.Tensor, pxyz_mu:torch.Tensor, pxyz_sig:torch.Tensor, 
                          pxyz_tar:torch.Tensor, mask:torch.BoolTensor, eps:float=1e-8) -> torch.Tensor:
        """
        Compute the Loss_loc term via the Gaussian Micture Model
        Args:
            p:              (N, H, W) the detection layer of the nn output
            detect_tar:     (N, H, W) the binary detection ground truth
            pxyz_mu:        (N, C=4, H, W) the pxyz_mu layer of the nn output
            pxyz_sig:       (N, C=4, H, W) the pxyz_sig layer of the output
            pxyz_tar:       (N, M, C=4) the groud truth for phot and xyz
            mask:           (N, M) the activation mask of the ground truth data
            eps:            small epsilon for numerical stability
        Returns:
            torch.Tensor:   (N,) the loss_loc per batch
        """
        batch_size, nch, imheight, imwidth = pxyz_mu.shape
        _f_type = pxyz_mu.dtype
        _device = pxyz_mu.device
        
        log_prob = 0.0
        
        # The loss_count term
        nlocs_mean = p.sum(-1).sum(-1) # -> (N,) average of number of emitters per batch
        nlocs_var = torch.clamp((p - p**2).sum(-1).sum(-1), min=eps) # -> (N,) variance of number of emitters per batch
        nlocs_dist = distributions.Normal(loc=nlocs_mean, scale=nlocs_var.sqrt())

        nlocs_tar = mask.sum(-1).float()
        log_prob = log_prob +  nlocs_dist.log_prob(nlocs_tar) * nlocs_tar
        
        if bool(mask.any()):
            pxyz_tar = to_tensor(pxyz_tar, dtype=_f_type, device=_device)
            if self.gmm_backend in {"mixture_same_family", "mixture", "torch"}:
                log_prob = log_prob + self._compute_mixture_same_family_log_gmm(
                    p,
                    pxyz_mu,
                    pxyz_sig,
                    pxyz_tar,
                    mask,
                    eps=eps,
                )
            elif self.gmm_backend in {"manual_chunked", "manual", "chunked"}:
                log_prob = log_prob + self._compute_manual_log_gmm(
                    p,
                    pxyz_mu,
                    pxyz_sig,
                    pxyz_tar,
                    mask,
                    eps=eps,
                )
            else:
                raise ValueError(f"Unsupported GMMLoss gmm_backend={self.gmm_backend!r}")
        
        return -1.0 * log_prob

    def _compute_mixture_same_family_log_gmm(
        self,
        p: torch.Tensor,
        pxyz_mu: torch.Tensor,
        pxyz_sig: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask: torch.BoolTensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        batch_size, nch, imheight, imwidth = pxyz_mu.shape
        _f_type = pxyz_mu.dtype
        _device = pxyz_mu.device
        prob_flat = torch.clamp(p.reshape(batch_size, -1), min=0.0)
        mix = distributions.Categorical(logits=torch.log(prob_flat + eps))

        px_xcenter = torch.arange(imwidth, dtype=_f_type, device=_device).repeat(imheight).view(imheight, imwidth) + 0.5
        px_ycenter = torch.arange(imheight, dtype=_f_type, device=_device).repeat_interleave(imwidth).view(imheight, imwidth) + 0.5
        mu = pxyz_mu.clone()
        mu[:, 1] += px_xcenter.unsqueeze(0) + float(self.offset_x)
        mu[:, 2] += px_ycenter.unsqueeze(0) + float(self.offset_y)
        mu = mu.permute(0, 2, 3, 1).reshape(batch_size, -1, nch)
        sig = torch.clamp(pxyz_sig.permute(0, 2, 3, 1).reshape(batch_size, -1, nch), min=eps)

        comp = distributions.Independent(distributions.Normal(mu, sig), 1)
        gmm = distributions.MixtureSameFamily(mix, comp)
        log_gmm = gmm.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
        return (log_gmm * mask.to(dtype=_f_type)).sum(-1)

    def _compute_manual_log_gmm(
        self,
        p: torch.Tensor,
        pxyz_mu: torch.Tensor,
        pxyz_sig: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask: torch.BoolTensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        batch_size, nch, imheight, imwidth = pxyz_mu.shape
        _f_type = pxyz_mu.dtype
        _device = pxyz_mu.device
        target_count = int(pxyz_tar.shape[1])
        if mask.ndim != 2 or mask.shape[1] != target_count:
            raise ValueError(f"mask shape mismatch: mask={tuple(mask.shape)} pxyz_tar={tuple(pxyz_tar.shape)}")
        active_cols = mask.any(dim=0) if target_count else None
        max_targets = 0
        if active_cols is not None and bool(active_cols.any()):
            max_targets = int(active_cols.nonzero(as_tuple=False).max().item()) + 1
        max_targets = max(0, min(max_targets, target_count))
        if max_targets <= 0:
            return torch.zeros((batch_size,), device=_device, dtype=_f_type)

        prob_flat = p.reshape(batch_size, -1)
        prob_flat = torch.clamp(prob_flat, min=0.0)
        prob_flat = torch.nan_to_num(prob_flat, nan=0.0, posinf=0.0, neginf=0.0)
        log_weights = torch.log(prob_flat + eps)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)

        px_xcenter = torch.arange(imwidth, dtype=_f_type, device=_device).repeat(imheight).view(imheight, imwidth) + 0.5
        px_ycenter = torch.arange(imheight, dtype=_f_type, device=_device).repeat_interleave(imwidth).view(imheight, imwidth) + 0.5
        mu = pxyz_mu.clone()
        mu[:, 1] += px_xcenter.unsqueeze(0) + float(self.offset_x)
        mu[:, 2] += px_ycenter.unsqueeze(0) + float(self.offset_y)
        mu = mu.permute(0, 2, 3, 1).reshape(batch_size, -1, nch)
        sig = torch.clamp(pxyz_sig.permute(0, 2, 3, 1).reshape(batch_size, -1, nch), min=eps)

        target_chunk = int(self.gmm_target_chunk) if self.gmm_target_chunk else max_targets
        target_chunk = max(1, target_chunk)
        comp_count = int(mu.shape[1])
        comp_chunk = int(self.gmm_component_chunk) if self.gmm_component_chunk else comp_count
        comp_chunk = max(1, comp_chunk)
        log_norm_const = torch.as_tensor(2.0 * torch.pi, dtype=_f_type, device=_device).log()
        log_gmm_acc = torch.zeros((batch_size,), device=_device, dtype=_f_type)
        neg_inf = torch.full((batch_size, target_chunk), -torch.inf, device=_device, dtype=_f_type)

        for target_start in range(0, max_targets, target_chunk):
            target_end = min(max_targets, target_start + target_chunk)
            mask_chunk = mask[:, target_start:target_end]
            if not bool(mask_chunk.any()):
                continue
            tar_chunk = pxyz_tar[:, target_start:target_end]
            running = neg_inf[:, : int(target_end - target_start)]
            for comp_start in range(0, comp_count, comp_chunk):
                comp_end = min(comp_count, comp_start + comp_chunk)
                mu_chunk = mu[:, comp_start:comp_end]
                sig_chunk = sig[:, comp_start:comp_end]
                weighted = log_weights[:, None, comp_start:comp_end].expand(
                    batch_size,
                    int(target_end - target_start),
                    int(comp_end - comp_start),
                ).clone()
                for dim in range(nch):
                    sig_d = sig_chunk[:, None, :, dim]
                    diff_d = tar_chunk[:, :, None, dim] - mu_chunk[:, None, :, dim]
                    weighted.add_(-0.5 * ((diff_d / sig_d) ** 2 + log_norm_const + 2.0 * torch.log(sig_d)))
                comp_log = torch.logsumexp(weighted, dim=-1)
                running = torch.logaddexp(running, comp_log)
            log_gmm_acc = log_gmm_acc + (running * mask_chunk.to(dtype=_f_type)).sum(-1)
        return log_gmm_acc
    
    @staticmethod
    def _compute_loss_bkg(bkg:torch.Tensor, bkg_tar:torch.Tensor) -> torch.Tensor:
        """
        Compute the Loss_bg term via MSELoss
        Args:
            bkg:            (N, H, W) the background layer of the nn output
            bkg_tar:        (N, H, W) the groud truth for background
        Returns:
            torch.Tensor:   (N,) the loss_bg per batch
        """
        loss_func = torch.nn.MSELoss(reduction='none')
        return loss_func(bkg, bkg_tar).sum(-1).sum(-1)

    def sanity_check(self, output:torch.Tensor, detect_tar:torch.Tensor, pxyz_tar:torch.Tensor, mask:torch.BoolTensor, bkg_tar:torch.Tensor):
        """sanity checking on the shape of the output and the target"""
        
        batch_size, nch, imheight, imwidth = output.shape
        if nch != 10:
            raise ValueError(f"the number of channels of output should be 10, but is given as {nch}")
        
        if detect_tar.ndim != 3:
            raise ValueError(f"the detect_tar should be shaped into (N, H, W), but is given as {detect_tar.shape}")
        if detect_tar.shape[0] != batch_size:
            raise ValueError(f"n_batch mismatch: detect_tar.shape[0]={detect_tar.shape[0]}, output.shape[0]={batch_size}")
        if detect_tar.shape[1] != imheight:
            raise ValueError(f"imheight mismatch: detect_tar.shape[1]={detect_tar.shape[1]}, output.shape[2]={imheight}")
        if detect_tar.shape[2] != imwidth:
            raise ValueError(f"imwidth mismatch: detect_tar.shape[1]={detect_tar.shape[1]}, output.shape[2]={imheight}") 
        
        if pxyz_tar.ndim != 3:
            raise ValueError(f"the pxyz_tar should be shaped into (N, M, 4), but is given as {pxyz_tar.shape}")
        if pxyz_tar.shape[0] != batch_size:
            raise ValueError(f"n_batch mismatch: pxyz_tar.shape[0]={pxyz_tar.shape[0]}, output.shape[0]={batch_size}")
        if pxyz_tar.shape[2] != 4:
            raise ValueError(f"the number of variables of target should be 4, but is given as {pxyz_tar.shape[2]}")
        if mask.shape[0] != batch_size:
            raise ValueError(f"n_batch mismatch: mask.shape[0]={mask.shape[0]}, output.shape[0]={batch_size}")
        if mask.shape[1] != pxyz_tar.shape[1]:
            raise ValueError(f"M mismatch: mask.shape[1]={mask.shape[1]}, pxyz_tar.shape[1]={pxyz_tar.shape[1]}")
        
        if bkg_tar is not None:
            if bkg_tar.shape[0] != batch_size:
                raise ValueError(f"n_batch mismatch: bkg_tar.shape[0]={bkg_tar.shape[0]}, output.shape[0]={batch_size}")
            if bkg_tar.shape[1] != imheight:
                raise ValueError(f"imheight mismatch: bkg_tar.shape[1]={bkg_tar.shape[1]}, output.shape[2]={imheight}")
            if bkg_tar.shape[2] != imwidth:
                raise ValueError(f"imwidth mismatch: bkg_tar.shape[1]={bkg_tar.shape[1]}, output.shape[2]={imheight}")
    
    def forward(self, output:torch.Tensor, detect_tar:torch.Tensor, pxyz_tar:torch.Tensor, mask:torch.BoolTensor, bkg_tar:torch.Tensor) -> torch.Tensor:
        """
        Compute the Loss_loc term via the Gaussian Micture Model
        Args:
            output:         (N, C=10, H, W) the nn output
            detect_tar:     (N, H, W) the binary detection ground truth
            pxyz_tar:       (N, M, C=4) the target phot and xyz processed from the Groundtruth (phot = GT.phot / GT.phot.max()) 
            mask:           (N, M) the activation mask of the ground truth data
            bkg_tar:        (N, H, W) the bkg frame processed from the Groundtruth (bkg = GT.bkg / GT.bkg.max()) 
        Returns:
            torch.Tensor:   (N, 3) the [[loss_count, loss_loc, loss_bkg],...] per batch
        """
        if self.forward_safety:
            self.sanity_check(output, detect_tar, pxyz_tar, mask, bkg_tar)
        
        p = output[:, 0]
        pxyz_mu = output[:, 1:5]
        pxyz_sig = output[:, 5:9]
        bkg = output[:, 9]

        _ch_weight = to_tensor(self._ch_weight, dtype=output.dtype, device=output.device)
        loss_gmm = self._compute_loss_gmm(p, detect_tar, pxyz_mu, pxyz_sig, pxyz_tar, mask)
        loss_bkg = self._compute_loss_bkg(bkg, bkg_tar)
        
        # Factor 2 because origianl impl. adds the two terms, but this way is better for logging
        loss = 2.0 * torch.cat((loss_gmm[:, None], loss_bkg[:, None]), dim=1) * _ch_weight[None, :]
        return loss
