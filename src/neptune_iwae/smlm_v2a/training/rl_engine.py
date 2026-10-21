import argparse
import json
import math
import os
import shutil
import time
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from . import train_setup, train_kernels
from ..utils.helpers import current_info
from ..utils.io import param_io, model_io
from ..data.smlm_dataset import SMLMRLDataset, SMLMLiveSampleRLDataset, SMLMPriorDataset, smlm_collate
from ..data.data_factory import FrameRescalar, BkgFrameRescalar, TargetProcess
from ..generic.localizations import FrameLocSet
from ..calibration.psf import CSplinePSF


LOG2PI = math.log(2.0 * math.pi)


def parse_args():
    p = argparse.ArgumentParser(description='Stage-2 RL (GRPO) training')
    p.add_argument('-p', '--param_file', required=True, help='Parameter file (.yml or .json)')
    p.add_argument('--ckpt_path', required=True, help='Best checkpoint from SFT')
    p.add_argument('--steps', type=int, default=1000, help='Number of RL update steps')
    p.add_argument('--epochs', type=int, default=None, help='Number of RL epochs (overrides steps)')
    p.add_argument('--rollouts_per_gpu', type=int, default=1, help='Number of rollouts per GPU per batch')
    p.add_argument('--log_folder', default='runs', help='TensorBoard log folder')
    p.add_argument('--experiment_comment', default='rl', help='Experiment comment')
    p.add_argument('--no_log', action='store_true', default=False, help='Disable logging')
    p.add_argument('--lr', type=float, default=3e-6, help='RL learning rate')
    p.add_argument('--clip_ratio', type=float, default=0.2, help='GRPO clip ratio')
    p.add_argument('--kl_coef', type=float, default=1e-3, help='KL penalty coefficient')
    p.add_argument('--ref_update_steps', type=int, default=400, help='Steps between reference updates')
    p.add_argument('--save_every', type=int, default=200, help='Steps between checkpoints')
    p.add_argument('--p_min', type=float, default=1e-4, help='Minimum p for mixture weights (fallback for legacy sampling)')
    p.add_argument('--p_th', type=float, default=None,
                   help='Detection threshold for RL sampling (defaults to PostProcessingParam.raw_th or 0.5)')
    p.add_argument('--count_penalty', type=float, default=1e-3, help='Emitter count penalty weight')
    p.add_argument('--phot_penalty', type=float, default=1e-3, help='Photon range penalty weight')
    p.add_argument('--bkg_loss_w', type=float, default=0.1, help='Background supervision loss weight')
    p.add_argument('--sim_device', default='cpu', help='Device for simulation (default: cpu)')
    p.add_argument('--sim_cache_device', default='cpu',
                   help="Device to cache simulated frames (cpu, cuda:0, or 'sim' to match sim_device)")
    p.add_argument('--sim_mode', default=None, choices=('acquisition', 'samples', 'apriori'),
                   help='Override Simulation.mode for RL')
    p.add_argument('--no_amp', action='store_true', default=False, help='Disable AMP')
    p.add_argument('--eval_every', type=int, default=0, help='Run eval every N steps (0 to disable)')
    return p.parse_args()


def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return rank, world_size, local_rank, device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return 0, 1, 0, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def broadcast_tensor(t: torch.Tensor, *, dtype: torch.dtype, device: torch.device, rank: int) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    shape_list = [t.shape if rank == 0 else None]
    dist.broadcast_object_list(shape_list, src=0)
    shape = shape_list[0]
    if rank != 0:
        t = torch.empty(shape, dtype=dtype, device=device)
    dist.broadcast(t, src=0)
    return t


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t.unsqueeze(0)
    world_size = dist.get_world_size()
    out = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(out, t)
    return torch.stack(out, dim=0)


def broadcast_int(value: int, device: torch.device, rank: int) -> int:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


def freeze_for_rl(model: torch.nn.Module, trainable_keys: Tuple[str, ...]) -> float:
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if any(key in name for key in trainable_keys):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    return trainable / max(total, 1)


def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return model.module.state_dict() if isinstance(model, DDP) else model.state_dict()


def rescale_mu_sig(pxyz_mu: torch.Tensor, pxyz_sig: torch.Tensor, phot_max: float, z_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = pxyz_mu.clone()
    sig = pxyz_sig.clone()
    if phot_max is not None:
        mu[:, 0] *= phot_max
        sig[:, 0] *= phot_max
    if z_max is not None:
        mu[:, 3] *= z_max
        sig[:, 3] *= z_max
    return mu, sig


def get_pixel_centers(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x_center = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    y_center = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    x_grid = x_center.repeat(height).view(height, width)
    y_grid = y_center.repeat_interleave(width).view(height, width)
    return x_grid.reshape(-1), y_grid.reshape(-1)


def sample_emitters(y_out: torch.Tensor, *, loc_meta: FrameLocSet, phot_max: float, z_max: float,
                    n_max: int, p_th: float, device: torch.device) -> Dict[str, object]:
    y_out = y_out.float()
    batch_size, _, height, width = y_out.shape
    p_raw = torch.clamp(y_out[:, 0], 0.0, 1.0)
    p_sum = p_raw.view(batch_size, -1).sum(-1).clamp(min=1e-8)

    pxyz_mu = y_out[:, 1:5]
    pxyz_sig = torch.clamp(y_out[:, 5:9], min=1e-4)
    pxyz_mu, pxyz_sig = rescale_mu_sig(pxyz_mu, pxyz_sig, phot_max, z_max)

    idx_list: List[torch.Tensor] = []
    val_list: List[torch.Tensor] = []
    logp_list: List[torch.Tensor] = []
    count_penalty: List[torch.Tensor] = []
    phot_penalty: List[torch.Tensor] = []
    nloc_list: List[int] = []

    for b in range(batch_size):
        mean = p_sum[b]
        active = (p_raw[b] >= p_th).view(-1)
        idx = active.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > n_max:
            p_vals = p_raw[b].view(-1)[idx]
            topk = torch.topk(p_vals, n_max, largest=True).indices
            idx = idx[topk]
        nloc = int(idx.numel())
        nloc_list.append(nloc)
        count_penalty.append((mean - float(nloc)).abs())

        if nloc == 0:
            idx_list.append(torch.empty((0,), dtype=torch.long))
            val_list.append(torch.empty((0, 4), dtype=torch.float32))
            logp_list.append(torch.tensor(0.0, device=device))
            phot_penalty.append(torch.tensor(0.0, device=device))
            continue

        mu_b = pxyz_mu[b].reshape(4, -1)[:, idx]
        sig_b = pxyz_sig[b].reshape(4, -1)[:, idx]
        eps = torch.randn_like(mu_b)
        vals = mu_b + sig_b * eps

        logp_feat = torch.distributions.Normal(mu_b, sig_b).log_prob(vals).sum()
        logp_list.append(logp_feat)

        if phot_max is None:
            phot_penalty.append(torch.tensor(0.0, device=device))
        else:
            phot_vals = vals[0]
            phot_pen = torch.relu(phot_vals - phot_max).mean() + torch.relu(-phot_vals).mean()
            phot_penalty.append(phot_pen)

        idx_list.append(idx.detach().cpu())
        val_list.append(vals.transpose(0, 1).detach().cpu())

    return {
        'idx': idx_list,
        'vals': val_list,
        'nloc': nloc_list,
        'logp': torch.stack(logp_list),
        'count_penalty': torch.stack(count_penalty),
        'phot_penalty': torch.stack(phot_penalty),
    }


def log_prob_from_output(y_out: torch.Tensor, sample_info: Dict[str, object], *, phot_max: float, z_max: float,
                          device: torch.device) -> torch.Tensor:
    y_out = y_out.float()
    batch_size, _, height, width = y_out.shape

    pxyz_mu = y_out[:, 1:5]
    pxyz_sig = torch.clamp(y_out[:, 5:9], min=1e-4)
    pxyz_mu, pxyz_sig = rescale_mu_sig(pxyz_mu, pxyz_sig, phot_max, z_max)

    logp_list: List[torch.Tensor] = []
    for b in range(batch_size):
        nloc = sample_info['nloc'][b]
        if nloc == 0:
            logp_list.append(torch.tensor(0.0, device=device))
            continue
        idx = sample_info['idx'][b].to(device)
        vals = sample_info['vals'][b].to(device)
        mu_b = pxyz_mu[b].reshape(4, -1)[:, idx].transpose(0, 1)
        sig_b = pxyz_sig[b].reshape(4, -1)[:, idx].transpose(0, 1)
        logp_feat = torch.distributions.Normal(mu_b, sig_b).log_prob(vals).sum()
        logp_list.append(logp_feat)
    return torch.stack(logp_list)


def build_locset(sample_info: Dict[str, object], b: int, *, loc_meta: FrameLocSet, device: torch.device,
                 x_center_flat: torch.Tensor, y_center_flat: torch.Tensor,
                 x_offset: float, y_offset: float) -> FrameLocSet:
    nloc = sample_info['nloc'][b]
    if nloc == 0:
        return None
    idx = sample_info['idx'][b]
    vals = sample_info['vals'][b]

    idx_dev = idx.to(device)
    x_abs = x_center_flat[idx_dev] + vals[:, 1].to(device) + x_offset
    y_abs = y_center_flat[idx_dev] + vals[:, 2].to(device) + y_offset
    phot = vals[:, 0].to(device)

    if loc_meta.ndim == 3:
        z = vals[:, 3].to(device)
        xyz = torch.stack((x_abs, y_abs, z), dim=1)
    else:
        xyz = torch.stack((x_abs, y_abs), dim=1)

    xyz = xyz.detach().cpu()
    phot = phot.detach().cpu()
    id_tensor = torch.arange(nloc, dtype=loc_meta.i_type)
    indf = torch.zeros(nloc, dtype=loc_meta.i_type)

    loc = FrameLocSet(**loc_meta.meta)
    loc.init_load(id=id_tensor, indf=indf, xyz=xyz, phot=phot)
    return loc


def compute_reward(sample_info: Dict[str, object], y_out_ref: torch.Tensor, raw_mid: torch.Tensor, *,
                   loc_meta: FrameLocSet, psf: CSplinePSF, camera: object, bkg_max: float,
                   count_penalty_w: float, phot_penalty_w: float, device: torch.device) -> torch.Tensor:
    y_out_ref = y_out_ref.float()
    raw_mid = raw_mid.float()
    batch_size, _, height, width = y_out_ref.shape

    bkg = torch.clamp(y_out_ref[:, 9], 0.0, 1.0) * bkg_max
    bkg = bkg.to(device)
    raw_mid = raw_mid.to(device)

    x_offset = loc_meta.xextent[0]
    y_offset = loc_meta.yextent[0]

    _A2D_tensor = torch.as_tensor(camera.A2D)
    _offset_tensor = torch.as_tensor(camera.offset)
    _var_tensor = torch.as_tensor(camera.var)
    if _A2D_tensor.ndim == 0:
        _A2D = _A2D_tensor.to(device)
        _offset = _offset_tensor.to(device)
        _var = _var_tensor.to(device)
    else:
        _A2D = _A2D_tensor[y_offset:y_offset + height, x_offset:x_offset + width].to(device)
        _offset = _offset_tensor[y_offset:y_offset + height, x_offset:x_offset + width].to(device)
        _var = _var_tensor[y_offset:y_offset + height, x_offset:x_offset + width].to(device)

    rewards = torch.zeros((batch_size,), device=device)
    x_center_flat, y_center_flat = get_pixel_centers(height, width, device)
    for b in range(batch_size):
        loc = build_locset(sample_info, b, loc_meta=loc_meta, device=device,
                           x_center_flat=x_center_flat, y_center_flat=y_center_flat,
                           x_offset=float(x_offset), y_offset=float(y_offset))
        if loc is None:
            phot = torch.zeros_like(raw_mid[b])
        else:
            phot = psf.psf_conv(loc, device=device)[0]

        mu_e = camera.QE * (phot + bkg[b]) + camera.spur_noise
        var_e = mu_e + _var
        mu_cam = mu_e * _A2D + _offset
        var_cam = torch.clamp(var_e * (_A2D ** 2), min=1e-6)

        nll = 0.5 * (((raw_mid[b] - mu_cam) ** 2) / var_cam + torch.log(var_cam) + LOG2PI)
        reward = -nll.mean()
        reward -= count_penalty_w * sample_info['count_penalty'][b]
        reward -= phot_penalty_w * sample_info['phot_penalty'][b]
        rewards[b] = reward

    return rewards


def main():
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed()
    torch.manual_seed(1234 + rank)
    print(f"[rank {rank}] init local_rank={local_rank} world_size={world_size} device={device}", flush=True)

    param, calibration = param_io.load_params(args.param_file)
    if args.sim_mode is not None:
        param.Simulation.mode = args.sim_mode
    sim_device = args.sim_device
    if sim_device in {'local', 'auto'}:
        sim_device = f'cuda:{local_rank}' if device.type == 'cuda' else 'cpu'
    param.Hardware.device_simulation = sim_device
    cache_device = args.sim_cache_device
    if cache_device == 'sim':
        cache_device = sim_device

    loc_meta = train_setup.setup_loc_meta(param, calibration)
    sim_train, sim_test = train_setup.setup_random_simulation(loc_meta, param, calibration)

    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    if args.p_th is None:
        p_th = getattr(getattr(param, 'PostProcessingParam', None), 'raw_th', None)
        if p_th is None:
            p_th = 0.5
    else:
        p_th = args.p_th
    if is_main_process(rank):
        print(f"RL sampling p_th={p_th:.4f}", flush=True)
    if is_main_process(rank):
        print(f"RL simulation device={sim_device}, cache_device={cache_device}, mode={param.Simulation.mode}", flush=True)
    if param.Simulation.mode == 'samples':
        train_dataset = SMLMLiveSampleRLDataset(sim_train, param.Simulation.train_ds_size,
                                                frame_proc=frame_proc, bkg_frame_proc=bkg_proc,
                                                frame_window=param.HyperParameter.channels_in,
                                                cache_device=cache_device)
    else:
        train_dataset = SMLMRLDataset(sim_train, frame_proc=frame_proc, bkg_frame_proc=bkg_proc,
                                      frame_window=param.HyperParameter.channels_in,
                                      cache_device=cache_device)

    train_loader = None
    train_loader_len = 0
    if is_main_process(rank):
        if param.Simulation.mode != 'samples':
            print("Sampling RL dataset...", flush=True)
            t0 = time.time()
            train_dataset.sample(verbose=True, nan_check=True)
            print(f"RL dataset sampled in {time.time() - t0:.2f}s", flush=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=param.HyperParameter.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=param.Hardware.num_worker_train,
            pin_memory=(cache_device == 'cpu'),
            collate_fn=smlm_collate,
        )
        train_loader_len = len(train_loader)

    train_loader_len = broadcast_int(train_loader_len, device=device, rank=rank)
    steps_target = args.steps if args.epochs is None else args.epochs * train_loader_len

    experiment_id = current_info(args.experiment_comment)
    experiment_root = param.ExpInOut.experiment_out or 'runs'
    experiment_path = os.path.join(experiment_root, experiment_id)
    os.makedirs(experiment_path, exist_ok=True)

    if is_main_process(rank):
        param_backup = os.path.join(experiment_path, f'param_run{os.path.splitext(args.param_file)[1]}')
        shutil.copy(args.param_file, param_backup)
        with open(os.path.join(experiment_path, 'rl_config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger = train_setup.setup_logger(args.no_log, args.log_folder, experiment_id)
    else:
        logger = train_setup.setup_logger(True, args.log_folder, experiment_id)

    eval_every = max(int(args.eval_every), 0)
    eval_dataset = None
    eval_loader = None
    eval_criterion = None
    post_processor = None
    matcher = None
    if is_main_process(rank) and eval_every > 0:
        target_proc = TargetProcess(
            disable_attr=param.DataFilter.disabled_attributes,
            phot_max=param.Scaling.phot_max,
            z_max=param.Scaling.z_max,
        )
        eval_dataset = SMLMPriorDataset(
            sim_test,
            frame_proc=frame_proc,
            bkg_frame_proc=bkg_proc,
            tar_proc=target_proc,
            frame_window=param.HyperParameter.channels_in,
            return_loc=False,
        )
        eval_dataset.sample(verbose=True, nan_check=True)
        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=param.HyperParameter.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_worker_train,
            pin_memory=False,
            collate_fn=smlm_collate,
        )
        eval_criterion = train_setup.setup_criterion(loc_meta, param)
        post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    model_out = os.path.join(experiment_path, 'rl_model.pt')
    model = train_setup.setup_fitter(param, model_out, device)[0]
    model_ls = model_io.LoadSaveModel(model, output_file=model_out, input_file=args.ckpt_path)
    model = model_ls.load_init(device)

    trainable_ratio = freeze_for_rl(model, trainable_keys=('mt_heads',))

    optimizer_cls = torch.optim.AdamW if param.HyperParameter.optimizer == 'AdamW' else torch.optim.Adam
    opt_kwargs = dict(param.HyperParameter.opt_param)
    opt_kwargs['lr'] = args.lr
    optimizer = optimizer_cls([p for p in model.parameters() if p.requires_grad], **opt_kwargs)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.train()

    model_ref = train_setup.setup_fitter(param, None, device)[0]
    model_ref_ls = model_io.LoadSaveModel(model_ref, output_file=None, input_file=args.ckpt_path)
    model_ref = model_ref_ls.load_init(device)
    for p in model_ref.parameters():
        p.requires_grad = False
    model_ref.eval()

    if is_main_process(rank):
        print(f"Trainable ratio: {trainable_ratio:.4f}")
        print(f"World size: {world_size}")

    psf = CSplinePSF.parse(calibration)
    camera = calibration.Camera

    amp_enabled = device.type == 'cuda' and (not args.no_amp)
    scaler = GradScaler(enabled=amp_enabled)

    step = 0
    epoch = 0
    data_iter = iter(train_loader) if is_main_process(rank) else None
    rollouts_per_gpu = max(args.rollouts_per_gpu, 1)
    while step < steps_target:
        if param.Simulation.mode != 'samples' and step % train_loader_len == 0 and step > 0:
            if is_main_process(rank):
                print("Sampling RL dataset...", flush=True)
                t0 = time.time()
                train_dataset.sample(verbose=True, nan_check=True)
                print(f"RL dataset sampled in {time.time() - t0:.2f}s", flush=True)
                data_iter = iter(train_loader)

        if is_main_process(rank):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x_proc, raw_mid, bkg_tar = batch
            x_proc = x_proc.to(device, non_blocking=True)
            raw_mid = raw_mid.to(device, non_blocking=True)
            bkg_tar = bkg_tar.to(device, non_blocking=True) if bkg_tar is not None else torch.empty(0, device=device)
        else:
            x_proc = torch.empty(0, device=device)
            raw_mid = torch.empty(0, device=device)
            bkg_tar = torch.empty(0, device=device)

        x_proc = broadcast_tensor(x_proc, dtype=torch.float32, device=device, rank=rank)
        raw_mid = broadcast_tensor(raw_mid, dtype=torch.float32, device=device, rank=rank)
        bkg_tar = broadcast_tensor(bkg_tar, dtype=torch.float32, device=device, rank=rank)
        if bkg_tar.numel() == 0:
            bkg_tar = None

        if is_main_process(rank) and step % 20 == 0:
            print(f"step {step}: sampling start (rollouts_per_gpu={rollouts_per_gpu})", flush=True)
        with torch.no_grad():
            y_ref = model_ref(x_proc)
            sample_infos = []
            rewards_list = []
            for _ in range(rollouts_per_gpu):
                sample_info = sample_emitters(
                    y_ref,
                    loc_meta=loc_meta,
                    phot_max=param.Scaling.phot_max,
                    z_max=param.Scaling.z_max,
                    n_max=param.DataFilter.max_number_targets,
                    p_th=p_th,
                    device=device,
                )
                rewards = compute_reward(
                    sample_info,
                    y_ref,
                    raw_mid,
                    loc_meta=loc_meta,
                    psf=psf,
                    camera=camera,
                    bkg_max=param.Scaling.bkg_max,
                    count_penalty_w=args.count_penalty,
                    phot_penalty_w=args.phot_penalty,
                    device=device,
                )
                sample_infos.append(sample_info)
                rewards_list.append(rewards)
            rewards_local = torch.stack(rewards_list, dim=0)
        if is_main_process(rank) and step % 20 == 0:
            print(f"step {step}: sampling done", flush=True)

        rewards_all = all_gather_tensor(rewards_local)
        rewards_group = rewards_all.reshape(-1, rewards_local.shape[1])
        reward_mean = rewards_group.mean(dim=0)
        reward_std = rewards_group.std(dim=0, unbiased=False).clamp(min=1e-6)
        adv = (rewards_local - reward_mean) / reward_std

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            y_new = model(x_proc)
            logp_new_list = []
            logp_old_list = []
            for sample_info in sample_infos:
                logp_new_list.append(log_prob_from_output(
                    y_new,
                    sample_info,
                    phot_max=param.Scaling.phot_max,
                    z_max=param.Scaling.z_max,
                    device=device,
                ))
                logp_old_list.append(sample_info['logp'].to(device))
            logp_new = torch.stack(logp_new_list, dim=0)
            logp_old = torch.stack(logp_old_list, dim=0)
            ratio = torch.exp(logp_new - logp_old)
            clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            obj = torch.min(ratio * adv, clipped * adv)
            kl = (logp_new - logp_old)
            loss = -obj.mean() + args.kl_coef * kl.mean()
            bkg_loss = None
            if bkg_tar is not None:
                bkg_pred = y_new[:, 9]
                bkg_loss = F.mse_loss(bkg_pred, bkg_tar, reduction='mean')
                loss = loss + args.bkg_loss_w * bkg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if is_main_process(rank) and step % 20 == 0:
            r_mean = rewards_local.mean().item()
            r_std = rewards_local.std(unbiased=False).item()
            print(f"step {step}: loss={loss.item():.4f} reward={r_mean:.4f} +/- {r_std:.4f}", flush=True)
            logger.add_scalar('rl/loss', loss.item(), step)
            logger.add_scalar('rl/reward_mean', r_mean, step)
            logger.add_scalar('rl/reward_std', r_std, step)
            logger.add_scalar('rl/kl_mean', kl.mean().item(), step)
            if bkg_loss is not None:
                logger.add_scalar('rl/bkg_loss', bkg_loss.item(), step)

        if eval_every > 0 and step % eval_every == 0:
            if is_main_process(rank):
                eval_model = model.module if isinstance(model, DDP) else model
                eval_model.eval()
                _, test_data = train_kernels.test_forward(
                    model=eval_model,
                    loss=eval_criterion,
                    dataloader=eval_loader,
                    epoch=step,
                    device=str(device),
                )
                loc_out = post_processor.forward(test_data.y_out)
                loc_tar = eval_dataset._loc_tar
                _, eval_sets = train_kernels.eval_forward(
                    cam_pxszs=param.Camera.cam_pxszs,
                    matcher=matcher,
                    loc_out=loc_out,
                    loc_tar=loc_tar,
                )
                eval_dict = eval_sets._asdict()
                print(
                    f"eval step {step}: rec={eval_dict['rec']:.3f}, "
                    f"prec={eval_dict['prec']:.3f}, jacc={eval_dict['jac']:.3f}, "
                    f"rmse_r={eval_dict['rmse_lat']:.3f}, rmse_z={eval_dict['rmse_ax']:.3f}",
                    flush=True,
                )
                logger.add_scalar_dict('eval/', eval_dict, step)
                model.train()
            if dist.is_initialized():
                dist.barrier()

        if is_main_process(rank) and step % args.save_every == 0 and step > 0:
            model_ls.save(model.module if isinstance(model, DDP) else model, metric_val=None)

        if step % args.ref_update_steps == 0 and step > 0:
            model_ref.load_state_dict(get_state_dict(model))
            model_ref.eval()

        step += 1

    if is_main_process(rank):
        model_ls.save(model.module if isinstance(model, DDP) else model, metric_val=None)


if __name__ == '__main__':
    main()
