# Gotchas

- 2026-06-24: Standard ROI-bank gamma update must use ROI-bank vector-PSF Poisson projection end to end. Do not reintroduce Gaussian sigma/blob projections or Gaussian quicklook fallbacks on the formal gamma route.
- 2026-06-24: When debugging the standard run, isolate epoch-30 ROI library construction first: old-route parity is half-FOV domain crop -> 128x128 tiled loc inference -> FiLM/SoftMoE domain condition -> p-threshold emitters -> 128x128 ROI bank. Do not advance gamma/reconstruction debugging until this stage is verified.
- 2026-06-26: Active SMLM 10-channel output must stay in old IWAE order: p, phot_mu, x_mu, y_mu, z_mu, phot_sig, x_sig, y_sig, z_sig, bg. GMM target/order is phot,x,y,z; do not silently switch back to x,y,z,phot.
- 2026-06-26: Formal old-IWAE GMMLoss metrics expose only loss_gmm, loss_bkg, and loss_total. Count and mixture-localization contributions are internal to loss_gmm; do not log them as separate training loss components.
