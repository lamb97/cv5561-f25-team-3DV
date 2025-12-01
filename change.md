# Feature 3DGS + LightGaussian (Semantic-Aware Pruning) Notes

## What was added/changed
- **Semantic gradient tracking**  
  - `semantic_grad_ema` stored on each Gaussian (`vanilla_gaussian.py`), saved/loaded with ckpt and pruned together with other properties.  
  - `feature_3dgs_renderer.py` updates this EMA from feature gradients (L2 norm), with `semantic_importance_beta` and `semantic_importance_warmup`.
- **Pruning score fusion**  
  - `gaussian_splatting.py::light_gaussian_prune` normalizes visibility score (`v_list`) and semantic score (`semantic_grad_ema`) separately, then combines with weights `semantic_importance_vis_weight` and `semantic_importance_weight`.  
  - Falls back to pure visibility score if semantic weight is 0 or no semantic data.
- **Pruning alignment**  
  - `feature_3dgs_renderer.py::prune_feature_parameters` keeps feature parameters/optimizer states aligned with Gaussian pruning (used only if pruning occurs while renderer has feature params).  
  - Density pruning uses `DensityController.Utils.prune_properties`; semantic EMA is pruned with other properties.
- **Checkpoint compatibility**  
  - `gaussian_model_loader.py` auto-fills missing `semantic_grad_ema` with zeros when loading old ckpts.
- **Configs added**  
  - `configs/light_gaussian/prune_finetune_semantic.yaml`: prune at step 1, short finetune, semantic-weighted score.  
  - `configs/light_gaussian/prune_finetune_semantic_static.yaml`: same but uses `StaticDensityController` (no densify after pruning).

## Config reference
- `configs/feature_3dgs/lseg-speedup.yaml`  
  Feature distillation only, no pruning, no semantic tracking.
- `configs/feature_3dgs/lseg_semantic_lightgaussian.yaml`  
  Feature distillation + semantic tracking + default prune at step 9000 (disable via CLI if not desired).
- `configs/light_gaussian/prune_finetune_semantic.yaml`  
  Prune at step 1, `prune_percent` default 0.5, finetune 6000 steps, semantic weights 1.0/1.0, densify enabled (VanillaDensityController).
- `configs/light_gaussian/prune_finetune_semantic_static.yaml`  
  Same as above but `density: StaticDensityController` (no densify after pruning).

## Recommended pipeline (office example)
1) **Feature distillation (record semantic EMA, no pruning)**  
   ```bash
   python main.py fit \
     --config configs/feature_3dgs/lseg_semantic_lightgaussian.yaml \
     --model.light_gaussian.prune_steps '[]' \
     --model.renderer.init_args.semantic_importance_warmup 2000 \
     --data.path /users/9/haoti002/data_replica/office4 \
     --model.initialize_from outputs/office4_gsplat/checkpoints/epoch=400-step=30000.ckpt \
     --save_iterations "[5000,8000]" \
     -n office4_lseg_sem_warmup2000
   ```

2) **Prune + finetune geometry (no densify)**  
   ```bash
   python main.py fit \
     --config configs/light_gaussian/prune_finetune_semantic_static.yaml \
     --data.path /users/9/haoti002/data_replica/office4 \
     --model.initialize_from outputs/office4_lseg_sem_warmup2000/checkpoints/epoch=160-step=12000.ckpt \
     --model.light_gaussian.prune_percent 0.5 \
     -n office4_rgb_prune_static
   ```
   - Prune at step 1 using normalized visibility + semantic scores; 6000-step finetune; point count stays fixed (no densify).

3) **Final feature distillation on pruned geometry**  
   ```bash
   python main.py fit \
     --config configs/feature_3dgs/lseg-speedup.yaml \
     --data.path /users/9/haoti002/data_replica/office4 \
     --model.initialize_from outputs/office4_rgb_prune_static/checkpoints/epoch=80-step=6000.ckpt \
     -n office4_lseg_final
   ```
   - Geometry frozen, no pruning; default 10k steps (override `--max_steps` if needed).

## Notes
- If you do **not** want pruning during feature distillation, always set `--model.light_gaussian.prune_steps '[]'` or use `lseg-speedup.yaml`.
- If you want pruning but **no densify** after pruning, use `prune_finetune_semantic_static.yaml`.  
  If you want to allow densify, use `prune_finetune_semantic.yaml`.
- Pruning score formula: `vis_norm = v_list / (max(v_list)+1e-9)`, `sem_norm = semantic_grad_ema / (max(sem)+1e-9)`, `score = vis_w*vis_norm + sem_w*sem_norm`.
