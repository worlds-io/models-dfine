"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import gc
import json
import os
import time

import torch

from ..misc import dist_utils
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg

        # Freeze backbone for stage 1 — params stay in optimizer but don't compute gradients.
        # This focuses stage 1 on the decoder/encoder and saves compute.
        self._freeze_backbone()

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f"Trainable params: {n_parameters:,}")

        early_stopping_patience = getattr(args, 'early_stopping_patience', 0)
        early_stopping_min_delta = getattr(args, 'early_stopping_min_delta', 0)
        # Minimum total epochs before patience may transition stages or stop the
        # run. The 80→N-class head remap re-initializes the score heads, so the
        # first epochs are spent re-learning the head — patience must not fire there.
        early_stopping_min_epochs = getattr(args, 'early_stopping_min_epochs', 0)

        top1 = 0
        best_stat = {"epoch": -1}

        # Per-stage best mAP, persisted to stage_metrics.json
        best_stg1_map = 0.0
        best_stg2_map = 0.0

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                False,
                max_val_images=args.max_val_images,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        epochs_without_improvement = 0
        stage = 1

        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()

            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler if stage == 1 else None,
                writer=self.writer,
                use_wandb=False,
                output_dir=self.output_dir,
                gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
                # First trained epoch runs BN in train mode so running stats adapt to
                # THIS checkpoint's activations before eval-freezing them. Transplanted
                # checkpoints (ONNX->pth) pair fused conv weights with the reference
                # model's pretrained BN stats — eval-mode BN on those unnormalized
                # activations overflows under AMP (NaN in the very first forward), and
                # no checkpoint field distinguishes trusted stats (num_batches_tracked
                # is 0 even in trained D-FINE checkpoints).
                after_train_mode=self._backbone_norm_eval if epoch > start_epoch else None,
            )

            if stage == 1 and (self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished()):
                # CosineAnnealingLR is periodic: past T_max the LR climbs back up.
                # Runs are early-stopped, not T_max-bounded, so hold at the floor
                # once the schedule completes instead of rebounding.
                t_max = getattr(self.lr_scheduler, "T_max", None)
                if t_max is None or self.lr_scheduler.last_epoch < t_max:
                    self.lr_scheduler.step()

            self.last_epoch += 1

            # last.pth is written in BOTH stages so a stage-2 crash is resumable
            # (previously stage-2 epochs were never checkpointed and a crash there
            # discarded the whole run). Periodic checkpoints stay stage-1-only.
            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if stage == 1 and (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                False,
                output_dir=self.output_dir,
                max_val_images=args.max_val_images,
            )

            # Track improvement for early stopping.
            # Requires improvement > min_delta to count as meaningful;
            # micro-improvements still update best_stat for checkpoint tracking.
            improved = False
            # Bind before the loop: an empty test_stats (e.g. the val pass produced
            # no predictions) previously left these unbound and the epoch print
            # below raised NameError instead of reporting the epoch.
            prev_best = best_stat.get("coco_eval_bbox", 0.0)
            current_map = 0.0
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                current_map = test_stats[k][0]
                if k in best_stat:
                    prev_best = best_stat[k]
                    if current_map > prev_best + early_stopping_min_delta:
                        # Only ratchet the "improvement target" when the gain clears
                        # min_delta. Previously this was a two-tier check (inner `if
                        # current_map > prev_best` updated best_stat on any rise), which
                        # silently moved the bar each epoch — a run of 0.001-sized
                        # improvements would end up failing patience even though mAP kept
                        # climbing. Checkpoint save below uses top1, not best_stat, so
                        # sub-delta bumps still get persisted if they're absolute maxes
                        improved = True
                        best_stat["epoch"] = epoch
                        best_stat[k] = current_map
                else:
                    prev_best = 0
                    best_stat["epoch"] = epoch
                    best_stat[k] = current_map
                    improved = True

                if current_map > top1:
                    top1 = current_map
                    if self.output_dir:
                        if stage == 2:
                            best_stg2_map = current_map
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            best_stg1_map = current_map
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")
                        if dist_utils.is_main_process():
                            with (self.output_dir / "stage_metrics.json").open("w") as f:
                                json.dump({"stg1_map": best_stg1_map, "stg2_map": best_stg2_map}, f)

            epoch_time = int(time.time() - epoch_start_time)
            if improved:
                print(f"mAP improved in epoch {epoch + 1} ({prev_best:.3f} -> {current_map:.3f}), completed in {epoch_time}s")
                epochs_without_improvement = 0
            else:
                print(f"mAP did not improve in epoch {epoch + 1} ({prev_best:.3f} -> {current_map:.3f}), completed in {epoch_time}s")
                epochs_without_improvement += 1

            # Early stopping: transition stages or stop training
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
                and (epoch + 1) >= early_stopping_min_epochs
            ):
                if stage == 1:
                    print(f"Stage 1 early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                    self._enter_stage2(epoch)
                    stage = 2
                    epochs_without_improvement = -1  # grace epoch for adapting to non-augmented data
                    top1 = 0
                    continue
                else:
                    print(f"Stage 2 early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                    break

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

            # Explicit gc.collect() at the end of each epoch — cyclic garbage from autograd
            # wrappers, etc. The end-of-evaluate() malloc_trim in det_engine.py returns
            # freed glibc arenas to the OS; without it the ~6 GB of eval-phase transient
            # allocations sit in the process heap until the next val rewrites them and
            # trigger k8s eviction long before the actual limit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final report: re-evaluate the checkpoint that will actually be exported
        # (best across stages) and persist metrics.json — per-class AP included.
        # Everything before this line reported the *last* epoch, not the artifact.
        try:
            self._write_final_metrics(args, best_stg1_map, best_stg2_map)
        except Exception as e:  # metrics are a report, never a reason to fail a run
            print(f"final metrics: FAILED ({type(e).__name__}: {e})")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def _enter_stage2(self, epoch):
        """Transition from stage 1 to stage 2: reload best checkpoint, disable augmentation,
        unfreeze late backbone, drop LR, refresh EMA."""
        print(f"Entering stage 2 at epoch {epoch + 1}")
        best_stg1_path = str(self.output_dir / "best_stg1.pth")
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        # best_stg1.pth only exists once stage 1 improved at least once; if it
        # never did (tiny/degenerate datasets), continue from current weights
        # rather than crashing.
        if os.path.exists(best_stg1_path):
            self.load_resume_state(best_stg1_path)
        else:
            print("No best_stg1.pth (stage 1 never improved) — continuing from current weights")

        # Disable augmentation for stage 2 refinement.
        # The COCO dataset stores its transform pipeline in `_transforms`
        # (torchvision's VisionDataset leaves `.transforms` as None), so the old
        # `dataset.transforms.policy` probe silently matched nothing and stage 2
        # trained with full augmentation. Probe both attribute conventions and
        # say out loud which way it went.
        self.train_dataloader.collate_fn.stop_epoch = epoch
        dataset_transforms = getattr(self.train_dataloader.dataset, '_transforms', None)
        if dataset_transforms is None:
            dataset_transforms = getattr(self.train_dataloader.dataset, 'transforms', None)
        if dataset_transforms is not None and hasattr(dataset_transforms, 'policy'):
            dataset_transforms.policy["epoch"] = epoch
            print(f"Stage 2: augmentation ops {dataset_transforms.policy.get('ops', [])} disabled")
        else:
            print("Stage 2: WARNING — transform policy not found; augmentation remains enabled")

        # Re-freeze backbone (load_resume_state doesn't restore requires_grad),
        # then unfreeze late stages for domain adaptation.
        self._freeze_backbone()
        self._unfreeze_backbone_late_stages()

        # Drop LR by 10x for fine-grained refinement
        for pg in self.optimizer.param_groups:
            pg['lr'] *= 0.1
        print(f"Stage 2 LR: {[pg['lr'] for pg in self.optimizer.param_groups]}")

        if self.ema:
            self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
            print(f"Refreshed EMA with decay {self.ema.decay}")

    def _freeze_backbone(self):
        """Freeze all backbone parameters. They remain in the optimizer but don't compute gradients."""
        model = dist_utils.de_parallel(self.model)
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False

    def _backbone_norm_eval(self):
        """Put frozen backbone BatchNorm layers in eval mode.

        `requires_grad = False` stops the affine params from training but the
        running mean/var still update every forward pass under model.train() —
        so the "frozen" backbone drifted with whatever physical batch size was
        selected. Called after each epoch's model.train() (via the
        `after_train_mode` hook): any backbone norm layer whose params are all
        frozen goes to eval; norm layers in stage-2-unfrozen stages keep training.
        """
        import torch.nn as nn

        model = dist_utils.de_parallel(self.model)
        if not hasattr(model, 'backbone'):
            return
        for m in model.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params = list(m.parameters(recurse=False))
                if params and not any(p.requires_grad for p in params):
                    m.eval()

    def _unfreeze_backbone_late_stages(self):
        """Unfreeze the last 2 backbone stages for domain-specific feature adaptation."""
        model = dist_utils.de_parallel(self.model)
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'stages'):
            # HGNetv2 has stages 0-3. Unfreeze stages 2-3 (high-level semantic features).
            for stage in model.backbone.stages[2:]:
                for param in stage.parameters():
                    param.requires_grad = True
            unfrozen = sum(p.numel() for s in model.backbone.stages[2:] for p in s.parameters() if p.requires_grad)
            print(f"Unfroze backbone stages 2-3 ({unfrozen:,} params)")

    def _write_final_metrics(self, args, best_stg1_map, best_stg2_map):
        """Evaluate the to-be-exported checkpoint and persist output_dir/metrics.json.

        Mirrors the Go exporter's stage choice (higher per-stage mAP wins, stage 2
        preferred on ties/missing metrics) so the reported numbers describe the
        artifact that ships, not the last epoch trained. Adds per-class AP — the
        headline mAP hid e.g. a cards-vs-chips gap entirely.
        """
        stg1 = self.output_dir / "best_stg1.pth"
        stg2 = self.output_dir / "best_stg2.pth"
        if stg1.exists() and stg2.exists():
            ckpt = stg1 if best_stg1_map > best_stg2_map else stg2
        elif stg2.exists():
            ckpt = stg2
        elif stg1.exists():
            ckpt = stg1
        else:
            print("final metrics: no best checkpoint written; skipping")
            return

        module = self.ema.module if self.ema else dist_utils.de_parallel(self.model)
        state = torch.load(str(ckpt), map_location="cpu")
        state_dict = state.get("ema", {}).get("module") or state.get("model")
        if state_dict is None:
            print(f"final metrics: {ckpt.name} has no ema/model state; skipping")
            return
        module.load_state_dict(state_dict)

        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
            max_val_images=args.max_val_images,
        )
        stats = test_stats.get("coco_eval_bbox")
        if not stats:
            print("final metrics: evaluation produced no stats; skipping")
            return

        names = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large',
                 'mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large']
        metrics = {names[i]: round(float(stats[i]), 4) for i in range(min(len(stats), len(names)))}
        metrics["checkpoint"] = ckpt.name
        metrics["background_excluded_from_gt"] = True
        # COCO size buckets above are in ORIGINAL-pixel areas (32²/96² at e.g. 4K),
        # not model-input space — flag it so consumers don't over-read map_small.
        metrics["size_buckets_space"] = "original_pixels"

        per_class = {}
        try:
            ce = coco_evaluator.coco_eval["bbox"]
            precision = ce.eval.get("precision") if isinstance(ce.eval, dict) else None
            if precision is not None:
                cat_ids = list(ce.params.catIds)
                cats = {}
                gt = getattr(ce, "cocoGt", None)
                if gt is not None and isinstance(getattr(gt, "dataset", None), dict):
                    for c in gt.dataset.get("categories", []):
                        cats[c.get("id")] = c.get("name", str(c.get("id")))
                for k, cat_id in enumerate(cat_ids):
                    # precision: [iou_thrs, recall, cat, area, max_dets]
                    p_all = precision[:, :, k, 0, -1]
                    p_50 = precision[0, :, k, 0, -1]
                    ap = float(p_all[p_all > -1].mean()) if (p_all > -1).any() else -1.0
                    ap50 = float(p_50[p_50 > -1].mean()) if (p_50 > -1).any() else -1.0
                    per_class[cats.get(cat_id, str(cat_id))] = {
                        "ap": round(ap, 4), "ap_50": round(ap50, 4),
                    }
        except Exception as e:
            print(f"final metrics: per-class AP unavailable ({type(e).__name__}: {e})")
        if per_class:
            metrics["per_class"] = per_class

        with (self.output_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        # Single-line, machine-parseable summary for log scrapers (the reviewer
        # parses the last such line; it now describes the exported checkpoint).
        print(f"final metrics: {json.dumps(metrics)}")

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
            max_val_images=self.cfg.max_val_images,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
