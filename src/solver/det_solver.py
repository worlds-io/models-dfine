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

        # AFSS: skip already-learned images each epoch and focus on the hard ones.
        # Single-GPU only — under DDP warp_loader swaps in a DistributedSampler.
        self.afss_sampler = None
        self._afss_force_refresh = False
        self._afss_scoring_loader = None
        if getattr(args, 'afss_enabled', False):
            if dist_utils.is_dist_available_and_initialized():
                print("[AFSS] disabled: not supported under distributed training")
            else:
                self._setup_afss(args)

        top1 = 0
        best_stat = {"epoch": -1}

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

            # AFSS: re-score sufficiency every refresh_period epochs (and right after a
            # stage transition, since the model/EMA was just reloaded), then set_epoch
            # below recomputes the subset for this epoch via the sampler fan-out.
            if self.afss_sampler is not None and (
                self.afss_sampler.needs_refresh(epoch) or self._afss_force_refresh
            ):
                self._afss_refresh(epoch)
                self._afss_force_refresh = False

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
            )

            if stage == 1 and (self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished()):
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and stage == 1:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
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
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

            epoch_time = int(time.time() - epoch_start_time)
            if improved:
                print(f"mAP improved in epoch {epoch + 1} ({prev_best:.3f} -> {current_map:.3f}), completed in {epoch_time}s")
                epochs_without_improvement = 0
            else:
                print(f"mAP did not improve in epoch {epoch + 1} ({prev_best:.3f} -> {current_map:.3f}), completed in {epoch_time}s")
                epochs_without_improvement += 1

            # Early stopping: transition stages or stop training
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
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

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def _setup_afss(self, args):
        """Rebuild the train loader to draw indices from an AFSSSampler.

        torch's DataLoader forbids reassigning sampler/batch_sampler after init, so we
        reconstruct an equivalent loader (same class, dataset, collate_fn, workers) with
        sampler= instead of shuffle=True. The AFSSState is hung on the solver so it's
        captured by BaseSolver.state_dict() for resume.
        """
        from ..data.afss import AFSSSampler

        loader = self.train_dataloader
        sampler = AFSSSampler(
            num_images=len(loader.dataset),
            easy_thresh=args.afss_easy_thresh,
            medium_thresh=args.afss_medium_thresh,
            easy_keep_frac=args.afss_easy_keep_frac,
            medium_keep_frac=args.afss_medium_keep_frac,
            hard_repeat=args.afss_hard_repeat,
            refresh_period=args.afss_refresh_period,
            score_only=args.afss_score_only,
        )
        kwargs = dict(
            batch_size=loader.batch_size,
            sampler=sampler,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers,
        )
        if loader.num_workers > 0:
            kwargs["prefetch_factor"] = loader.prefetch_factor
        new_loader = type(loader)(loader.dataset, **kwargs)
        new_loader.shuffle = False
        new_loader.afss_sampler = sampler

        self.train_dataloader = new_loader
        self.afss_sampler = sampler
        self.afss_state = sampler.state
        print(
            f"[AFSS] enabled: {len(loader.dataset)} train images, refresh every "
            f"{sampler.refresh_period} epochs, score_only={sampler.score_only}"
        )

    def _afss_refresh(self, epoch):
        """Score the training set (on the EMA model) and update the sampler's tiers."""
        from ..data.afss import build_scoring_loader, score_train_set

        args = self.cfg
        if self._afss_scoring_loader is None:
            self._afss_scoring_loader = build_scoring_loader(
                self.train_dataloader.dataset,
                batch_size=self.train_dataloader.batch_size,
                num_workers=self.train_dataloader.num_workers,
                collate_fn=self.train_dataloader.collate_fn,
                max_images=args.afss_max_score_images,
            )
        module = self.ema.module if self.ema else self.model
        scores = score_train_set(
            module,
            self.postprocessor,
            self._afss_scoring_loader,
            self.device,
            conf_thresh=args.afss_conf_thresh,
            iou_thresh=args.afss_iou_thresh,
        )
        self.afss_sampler.update_sufficiency(scores, epoch)

    def _enter_stage2(self, epoch):
        """Transition from stage 1 to stage 2: reload best checkpoint, disable augmentation,
        unfreeze late backbone, drop LR, refresh EMA."""
        print(f"Entering stage 2 at epoch {epoch + 1}")
        # Sufficiency computed before this reload is stale (model/EMA just changed);
        # force an AFSS re-score on the next epoch.
        self._afss_force_refresh = True
        best_stg1_path = str(self.output_dir / "best_stg1.pth")
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        self.load_resume_state(best_stg1_path)

        # Disable augmentation for stage 2 refinement
        self.train_dataloader.collate_fn.stop_epoch = epoch
        if hasattr(self.train_dataloader.dataset, 'transforms') and \
                hasattr(self.train_dataloader.dataset.transforms, 'policy'):
            self.train_dataloader.dataset.transforms.policy["epoch"] = epoch

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
