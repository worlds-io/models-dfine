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
from .det_engine import DataLoaderIterator, evaluate, train_steps


class DetSolver(BaseSolver):
    def fit(self):
        """Step-based training loop.

        The total compute budget is `max_steps`, chunked into windows of
        `eval_every_steps`. After each window we validate, checkpoint, and check for
        early stopping / stage-2 transition. The DataLoader is iterated by a wrapping
        iterator that reshuffles on exhaustion, so the loop is agnostic to dataset size.
        """
        self.train()
        args = self.cfg

        # Freeze backbone for stage 1 — params stay in optimizer but don't compute gradients.
        # Stage 2 unfreezes late backbone stages for domain-specific refinement.
        self._freeze_backbone()

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f"Trainable params: {n_parameters:,}")

        max_steps = int(args.max_steps)
        eval_every_steps = int(args.eval_every_steps)
        checkpoint_every_steps = int(getattr(args, "checkpoint_every_steps", eval_every_steps))
        early_stopping_patience = getattr(args, "early_stopping_patience", 0)
        early_stopping_min_delta = getattr(args, "early_stopping_min_delta", 0)

        top1 = 0.0
        best_stat = {"step": -1}

        # If resuming, run an initial eval at the recovered step so we have a baseline for
        # improvement tracking
        if self.last_step > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_step,
                False,
            )
            for k in test_stats:
                best_stat["step"] = self.last_step
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]

        start_time = time.time()
        step = max(self.last_step, 0)
        evals_without_improvement = 0
        stage = 1

        data_iter = DataLoaderIterator(self.train_dataloader)

        while step < max_steps:
            window_start = time.time()
            window_end = min(step + eval_every_steps, max_steps)
            num_window_steps = window_end - step

            train_stats, step = train_steps(
                self.model,
                self.criterion,
                data_iter,
                self.optimizer,
                self.device,
                start_step=step,
                num_steps=num_window_steps,
                max_steps=max_steps,
                use_wandb=False,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler if stage == 1 else None,
                lr_scheduler=self.lr_scheduler,
                writer=self.writer,
                output_dir=self.output_dir,
                gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
            )

            self.last_step = step

            # Checkpoint ("last" always; numbered checkpoint every N steps)
            if self.output_dir and stage == 1:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if step % checkpoint_every_steps == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint_step{step:06}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # Validate the EMA model (better generalization than the raw model)
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                step,
                False,
                output_dir=self.output_dir,
            )

            # Track improvement for early stopping. Requires improvement > min_delta to
            # reset patience; micro-improvements still update best_stat for checkpointing
            improved = False
            prev_best = 0.0
            current_map = 0.0
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}", v, step)

                current_map = test_stats[k][0]
                if k in best_stat:
                    prev_best = best_stat[k]
                    if current_map > prev_best + early_stopping_min_delta:
                        improved = True
                    if current_map > prev_best:
                        best_stat["step"] = step
                        best_stat[k] = current_map
                else:
                    prev_best = 0
                    best_stat["step"] = step
                    best_stat[k] = current_map
                    improved = True

                if current_map > top1:
                    top1 = current_map
                    if self.output_dir:
                        if stage == 2:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

            window_time = int(time.time() - window_start)
            if improved:
                print(f"mAP improved at step {step}/{max_steps} ({prev_best:.3f} -> {current_map:.3f}), window completed in {window_time}s")
                evals_without_improvement = 0
            else:
                print(f"mAP did not improve at step {step}/{max_steps} ({prev_best:.3f} -> {current_map:.3f}), window completed in {window_time}s")
                evals_without_improvement += 1

            # Early stopping: transition stages or stop training
            if early_stopping_patience > 0 and evals_without_improvement >= early_stopping_patience:
                if stage == 1:
                    print(f"Stage 1 early stopping at step {step} (no improvement for {early_stopping_patience} evals)")
                    self._enter_stage2(step)
                    stage = 2
                    evals_without_improvement = -1  # grace eval for adapting to non-aug data
                    top1 = 0
                    continue
                else:
                    print(f"Stage 2 early stopping at step {step} (no improvement for {early_stopping_patience} evals)")
                    break

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "step": step,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            self.output_dir / "eval" / "latest.pth",
                        )

            # Force GC + CUDA cache release after each eval window. Defense in depth
            # against slow heap accumulation on long-running jobs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def _enter_stage2(self, step):
        """Transition from stage 1 to stage 2: reload best checkpoint, disable augmentation,
        unfreeze late backbone, drop LR, refresh EMA."""
        print(f"Entering stage 2 at step {step}")
        best_stg1_path = str(self.output_dir / "best_stg1.pth")
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.barrier()
        self.load_resume_state(best_stg1_path)

        # Disable augmentation for stage 2 refinement. Freeze the policy at the current
        # dataloader epoch so the `epoch >= policy_epoch` check in the augmentation forward
        # path skips the configured ops from here on
        current_dataloader_epoch = getattr(self.train_dataloader, "_epoch", 0)
        self.train_dataloader.collate_fn.stop_epoch = current_dataloader_epoch
        if hasattr(self.train_dataloader.dataset, "transforms") and \
                hasattr(self.train_dataloader.dataset.transforms, "policy"):
            self.train_dataloader.dataset.transforms.policy["epoch"] = current_dataloader_epoch

        # Re-freeze backbone (load_resume_state doesn't restore requires_grad),
        # then unfreeze late stages for domain adaptation
        self._freeze_backbone()
        self._unfreeze_backbone_late_stages()

        # Drop LR by 10x for fine-grained refinement
        for pg in self.optimizer.param_groups:
            pg["lr"] *= 0.1
        print(f"Stage 2 LR: {[pg['lr'] for pg in self.optimizer.param_groups]}")

        if self.ema:
            self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
            print(f"Refreshed EMA with decay {self.ema.decay}")

    def _freeze_backbone(self):
        model = dist_utils.de_parallel(self.model)
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False

    def _unfreeze_backbone_late_stages(self):
        model = dist_utils.de_parallel(self.model)
        if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
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
            step=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
