"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
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
            )

            # Track improvement for early stopping.
            # Requires improvement > min_delta to count as meaningful;
            # micro-improvements still update best_stat for checkpoint tracking.
            improved = False
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    if test_stats[k][0] > best_stat[k] + early_stopping_min_delta:
                        improved = True
                    if test_stats[k][0] > best_stat[k]:
                        best_stat["epoch"] = epoch
                        best_stat[k] = test_stats[k][0]
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]
                    improved = True

                if test_stats[k][0] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = test_stats[k][0]

                    if self.output_dir:
                        if stage == 2:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                best_stat_print[k] = max(best_stat.get(k, 0), top1)
                print(f"best_stat: {best_stat_print}")

            epoch_time = int(time.time() - epoch_start_time)
            print(f"Finished epoch {epoch + 1} in {epoch_time} seconds")

            if improved:
                epochs_without_improvement = 0
            else:
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
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
