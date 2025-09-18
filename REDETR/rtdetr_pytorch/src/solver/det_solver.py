'''
by lyuwenyu
'''
import time 
import json
import datetime
import numpy as np

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class DetSolver(BaseSolver):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # 使用cfg中的output_dir
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 早停机制初始化
        self.early_stop_patience = getattr(cfg, 'early_stop_patience', 10)  # 默认10轮
        self.early_stop_delta = getattr(cfg, 'early_stop_delta', 0.001)     # 最小提升阈值
        self.best_map = 0.0                                                 # 最佳mAP值
        self.patience_counter = 0                                           # 当前等待轮数
        self.early_stop_triggered = False                                   # 早停标志
    
    def fit(self):
        print("Start training")
        print(f"Early stopping patience: {self.early_stop_patience} epochs")
        print(f"Early stopping minimum delta: {self.early_stop_delta}")
        
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # ------------- TensorBoard 初始化 -------------
        if dist.is_main_process():
            tb_dir = self.output_dir / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard logs will be saved to: {tb_dir}")
        # ---------------------------------------------

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, }

        start_time = time.time()
        
        # 修改循环条件，加入早停判断
        epoch = self.last_epoch + 1
        while epoch < args.epoches and not self.early_stop_triggered:
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # ------------- 早停机制判断 -------------
            current_map = test_stats.get('coco_eval_bbox', [0])[0]  # 获取当前mAP值
            
            if dist.is_main_process():
                if current_map > self.best_map + self.early_stop_delta:
                    # 性能提升，重置计数器并更新最佳值
                    self.best_map = current_map
                    self.patience_counter = 0
                    print(f"✓ mAP improved to {current_map:.4f}, resetting patience counter")
                    
                    # 保存最佳模型
                    best_model_path = self.output_dir / 'best_model.pth'
                    dist.save_on_master(self.state_dict(epoch), best_model_path)
                    print(f"✓ Best model saved to {best_model_path}")
                    
                else:
                    # 性能没有显著提升，增加计数器
                    self.patience_counter += 1
                    print(f"✗ mAP did not improve ({current_map:.4f} vs best {self.best_map:.4f}), "
                          f"patience: {self.patience_counter}/{self.early_stop_patience}")
                    
                    # 检查是否触发早停
                    if self.patience_counter >= self.early_stop_patience:
                        self.early_stop_triggered = True
                        print(f"⏹️ Early stopping triggered at epoch {epoch}! "
                              f"No improvement for {self.early_stop_patience} consecutive epochs.")
                        print(f"Best mAP: {self.best_map:.4f} at epoch {best_stat['epoch']}")
            # -----------------------------------------

            # 更新最佳统计
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'early_stop_patience': self.patience_counter,  # 记录早停状态
                        'best_map': self.best_map}
            
            # ------------- TensorBoard 记录 -------------
            if dist.is_main_process():
                # 记录标量指标
                for k, v in log_stats.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(k, v, epoch)
                
                # 记录学习率
                for i, lr in enumerate(self.lr_scheduler.get_last_lr()):
                    self.writer.add_scalar(f'learning_rate/group_{i}', lr, epoch)
                
                # 记录早停相关指标
                self.writer.add_scalar('early_stop/patience_counter', self.patience_counter, epoch)
                self.writer.add_scalar('early_stop/best_map', self.best_map, epoch)
                
                # 添加mAP指标记录
                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    try:
                        coco_eval = coco_evaluator.coco_eval["bbox"]
                        if hasattr(coco_eval, 'stats') and len(coco_eval.stats) >= 6:
                            stats = coco_eval.stats
                            self.writer.add_scalar('mAP/AP', stats[0], epoch)
                            self.writer.add_scalar('mAP/AP50', stats[1], epoch)
                            self.writer.add_scalar('mAP/AP75', stats[2], epoch)
                            self.writer.add_scalar('mAP/AP_small', stats[3], epoch)
                            self.writer.add_scalar('mAP/AP_medium', stats[4], epoch)
                            self.writer.add_scalar('mAP/AP_large', stats[5], epoch)
                            print(f"Epoch {epoch}: mAP@[0.5:0.95] = {stats[0]:.4f}, mAP@0.5 = {stats[1]:.4f}")
                    except Exception as e:
                        print(f"Error recording mAP metrics: {e}")
            # --------------------------------------------
            
            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)
            
            epoch += 1  # 手动增加epoch计数
        
        # ------------- 关闭 TensorBoard -------------
        if dist.is_main_process():
            self.writer.close()
        # --------------------------------------------
                    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # 训练结束提示
        if self.early_stop_triggered:
            print(f"🏁 Training stopped early at epoch {epoch-1}")
        else:
            print(f"🏁 Training completed all {args.epoches} epochs")
        print(f"🎯 Best mAP achieved: {self.best_map:.4f}")

    def val(self):
        self.eval()
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return