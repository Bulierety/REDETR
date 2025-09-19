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
        
        # 早停机制初始化 - 修改为20轮
        self.early_stop_patience = getattr(cfg, 'early_stop_patience', 20)  # 修改为20轮
        self.early_stop_delta = getattr(cfg, 'early_stop_delta', 0.001)     # 最小提升阈值
        self.best_map = 0.0                                                 # 最佳mAP值
        self.patience_counter = 0                                           # 当前等待轮数
        self.early_stop_triggered = False                                   # 早停标志
        
        # 日志格式化相关
        self.epoch_width = len(str(cfg.epoches))
        self.log_separator = "=" * 80
    
    def _print_header(self, title):
        """打印标题头"""
        print(f"\n{self.log_separator}")
        print(f"{title:^80}")
        print(f"{self.log_separator}")
    
    def _print_section(self, title):
        """打印章节标题"""
        print(f"\n{' ' + title + ' ':-^80}")
    
    def _print_info(self, key, value, indent=0):
        """格式化打印信息"""
        indent_str = " " * indent
        print(f"{indent_str}{key:<25}: {value}")
    
    def _print_epoch_progress(self, epoch, total_epochs, train_stats, test_stats):
        """打印epoch进度信息"""
        current_epoch_str = f"Epoch [{epoch:>{self.epoch_width}}/{total_epochs}]"
        print(f"\n{current_epoch_str:-^80}")
        
        # 训练统计信息
        if train_stats:
            self._print_section("Training Statistics")
            for k, v in train_stats.items():
                if isinstance(v, (int, float)):
                    self._print_info(k, f"{v:.6f}", indent=2)
        
        # 测试统计信息
        if test_stats:
            self._print_section("Validation Statistics")
            for k, v in test_stats.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    self._print_info(k, f"{v[0]:.6f}", indent=2)
                elif isinstance(v, (int, float)):
                    self._print_info(k, f"{v:.6f}", indent=2)
    
    def _print_early_stop_info(self, current_map, improved=False):
        """打印早停相关信息"""
        self._print_section("Early Stopping Status")
        self._print_info("Current mAP", f"{current_map:.4f}", indent=2)
        self._print_info("Best mAP", f"{self.best_map:.4f}", indent=2)
        self._print_info("Patience", f"{self.patience_counter}/{self.early_stop_patience}", indent=2)
        
        if improved:
            print(f"{' ':2}🎯 mAP improved! Best model saved.")
        else:
            print(f"{' ':2}⏳ No improvement, patience counter increased.")
    
    def fit(self):
        self._print_header("RT-DETR TRAINING STARTED")
        self._print_info("Early stopping patience", f"{self.early_stop_patience} epochs")
        self._print_info("Early stopping delta", f"{self.early_stop_delta}")
        self._print_info("Total epochs", f"{self.cfg.epoches}")
        print(self.log_separator)
        
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._print_info("Trainable parameters", f"{n_parameters:,}")
        print(self.log_separator)

        # ------------- TensorBoard 初始化 -------------
        if dist.is_main_process():
            tb_dir = self.output_dir / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            self._print_info("TensorBoard directory", str(tb_dir))
        # ---------------------------------------------

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, }

        start_time = time.time()
        training_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._print_info("Training start time", training_start_time)
        print(self.log_separator)
        
        # 修改循环条件，加入早停判断
        epoch = self.last_epoch + 1
        while epoch < args.epoches and not self.early_stop_triggered:
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # 训练一个epoch
            epoch_start_time = time.time()
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)
            epoch_train_time = time.time() - epoch_start_time

            self.lr_scheduler.step()
            
            # 保存checkpoint
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            # 验证
            eval_start_time = time.time()
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )
            epoch_eval_time = time.time() - eval_start_time

            # 打印epoch进度信息
            self._print_epoch_progress(epoch, args.epoches, train_stats, test_stats)
            self._print_info("Epoch time", f"{epoch_train_time:.2f}s (train) + {epoch_eval_time:.2f}s (eval)", indent=2)

            # ------------- 早停机制判断 -------------
            current_map = test_stats.get('coco_eval_bbox', [0])[0]  # 获取当前mAP值
            
            if dist.is_main_process():
                improved = False
                if current_map > self.best_map + self.early_stop_delta:
                    # 性能提升，重置计数器并更新最佳值
                    self.best_map = current_map
                    self.patience_counter = 0
                    improved = True
                    
                    # 保存最佳模型
                    best_model_path = self.output_dir / 'best_model.pth'
                    dist.save_on_master(self.state_dict(epoch), best_model_path)
                    self._print_info("Best model saved", str(best_model_path), indent=2)
                    
                else:
                    # 性能没有显著提升，增加计数器
                    self.patience_counter += 1
                    
                    # 检查是否触发早停
                    if self.patience_counter >= self.early_stop_patience:
                        self.early_stop_triggered = True
                        self._print_section("EARLY STOPPING TRIGGERED")
                        self._print_info("Stopped at epoch", epoch, indent=2)
                        self._print_info("Best mAP", f"{self.best_map:.4f}", indent=2)
                        self._print_info("Best epoch", best_stat['epoch'], indent=2)
                
                # 打印早停信息
                self._print_early_stop_info(current_map, improved)
            # -----------------------------------------

            # 更新最佳统计
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            
            self._print_section("Best Statistics So Far")
            for k, v in best_stat.items():
                if k != 'epoch':
                    self._print_info(k, f"{v:.6f}", indent=2)
                else:
                    self._print_info(k, v, indent=2)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'early_stop_patience': self.patience_counter,
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
            print(self.log_separator)
        
        # ------------- 关闭 TensorBoard -------------
        if dist.is_main_process():
            self.writer.close()
        # --------------------------------------------
                    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        self._print_header("TRAINING COMPLETED")
        self._print_info("Total training time", total_time_str)
        self._print_info("Best mAP achieved", f"{self.best_map:.4f}")
        self._print_info("Best epoch", best_stat['epoch'])
        
        if self.early_stop_triggered:
            self._print_info("Stopped reason", "Early stopping")
        else:
            self._print_info("Stopped reason", "Completed all epochs")
        
        print(self.log_separator)

    def val(self):
        self._print_header("VALIDATION STARTED")
        self.eval()
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        val_start_time = time.time()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
        val_time = time.time() - val_start_time
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        self._print_section("Validation Results")
        for k, v in test_stats.items():
            if isinstance(v, (list, tuple)) and len(v) > 0:
                self._print_info(k, f"{v[0]:.6f}")
            elif isinstance(v, (int, float)):
                self._print_info(k, f"{v:.6f}")
        
        self._print_info("Validation time", f"{val_time:.2f}s")
        print(self.log_separator)
        
        return