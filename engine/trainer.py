import torch
from tqdm import tqdm
from torch.amp import autocast
class BraTSTrainer:
    def __init__(self, model, ema_model, optimizer, criterion, scaler, device, 
                 accumulation_steps, ema_decay, clip_grad, 
                 scheduler=None, warmup_epochs=0, base_lr=3e-4):
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.ema_decay = ema_decay
        self.clip_grad = clip_grad
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr

    def _update_ema(self):
        
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.mul_(self.ema_decay).add_(p, alpha=1.0 - self.ema_decay)
            for ema_b, model_b in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_b.copy_(model_b)

    def train_epoch(self, loader, epoch_idx):
        self.model.train()
        running_loss = 0.0

        if epoch_idx < self.warmup_epochs:
            warm_lr = self.base_lr * float(epoch_idx + 1) / float(self.warmup_epochs)
            for g in self.optimizer.param_groups:
                g["lr"] = warm_lr

        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(tqdm(loader, desc=f"Train Epoch {epoch_idx+1}")):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).long()

            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                self._update_ema()

            running_loss += loss.item() * self.accumulation_steps

        if self.scheduler is not None and epoch_idx >= self.warmup_epochs:
            self.scheduler.step()

        return running_loss / max(len(loader), 1)