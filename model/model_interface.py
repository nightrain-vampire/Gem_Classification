import pytorch_lightning as pl
import torch
import torchvision.models as models
import torch.optim.lr_scheduler as lrs
from torchmetrics import Accuracy
import numpy as np


class MInterface(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.accuracy = Accuracy(task='multiclass', 
                                 num_classes=self.hparams.num_classes)
    
    def load_model(self):
        self.model = getattr(models, self.hparams.model_name)(pretrained=False, num_classes=self.hparams.num_classes)

    def configure_loss(self):
        self.loss_function = torch.nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0.0
        optimizer = torch.optim.AdamW(self.parameters(), 
                                     lr=self.hparams.lr, weight_decay=weight_decay)
        
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        image = batch[0]
        label = batch[1]
        predict = self(image)
        loss = self.loss_function(predict, label)
        acc = self.accuracy(predict, label)
        values = {'epoch': batch_idx, 'train_loss': loss.item(), 'train_acc': acc.item()}
        self.log_dict(dictionary=values, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.accuracy.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.print('one training epoch finished!')
        self.accuracy.reset()
        
    def validation_step(self, batch, batch_idx):
        image = batch[0]
        label = batch[1]
        predict = self(image)
        valid_acc = self.accuracy(predict, label)
        self.log('val_acc', valid_acc.item(),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return predict
    
    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.accuracy.reset()
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image = batch[0]
        predict = self(image)
        return predict

