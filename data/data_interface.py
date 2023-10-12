import pytorch_lightning as pl
from .datautils import unzip_data, get_data_list
from .datareader import Reader
import os
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8, batch_size = 4, src_path=None, 
                 target_path=None, train_list_path=None, eval_list_path=None, img_size=224, **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        self.configure_dataset()

    def configure_dataset(self):
        if os.path.exists(self.hparams.target_path):
            print("dataset already exists!")
        else:
            unzip_data(self.hparams.src_path, self.hparams.target_path)
        
        if self.hparams.regen:
            if os.path.exists(self.hparams.train_list_path):
                with open(self.hparams.train_list_path, 'w') as f:
                    f.seek(0)
                    f.truncate()
            
            if os.path.exists(self.hparams.eval_list_path):
                with open(self.hparams.eval_list_path, 'w') as f:
                    f.seek(0)
                    f.truncate()
            
            get_data_list(self.hparams.target_path, 
                        self.hparams.train_list_path, self.hparams.eval_list_path)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Reader(self.hparams.train_list_path, self.hparams.img_size)
            self.eval_dataset = Reader(self.hparams.eval_list_path, self.hparams.img_size)
        if stage == 'test' or stage is None:
            self.test_dataset = Reader(self.hparams.eval_list_path, self.hparams.img_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, shuffle=False)
