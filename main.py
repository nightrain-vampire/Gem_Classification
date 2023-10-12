from config import get_args_parser
from model import MInterface
from data import DInterface
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import TensorBoardLogger


def main(args):
    data_module = DInterface(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        src_path=args.src_path,
        target_path=args.target_path,
        train_list_path=args.train_list_path,
        eval_list_path=args.eval_list_path,
        img_size=args.img_size,
        regen=args.regen,
    )
    model_module = MInterface(
        model_name=args.model,
        num_classes=args.num_classes,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_min_lr=args.lr_decay_min_lr,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc_epoch",
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        save_top_k = 5,
        monitor="val_acc_epoch",
        mode="max",
        save_weights_only=False,
    )
    

    if args.test:
        ddp_Trainer = pl.Trainer(
            fast_dev_run=False,
            max_epochs=args.num_epochs,
            accelerator="gpu",  devices=1,
            precision=32,   log_every_n_steps=1,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
            callbacks=[RichProgressBar(leave=True)],
        )

        best_model = model_module.load_from_checkpoint(args.resume)
        ddp_Trainer.test(best_model, data_module)
    else:
        logger = TensorBoardLogger(save_dir='./tf_log', name=f"{args.model}")
        ddp_Trainer = pl.Trainer(
            fast_dev_run=False,
            max_epochs=args.num_epochs,
            accelerator="gpu",  devices=2, strategy="ddp",
            precision=32,   log_every_n_steps=1,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
            callbacks=[early_stop_callback, RichProgressBar(leave=True),checkpoint_callback],
            logger=logger,
        )

        ddp_Trainer.fit(model_module, data_module)
    

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
