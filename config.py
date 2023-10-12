import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Gem Classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='resnet18', type=str,
                        help='Name of model to train, you can choose resnet18, resnet50, resnet101, resnet152')
    parser.add_argument('--num_classes', default=25, type=int)

    # LR parameters
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=10, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Dataset parameters
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--src_path', default="./data/data55032/archive_train.zip",
                         type=str)
    parser.add_argument('--target_path', default="./data/dataset", type=str)
    parser.add_argument('--train_list_path', default="./data/train_list.txt", type=str)
    parser.add_argument('--eval_list_path', default="./data/eval_list.txt", type=str)

    # Training parameters
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)

    # Testing parameters
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--resume', default='./checkpoints/epoch=5-step=492.ckpt', type=str)
    parser.add_argument('--regen', default=True, type=bool)

    return parser
