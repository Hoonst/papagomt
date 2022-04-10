import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # default hparams
    parser.add_argument("--root-path", type=str, default="./data")
    parser.add_argument("--ckpt-path", type=str, default="./src/checkpoints/")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for reproducibility"
    )
    parser.add_argument("--experiment-name", type=str)
    # parser.add_argument("--save-path", type=str, default="./src/model_save")
    parser.add_argument("--log-step", type=int, default=10)
    parser.add_argument("--rnn-cell-type", type=str, default="lstm")
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Evaluation will be done at the end of epoch if set to 0.0",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.0,
        help="Evaluation will be done at the end of epoch if set to 0.0",
    )
    parser.add_argument("--init-type", type=str, default="xavier")

    # model params
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--pad-idx", type=int, default=2)
    
    # use attention / use transformer : 1
    parser.add_argument("--use-attention", type=int, default=0)
    parser.add_argument("--use-transformer", type=int, default=0)

    # transformer params
    parser.add_argument("--trans-layers", type=int, default=3)
    parser.add_argument("--n-heads" , type=int, default=8)
    parser.add_argument("--pf-dim", type=int, default=512)

    # training hparams
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-step", type=int, default=1)
    parser.add_argument("--clip_param", type=float, default=1.0)

    args = parser.parse_args()
    return args
