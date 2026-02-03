import argparse

from moeint.moe_null_sim_harness import Trainer


def main(batch_size: int, seq_len: int, lr: float, max_steps: int):
    trainer = Trainer(batch_size, seq_len, lr)
    trainer.train(max_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    main(args.batch_size, args.seq_len, args.lr, args.max_steps)
