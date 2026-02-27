import argparse

from moeint.moe_null_sim_harness import Trainer


def main(data_file: str, tokenizer_dir: str, batch_size: int, seq_len: int, lr: float, max_steps: int):
    trainer = Trainer(data_file, tokenizer_dir, batch_size, seq_len, lr)
    trainer.train(max_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--tokenizer-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    print(f"Args are: {args}")

    main(args.data_file, args.tokenizer_dir, args.batch_size, args.seq_len, args.lr, args.max_steps)

    """To run
    python sim.py --data-file /work/courses/capstone_era4/Data-benchmark/small_shard.parquet --tokenizer-dir /work/courses/capstone_era4/6_Tokenizer/tsai_131k_tokenizer --max-steps 300 --seq-length 1024
    """