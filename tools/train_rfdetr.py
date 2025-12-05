from rfdetr import RFDETRBase
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Create a new project from your locally trained model")
    parser.add_argument("--DATASET_ROOT", type=str, default="./football-players-detection-18/", help="Local dataset root path")
    parser.add_argument("--EPOCHS", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--BATCH_SIZE", type=int, default=16, help="Number of samples per iteration")
    parser.add_argument("--GRAD_ACCUM_STEPS", type=int, default=1, help="scales batch size based on this value")
    parser.add_argument("--LR", type=float, default=1e-4, help="Increment for search space exploration")
    parser.add_argument("--OUTPUT_DIR", type=str, default="./outputs/", help="Output files from training")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    model = RFDETRBase()
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)

    model.train(
        dataset_dir=args.DATASET_ROOT,
        epochs=args.EPOCHS,
        batch_size=args.BATCH_SIZE,
        grad_accum_steps=args.GRAD_ACCUM_STEPS,
        lr=args.LR,
        output_dir=args.OUTPUT_DIR
    )

if __name__ == "__main__":
    main()