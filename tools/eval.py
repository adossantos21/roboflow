import os
import roboflow
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detector")
    parser.add_argument("--THRESHOLD", type=float, default=0.5, help="Threshold for generating bbox predictions.")
    parser.add_argument("--ROBOFLOW_API_KEY", type=str, default="insert_api_key", help="Roboflow API Key for account recognition")
    parser.add_argument("--WORKSPACE_ID", type=str, default="specified_workspace", help="Roboflow Workspace ID")
    parser.add_argument("--PROJECT_ID", type=str, default="football-players-detection-zolkr", help="Your project name")
    parser.add_argument("--MODEL_TYPE", type=str, default="rfdetr-base", help="Model architecture")
    parser.add_argument("--VERSION_NUMBER", type=int, default=1, help="Dataset version number")
    parser.add_argument("--WEIGHTS_DIR", type=str, default="./checkpoints/", help="Path containing designated checkpoint file")
    parser.add_argument("--CKPT_NAME", type=str, default="checkpoint.pth", help="Name of checkpoint file in weights directory")
    parser.add_argument("--SERVER_TEST_IMAGE", type=str, default="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFaXxL6kgRtQ3NcpKzp3A1ghqJ4XiYEN1crQ&s", help="URL to image address")
    parser.add_argument("--TEST_IMAGE", type=str, default="./football-players-detection-18/test/4b770a_1_4_png.rf.af9607e58333ddc25aef684b88d5e54a.jpg", help="Path to local test image")
    parser.add_argument("--LOCAL", action="store_true", default=False, help="Whether to run inference locally or on Roboflow's server")
    args = parser.parse_args()

    return args

def upload_weights(args):
    """Load trained RF-DETR weights to Roboflow for deployment."""
    
    print("Connecting to Roboflow...")
    rf = roboflow.Roboflow(api_key=args.ROBOFLOW_API_KEY)
    
    workspace = rf.workspace(args.WORKSPACE_ID)
    project = workspace.project(args.PROJECT_ID)
    version = project.version(args.VERSION_NUMBER)
    
    # Verify weights exist
    weights_path = os.path.join(args.WEIGHTS_DIR, args.CKPT_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            "Train your model first using train_rfdetr.py"
        )
    
    print(f"Loading RF-DETR weights from {args.WEIGHTS_DIR}...")
    
    # Load RF-DETR weights
    version.deploy(
        model_type=args.MODEL_TYPE,
        model_path=args.WEIGHTS_DIR,
        filename=args.CKPT_NAME
    )
    
    print("Weights loaded successfully!")
    print(f"Version type: {type(version)}")
    return version


def run_inference(args, version):
    """Run inference using the deployed model."""
    import base64
    
    print(f"\nRunning inference on: {args.TEST_IMAGE}")
    
    model = version.model
    
    # Run prediction
    # NOTE: hosted=True runs on Roboflow's servers
    # NOTE: hosted=False runs locally (requires inference package)
    predictions = model.predict(args.SERVER_TEST_IMAGE, hosted=True).json()
    
    print("\n" + "="*50)
    print("PREDICTIONS")
    print("="*50)
    
    if "predictions" in predictions:
        detections = predictions["predictions"]
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            print(f"  - {det}")
    else:
        print(predictions)
    
    return predictions


def run_local_inference(args):
    """Run inference locally using rfdetr package (no upload needed)."""
    from rfdetr import RFDETRBase
    from PIL import Image
    
    print("Running local inference with RF-DETR...")
    
    # Load trained model
    model = RFDETRBase(pretrain_weights=os.path.join(args.WEIGHTS_DIR, args.CKPT_NAME))
    
    # Run inference
    image = Image.open(args.TEST_IMAGE)
    detections = model.predict(image, threshold=args.THRESHOLD, hosted=False)
    
    print(f"\nDetected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det}")
    
    return detections


def main():
    print("="*50)
    print("RF-DETR Deployment & Inference")
    print("="*50)
    args = parse_args()

    if args.LOCAL:
        # Option 1: Evaluate locally
        print(f"Running inference locally")
        run_local_inference(args)
    else:
        # Option 2: Upload to Roboflow and run hosted inference on their server
        print(f"Running inference on Roboflow server")
        version = upload_weights(args)
        run_inference(args, version)


if __name__ == "__main__":
    main()