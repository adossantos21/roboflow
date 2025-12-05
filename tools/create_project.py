import roboflow
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Create a new project from your locally trained model")
    parser.add_argument("--ROBOFLOW_API_KEY", type=str, default="insert_api_key", help="Roboflow API Key for account recognition")
    parser.add_argument("--WORKSPACE_ID", type=str, default="specified_workspace", help="Roboflow Workspace ID")
    parser.add_argument("--PROJECT_ID", type=str, default="football-players-detection", help="Your project name")
    parser.add_argument("--PROJECT_TYPE", type=str, default="object-detection", help="Project task")
    parser.add_argument("--ANNOTATION", type=str, default="football-players", help="The objects you are annotating")
    parser.add_argument("--DATASET_ROOT", type=str, default="./football-players-detection-18/", help="Local dataset root path")
    parser.add_argument("--DATASET_FORMAT", type=str, default="coco", help="Dataset format structure")
    parser.add_argument("--MODEL_TYPE", type=str, default="rfdetr-base", help="Model architecture")
    parser.add_argument("--WEIGHTS_DIR", type=str, default="./checkpoints/", help="Path containing designated checkpoint file")
    parser.add_argument("--CKPT_NAME", type=str, default="checkpoint.pth", help="Name of checkpoint file in weights directory")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    rf = roboflow.Roboflow(api_key=args.ROBOFLOW_API_KEY)
    workspace = rf.workspace(args.WORKSPACE_ID)

    # Step 1: Create a new project
    project = workspace.create_project(
        project_name=args.PROJECT_ID,
        project_type=args.PROJECT_TYPE,
        project_license="MIT",  # or "private"
        annotation=args.ANNOTATION
    )
    project_id = project.id.split("/")[-1]

    # Step 2: You probably downloaded the dataset locally. Upload it to the project here. This is the equivalent to forking it on the Roboflow interface
    workspace.upload_dataset(
        dataset_path=args.DATASET_ROOT,  # Your downloaded dataset
        project_name=project_id,
        num_workers=10,
        dataset_format=args.DATASET_FORMAT, # coco .json format
        project_type=args.PROJECT_TYPE
    )

    # Step 3: Generate a project version (1 since you initialized it here)
    project = workspace.project(project_id) # Re-fetch project to ensure it has the uploaded data
    version_number = project.generate_version(settings={
        "preprocessing": {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"}
        },
        "augmentation": {}  # No augmentation needed - you already trained
    })
    print("  Waiting for version generation to complete...")
    time.sleep(10)  # Give it a moment to process
    
    # Step 4: Get the version object and upload weights
    print("\nStep 4: Uploading model weights...")
    version = project.version(version_number)
    
    version.deploy(
        model_type=args.MODEL_TYPE,  # or appropriate model type
        model_path=args.WEIGHTS_DIR,
        filename=args.CKPT_NAME
    )
    print("Weights uploaded!")

    # Step 5: Print success messages
    print("\n" + "="*50)
    print("SUCCESS! Your model is deployed.")
    print(f"  Project: {project.id}")
    print(f"  Version: {version_number}")
    print(f"  Model ID: {project.id}/{version_number}")
    print("="*50)


if __name__ == "__main__":
    main()