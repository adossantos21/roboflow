
from roboflow import Roboflow
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create a new project from your locally trained model")
    parser.add_argument("--ROBOFLOW_API_KEY", type=str, default="insert_api_key", help="Roboflow API Key for account recognition")
    parser.add_argument("--WORKSPACE_ID", type=str, default="roboflow-jvuqo", help="Roboflow Workspace ID for the dataset you are downloading")
    parser.add_argument("--PROJECT_ID", type=str, default="football-players-detection-3zvbc", help="Your project name")
    parser.add_argument("--VERSION_NUMBER", type=int, default=18, help="Dataset version number")
    parser.add_argument("--DATASET_FORMAT", type=str, default="coco", help="Dataset format structure")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    rf = Roboflow(api_key=args.ROBOFLOW_API_KEY)
    project = rf.workspace(args.WORKSPACE_ID).project(args.PROJECT_ID)
    version = project.version(args.VERSION_NUMBER)
    dataset = version.download(args.DATASET_FORMAT)
    print(dataset.location)

if __name__ == "__main__":
    main()