import os
from pipeline.dummy_model import dummy_predict


def load_scene(scene_path):
    # Check if file exists
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Error: File '{scene_path}' not found.")

    # Check if it is a .tif file
    if not scene_path.lower().endswith(".tif"):
        raise ValueError("Error: Input file must be a .tif file.")

    print(f"Loading scene: {scene_path}")
    return scene_path


def call_model(scene):
    print("Running dummy model...")
    return dummy_predict(scene)


def save_output(result, out_path):
    print(f"Output saved to: {out_path}")


def main():
    scene = load_scene("pipeline/sample_scene.tif")
    result = call_model(scene)
    save_output(result, "output.tif")
    print("Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()