from dummy_model import dummy_predict


def load_scene(path: str):
    print(f"Loading scene: {path}")
    return path


def call_model(scene):
    return dummy_predict(scene)


def save_output(result, out_path: str):
    print(f"Output saved to: {out_path}")


def main():
    scene = load_scene("sample_scene.tif")
    result = call_model(scene)
    save_output(result, "output.tif")
    print("Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()