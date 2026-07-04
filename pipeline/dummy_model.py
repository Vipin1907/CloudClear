import numpy as np

def dummy_predict(input_scene_path: str):
    """
    Dummy model for testing.
    It returns a fake image (100x100x3 array).
    """

    print(f"Processing: {input_scene_path}")

    fake_output = np.zeros((100, 100, 3), dtype=np.uint8)

    return fake_output