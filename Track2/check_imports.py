modules = [
    "torch",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "sklearn",
    "PIL",
]


def module_exists(module_name):
    try:
        __import__(module_name)
    except Exception:
        return False
    else:
        return True


if __name__ == "__main__":
    for module in modules:
        print(f"{module}: {module_exists(module)}")
