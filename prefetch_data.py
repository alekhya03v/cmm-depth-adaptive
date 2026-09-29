from datasets import load_dataset
specs = {
    "sun397": dict(path="tanganke/sun397"),
    "stanford-cars": dict(path="tanganke/stanford_cars"),
    "resisc45": dict(path="tanganke/resisc45"),
    "eurosat": dict(path="tanganke/eurosat"),
    "svhn": dict(path="svhn", name="cropped_digits"),
    "gtsrb": dict(path="tanganke/gtsrb"),
    "mnist": dict(path="mnist"),
    "dtd": dict(path="tanganke/dtd"),
}
for k, s in specs.items():
    ds = load_dataset(**s, split="test")
    print(f"OK {k}: {len(ds)} test images", flush=True)
