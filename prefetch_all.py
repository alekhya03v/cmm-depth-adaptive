import os, sys
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from huggingface_hub import snapshot_download, HfApi

def fetch(repo):
    files = HfApi().list_repo_files(repo)
    w = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
    snapshot_download(repo, allow_patterns=["*.json", "*.txt"] + w)

model = sys.argv[1] if len(sys.argv) > 1 else "clip-vit-base-patch32"
models_only = "--models-only" in sys.argv
with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="fabric_model_fusion",
                  overrides=[f"modelpool=CLIPVisionModelPool/{model}_TALL20_model_only",
                             "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20",
                             f"taskpool.base_model=openai/{model}"])

names = list(cfg.modelpool.models.keys())
for i, name in enumerate(names, 1):
    path = cfg.modelpool.models[name]
    path = path if isinstance(path, str) else path.get("pretrained_model_name_or_path")
    fetch(path)
    print(f"[{i}/{len(names)}] MODEL OK {name}: {path}", flush=True)
fetch(f"openai/{model}")
print(f"MODEL OK text encoder: openai/{model}", flush=True)

if not models_only:
    ds_names = list(cfg.taskpool.test_datasets.keys())
    for i, name in enumerate(ds_names, 1):
        ds = instantiate(cfg.taskpool.test_datasets[name])
        print(f"[{i}/{len(ds_names)}] DATA OK {name}: {len(ds)} test images", flush=True)
print(f"ALL DOWNLOADS DONE for {model}")
