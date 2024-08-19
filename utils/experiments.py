import json
import os

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def save_checkpoint(config, w_glob_client, model_server, train_step, num_clients):
    model_state_dict = {}

    for key, value in w_glob_client.items():
        new_key = ""
        if key.startswith("client_transformer"):
            new_key = key.replace("client_transformer", "module.transformer")
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    for key, value in model_server.state_dict().items():
        new_key = ""
        if key.startswith("module.server_transformer"):
            new_key = key.replace("module.server_transformer", "module.transformer")
        else:
            model_state_dict[key] = value

        if new_key.startswith("module.transformer.h."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            new_key = ".".join(["module.transformer.h", str(layer_idx + 3)] + parts[4:])
            model_state_dict[new_key] = value
        else:
            model_state_dict[new_key] = value

    model_path = os.path.join(
        config["training"]["work_dir"],
        f'model_sfl.{train_step}_r={config["lora"]["lora_dim"]}_c={config["model"]["split_point"]}_num={num_clients}_block=3.pt',
    )
    print("Saving checkpoint to ", model_path)
    torch.save({"model_state_dict": model_state_dict}, model_path)
