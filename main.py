import argparse
import copy
import itertools
import math
import os
import time

import loralib as lora
import torch
import wandb

from datasets import get_dataloaders
from networks import ClientGPT2LMModel, GPT2Config, ServerGPT2LMModel
from utils.experiments import AverageMeter, load_config, save_checkpoint
from utils.optimizer import get_optimizer, get_scheduler


def fed_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def optimizer_step(
    _loss,
    server_optimizer,
    server_model,
    client_optimizer,
    _schedule,
    client_hidden_states,
    hidden_states,
    is_update=True,
):
    _loss.backward()

    dfx_client = client_hidden_states.grad.clone().detach()

    if is_update and config["training"]["clip"] > 0:
        torch.nn.utils.clip_grad_norm_(
            server_model.parameters(), config["training"]["clip"]
        )
    server_optimizer.step()
    server_optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()

    hidden_states.backward(dfx_client)
    client_optimizer.step()
    client_optimizer.zero_grad()


def evaluate(device, client_model, server_model, valid_dl):
    device = device
    client_model.eval()
    server_model.eval()
    server_model = server_model.to(device)

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_dl):
            data = {key: value.to(device) for key, value in data.items()}

            _input = data["input"]
            _target = data["target"]
            _msk = data["mask"]

            hidden_states, presents, _ = client_model(_input)
            _, _loss = server_model(
                _input.shape, hidden_states, presents, lm_labels=_target, lm_mask=_msk
            )
            loss = _loss.mean()
            avg_lm_loss.update(loss.item())
            if idx % 100 == 0:
                print("eval samples:", idx, "loss:", loss.float())
        print("Average loss", avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train(
    config,
    device,
    client_model,
    server_model,
    client_models,
    optimizers,
    server_optimizer,
    server_scheduler,
    train_dl_c0,
    train_dl_c1,
    train_dl_c2,
    valid_dl,
    train_step=0,
    epoch=0,
):
    client_model.train()
    server_model.train()

    avg_lm_loss = AverageMeter()
    print("Training Start", epoch)
    log_start_time = time.time()

    best_val_ppl = None

    device = device

    global_client_net = ClientGPT2LMModel(model_configuration)
    global_client_net = global_client_net.to(device)

    global_client_net.load_weight(state_dict)
    lora.mark_only_lora_as_trainable(global_client_net)
    global_client_net.train()
    global_client_weight = global_client_net.state_dict()

    aggregate_step = 100
    w_locals_client = []

    for idx, data in enumerate(zip(train_dl_c0, train_dl_c1, train_dl_c2)):
        for i in range(num_clients):
            client_data = {key: value.to(device) for key, value in data[i].items()}

            _input = client_data["input"]
            _target = client_data["target"]
            _msk = client_data["mask"]

            client_models[i].train()

            _input = _input.to(device)

            hidden_states, presents, w_client = client_models[i](_input)
            train_step += 1

            if (train_step + num_clients) % aggregate_step <= num_clients:
                w_locals_client.append(copy.deepcopy(w_client))

            client_hidden_states = hidden_states.clone().detach().requires_grad_(True)

            _, _lm_loss = server_model(
                _input.shape,
                client_hidden_states,
                presents,
                lm_labels=_target,
                lm_mask=_msk,
                label_smooth=config["model"]["label_smooth"],
            )

            _lm_loss = _lm_loss.mean()

            is_update = train_step % config["training"]["grad_acc"] == 0
            avg_lm_loss.update(_lm_loss.item())

            optimizer_step(
                _lm_loss / config["training"]["grad_acc"],
                server_optimizer,
                server_model,
                optimizers[i],
                server_scheduler,
                client_hidden_states,
                hidden_states,
                is_update=is_update,
            )

            if train_step % aggregate_step == 0:
                temp_dict = {}
                w_locals_client_lora = []
                for w_client in w_locals_client:
                    for key, value in w_client.items():
                        if key.endswith("lora_A"):
                            temp_dict[key] = value
                        if key.endswith("lora_B"):
                            temp_dict[key] = value
                    w_locals_client_lora.append(copy.deepcopy(temp_dict))
                w_glob_client_lora = fed_avg(w_locals_client_lora)

                w_glob_client_lora_new = {}
                for key, value in w_glob_client_lora.items():
                    new_key = "client_transformer." + key
                    w_glob_client_lora_new[new_key] = value
                for key, _ in global_client_weight.items():
                    if key.endswith("lora_A"):
                        global_client_weight[key] = w_glob_client_lora_new[key]
                    if key.endswith("lora_B"):
                        global_client_weight[key] = w_glob_client_lora_new[key]

                global_client_net.load_state_dict(global_client_weight)
                for client_model_ in client_models:
                    client_model_.load_state_dict(global_client_weight)

                w_locals_client = []

            if train_step % config["training"]["log_interval"] == 0:
                elapsed = time.time() - log_start_time
                lr = server_optimizer.param_groups[0]["lr"]
                log_str = (
                    f"| epoch {epoch:3d} step {train_step:>8d} | {idx*3 + 1:>6d} batches | "
                    f"lr {lr:.3g} | ms/batch {elapsed * 1000 / config['training']['log_interval']:5.2f} | "
                    f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                    f"ppl {math.exp(avg_lm_loss.avg):5.2f}"
                )
                print(log_str)
                if config["wandb"]["logging"]:
                    wandb.log(
                        {
                            "train/step/train_loss": avg_lm_loss.val,
                            "train/step/lr": lr,
                            "train/step/ppl": math.exp(avg_lm_loss.val),
                        },
                        step=train_step,
                    )
                log_start_time = time.time()
                avg_lm_loss.reset()

            # save checkpoint at each save_interval
            if train_step % config["training"]["save_interval"] == 0:
                save_checkpoint(
                    config, global_client_weight, server_model, train_step, num_clients
                )

            if train_step % config["training"]["eval_interval"] == 0:
                eval_start_time = time.time()
                valid_loss, valid_ppl = evaluate(
                    device, global_client_net, server_model, valid_dl
                )
                if best_val_ppl is None or valid_ppl < best_val_ppl:
                    best_val_ppl = valid_ppl

                log_str = (
                    f"| Eval {train_step //  config['training']['eval_interval']:3d} at step {train_step:>8d} | "
                    f"time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | "
                    f"valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} "
                )

                print("=" * 100)
                print(log_str)
                print("=" * 100)
                if config["wandb"]["logging"]:
                    wandb.log(
                        {"valid/loss": valid_loss, "valid/ppl": math.exp(valid_ppl)}
                    )

                global_client_net.train()
                server_model.train()

            if train_step == config["scheduler"]["max_step"]:
                break

    if train_step == config["scheduler"]["max_step"]:
        save_checkpoint(
            config, global_client_weight, server_model, train_step, num_clients
        )

    return train_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SplitLoRA Script")
    parser.add_argument(
        "--config", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config["wandb"]["logging"]:
        wandb.init(project="splitlora-experiments", name=config["wandb"]["run_name"])

    torch.manual_seed(config["distributed"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config["training"]["work_dir"]):
        os.makedirs(config["training"]["work_dir"])

    num_batches, train_dl_c0, train_dl_c1, train_dl_c2, valid_dl = get_dataloaders(
        config=config
    )

    model_configuration = GPT2Config(
        n_embd=768,
        n_layer=12,
        n_head=12,
        lora_attn_dim=config["lora"]["lora_dim"],
        lora_attn_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        split_point=config["model"]["split_point"],
    )
    gpt_client = ClientGPT2LMModel(model_configuration)
    gpt_server = ServerGPT2LMModel(model_configuration)

    state_dict = torch.load(config["model"]["init_checkpoint"])
    if config["model"]["init_checkpoint"] is not None:
        print("Loading pre-trained weight from", config["model"]["init_checkpoint"])
        gpt_client.load_weight(state_dict)
        gpt_server.load_weight(state_dict)

    gpt_client = gpt_client.to(device=device)
    gpt_server = gpt_server.to(device=device)

    lora.mark_only_lora_as_trainable(gpt_client)
    lora.mark_only_lora_as_trainable(gpt_server)

    client_optimizer = get_optimizer(gpt_client, config)
    server_optimizer = get_optimizer(gpt_server, config)

    # nums of clients:
    num_clients = 3
    client_models = []
    optimizers = []

    # Create client models for different clients
    for _ in range(num_clients):
        client_model = ClientGPT2LMModel(model_configuration)
        client_model.load_weight(state_dict)
        client_model = client_model.to(device=device)
        lora.mark_only_lora_as_trainable(client_model)
        optimizer = get_optimizer(client_model, config)
        client_models.append(client_model)
        optimizers.append(optimizer)

    if config["scheduler"]["max_step"] is None:
        config["scheduler"]["max_step"] = (
            config["scheduler"]["max_epoch"] * num_batches * 3
        )
        print("set max_step:", config["scheduler"]["max_step"])

    client_scheduler = get_scheduler(client_optimizer, config)
    server_scheduler = get_scheduler(server_optimizer, config)

    train_step = 0
    for epoch in itertools.count(start=1):
        train_step = train(
            config=config,
            device=device,
            client_model=gpt_client,
            server_model=gpt_server,
            client_models=client_models,
            optimizers=optimizers,
            server_optimizer=server_optimizer,
            server_scheduler=server_scheduler,
            train_dl_c0=train_dl_c0,
            train_dl_c1=train_dl_c1,
            train_dl_c2=train_dl_c2,
            valid_dl=valid_dl,
            train_step=train_step,
            epoch=epoch,
        )

        if train_step >= config["scheduler"]["max_step"] or (
            config["scheduler"]["max_epoch"] is not None
            and epoch >= config["scheduler"]["max_epoch"]
        ):
            print("End of the training session.")
            break
