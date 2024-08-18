import argparse
import json
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from datasets import get_test_dataloaders
from networks import GPT2Config, GPT2LMModel


def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(
        layer_past.index_select(1, beam_idx).contiguous().detach()
        for layer_past in past
    )


def _calc_banned_ngram_tokens(
    prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int
) -> None:
    if cur_len + 1 < no_repeat_ngram_size:
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []
            ) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(
    lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty
):
    for i in range(batch_size * num_beams):
        print("prev_output_tokens.shape", prev_output_tokens.shape)
        print("prev_output_tokens[i].shape", prev_output_tokens[i].shape)
        for previous_token in set(prev_output_tokens[i].tolist()):
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def _postprocess_next_token_scores(
    scores,
    history,
    cur_len,
    batch_size,
    num_beams,
    repetition_penalty=1.0,
    no_repeat_ngram_size=4,
    bad_words_ids=None,
    min_length=0,
    max_length=100,
    eos_token_id=None,
):
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(
            scores, batch_size, num_beams, history, repetition_penalty
        )
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")
    if no_repeat_ngram_size > 0 and history is not None:
        num_batch_hypotheses = batch_size * num_beams
        banned_batch_tokens = _calc_banned_ngram_tokens(
            history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")
    return scores


def _add_beam_candidate(
    best_score,
    best_sequence,
    batch_size,
    num_beams,
    beam_scores,
    history,
    eos_token_id=None,
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = (
                beam_scores.view(-1)[_i]
                / cur_len ** config["generation"]["length_penalty"]
            )

            batch_id = _i // num_beams

            if batch_id not in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


def beam(config, device, net, test_dl):
    net.eval()

    all_predictions = {}
    with torch.no_grad():
        for idx, data in enumerate(test_dl):
            data = {key: value for key, value in data.items()}

            _id = data["id"].to(device)
            _query = data["query"].to(device)
            _query_len = data["query_len"].to(device)

            output = None

            batch_size = _id.size(0)
            num_beams = config["generation"]["beam"]

            _batch = torch.arange(0, _id.size(0), device=device, dtype=torch.long)

            past = None
            len_past = None

            _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

            _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)

            # scores for each sentence in the beam
            beam_scores = torch.zeros(
                (batch_size, num_beams), dtype=torch.float, device=_query.device
            )

            best_sequence = torch.zeros(
                (batch_size, config["data"]["eval_len"]),
                dtype=torch.long,
                device=_query.device,
            )
            best_score = {}

            history = None
            with torch.no_grad():
                for i in range(0, config["data"]["eval_len"]):
                    if i == 0:
                        logits, past = net(_query)
                        logits = logits[
                            _bbatch, (_query_len - 1).long(), :
                        ]  # batch_size * beam, vocab
                    else:
                        # print('token_id.shape', token_id.shape, token_id)
                        # print('past.shape', past[0].shape)
                        # print('len_past.shape', len_past.shape, len_past)

                        logits, past = net(token_id, past=past, len_past=len_past)
                        logits = logits[:, -1, :]  # batch_size * beam, vocab

                    logits = _postprocess_next_token_scores(
                        logits,
                        history,
                        i,
                        batch_size,
                        num_beams,
                        repetition_penalty=config["generation"]["repetition_penalty"],
                        no_repeat_ngram_size=config["generation"][
                            "no_repeat_ngram_size"
                        ],
                        min_length=config["data"]["min_length"],
                        eos_token_id=config["generation"]["eos_token_id"],
                    )

                    softmax_probs = F.softmax(logits, dim=-1)
                    ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

                    vocab_size = softmax_probs.shape[-1]

                    _logprob = torch.log(softmax_probs)  # batch_size * beam, vocab
                    if i == 0:
                        next_scores = _logprob.view(batch_size, num_beams, -1)[
                            :, 0, :
                        ]  # batch_size, vocab

                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.view(
                            batch_size, num_beams, -1
                        )
                        next_scores = next_scores.view(
                            batch_size, -1
                        )  # batch_size, beam * vocab

                    next_scores, next_tokens = torch.topk(
                        next_scores, num_beams, dim=1, largest=True, sorted=True
                    )  # batch_size, num_beams

                    beam_id = (next_tokens // vocab_size).view(
                        -1
                    )  # batch_size * num_beams
                    token_id = (
                        (next_tokens % vocab_size).view(-1).unsqueeze(-1)
                    )  # batch_size, num_beams

                    beam_idx = beam_id.view(batch_size, num_beams) + (
                        _batch * num_beams
                    ).unsqueeze(-1)
                    past = _reorder_cache(past, beam_idx.view(-1))
                    beam_scores = next_scores  # batch_size, num_beams
                    len_past = (_query_len + i).long()

                    if history is None:
                        history = token_id.detach()
                    else:
                        history = torch.cat(
                            (history[beam_idx.view(-1)], token_id.detach()), dim=1
                        ).detach()

                    _add_beam_candidate(
                        best_score,
                        best_sequence,
                        batch_size,
                        num_beams,
                        beam_scores,
                        history,
                        eos_token_id=config["generation"]["eos_token_id"],
                    )

                _add_beam_candidate(
                    best_score,
                    best_sequence,
                    batch_size,
                    num_beams,
                    beam_scores,
                    history,
                )

            with torch.no_grad():
                output = best_sequence

            _id = _id.view(-1).cpu()
            output = output.view(-1, output.shape[-1]).cpu()

            for _b in range(0, _id.shape[-1]):
                _i = int(_id[_b].item())
                all_predictions[_i] = {}
                all_predictions[_i]["id"] = _i
                all_predictions[_i]["predict"] = output[_b].tolist()

            if idx % 10 == 0:
                print("inference samples", idx)

    pred_file = os.path.join(
        config["training"]["work_dir"], config["generation"]["output_file"]
    )
    print("Saving inference file", pred_file)
    with open(pred_file, "w") as writer:
        for _i in all_predictions:
            writer.write(json.dumps(all_predictions[_i]) + "\n")


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SplitLoRA Script")
    parser.add_argument(
        "--config", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batches, test_dl = get_test_dataloaders(config)

    model_configuration = GPT2Config(
        n_embd=768,
        n_layer=12,
        n_head=12,
        lora_attn_dim=config["lora"]["lora_dim"],
        lora_attn_alpha=config["lora"]["lora_alpha"],
    )

    net = GPT2LMModel(model_configuration)
    if config["model"]["init_checkpoint"] is not None:
        print(
            "loading model pretrained weight from", config["model"]["init_checkpoint"]
        )
        checkpoint = torch.load(
            config["model"]["init_checkpoint"], map_location=torch.device("cpu")
        )
        net.load_weight(checkpoint)
    net = net.to(device=device)

    print("Starting Model Sampling.")
    beam(config, device, net, test_dl)
