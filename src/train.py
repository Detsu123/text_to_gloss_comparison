import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data.dataset import loader, load_vocab
from .data.vocabulary import PAD
from .models.seq2seq import build_model
from .metrics import score
from .utils import (
    setup,
    load_config,
    dump,
    fingerprint,
    environment,
)


@torch.no_grad()
def validate(
    model,
    batches,
    device,
    tgt_vocab,
    max_len,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    hypotheses = []
    references = []

    for x, y, rows in batches:
        x = x.to(device)
        y = y.to(device)

        decoder_input = y[:, :-1]
        labels = y[:, 1:]

        logits = model(
            x,
            decoder_input,
        )

        loss_sum = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            labels.reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        )

        total_loss += loss_sum.item()

        total_tokens += (
            labels.ne(PAD)
            .sum()
            .item()
        )

        generated = model.generate(
            x,
            max_len,
        )

        hypotheses.extend(
            tgt_vocab.decode(ids)
            for ids
            in generated.tolist()
        )

        references.extend(
            row["gloss"]
            for row in rows
        )

    metrics, _ = score(
        hypotheses,
        references,
    )

    return (
        total_loss
        / max(total_tokens, 1),
        metrics,
    )


def build_optimizer(
    kind,
    model,
    cfg,
):
    if kind == "transformer":
        lr = cfg[
            "transformer_learning_rate"
        ]
    else:
        lr = cfg[
            "learning_rate"
        ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg[
            "weight_decay"
        ],
    )

    return optimizer


def build_scheduler(
    kind,
    optimizer,
    train_loader,
    cfg,
):
    if kind != "transformer":
        return None

    updates_per_epoch = math.ceil(
        len(train_loader)
        / cfg["grad_accumulation"]
    )

    total_steps = (
        updates_per_epoch
        * cfg["max_epochs"]
    )

    warmup_steps = min(
        cfg[
            "transformer_warmup_steps"
        ],
        max(
            1,
            total_steps // 3,
        ),
    )

    min_ratio = cfg[
        "transformer_min_lr_ratio"
    ]

    def schedule(step):
        if step < warmup_steps:
            return float(
                step + 1
            ) / float(
                max(
                    1,
                    warmup_steps,
                )
            )

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps
            - warmup_steps,
        )

        return max(
            min_ratio,
            1.0 - progress,
        )

    return (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            schedule,
        )
    )


def train(
    kind,
    cfg,
):
    device = setup(cfg)

    out = (
        Path(cfg["output"])
        / f"{kind}_seed{cfg['seed']}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (out / "best.pt").exists():
        raise FileExistsError(
            f"{out} already contains "
            "a trained experiment."
        )

    # =========================================
    # Dataset / vocabulary
    # =========================================

    src_vocab, tgt_vocab = load_vocab(
        cfg["processed"]
    )

    train_loader = loader(
        cfg["processed"],
        "train",
        cfg,
        True,
    )

    dev_loader = loader(
        cfg["processed"],
        "dev",
        cfg,
        False,
    )

    # =========================================
    # Model
    # =========================================

    model = build_model(
        kind,
        len(src_vocab),
        len(tgt_vocab),
        cfg,
    ).to(device)

    optimizer = build_optimizer(
        kind,
        model,
        cfg,
    )

    scheduler = build_scheduler(
        kind,
        optimizer,
        train_loader,
        cfg,
    )

    use_amp = bool(
        cfg["amp"]
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    config = {
        **cfg,
        "model": kind,
        "actual_device": str(device),
        "actual_amp": use_amp,
    }

    dump(
        out / "config.json",
        config,
    )

    dump(
        out / "environment.json",
        environment(device),
    )

    identity = fingerprint(
        cfg["processed"]
    )

    prep = json.loads(
        (
            Path(cfg["processed"])
            / "preprocessing.json"
        ).read_text()
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    # =========================================
    # Tracking
    # =========================================

    best_bleu = -float("inf")

    best_dev_loss = float(
        "inf"
    )

    bad_bleu = 0
    bad_loss = 0

    best_epoch = 0

    logs = []

    start_time = (
        time.perf_counter()
    )

    if kind == "transformer":
        patience = cfg[
            "transformer_patience"
        ]

        min_epochs = cfg[
            "transformer_min_epochs"
        ]
    else:
        patience = cfg[
            "patience"
        ]

        min_epochs = 1

    # =========================================
    # Epoch loop
    # =========================================

    for epoch in range(
        1,
        cfg["max_epochs"] + 1,
    ):
        epoch_start = (
            time.perf_counter()
        )

        model.train()

        total_loss = 0.0
        total_tokens = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        window_tokens = 0

        for step, (
            x,
            y,
            _,
        ) in enumerate(
            train_loader,
            1,
        ):
            x = x.to(device)
            y = y.to(device)

            labels = y[:, 1:]

            n_tokens = (
                labels.ne(PAD)
                .sum()
                .item()
            )

            decoder_input = (
                y[:, :-1]
            )

            if kind == "transformer":
                label_smoothing = cfg[
                    "transformer_label_smoothing"
                ]
            else:
                label_smoothing = 0.0

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(
                    x,
                    decoder_input,
                    teacher_forcing=
                        cfg[
                            "teacher_forcing"
                        ],
                )

                loss_sum = (
                    F.cross_entropy(
                        logits.reshape(
                            -1,
                            logits.size(-1),
                        ),
                        labels.reshape(-1),
                        ignore_index=PAD,
                        reduction="sum",
                        label_smoothing=
                            label_smoothing,
                    )
                )

            if not torch.isfinite(
                loss_sum
            ):
                raise (
                    FloatingPointError(
                        "Non-finite loss "
                        "detected."
                    )
                )

            scaler.scale(
                loss_sum
            ).backward()

            total_loss += (
                loss_sum.item()
            )

            total_tokens += (
                n_tokens
            )

            window_tokens += (
                n_tokens
            )

            should_step = (
                step
                % cfg[
                    "grad_accumulation"
                ]
                == 0
                or step
                == len(
                    train_loader
                )
            )

            if should_step:
                scaler.unscale_(
                    optimizer
                )

                for param in (
                    model.parameters()
                ):
                    if (
                        param.grad
                        is not None
                    ):
                        param.grad.div_(
                            max(
                                1,
                                window_tokens,
                            )
                        )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg[
                        "clip_grad"
                    ],
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                window_tokens = 0

        train_seconds = (
            time.perf_counter()
            - epoch_start
        )

        # =====================================
        # Validation
        # =====================================

        dev_loss, metrics = (
            validate(
                model,
                dev_loader,
                device,
                tgt_vocab,
                cfg[
                    "max_decode_len"
                ],
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        train_loss = (
            total_loss
            / max(
                total_tokens,
                1,
            )
        )

        row = {
            "epoch": epoch,
            "train_loss":
                train_loss,
            "dev_loss":
                dev_loss,
            "dev_bleu4":
                metrics["bleu4"],
            "dev_chrf":
                metrics["chrf"],
            "learning_rate":
                current_lr,
            "train_seconds":
                train_seconds,
            "epoch_seconds":
                time.perf_counter()
                - epoch_start,
        }

        logs.append(row)

        with (
            out / "history.csv"
        ).open(
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=
                    list(row.keys()),
            )

            writer.writeheader()
            writer.writerows(
                logs
            )

        # =====================================
        # Best checkpoint
        # =====================================

        bleu_improved = (
            metrics["bleu4"]
            > best_bleu
        )

        if bleu_improved:
            best_bleu = (
                metrics[
                    "bleu4"
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "state_dict":
                        model.state_dict(),
                    "config":
                        config,
                    "src_vocab":
                        src_vocab.tokens,
                    "tgt_vocab":
                        tgt_vocab.tokens,
                    "train_dev_fingerprint":
                        identity,
                    "dataset":
                        prep["dataset"],
                    "best_epoch":
                        epoch,
                    "dev_bleu4":
                        best_bleu,
                    "dev_loss":
                        dev_loss,
                },
                out / "best.pt",
            )

        # =====================================
        # Early stopping logic
        # =====================================

        if kind == "transformer":
            if (
                dev_loss
                < best_dev_loss
                - cfg[
                    "min_delta"
                ]
            ):
                best_dev_loss = (
                    dev_loss
                )

                bad_loss = 0

            else:
                bad_loss += 1

        else:
            if bleu_improved:
                bad_bleu = 0
            else:
                bad_bleu += 1

        print(
            f"{kind} "
            f"epoch={epoch} "
            f"train="
            f"{train_loss:.4f} "
            f"dev="
            f"{dev_loss:.4f} "
            f"BLEU4="
            f"{metrics['bleu4']:.3f} "
            f"lr="
            f"{current_lr:.6g} "
            f"seconds="
            f"{row['epoch_seconds']:.1f}",
            flush=True,
        )

        # =====================================
        # Stop
        # =====================================

        if kind == "transformer":
            if (
                epoch >= min_epochs
                and bad_loss
                >= patience
            ):
                print(
                    "Transformer "
                    "early stopping: "
                    "validation loss "
                    "stopped improving.",
                    flush=True,
                )
                break

        else:
            if (
                epoch >= min_epochs
                and bad_bleu
                >= patience
            ):
                print(
                    f"{kind} "
                    "early stopping: "
                    "BLEU-4 stopped "
                    "improving.",
                    flush=True,
                )
                break

    # =========================================
    # Summary
    # =========================================

    summary = {
        "model":
            kind,

        "seed":
            cfg["seed"],

        "dataset":
            prep["dataset"],

        "experiment":
            cfg["experiment"],

        "parameters":
            sum(
                p.numel()
                for p
                in model.parameters()
            ),

        "best_epoch":
            best_epoch,

        "epochs_run":
            len(logs),

        "best_dev_bleu4":
            best_bleu,

        "training_wall_seconds":
            time.perf_counter()
            - start_time,

        "train_only_seconds":
            sum(
                row[
                    "train_seconds"
                ]
                for row in logs
            ),

        "peak_gpu_memory_mib":
            (
                torch.cuda
                .max_memory_allocated(
                    device
                )
                / 2**20
                if device.type
                == "cuda"
                else None
            ),

        "device":
            str(device),

        "train_dev_fingerprint":
            identity,
    }

    dump(
        out / "summary.json",
        summary,
    )

    del model
    del optimizer

    if scheduler is not None:
        del scheduler

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "rnn",
            "gru",
            "lstm",
            "transformer",
        ],
    )

    parser.add_argument(
        "--config",
        default=
            "config/default.yaml",
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    if args.seed is not None:
        cfg["seed"] = (
            args.seed
        )

    train(
        args.model,
        cfg,
    )