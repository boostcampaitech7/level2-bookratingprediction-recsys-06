import os
from typing import Any

from tqdm import tqdm
import torch
from src.loss import loss as loss_module
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module

METRIC_NAMES = {"RMSELoss": "RMSE", "MSELoss": "MSE", "MAELoss": "MAE"}


def train(cfg, model, dataloader, logger, setting):
    if cfg.wandb.use:
        import wandb

    minimum_loss = None

    loss_fn = getattr(loss_module, cfg.loss)().to(cfg.device)
    cfg.metrics = sorted([metric for metric in set(cfg.metrics) if metric != cfg.loss])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optimizer_module, cfg.optimizer.type)(
        trainable_params, **cfg.optimizer.args
    )

    if cfg.lr_scheduler.use:
        cfg.lr_scheduler.args = {
            k: v
            for k, v in cfg.lr_scheduler.args.items()
            if k
            in getattr(
                scheduler_module, cfg.lr_scheduler.type
            ).__init__.__code__.co_varnames
        }
        lr_scheduler = getattr(scheduler_module, cfg.lr_scheduler.type)(
            optimizer, **cfg.lr_scheduler.args
        )
    else:
        lr_scheduler = None

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss, train_len = 0, len(dataloader["train_dataloader"])

        for data in tqdm(
            dataloader["train_dataloader"],
            desc=f"[Epoch {epoch + 1:02d}/{cfg.train.epochs:02d}]",
        ):
            x, y, y_hat = _train(data, cfg, model)
            loss = loss_fn(y_hat, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if cfg.lr_scheduler.use and cfg.lr_scheduler.type != "ReduceLROnPlateau":
            lr_scheduler.step()

        msg = ""
        train_loss = total_loss / train_len
        msg += f"\tTrain Loss ({METRIC_NAMES[cfg.loss]}): {train_loss:.3f}"
        if cfg.dataset.valid_ratio != 0:  # valid 데이터가 존재할 경우
            valid_loss = valid(cfg, model, dataloader["valid_dataloader"], loss_fn)
            msg += f"\n\tValid Loss ({METRIC_NAMES[cfg.loss]}): {valid_loss:.3f}"
            if cfg.lr_scheduler.use and cfg.lr_scheduler.type == "ReduceLROnPlateau":
                lr_scheduler.step(valid_loss)

            valid_metrics = dict()
            for metric in cfg.metrics:
                metric_fn = getattr(loss_module, metric)().to(cfg.device)
                valid_metric = valid(
                    cfg, model, dataloader["valid_dataloader"], metric_fn
                )
                valid_metrics[f"Valid {METRIC_NAMES[metric]}"] = valid_metric
            for metric, value in valid_metrics.items():
                msg += f" | {metric}: {value:.3f}"
            print(msg)
            logger.log(
                epoch=epoch + 1,
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_metrics=valid_metrics,
            )
            if cfg.wandb.use:
                wandb.log(
                    {
                        f"Train {METRIC_NAMES[cfg.loss]}": train_loss,
                        f"Valid {METRIC_NAMES[cfg.loss]}": valid_loss,
                        **valid_metrics,
                    }
                )
        else:  # valid 데이터가 없을 경우
            print(msg)
            logger.log(epoch=epoch + 1, train_loss=train_loss)
            if cfg.wandb.use:
                wandb.log({f"Train {METRIC_NAMES[cfg.loss]}": train_loss})

        if cfg.train.save_best_model:
            best_loss = valid_loss if cfg.dataset.valid_ratio != 0 else train_loss
            if minimum_loss is None or minimum_loss > best_loss:
                minimum_loss = best_loss
                os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"{cfg.train.ckpt_dir}/{setting.save_time}_{cfg.model}_best.pt",
                )
        else:
            os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{cfg.train.ckpt_dir}/{setting.save_time}_{cfg.model}_e{epoch:02}.pt",
            )

    logger.close()

    return model


def _train(data: Any, args, model):
    if args.model_args[args.model].datatype == "image":
        x, y = [
            data["user_book_vector"].to(args.device),
            data["img_vector"].to(args.device),
        ], data["rating"].to(args.device)
    elif args.model_args[args.model].datatype == "text":
        x, y = [
            data["user_book_vector"].to(args.device),
            data["user_summary_vector"].to(args.device),
            data["book_summary_vector"].to(args.device),
        ], data["rating"].to(args.device)
    else:
        x, y = data[0].to(args.device), data[1].to(args.device)
    return x, y, model(x)


def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0

    for data in dataloader:
        x, y, y_hat = _train(data, args, model)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(args, model, dataloader, setting, checkpoint=None):
    predicts = list()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        if args.train.save_best_model:
            model_path = (
                f"{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt"
            )
        else:
            # best가 아닐 경우 마지막 에폭으로 테스트하도록 함
            model_path = f"{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{args.train.epochs - 1:02d}.pt"
        model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    for data in dataloader["test_dataloader"]:
        if args.model_args[args.model].datatype == "image":
            x = [
                data["user_book_vector"].to(args.device),
                data["img_vector"].to(args.device),
            ]
        elif args.model_args[args.model].datatype == "text":
            x = [
                data["user_book_vector"].to(args.device),
                data["user_summary_vector"].to(args.device),
                data["book_summary_vector"].to(args.device),
            ]
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts
