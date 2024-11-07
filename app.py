from omegaconf import OmegaConf, DictConfig, ListConfig
import pandas as pd
import torch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module

from src.data.interface import DataInterface
from src.utils import Logger, Setting
import src.data as data_module
from src.train import train, predict
import src.models as model_module


def run_model(cfg: DictConfig | ListConfig):
    Setting.seed_everything(cfg.seed)

    ######################## LOAD DATA
    datatype = cfg.model_args[cfg.model].datatype
    data_class: DataInterface = getattr(data_module, f"{datatype}Data")(cfg)

    print(f"--------------- {cfg.model} Load Data ---------------")
    data_class.data_load()

    print(f"--------------- {cfg.model} Train/Valid Split ---------------")
    data_class.data_split()
    data_class.data_loader()
    data = data_class.get_data()
    ####################### Setting for Log
    setting = Setting()

    if not cfg.predict:
        log_path = setting.get_log_path(cfg)
        logger = Logger(cfg, log_path)
        logger.save_args()

    ######################## Model
    print(f"--------------- INIT {cfg.model} ---------------")
    # models > __init__.py 에 저장된 모델만 사용 가능
    # model = FM(args.model_args.FM, data).to('cuda')와 동일한 코드
    model = getattr(model_module, cfg.model)(cfg.model_args[cfg.model], data).to(
        cfg.device
    )

    # 만일 기존의 모델을 불러와서 학습을 시작하려면 resume을 true로 설정하고 resume_path에 모델을 지정하면 됨
    if cfg.train.resume:
        model.load_state_dict(torch.load(cfg.train.resume_path, weights_only=True))

    ######################## TRAIN
    if not cfg.predict:
        print(f"--------------- {cfg.model} TRAINING ---------------")
        model = train(cfg, model, data, logger, setting)

    ######################## INFERENCE
    if not cfg.predict:
        print(f"--------------- {cfg.model} PREDICT ---------------")
        predicts = predict(cfg, model, data, setting)
    else:
        print(f"--------------- {cfg.model} PREDICT ---------------")
        predicts = predict(cfg, model, data, setting, cfg.checkpoint)

    ######################## SAVE PREDICT
    print(f"--------------- SAVE {cfg.model} PREDICT ---------------")
    submission = pd.read_csv(cfg.dataset.data_path + "sample_submission.csv")
    submission["rating"] = predicts

    filename = setting.get_submit_filename(cfg)
    print(f"Save Predict: {filename}")
    submission.to_csv(filename, index=False)


def load_config():
    cfg = OmegaConf.load("config.yaml")
    # 사용되지 않는 정보 삭제 (학습 시에만)
    if not cfg.predict:
        del cfg.checkpoint

        if not cfg.wandb.use:
            del cfg.wandb.project, cfg.wandb.run_name

        cfg.model_args = OmegaConf.create({cfg.model: cfg.model_args[cfg.model]})

        cfg.optimizer.args = {
            k: v
            for k, v in cfg.optimizer.args.items()
            if k
            in getattr(
                optimizer_module, cfg.optimizer.type
            ).__init__.__code__.co_varnames
        }

        if not cfg.lr_scheduler.use:
            del cfg.lr_scheduler.type, cfg.lr_scheduler.args
        else:
            cfg.lr_scheduler.args = {
                k: v
                for k, v in cfg.lr_scheduler.args.items()
                if k
                in getattr(
                    scheduler_module, cfg.lr_scheduler.type
                ).__init__.__code__.co_varnames
            }

        if not cfg.train.resume:
            del cfg.train.resume_path

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(cfg))
    return cfg


def run_app():
    cfg = load_config()

    if cfg.wandb.use:
        import wandb

        w_cfg = cfg.wandb
        wandb.login(key=cfg.wandb.api_key)
        # wandb.require("core")
        # https://docs.wandb.ai/ref/python/init 참고
        wandb.init(
            project=w_cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=w_cfg.run_name,
            notes=w_cfg.memo,
            tags=[cfg.model],
            resume="allow",
        )
        cfg.run_href = wandb.run.get_url()

        wandb.run.log_code(
            "./src"
        )  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능
    run_model(cfg)

    if cfg.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    run_app()
