#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib


def _get_cksum(data: torch.Tensor):
    byte_data = data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(byte_data)

    return hasher.hexdigest()


class MegatronCollector:
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))

    def __init__(self):
        raise RuntimeError("Use initialize to init Megatron Core Collector")

    @classmethod
    def initialize(cls):
        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/var/log")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        assert hasattr(cls, "ranks_info_"), "the rank information must be set"

        db_path = os.path.join(
            root_dir,
            "Collector/coredump_dp{}_tp{}_pp{}_ep{}.db".format(
                cls.ranks_info_["dp"],
                cls.ranks_info_["tp"],
                cls.ranks_info_["pp"],
                cls.ranks_info_["ep"],
            ),
        )
        cls.db_ = duckdb.connect(db_path)

        cls.db_.execute(
            """CREATE TABLE IF NOT EXISTS coredump(
                  step INTEGER,
                  stage TEXT,
                  data JSON);"""
        )

    @classmethod
    def set_process_group_info(cls, ranks_info):
        cls.ranks_info_ = ranks_info

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        cls.initialize()

    @classmethod
    def should_dump(cls):
        return cls.step_ <= cls.dump_step_

    @classmethod
    def dump_main_grad(
        cls, param, param_name: str, stage_name: str = "main-grad-in-bwd"
    ):
        if not cls.should_dump():
            return

        param_info = {
            "name": param_name,
            "cksum": _get_cksum(param.main_grad),
            "mean": float(param.main_grad.mean().item()),
            "std": float(param.main_grad.std().item()),
            "shape": list(param.main_grad.shape),
            "type": str(param.main_grad.type()),
        }
        param_info.update(cls.ranks_info_)

        try:
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, json.dumps(param_info)),
            )
        except Exception as e:
            print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_main_param(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                main_param_exist = (
                    hasattr(param, "main_param") and param.main_param is not None
                )

                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param.main_param) if main_param_exist else None,
                    "mean": float(param.main_param.mean().item())
                    if main_param_exist
                    else None,
                    "std": float(param.main_param.std().item())
                    if main_param_exist
                    else None,
                    "shape": list(param.main_param.shape) if main_param_exist else None,
                    "type": str(param.main_param.type())
                    if hasattr(param, "main_param") and param.main_param is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, json.dumps(param_info)),
                    )
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_model(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param),
                    "mean": float(param.mean().item()),
                    "std": float(param.std().item()),
                    "shape": list(param.shape),
                    "type": str(param.type()),
                    "requires_grad": param.requires_grad,
                    "grad_cksum": _get_cksum(param.grad)
                    if param.grad is not None
                    else None,
                    "grad_shape": list(param.grad.shape)
                    if param.grad is not None
                    else None,
                    "grad_type": str(param.grad.type())
                    if param.grad is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, json.dumps(param_info)),
                    )
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")

    @classmethod
    def step(cls):
        cls.step_ += 1
