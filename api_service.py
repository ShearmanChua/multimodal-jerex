import json
from operator import sub
import pandas as pd
import ast
import os
import requests

from fastapi import FastAPI, Request

import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TestConfig
from jerex import model, util

app = FastAPI()

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@hydra.main(config_name='test', config_path='configs/docred_joint')
def load_configs(cfg: TestConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    global configs

    util.config_to_abs_paths(cfg.dataset, 'test_path')
    util.config_to_abs_paths(cfg.dataset, 'save_path')
    util.config_to_abs_paths(cfg.dataset, 'csv_path')
    util.config_to_abs_paths(cfg.dataset, 'types_path')
    util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    configs = cfg

def inference(cfg: TestConfig, docs):
    results_df = model.api_call_single(cfg)
    df_json = results_df.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json

@app.post("/single_inference")
async def link(request: Request):
    dict_str = await request.json()
    json_dict = dict_str

    data = json_dict['text'].split("\n")
    data = [text for text in data if len(text) > 5]

    json_string = inference(configs, data)

    json_string = json.dumps(json_string)

    return json_string

if __name__ == '__main__':

    load_configs()
