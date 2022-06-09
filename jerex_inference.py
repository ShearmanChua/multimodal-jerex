import json
from operator import sub
import pandas as pd

import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TestConfig
from jerex import model, util

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)


@hydra.main(config_name='test', config_path='configs/docred_joint')
def inference(cfg: TestConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.dataset, 'test_path')
    util.config_to_abs_paths(cfg.dataset, 'types_path')
    util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    model.test_on_df(cfg)
    # results_df = model.test_on_fly(cfg)


    # if results_df is not None:
    #     nodes_df, relations_df, triples_df = generate_neo4j_dfs(cfg,results_df)
    #     idx_to_node = dict(zip(nodes_df.node_id, nodes_df.node_name))
    #     idx_to_relation = dict(zip(relations_df.relation_id, relations_df.relation))

    #     for idx, (subject, relation, object) in triples_df.iterrows():
    #         print(idx_to_node[subject],idx_to_relation[relation],idx_to_node[object])


def generate_neo4j_dfs(cfg: TestConfig, results_df):
    json_path = cfg.dataset.types_path

    with open(json_path, 'r') as f:
        types_dict = json.load(f)

    relations_types = types_dict['relations']

    relations_df = pd.DataFrame(columns=['relation','relation_id'])
    relation_to_idx = {}

    for relation,relation_dict in relations_types.items():
        relations_df.loc[-1] = [relation_dict['verbose'],relation.strip()] # adding a row
        relations_df.index = relations_df.index + 1  # shifting index
        relations_df = relations_df.sort_index()  # sorting by index
        relation_to_idx[relation_dict['verbose']] = relation.strip()

    # nodes_df = pd.DataFrame(columns=['name','entity_type'])
    triples_df = pd.DataFrame(columns=['subject','relation','object'])

    head_nodes_df = results_df[['head', 'head_type']]
    head_nodes_df = head_nodes_df.drop_duplicates()
    head_nodes_df.columns = ['node_name', 'entity_type']
    tail_nodes_df = results_df[['tail', 'tail_type']]
    tail_nodes_df = tail_nodes_df.drop_duplicates()
    tail_nodes_df.columns = ['node_name', 'entity_type']
    nodes_df = pd.concat([head_nodes_df, tail_nodes_df]).reset_index(drop=True)

    nodes_df['node_id'] = nodes_df.index
    node_to_idx = dict(zip(nodes_df.node_name, nodes_df.node_id))

    for idx, (subject, subject_type, object, object_type,relation) in results_df.iterrows():
        triples_df.loc[-1] = [node_to_idx[subject],relation_to_idx[str(relation)],node_to_idx[object]] # adding a row
        triples_df.index = triples_df.index + 1  # shifting index
        triples_df = triples_df.sort_index()

    print(nodes_df.head())
    print(triples_df.head())
    print(relations_df.head())

    return nodes_df, relations_df, triples_df
    
if __name__ == '__main__':

    # try:
    #     torch.multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    inference()
    