import requests
import json
from typing import List, Dict
import pandas as pd

def predict_templates(dataset: Dict):
    # response = requests.post('http://0.0.0.0:8080/single_inference', json = dataset)
    response = requests.post('http://0.0.0.0:8080/df_link', json = dataset)
    #ipdb.set_trace()
    # print([doc for doc in response.iter_lines()])
    # response = response.json()

    df_json = json.dumps(response.json())
    df = pd.read_json(df_json, orient="records")

    print(df.head())
    print(df.info())

    return df


if __name__ == '__main__':
    # dataset = {"sentence":"Apple computers"}
    dataset = {
                "context_left": "".lower(),
                "mention": "Shakespeare".lower(),
                "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
              }
    entity_linking_df = pd.read_csv("data/entity_linking.csv")

    df_json = entity_linking_df.to_json(orient="records")
    df_json = json.loads(df_json)
    results = predict_templates(df_json)
    # results = predict_templates(dataset)
    print(results)