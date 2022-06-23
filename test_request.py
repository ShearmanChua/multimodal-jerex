import requests
import json
from typing import List, Dict
import pandas as pd

def predict_templates(dataset: Dict):
    response = requests.post('http://0.0.0.0:8080/single_inference', json = dataset)
    
    #ipdb.set_trace()
    print([doc for doc in response.iter_lines()])
    response = response.json()

    # response = requests.post('http://0.0.0.0:8080/df_link', json = dataset)
    # df_json = json.dumps(response.json())
    # df = pd.read_json(df_json, orient="records")

    # print(df.head())
    # print(df.info())

    return response


if __name__ == '__main__':
    # dataset = {"sentence":"Apple computers"}
    # dataset = {
    #             "context_left": "Who manufactured the".lower(),
    #             "mention": "Oerlikon cannon",
    #             "context_right": "".lower(),
    #           }
    dataset = {
        "text": "Red Bull's Max Verstappen took his first Formula One title, denying rival Lewis Hamilton a record eighth, with a last-lap overtake to win a season-ending Abu Dhabi Grand Prix that started and ended amid high drama and controversy on Sunday (Dec 12). Hamilton's Mercedes team, who won the constructors' championship for an unprecedented eighth successive year but had their run of double dominance ended, protested about the result within half-an-hour of the chequered flag.It's insane,said Verstappen, Formula One's first Dutch world champion, of a race that started with the 24-year-old on pole position and level on points with Hamilton and ended in uproar. Verstappen's hopes had sunk and risen as the stewards first refused to intervene and then decisively pushed the boundaries."
    }
    entity_linking_df = pd.read_csv("data/entity_linking.csv")

    df_json = entity_linking_df.to_json(orient="records")
    df_json = json.loads(df_json)
    # results = predict_templates(df_json)
    results = predict_templates(dataset)
    print(results)