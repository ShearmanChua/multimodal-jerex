import requests
import json
from typing import List, Dict
import pandas as pd

def predict_templates(dataset: Dict):
    # response = requests.post('http://0.0.0.0:8080/single_inference', json = dataset)
    
    #ipdb.set_trace()
    # print([doc for doc in response.iter_lines()])
    # response = response.json()

    response = requests.post('http://0.0.0.0:8080/df_link', json = dataset)
    df_json = json.dumps(response.json())
    df = pd.read_json(df_json, orient="records")

    print(df.head())
    print(df.info())

    return response


if __name__ == '__main__':
    # dataset = {"sentence":"Apple computers"}
    # dataset = {
    #             "context_left": "Who manufactured the".lower(),
    #             "mention": "Oerlikon cannon",
    #             "context_right": "".lower(),
    #           }
    dataset = {
            "text": '''Kimi Raikkonen said on Thursday he was looking forward to retirement after 19 seasons and 349 races in Formula One and would be less emotional about it than his wife Minttu.
            Ferrari's 2007 world champion, now driving for Alfa Romeo, has one more race in Abu Dhabi on Sunday before he calls it quits at the age of 42.
            "I'm looking forward to get the season done," the Finnish 'Iceman', who made his Formula One debut in 2001, told reporters at Yas Marina.
            "It's nice that it comes to an end and I'm looking forward to the normal life after.
            "I think for sure my wife will be more emotional about it," added the poker-faced winner of 21 races.
            "I doubt that the kids will really care either way, I think they will find other things to do that are more interesting. They like coming to a warm country and be in a pool and other things but it’s nice to have them here."
            Raikkonen announced in September he would be retiring at the end of the season, with the Swiss-based team signing compatriot Valtteri Bottas as his replacement from Mercedes.
            Alfa Romeo will be marking his final race with a special livery tribute on the side of his car declaring "Dear Kimi, we will leave you alone now".
            Raikkonen famously uttered the phrase "Just leave me alone, I know what I'm doing" over the radio while heading to victory in Abu Dhabi with Lotus in 2012, a comment that spawned a range of merchandise and social media memes.
            Ever popular with the fans, the Finn said he was looking forward to life without a rigid schedule.
            "Right now I’m not looking at anything apart from finishing the year," he said.
            "We’ll see if there’s some interesting things that comes out, if it makes sense maybe I’ll do it, but I have zero plans right now."'''
        }
    entity_linking_df = pd.read_csv("data/articles.csv")

    df_json = entity_linking_df.to_json(orient="records")
    df_json = json.loads(df_json)
    results = predict_templates(df_json)
    # results = predict_templates(dataset)
    print(results)