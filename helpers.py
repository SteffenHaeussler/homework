from typing import Dict, List

import json

import pandas as pd
import numpy as np


def load_json(filename: str) -> List[Dict]:

    with open(filename, 'r', encoding='utf-8') as input:
        data = json.load(input)

    return data


def get_word_data(data: List[Dict]) -> pd.DataFrame:

    words_data = pd.json_normalize(data=data, record_path=['words'], meta=['id'])
    words_data['indices'] = words_data.groupby('id').cumcount()
    words_data["region.page"] = words_data["region.page"].astype(np.int64)

    return words_data


def get_entities_data(data: List[Dict]) -> pd.DataFrame:

    entities_data = pd.json_normalize(data=data, record_path=['entities'], meta=['id'])
    entities_data['n_idx_item'] = entities_data.groupby('id').cumcount()
    entities_data = entities_data.explode("indices")

    return entities_data


def merge_words_entities(words_data: pd.DataFrame, entities_data: pd.DataFrame) -> pd.DataFrame:
    # merge the words and entities based on the id, indices and page
    data = pd.merge(words_data, entities_data,
                    left_on=["id", "region.page", "indices"],
                    right_on=["id", "metaData.region.page", "indices"],
                    how = 'left')

    return data


def store_file(filename: str, data: pd.DataFrame) -> None:

    data.to_csv(filename, index=False)


def data_preparation_strategy(filename: str) -> None:

    data = load_json(filename + ".json")
    words = get_word_data(data)
    entities = get_entities_data(data)
    data = merge_words_entities(words, entities)
    store_file(filename  + ".csv", data)

    return None
