import json
import pandas as pd


def analyzeASLData():
    classFile = open("data/asl/MSASL_classes.json", encoding="utf-8")
    classes = json.load(classFile)
    print("Classes: ", len(classes))

    synonymFile = open("data/asl/MSASL_synonym.json",encoding="utf-8")
    synonym = json.load(synonymFile)
    print("Synonym: ", len(synonym))



analyzeASLData()


def analyzeVSLData():
    ## Problem: Split raw data to different datasets for model accuracy testing.

    f = open("data/full_dict.json", encoding="utf-8")

    data = json.load(f)["data"]

    json_loader = pd.DataFrame.from_dict(data)
    print("Entire dataset: ", json_loader.shape[0])

    numbers_data = json_loader[json_loader["type"] == 1]
    print("Numbers dataset: ", numbers_data.shape[0])

    alphabet_data = json_loader[json_loader["type"] == 2]
    print("Alphabet dataset: ", alphabet_data.shape[0])

    ## Split words data to 2 type: complex words (has many meanings, long words) and simple words.
    words_data = json_loader[json_loader["type"] == 0]
    print("Words dataset: ", words_data.shape[0])

    words_data["word"] = words_data["word"].astype("str")

    complex_words = words_data.loc[json_loader["word"].str.len() > 10]
    print("Complex words dataset: ", complex_words.shape[0])

    simple_words = words_data.loc[json_loader["word"].str.len() < 10]
    print("Simple words dataset: ", simple_words.shape[0])
