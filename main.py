import csv
import pandas as pd

from code.Helpers import getArgs
from code.Sampling import sampling

types_url = "./data/heloc"
types_dict_file_name = f"{types_url}/heloc_types_alt.csv"
types_dict_c_file_name = f"{types_url}/heloc_types_c_alt.csv"

class MockClassifier():
    def __init__(self):
        return
    
    def predict(x):
        print(x)
        return [x]

def csv_to_dict(file_name):
    df = pd.read_csv(file_name)
    records = df.to_dict(orient='records')
    return records        

def main():
    classifier = MockClassifier()
    args = getArgs()
    out = {"training": [0,0,0], "test_counter": [0,0,0]}
    types_dict = csv_to_dict(types_dict_file_name)
    types_dict_c = csv_to_dict(types_dict_c_file_name)
    sampling(
        settings="",
        types_dict=types_dict,
        types_dict_c=types_dict_c,
        out=out,
        ncounterfactuals=args.ncounterfactuals,
        classifier=classifier,
        n_batches_train=1,
        n_samples_train=1,
        k=2,
        n_input=23,
        degree_active=args.degree_active
    )

if __name__ == "__main__":
    main()
