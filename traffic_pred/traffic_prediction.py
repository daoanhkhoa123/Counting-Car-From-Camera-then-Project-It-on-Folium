import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

encode_map = {'low': 2, 'normal': 3, 'heavy': 0, 'high': 1}
decode_map = {2: 'low', 3: 'normal', 0: 'heavy', 1: 'high'}


class Traffic_Situation_Encoder:
    def to_number(self, string: str):
        return encode_map[string]

    def to_string(self, number: int):
        return decode_map[number]


class Traffic_Classifier:
    """Conslt at my traffic_prediction.ipynb

    Mainly dependent on traffic_prediction_dapassignment binary
    """

    def __init__(self) -> None:
        # with open("traffic_prediction_dapassignment", "rb") as file:
        with open(r"traffic_pred\traffic_prediction_dapassignment", "rb") as file:
            self.grid: object = pickle.load(file)

    def predict_num(self, inps):
        return self.grid.predict(inps)

    def predict_text(self, inps):
        return np.vectorize(decode_map.get)(self.grid.predict(inps))




if __name__ == "__main__":
    tse = Traffic_Situation_Encoder()
    print(tse.to_number("low"))
    print(tse.to_string(2))

    tc = Traffic_Classifier()
    print(tc.grid)
    data = np.array([5, 10, 7, 9, 31]).reshape(1,-1)
    # data = data[:,np.newaxis]
    print(tc.predict_text(data))