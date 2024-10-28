from tools.predict.predict_no_tile import PredictNoTile
from tools.predict.predict_tile import PredictTile


class PredictFactory:
    def __call__(self, type: str):
        if type == "tile":
            result = PredictTile
        elif type == "no_tile":
            result = PredictNoTile
        else:
            raise ValueError("Unknown predict type: " + type)

        return result
