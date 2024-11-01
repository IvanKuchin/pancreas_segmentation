from tools.predict.predict_no_tile import PredictNoTile
from tools.predict.predict_tile import PredictTile


def predict_factory(config: dict):
        if config.IS_TILE:
            result = PredictTile
        elif config.IS_TILE == False:
            result = PredictNoTile
        else:
            raise ValueError("Unknown predict type: " + type)

        return result
