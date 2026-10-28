
class EmbeddingCalcError(Exception):
    def __init__(self, message=f"Embedding could not be obtained neither from cache nor from API call."):
        super().__init__(message)

class UnknownCalcScoreOption(Exception):
    def __init__(self, message=f"The selected option to calculate scores is not supported."):
        super().__init__(message)
