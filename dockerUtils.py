import os
from urllib.parse import urlparse

def LoadModelsDict():

    modelsFilePath = os.environ['DOWNLOADED_MODELS_FILEPATH']
    modelsFolder = os.path.dirname(modelsFilePath)

    models_dict = {}
    with open(modelsFilePath) as f:
        for line in f:
            (key, val) = line.split()
            filename = os.path.basename(urlparse(val).path)
            val = os.path.join(modelsFolder, filename)
            models_dict[key] = val

    return models_dict

def GetModelPath(model):
    models = LoadModelsDict()

    if model.lower() in models:
        return models[model.lower()]
    else:
        return model
