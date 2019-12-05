#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import urllib
import pandas as pd
from object_detection.utils import config_util


def download_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    
    urllib.request.urlretrieve(base_url + model_file, model_file)
    
    tarfile.open(model_file).extractall()
    
    os.remove(model_file)


def set_number_of_classes(model_config, n_classes):
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")


def override_pipeline_configs(config_file, overrides, out_dir=""):
    configs = config_util.get_configs_from_pipeline_file(config_file)

    for field, value in overrides.items():
        if field == "num_classes":
            set_number_of_classes(configs['model'], value)

        elif not config_util._maybe_update_config_with_key_value(configs, field, value):
            try:
                config_util._update_generic(configs, field, value)
            except ValueError as ex:
                if field == "train_config.fine_tune_checkpoint":
                    configs['train_config'].fine_tune_checkpoint = value
                else:
                    raise

    config_util.save_pipeline_config(config_util.create_pipeline_proto_from_configs(configs), out_dir)


if __name__ == '__main__':

    model_name = "ssd_mobilenet_v2_oid_v4_2018_12_12"

    download_model(model_name)

    overrides = {"train_config.fine_tune_checkpoint": "model.ckpt",
                 "label_map_path": "labelMap.pbtxt",
                 "eval_input_path": "validation/validation.record-????-of-0010",
                 "train_input_path": "train/train.record-????-of-0010",
                 "train_shuffle":True, "num_classes":7}

    override_pipeline_configs(model_name+"/pipeline.config", overrides, model_name)

