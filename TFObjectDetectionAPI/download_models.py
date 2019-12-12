#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import urllib
import pandas as pd
from glob import glob
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


def get_record_file_patten(dataset_dir, split):
    records = glob(f"{dataset_dir}/{split}/{split}.record*")

    if len(records) == 1:
        return records[0]

    pattern = records[0]
    to_replace = pattern.split('-')[-3]

    return pattern.replace(to_replace, "?" * len(to_replace))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Downloads pre-trained Tensorflow models")
    parser.add_argument('-d', '--dataset', required=True,
                        help="Full path to root directory of the dataset.")
    parser.add_argument('-m', '--models_csv', required=False, default="models.csv",
                        help="Path to csv file containing the model names on Tensorflow download website. "
                             "See https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")
    parser.add_argument('-n', '--n_classes', required=False, default=7, type=int, help="Number of classes")
    args = parser.parse_args()

    dataset_dir = args.dataset
    models_csv = args.models_csv
    n_classes = args.n_classes

    pd.set_option("display.max_colwidth", 10000)
    models = pd.read_csv(models_csv)

    overrides = {"train_config.fine_tune_checkpoint": "model.ckpt",
                 "label_map_path": f"{dataset_dir}/labelMap.pbtxt",
                 "eval_input_path": f"{get_record_file_patten(dataset_dir, 'validation')}",
                 "train_input_path": f"{get_record_file_patten(dataset_dir, 'train')}",
                 "train_shuffle": True, "num_classes": n_classes}

    for model_id, model_name in models.itertuples(index=False):
        download_model(model_name)
        override_pipeline_configs(model_name+"/pipeline.config", overrides, model_name)

