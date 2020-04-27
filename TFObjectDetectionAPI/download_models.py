#!/usr/bin/env python
# coding: utf-8

import os
import re
import shutil
import tarfile
import urllib
import pandas as pd
from glob import glob
from object_detection.utils import config_util


def download_model(model_name, out_dir=None):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'

    urllib.request.urlretrieve(base_url + model_file, model_file)

    if not out_dir: out_dir=model_name

    tarfile.open(model_file).extractall()
    
    os.remove(model_file)
    os.rename(model_name, out_dir)


def get_model_architecture(model_config):
    return model_config.WhichOneof("model")


def get_image_resizer_type(resizer):
    return resizer.WhichOneof('image_resizer_oneof')


def set_keep_aspect_ratio_resizer_dimensions(resizer, min_dimension, max_dimension):
    resizer.keep_aspect_ratio_resizer.min_dimension = min_dimension
    resizer.keep_aspect_ratio_resizer.max_dimension = max_dimension


def set_fixed_shape_resizer_dimensions(resizer, width, height):
    resizer.fixed_shape_resizer.width = width
    resizer.fixed_shape_resizer.height = height


def set_resizer_width_height(model_config, width, height):
    meta_architecture = get_model_architecture(model_config)

    resizer = model_config.faster_rcnn.image_resizer if meta_architecture == "faster_rcnn" else model_config.ssd.image_resizer
    resizer_type = get_image_resizer_type(resizer)

    if resizer_type == "keep_aspect_ratio_resizer":
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        set_keep_aspect_ratio_resizer_dimensions(resizer, min_dimension, max_dimension)

    elif resizer_type == "fixed_shape_resizer":
        set_fixed_shape_resizer_dimensions(resizer, width, height)

def set_number_of_classes(model_config, n_classes):
    meta_architecture = get_model_architecture(model_config)
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")


def override_pipeline_configs(config_file, overrides, out_dir=""):
    configs = config_util.get_configs_from_pipeline_file(config_file)

    configs['train_config'].from_detection_checkpoint = True

    for field, value in overrides.items():
        if field == "num_classes":
            set_number_of_classes(configs['model'], value)
        elif field == "width_height":
            set_resizer_width_height(configs['model'], value[0], value[1])
        elif not config_util._maybe_update_config_with_key_value(configs, field, value):
            try:
                config_util._update_generic(configs, field, value)
            except ValueError as ex:
                if field == "train_config.fine_tune_checkpoint":
                    configs['train_config'].fine_tune_checkpoint = value
                else:
                    raise

    config_util.save_pipeline_config(config_util.create_pipeline_proto_from_configs(configs), out_dir)

def post_process_pipeline_file(filename):
    with open(filename, 'r+') as f:
        text = f.read()

        text = text.replace("open_images_V2_detection_metrics", "oid_V2_detection_metrics")
        text = re.sub(r'keep_checkpoint_every_n_hours:[\s]*[\d.]*', '', text)

        f.seek(0)
        f.write(text)
        f.truncate()

def get_record_file_patten(dataset_dir, split):
    records_pattern = os.path.join(dataset_dir, split, f"{split}.record*")
    records = glob(records_pattern)

    if not records:
        print(f"No record files found in {os.path.join(dataset_dir, split)}")
        return "placeholder.record"

    if len(records) == 1:
        return records[0]

    pattern = records[0]
    to_replace = pattern.split('-')[-3]

    return pattern.replace(to_replace, "?" * len(to_replace))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Downloads pre-trained Tensorflow models")
    parser.add_argument('-d', '--dataset', required=False, default="../OpenImagesDataset",
                        help="Full path to root directory of the dataset. Defaults to '../OpenImagesDataset'")
    parser.add_argument('-m', '--models_csv', required=False, default="models.csv",
                        help="Path to csv file containing the model names on Tensorflow download website. "
                             "See https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")
    parser.add_argument('-n', '--n_classes', required=False, default=7, type=int, help="Number of classes")
    parser.add_argument('-b', '--batch_size', required=False, default=1, type=int, help="Batch size. Default 1")
    parser.add_argument('--width', required=False, default=640, type=int, help="Image width. Default 640")
    parser.add_argument('--height', required=False, default=480, type=int, help="Image height. Default 480")
    parser.add_argument('--override_only', required=False, action='store_true', help="Skip downloading the files. Used for fixing config files")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    models_csv = args.models_csv
    n_classes = args.n_classes
    batch_size = args.batch_size
    width_height = (args.width, args.height)

    pd.set_option("display.max_colwidth", 10000)
    models = pd.read_csv(models_csv)

    label_map_path = os.path.join(dataset_dir, "labelMap.pbtxt")

    for model_id, model_name in models.itertuples(index=False):
        checkpoint_path = os.path.join(model_id, "model.ckpt")
        overrides = {"train_config.fine_tune_checkpoint": f"{checkpoint_path}",
                     "label_map_path": f"{label_map_path}",
                     "eval_input_path": f"{get_record_file_patten(dataset_dir, 'validation')}",
                     "train_input_path": f"{get_record_file_patten(dataset_dir, 'train')}",
                     "batch_size": batch_size, "train_shuffle": True,
                     "num_classes": n_classes, "width_height": width_height}
        if not args.override_only:
            download_model(model_name, model_id)

        config_file = os.path.join(model_id, "pipeline.config")
        config_bkp = config_file + ".bkp"
        if not os.path.isfile(config_bkp):
            shutil.copy(config_file, config_bkp)
        override_pipeline_configs(config_file, overrides, model_id)
        post_process_pipeline_file(config_file)

