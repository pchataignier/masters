import os
import math
import argparse
import pandas as pd
import tensorflow as tf
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util, label_map_util


def create_label_map(label_map_path, labels):
    with open(label_map_path, 'w+') as f:
        for index, row in labels.iterrows():
            line = f'item {{\n' \
                            f'id: {index + 1}\n' \
                            f'name: "{row.LabelName}"\n' \
                            f'display_name: "{row.ClassName}"}}\n'
            f.write(line)


def create_tf_example(example, split):
    height = example.Height # Image height
    width = example.Width # Image width
    filename = bytes(os.path.join(split, os.path.basename(example.FilePath)), 'utf-8') # Filename of the image. Empty if image is not from file

    with tf.io.gfile.GFile(example.FilePath, 'rb') as fid:
        encoded_image_data = bytes(fid.read()) # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = example.XMin # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = example.XMax # List of normalized right x coordinates in bounding box (1 per box)
    ymins = example.YMin # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = example.YMax # List of normalized bottom y coordinates in bounding box (1 per box)

    classes_text = [x.encode() for x in example.LabelName] # List of string class name of bounding box (1 per box)
    classes = example.LabelID # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generates Tensorflow '.record' files from downloaded images")
    parser.add_argument('-i', '--input_dir', required=False, default="",
                        help="Path to input directory. Default is current directory")
    parser.add_argument('-o', '--output_dir', required=False, default="",
                        help="Path to desired output directory. "
                             "If specified, saves files in <outdir>/[train, test, validation], "
                             "otherwise saves them to <working directory>/[train, test, validation]")
    parser.add_argument('-s', '--shards', required=False, type=int)
    args = parser.parse_args()

    pd.set_option("display.max_colwidth", 10000)

    # Get class labels
    class_label_file = os.path.join(args.input_dir, "filteredLabels.csv")
    labels = pd.read_csv(class_label_file)

    # Create labelMap.pbtxt
    label_map_path = os.path.join(args.output_dir, "labelMap.pbtxt")
    create_label_map(label_map_path, labels)
    label_dict = label_map_util.get_label_map_dict(label_map_util.load_labelmap(label_map_path))


    for split in ["validation", "test", "train"]:

        # Load bboxes
        bbox_file = os.path.join(args.input_dir, split, f"{split}-filtered-bbox.csv")
        bboxes = pd.read_csv(bbox_file, escapechar='"')

        # Load download summary (from download_images.py)
        download_summary_file = os.path.join(args.input_dir, split, f"{split}-download-summary.csv")
        download_summary = pd.read_csv(download_summary_file)

        # Filter bboxes and generate examples table
        bboxes = bboxes[bboxes.ImageID.isin(download_summary[~download_summary.FilePath.isnull()].ImageID)]
        bboxes["LabelID"] = bboxes.LabelName.apply(lambda x: label_dict[x])
        examples = bboxes.groupby("ImageID").agg({"LabelName": list, "LabelID":list,
                                                  "XMin": list, "XMax": list,
                                                  "YMin": list, "YMax": list}).reset_index()
        examples = pd.merge(examples, download_summary, on="ImageID")

        # Duplicate examples of classes that are un-balanced
        if split == 'train':
            class_load = bboxes.groupby('LabelID').agg({"LabelName":'count', "ImageID":list}).reset_index()

            target = class_load.LabelName.max()
            examples_clone = examples.copy()
            to_append = []

            for class_id, num_examples, img_ids in class_load.itertuples(index=False):
                factor = math.ceil(target / num_examples) - 1
                if factor <= 0: continue

                to_replicate = examples_clone[examples_clone.ImageID.isin(img_ids)]
                to_append += [to_replicate] * factor

            examples = examples.append(to_append, ignore_index=True)

        # Shuffle 'examples' rows
        examples = examples.sample(frac=1, random_state=22).reset_index(drop=True)

        record_path = os.path.join(args.output_dir, split, f"{split}.record")

        num_shards = math.ceil(examples.shape[0]/args.shards) if args.shards else 1

        # Single record file
        if not args.shards or num_shards==1:
            writer = tf.io.TFRecordWriter(record_path)
            for example in examples.itertuples(index=False):
                tf_example = create_tf_example(example, split)
                writer.write(tf_example.SerializeToString())

            writer.close()

        # Sharded records
        else:
            with contextlib2.ExitStack() as tf_record_close_stack:
                output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, record_path, num_shards)

                for example in examples.itertuples():
                    tf_example = create_tf_example(example, split)
                    output_shard_index = example.Index % num_shards
                    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())