import os
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util


flags = tf.app.flags
flags.DEFINE_string('output_dir', '', 'Path to output directory')
flags.DEFINE_string('input_dir', '', 'Path to data directory')
FLAGS = flags.FLAGS

def create_label_map(label_map_path, labels):
    with open(label_map_path, 'w+') as f:
        for index, row in labels.iterrows():
            line = f'item {{\n' \
                            f'id: {index + 1}\n' \
                            f'name: "{row.LabelName}"\n' \
                            f'display_name: "{row.ClassName}"}}\n'
            f.write(line)

def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = None # Image height
    width = None # Image width
    filename = None # Filename of the image. Empty if image is not from file
    encoded_image_data = None # Encoded image bytes
    image_format = None # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
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


def main(_):
    split = "validation"

    record_path = os.path.join(FLAGS.output_dir, split, f"{split}.record")
    writer = tf.python_io.TFRecordWriter(record_path)

    class_label_file = os.path.join(FLAGS.input_dir, "filteredLabels.csv")
    bbox_file = os.path.join(FLAGS.input_dir, split, f"{split}-filtered-bbox.csv")
    img_ids_file = os.path.join(FLAGS.input_dir, split, f"{split}-filtered-image-ids.csv")
    download_summary_file = os.path.join(FLAGS.input_dir, split, f"{split}_download_summary.csv")

    labels = pd.read_csv(class_label_file)
    bboxes = pd.read_csv(bbox_file, escapechar='"')
    img_ids = pd.read_csv(img_ids_file)
    download_summary = pd.read_csv(download_summary_file)

    label_map_path = os.path.join(FLAGS.output_dir, "labelMap.pbtxt")
    create_label_map(label_map_path, labels)
    label_dict = label_map_util.get_label_map_dict(label_map_util.load_labelmap(label_map_path))
    # TODO(user): Write code to read in your dataset to examples variable

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()