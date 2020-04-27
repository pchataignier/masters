import os
import cv2
import math
import imutils
import argparse
import contextlib2
import pandas as pd
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import label_map_util

pd.set_option("display.max_colwidth", 10000)

train_annotations_csv = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
train_imgs_url_csv = "https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-images-boxable.csv"

val_annotations_csv = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
val_imgs_url_csv = "https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-images.csv"

test_annotations_csv = "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
test_imgs_url_csv = "https://datasets.appen.com/appen_datasets/open-images/test-images.csv"

# # Helper Functions
def get_filtered_bboxes(annotations_path, labels_df):
    if not os.path.isfile(FILTERED_ANNO_CSV):
        anno_df = pd.read_csv(annotations_path)
        anno_df = anno_df[(anno_df.IsDepiction == 0) &
                                     anno_df.LabelName.isin(labels_df.LabelName)]
        anno_df = pd.merge(anno_df, labels_df, on="LabelName")
        if SAVE_FILTERED_CSV:
            anno_df.to_csv(FILTERED_ANNO_CSV, index=False)
    else:
        anno_df = pd.read_csv(FILTERED_ANNO_CSV)
    
    return anno_df

def get_filtered_urls(imgs_path, bboxes):
    if not os.path.isfile(FILTERED_URL_CSV):
        url_df = pd.read_csv(imgs_path)
        url_df["ImageID"] = url_df.image_name.apply(lambda x: x.split('.')[0])
        url_df.drop('image_name', axis=1, inplace=True)
        
        url_df = url_df[url_df.ImageID.isin(bboxes.ImageID)]
        
        if SAVE_FILTERED_CSV:
            url_df.to_csv(FILTERED_URL_CSV, index=False)
    else:
        url_df = pd.read_csv(FILTERED_URL_CSV)
    
    return url_df

def get_class_load_and_factors(bboxes):
    class_load = bboxes.groupby('ClassName').agg({"LabelName":'count', "ImageID":'unique'}).reset_index()
    class_load.rename(columns={'LabelName': 'Instances'}, inplace=True)

    total_boxes = class_load.Instances.sum()
    class_load["Normalized"] = class_load.Instances.apply(lambda x: 100 * x/total_boxes)

    target = class_load.Instances.max()
    class_load["Factor"] = class_load.Instances.apply(lambda x: math.ceil(target / x))

    factors = pd.DataFrame(columns=["ImageID", "Factor"])
    for i, row in class_load.iterrows():
        df = pd.DataFrame(row.ImageID, columns=["ImageID"], dtype=str)
        df['Factor'] = row.Factor
        factors = pd.concat([factors, df], ignore_index=True)

    factors = factors.sort_values("Factor", ascending=False).drop_duplicates("ImageID").reset_index(drop=True)
    
    return class_load, factors

# def plot_class_load(class_load):
#     ax = class_load.plot.bar("ClassName", "Normalized", rot=45, figsize=(10, 5))
#     ax.set_xlabel("")
#     ax.set_ylabel("Number of boxes (%)", fontsize=12)
#     ax.legend_.remove()
#     #ax.set_ylim([0, 100])
#
#     (y_bottom, y_top) = ax.get_ylim()
#     y_height = y_top - y_bottom
#
#     rects = ax.patches
#     labels = [f"{row.Instances:,}" for _,row in class_load.iterrows()]
#
#     for rect, label in zip(rects, labels):
#         height = rect.get_height()
#         p_height = (height / y_height)
#
#         if p_height > 0.95: # arbitrary; 95% looked good to me.
#             label_position = height - (y_height * 0.05)
#         else:
#             label_position = height + (y_height * 0.01)
#
#         ax.text(rect.get_x() + rect.get_width() / 2, label_position, label.replace(",", " "),
#                 ha='center', va='bottom')
#     plt.tight_layout()
#     return ax

def get_encoded_image_bytes_from_url(url):
    img = imutils.url_to_image(url)
    img = cv2.resize(img, (640, 480))
    success, encoded_image = cv2.imencode('.jpg', img)
    
    return encoded_image.tobytes()

def decode_jpg_image(encoded_jpg):
    bytess = tf.placeholder(tf.string)
    decode_img = tf.image.decode_image(bytess, channels=3)
    session = tf.Session()
    image = session.run(decode_img, feed_dict={bytess: encoded_jpg})
    return image

def create_label_map(label_map_path, labels):
    with open(label_map_path, 'w+') as f:
        for index, row in labels.iterrows():
            line = f'item {{\n'\
                   f'id: {index + 1}\n'\
                   f'name: "{row.LabelName}"\n'\
                   f'display_name: "{row.ClassName}"}}\n'
            f.write(line)


# ## Fix for as_matrix() deprecation
from object_detection.core import standard_fields
from object_detection.utils import dataset_util


def tf_example_from_annotations_data_frame(annotations_data_frame, label_map, encoded_image):
    """Populates a TF Example message with image annotations from a data frame.
    Args:
    annotations_data_frame: Data frame containing the annotations for a single
      image.
    label_map: String to integer label map.
    encoded_image: The encoded image string
    Returns:
    The populated TF Example, if the label of at least one object is present in
    label_map. Otherwise, returns None.
    """

    filtered_data_frame = annotations_data_frame[annotations_data_frame.LabelName.isin(label_map)]
    filtered_data_frame_boxes = filtered_data_frame[~filtered_data_frame.YMin.isnull()]
    filtered_data_frame_labels = filtered_data_frame[filtered_data_frame.YMin.isnull()]
    image_id = annotations_data_frame.ImageID.iloc[0]

    feature_map = {
      standard_fields.TfExampleFields.object_bbox_ymin:
          dataset_util.float_list_feature(filtered_data_frame_boxes.YMin.to_numpy()),
      standard_fields.TfExampleFields.object_bbox_xmin:
          dataset_util.float_list_feature(filtered_data_frame_boxes.XMin.to_numpy()),
      standard_fields.TfExampleFields.object_bbox_ymax:
          dataset_util.float_list_feature(filtered_data_frame_boxes.YMax.to_numpy()),
      standard_fields.TfExampleFields.object_bbox_xmax:
          dataset_util.float_list_feature(filtered_data_frame_boxes.XMax.to_numpy()),
      standard_fields.TfExampleFields.object_class_text:
          dataset_util.bytes_list_feature(filtered_data_frame_boxes.LabelName.to_numpy().astype("bytes")),
      standard_fields.TfExampleFields.object_class_label:
          dataset_util.int64_list_feature(filtered_data_frame_boxes.LabelName.map(lambda x: label_map[x]).to_numpy()),
      standard_fields.TfExampleFields.filename:
          dataset_util.bytes_feature('{}.jpg'.format(image_id).encode("utf-8")),
      standard_fields.TfExampleFields.source_id:
          dataset_util.bytes_feature(image_id.encode("utf-8")),
      standard_fields.TfExampleFields.image_encoded:
          dataset_util.bytes_feature(encoded_image),
  }

    if 'IsGroupOf' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_group_of] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsGroupOf.to_numpy().astype(int))
    if 'IsOccluded' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_occluded] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsOccluded.to_numpy().astype(int))
    if 'IsTruncated' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_truncated] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsTruncated.to_numpy().astype(int))
    if 'IsDepiction' in filtered_data_frame.columns:
        feature_map[standard_fields.TfExampleFields.object_depiction] = dataset_util.int64_list_feature(filtered_data_frame_boxes.IsDepiction.to_numpy().astype(int))

    return tf.train.Example(features=tf.train.Features(feature=feature_map))

parser = argparse.ArgumentParser(description="Generates Tensorflow '.record' files")
parser.add_argument('-o', '--output_dir', required=False, default="",
                    help="Path to desired output directory. "
                         "If specified, saves files in <outdir>/[train, test, validation], "
                         "otherwise saves them to <working directory>/[train, test, validation]")
args = parser.parse_args()
OUT_DIR = args.output_dir

# ## Settings and File Paths
SPLITS = {"validation":100, "train":1000}
LABELS_CSV = "filteredLabels.csv"
LABEL_MAP_PATH = "labelMap.pbtxt"
SAVE_FILTERED_CSV = True

# ### Label Map pbtxt file
labels = pd.read_csv(LABELS_CSV)
if not os.path.isfile(LABEL_MAP_PATH):
    create_label_map(LABEL_MAP_PATH, labels)
label_dict = label_map_util.get_label_map_dict(LABEL_MAP_PATH)

for SPLIT, NUM_SHARDS in SPLITS.items():
    split_out_dir = os.path.join(OUT_DIR, SPLIT)
    if not os.path.exists(split_out_dir):
        os.makedirs(split_out_dir)

    RECORDS_FILEPATH = os.path.join(split_out_dir, f"{SPLIT}.tfrecord")
    ANNO_CSV = ""
    URLS_CSV = ""

    FILTERED_ANNO_CSV = f"{SPLIT}/{SPLIT}-filtered-annotations-v2.csv"
    FILTERED_URL_CSV = f"{SPLIT}/{SPLIT}-filtered-img-url-v2.csv"
    BALANCE_EXAMPLES = False if SPLIT == "test" else True

    if not ANNO_CSV:
        if SPLIT == "train":
            ANNO_CSV = train_annotations_csv
        elif SPLIT == "test":
            ANNO_CSV = test_annotations_csv
        else:
            ANNO_CSV = val_annotations_csv

    if not URLS_CSV:
        if SPLIT == "train":
            URLS_CSV = train_imgs_url_csv
        elif SPLIT == "test":
            URLS_CSV = test_imgs_url_csv
        else:
            URLS_CSV = val_imgs_url_csv

    # ## Main code


    # ### Loading DataFrames
    bboxes = get_filtered_bboxes(ANNO_CSV, labels)
    urls = get_filtered_urls(URLS_CSV, bboxes)
    class_load, factors = get_class_load_and_factors(bboxes)

    # ### Checkpoint file
    # Saves the image bytes and errors.
    # Used for continuing without re-downloading.
    ckpt_file = os.path.join(split_out_dir, f"{SPLIT}-download-summary-v2.csv")
    ckpt = None
    if os.path.isfile(ckpt_file):
        ckpt = pd.read_csv(ckpt_file)
    else:
        ckpt = pd.DataFrame([], columns=["ImageID", "Error", "ErrorMessage", "EncodedJpg"])
        ckpt.to_csv(ckpt_file, index=False)


    # ### Writing the Record files
    total = len(bboxes.ImageID.unique())
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                 RECORDS_FILEPATH,
                                                                                 NUM_SHARDS)

        for counter, image_data in enumerate(bboxes.groupby('ImageID')):

            image_id, image_annotations = image_data
            print(f"Processing image {counter+1}/{total} at {image_id}")

            error = False
            error_msg = ""
            try:
                if image_id in ckpt.ImageID.unique():
                    print("\tAlready seen")
                    if ckpt[ckpt.ImageID == image_id].iloc[0].Error: continue
                    else:
                        encoded_jpg = ckpt[ckpt.ImageID == image_id].iloc[0].EncodedJpg
                else:
                    url = urls[urls.ImageID == image_id].iloc[0].image_url
                    encoded_jpg = get_encoded_image_bytes_from_url(url)

                tf_example = tf_example_from_annotations_data_frame(image_annotations, label_dict, encoded_jpg)

                if tf_example:
                    if BALANCE_EXAMPLES:
                        factor = factors[factors.ImageID == image_id].iloc[0].Factor
                        for i in range(factor):
                            shard_idx = (counter + i) % NUM_SHARDS
                            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
                    else:
                        shard_idx = counter % NUM_SHARDS
                        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except Exception as ex:
                error = True
                error_msg = str(ex)
                encoded_jpg = b''
                print(error_msg)

            ckpt = ckpt.append({"ImageID":image_id, "Error":error, "ErrorMessage":error_msg, "EncodedJpg":encoded_jpg},
                               ignore_index=True)
            ckpt.to_csv(ckpt_file, index=False)

print("Finished!")