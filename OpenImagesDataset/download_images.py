import pandas as pd
import os
import cv2
import argparse
from urllib.request import urlopen
from imread import imread_from_blob, imsave
from concurrent.futures import ThreadPoolExecutor

def download_and_resize(url, filename, output_dir="", w_h_shape=None):
    try:
        req = urlopen(url)
        file_type = req.headers.get('Content-Type').split('/')[1]

        image = imread_from_blob(req.read(), file_type)
        if w_h_shape:
            image = cv2.resize(image, w_h_shape)

        if output_dir and not os.path.exists(output_dir):
            os.mkdir(output_dir)

        file_path = os.path.join(output_dir, filename+"."+file_type)
        imsave(file_path, image, formatstr=file_type)

        height, width, _ = image.shape
        return filename, file_path, None, width, height
    except Exception as ex:
        return filename, None, ex, -1, -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads the OpenImages dataset from csv files')
    parser.add_argument('--width', required=False, default=None, type=int,
                                help='Desired image width. If specified, resizes image after download')
    parser.add_argument('--height', required=False, default=None, type=int,
                                help='Desired image height. If specified, resizes image after download')
    parser.add_argument('-o', '--outdir', required=False, default="",
                                help="Path to desired output directory. "
                                     "If specified, saves files in <outdir>/[train, test, validation], "
                                     "otherwise saves them to <working directory>/[train, test, validation]")

    args = parser.parse_args()
    desired_width = args.width
    desired_height = args.height
    root_out_dir = args.outdir

    pd.set_option("display.max_colwidth", 10000)

    splits=["validation", "test", "train"]
    desired_shape = (desired_width, desired_height) if (desired_width or desired_height) else None

    for split in splits:
        print(f"Processing {split} subset...")
        output_dir = os.path.join(root_out_dir, split)

        class_label_file = "filteredLabels.csv"
        bbox_file = f"{split}/{split}-filtered-bbox.csv"
        img_ids_file = f"{split}/{split}-filtered-image-ids.csv"

        labels = pd.read_csv(class_label_file)
        bboxes = pd.read_csv(bbox_file, escapechar='"')
        img_ids = pd.read_csv(img_ids_file)

        img_ids.Thumbnail300KURL = img_ids.Thumbnail300KURL.fillna(img_ids.OriginalURL)

        download_summary = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: download_and_resize(x[1].Thumbnail300KURL, x[1].ImageID, output_dir=output_dir, w_h_shape=desired_shape), img_ids.iterrows())

            for fileId, filePath, ex, width, height in results:
                if ex:
                    print(f"Download of image {fileId} failed with error '{ex}'")
                else:
                    print(f"Image {fileId} downloaded to '{filePath}'")

                download_summary.append((fileId, filePath, str(ex) if ex else None, width, height))

        df = pd.DataFrame(download_summary, columns=["ImageID", "FilePath", "Error", "Width", "Height"])
        summary_path = os.path.join(output_dir, f"{split}-download-summary.csv")
        df.to_csv(summary_path, index=False)