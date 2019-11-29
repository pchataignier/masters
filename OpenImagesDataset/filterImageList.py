import pandas as pd

pd.set_option("display.max_colwidth", 10000)

splits=["validation", "test", "train"]

for split in splits:
    class_label_file = "filteredLabels.csv"
    bbox_file = f"{split}/{split}-annotations-bbox.csv"
    img_ids_file = f"{split}/{split}-image-ids.csv"

    labels = pd.read_csv(class_label_file)
    bboxes = pd.read_csv(bbox_file, escapechar='"')
    img_ids = pd.read_csv(img_ids_file)

    filtered_bboxes = bboxes[(bboxes.IsDepiction == 0) &
                             (bboxes.IsGroupOf == 0) &
                             bboxes.LabelName.isin(labels.LabelName)]

    imgs = img_ids[img_ids.ImageID.isin(filtered_bboxes.ImageID)]

    filtered_bboxes.to_csv(f"{split}/{split}-filtered-bbox.csv", index=False)
    imgs.to_csv(f"{split}/{split}-filtered-image-ids.csv", index=False)
