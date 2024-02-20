import pandas as pd

import shutil


control_dir = "./cataracts/Train_data/control/"
treatment_dir = "./cataracts/Train_data/treatment/"


image_mapping = pd.read_excel("./cataracts/images.xlsx")

left_eye_imgs = image_mapping["Left-Fundus"]
right_eye_imgs = image_mapping["Right-Fundus"]

left_eye_labels = image_mapping["Left-Diagnostic Keywords"]
right_eye_labels = image_mapping["Right-Diagnostic Keywords"]


# left eye images first
for i, eye_img_file_name in enumerate(left_eye_imgs):
    if left_eye_labels[i] == "cataract":
        shutil.move(f"./cataracts/original/{eye_img_file_name}", control_dir + eye_img_file_name)
    if left_eye_labels[i] == "normal fundus":
        shutil.move(f"./cataracts/original/{eye_img_file_name}", treatment_dir + eye_img_file_name)

# right eye images
for i, eye_img_file_name in enumerate(right_eye_imgs):
    if right_eye_labels[i] == "cataract":
        shutil.move(f"./cataracts/original/{eye_img_file_name}", control_dir + eye_img_file_name)
    if right_eye_labels[i] == "normal fundus":
        shutil.move(f"./cataracts/original/{eye_img_file_name}", treatment_dir + eye_img_file_name)





