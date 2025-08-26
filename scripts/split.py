import os

img_dir = r"C:\Users\bilal\OneDrive\Desktop\Agile\Emirates stuff\annotateID\images\val"
label_dir = r"C:\Users\bilal\OneDrive\Desktop\Agile\Emirates stuff\annotateID\labels\val"

image_exts = [".jpg", ".jpeg", ".png"]
image_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in image_exts]
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

image_basenames = set(os.path.splitext(f)[0] for f in image_files)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)

images_without_labels = sorted([f for f in image_files if os.path.splitext(f)[0] not in label_basenames])
labels_without_images = sorted([f for f in label_files if os.path.splitext(f)[0] not in image_basenames])

print(f"Total images: {len(image_files)}")
print(f"Total labels: {len(label_files)}")
print(f"Images without labels: {len(images_without_labels)}")
print(f"Labels without images: {len(labels_without_images)}")
