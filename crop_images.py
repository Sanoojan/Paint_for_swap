from PIL import Image

# Load the image
img = Image.open("Reverse_find_folder/44.png")

# Get image dimensions
width, height = img.size


# Calculate height of each individual image
single_image_height = height // 2

# Split the image into 6 parts
images = [img.crop((0, i * single_image_height, width, (i + 1) * single_image_height)) for i in range(2)]

# Save each image separately
for i, image in enumerate(images):
    file_path = f"Reverse_find_folder/{i + 1}.png"
    image.save(file_path)
    print(f"Saved: {file_path}")