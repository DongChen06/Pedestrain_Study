import os

def count_images_in_folder(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}  # Add or remove extensions as needed.
    image_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_count += 1

    return image_count

# Example usage:
folder_path = '../Data/train'
print(f"Total number of images: {count_images_in_folder(folder_path)}")
