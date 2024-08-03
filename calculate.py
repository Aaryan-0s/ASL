import os


def count_images_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(
            [
                file
                for file in files
                if file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]
        )
    return count


def count_images_in_dataset(base_directory):
    train_directory = os.path.join(base_directory, "train")
    test_directory = os.path.join(base_directory, "test")

    num_train_images = count_images_in_directory(train_directory)
    num_test_images = count_images_in_directory(test_directory)

    return num_train_images, num_test_images


# Base directory containing 'train' and 'test' folders
dataset_directory = "data2"
num_train_images, num_test_images = count_images_in_dataset(dataset_directory)

print(f"Number of training images: {num_train_images}")
print(f"Number of test images: {num_test_images}")

batch_size = 10
steps_per_epoch = num_train_images // batch_size
validation_steps = num_test_images // batch_size

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
