import os
import random
import shutil
from PIL import Image, ImageEnhance 
import numpy as np

def generate_gender_balanced_dataset(source_dir, dest_dir, target_samples_per_gender):
    # 1. Create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 2. Define gender groups (assuming they are subdirectories in source_dir)
    gender_groups = ['Male', 'Female']

    for gender in gender_groups:
        source_gender_dir = os.path.join(source_dir, gender)
        dest_gender_dir = os.path.join(dest_dir, gender)

        # Create destination directory for the gender group
        if not os.path.exists(dest_gender_dir):
            os.makedirs(dest_gender_dir)

        # 3. Get list of images in the source gender group directory
        try:
            images = [f for f in os.listdir(source_gender_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
        except FileNotFoundError:
            print(f"Error: Directory not found: {source_gender_dir}. Skipping this gender.")
            continue

        num_images = len(images)

        if num_images == 0:
            print(f"Warning: No images found in {source_gender_dir}. Skipping this gender.")
            continue

        # 4. Sample images for the gender group
        if num_images > target_samples_per_gender:
            # Randomly sample images without replacement
            sampled_images = random.sample(images, target_samples_per_gender)
            print(f"Downsampling {gender} from {num_images} to {target_samples_per_gender} images.")
        elif num_images < target_samples_per_gender:
            # Handle the case where there are fewer images than the target number.
            sampled_images = images * (target_samples_per_gender // num_images) + random.sample(images, target_samples_per_gender % num_images)
            print(f"Upsampling {gender} from {num_images} to {target_samples_per_gender} images.")
        else:
            # No sampling needed
            sampled_images = images
            print(f"No change in sample number for {gender}. {num_images} images.")

        # 5. Define augmentation parameters based on level
        augment_params = {
            'rotate': (-15, 15),
            'translate': ((0, -20), (0, 20)),
            'scale': (0.9, 1.1),
            'brightness': (0.9, 1.1),
            'contrast': (0.9, 1.1),
        }


        # 6. Copy and augment sampled images
        for i, image_name in enumerate(sampled_images):
            source_image_path = os.path.join(source_gender_dir, image_name)
            dest_image_path = os.path.join(dest_gender_dir, f"{os.path.splitext(image_name)[0]}_augmented_{i}.jpg")

            try:
                source_image = Image.open(source_image_path)
                augmented_image = source_image.copy() #start with a copy

                # Apply augmentations
                if augment_params: # Only augment if params are not empty
                    if 'rotate' in augment_params and random.random() < 0.8:
                        angle = random.uniform(augment_params['rotate'][0], augment_params['rotate'][1])
                        augmented_image = augmented_image.rotate(angle, resample=Image.BICUBIC)

                    if 'translate' in augment_params and random.random() < 0.8:
                        x_translate = random.randint(augment_params['translate'][0][0], augment_params['translate'][0][1])
                        y_translate = random.randint(augment_params['translate'][1][0], augment_params['translate'][1][1])
                        augmented_image = augmented_image.transform(augmented_image.size, Image.AFFINE,
                                                                    (1, 0, x_translate, 0, 1, y_translate),
                                                                    resample=Image.BICUBIC)

                    if 'scale' in augment_params and random.random() < 0.8:
                        scale = random.uniform(augment_params['scale'][0], augment_params['scale'][1])
                        new_size = (int(augmented_image.size[0] * scale), int(augmented_image.size[1] * scale))
                        augmented_image = augmented_image.resize(new_size, resample=Image.BICUBIC)

                    if 'brightness' in augment_params and random.random() < 0.5:
                        brightness = random.uniform(augment_params['brightness'][0], augment_params['brightness'][1])
                        enhancer = ImageEnhance.Brightness(augmented_image)
                        augmented_image = enhancer.enhance(brightness)

                    if 'contrast' in augment_params and random.random() < 0.5:
                        contrast = random.uniform(augment_params['contrast'][0], augment_params['contrast'][1])
                        enhancer = ImageEnhance.Contrast(augmented_image)
                        augmented_image = enhancer.enhance(contrast)

                    if 'sharpness' in augment_params and random.random() < 0.5:
                        sharpness = random.uniform(augment_params['sharpness'][0], augment_params['sharpness'][1])
                        enhancer = ImageEnhance.Sharpness(augmented_image)
                        augmented_image = enhancer.enhance(sharpness)

                augmented_image.save(dest_image_path, "JPEG")
                print(f"Saved augmented image: {dest_image_path}")

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    print("Dataset generation and balancing complete.")



def main():
    """
    Main function to execute the dataset generation and balancing process.
    """
    # Specify the path to your source directory containing the gender-grouped subdirectories
    source_directory = 'content/CXR_Fairness/CheXpert-v1.0.0-small'  # Replace with your actual source directory
    # Specify the path to the destination directory where the balanced dataset will be created
    destination_directory = 'content/CXR_Fairness/CheXpert-bias'  # Replace with your desired destination directory

    target_samples = 500  # Example: 500 images per gender group

    # Create the gender-balanced and augmented dataset
    generate_gender_balanced_dataset(source_directory, destination_directory, target_samples)



if __name__ == "__main__":
    main()