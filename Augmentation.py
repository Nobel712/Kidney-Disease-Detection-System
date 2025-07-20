# Settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
N= 300
OUTPUT_DIR = "augmented_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image from path and resize
def decode_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Apply specified augmentations
def augment_image(image):
    # Cropping (random crop after padding)
    image = tf.image.resize_with_crop_or_pad(image, 256, 256)  # Add border
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    # Scaling (simulate zoom-in/out by resizing then cropping/padding back)
    scale = random.uniform(0.9, 1.1)
    new_size = [int(IMG_HEIGHT * scale), int(IMG_WIDTH * scale)]
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)

    # Resizing (already done at decode, but reassert)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # Rotation (90°, 180°, or 270°)
    angle = random.choice([0, 90, 180, 270])
    image = tf.image.rot90(image, k=angle // 90)

    # Brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.3)

    return tf.clip_by_value(image, 0.0, 1.0)

# Remove duplicate images by hashing file bytes
def remove_duplicates_tf(image_paths):
    seen = set()
    unique_paths = []
    for path in image_paths:
        img_bytes = tf.io.read_file(path).numpy()
        img_hash = hash(img_bytes)
        if img_hash not in seen:
            seen.add(img_hash)
            unique_paths.append(path)
    return unique_paths

# Save image to disk
def save_image(image_tensor, index):
    image = (image_tensor.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(os.path.join(OUTPUT_DIR, f"aug_img_{index:03d}.png"))

# Main function to generate 300 images
def generate_augmented_images(input_folder, target_count=300):
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    image_paths = remove_duplicates_tf(image_paths)
    print(f"Found {len(image_paths)} unique images.")

    base_images = [decode_img(p) for p in image_paths]
    total, idx = 0, 0

    while total < target_count:
        for img in base_images:
            aug_img = augment_image(img)
            save_image(aug_img, idx)
            total += 1
            idx += 1
            if total >= target_count:
                break
    print(f"Saved {total} augmented images to '{OUTPUT_DIR}'")

# Run
if __name__ == "__main__":
    generate_augmented_images("path", target_count=N)
