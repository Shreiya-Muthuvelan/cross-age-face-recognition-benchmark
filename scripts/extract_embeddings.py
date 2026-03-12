
# Preprocessing and embedding extraction for FGNET and CACD datasets.

import os
import cv2
import numpy as np
import json
import keras  

# --------------------
# Preprocessing
# --------------------

def preprocess_image_facenet(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_image_vggface(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    mean = np.array([93.5940, 104.7624, 129.1863])
    img -= mean
    return np.expand_dims(img, axis=0)


def preprocess_image_arcface(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype('float32')
    mean = np.array([127.5, 127.5, 127.5])
    img = (img - mean) / 128.0
    return np.expand_dims(img, axis=0)


def preprocess_image_deepface(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (152, 152))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_image_openface(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


# --------------------
# Embedding extractors
# --------------------

def get_face_embedding(model, image_path):
    return model.model.predict(preprocess_image_facenet(image_path))


def get_vggface_embedding(model, image_path):
    return model.model.predict(preprocess_image_vggface(image_path))


def get_arcface_embedding(model, image_path):
    return model.model.predict(preprocess_image_arcface(image_path))


def get_deepface_embedding(model, image_path):
    return model.model.predict(preprocess_image_deepface(image_path))


def get_openface_embedding(model, image_path):
    return model.model.predict(preprocess_image_openface(image_path))


embedding_extractors = {
    "facenet":  get_face_embedding,
    "vggface":  get_vggface_embedding,
    "arcface":  get_arcface_embedding,
    "deepface": get_deepface_embedding,
    "openface": get_openface_embedding,
}

# --------------------
# Dataset iterators
# --------------------

def iterate_fgnet_images(base_dir):
    """
    Yields (image_path, subject_id, age) for every .jpg under base_dir.
    FGNET filename convention: <subject_id><gender><age_zero_padded>.jpg
    e.g. 001A05.jpg -> subject='001', age=5
    """
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith(".jpg"):
                try:
                    subject_id = filename[:3]
                    age = int(filename[4:6])
                    yield os.path.join(root, filename), subject_id, age
                except Exception as e:
                    print(f"Skipping {filename}: {e}")


def iterate_cacd_images(base_dir):
    """
    Yields (image_path, subject, age) for every .jpg under base_dir.
    CACD filename convention: <age>_<FirstName>_<LastName>_<index>.jpg
    e.g. 32_Aaron_Eckhart_0001.jpg -> subject='Aaron_Eckhart', age=32
    """
    for subject in os.listdir(base_dir):
        subject_dir = os.path.join(base_dir, subject)
        if os.path.isdir(subject_dir):
            for filename in os.listdir(subject_dir):
                if filename.lower().endswith(".jpg"):
                    age = int(filename.split("_")[0])
                    yield os.path.join(subject_dir, filename), subject, age


# --------------------
# Main extraction
# --------------------

def extract_embeddings_and_save(model_name, model,
                                dataset="fgnet",
                                input_dir=os.path.join("data", "processed"),
                                output_dir=os.path.join("data", "embeddings")):
    """
    Extracts per-image embeddings and saves them as individual .npy files.
    Also writes a combined <dataset>_embeddings.npy and <dataset>_metadata.json
    for the entire dataset.

    """
    extractor = embedding_extractors[model_name]

    if dataset.lower() == "fgnet":
        iterator = iterate_fgnet_images(os.path.join(input_dir, "FGNET"))
    elif dataset.lower() == "cacd":
        iterator = iterate_cacd_images(os.path.join(input_dir, "CACD"))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    save_dir = os.path.join(output_dir, model_name, dataset)
    os.makedirs(save_dir, exist_ok=True)

    # Resume from existing combined files if present
    embeddings_file = os.path.join(save_dir, f"{dataset}_embeddings.npy")
    metadata_file   = os.path.join(save_dir, f"{dataset}_metadata.json")

    embeddings = []
    metadata   = []
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        print("Resuming from previous progress...")
        embeddings = np.load(embeddings_file).tolist()
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    iterator = list(iterator)
    print(f"Found {len(iterator)} images in {dataset}")

    for img_path, identity, age in iterator:
        emb_filename = os.path.splitext(os.path.basename(img_path))[0] + ".npy"
        emb_filepath = os.path.join(save_dir, emb_filename)

        if os.path.exists(emb_filepath):
            print(f"Skipping (already processed): {img_path}")
            continue

        print("Processing:", img_path)
        try:
            emb = extractor(model, img_path)
            emb = emb.flatten()

            np.save(emb_filepath, emb)

            embeddings.append(emb.tolist())
            metadata.append({
                "file":            img_path,
                "id":              identity,
                "age":             age,
                "embedding_file":  emb_filename,
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Persist combined files
    np.save(embeddings_file, np.array(embeddings))
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {len(embeddings)} embeddings for {dataset} using {model_name}")
