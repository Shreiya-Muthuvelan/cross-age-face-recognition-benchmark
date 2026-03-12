# scripts/run_embeddings.py
# Loads all four DeepFace models and extracts embeddings for both datasets.
# Run from project root:  python scripts/run_embeddings.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepface import DeepFace
from scripts.extract_embeddings import extract_embeddings_and_save

# ---------- Load models ----------
facenet  = DeepFace.build_model('Facenet')
arcface  = DeepFace.build_model('ArcFace')
vggface  = DeepFace.build_model('VGG-Face')
openface = DeepFace.build_model('OpenFace')

print("All models loaded successfully!")

# ---------- FGNET ----------
extract_embeddings_and_save("facenet",  facenet,  dataset="fgnet")
extract_embeddings_and_save("arcface",  arcface,  dataset="fgnet")
extract_embeddings_and_save("vggface",  vggface,  dataset="fgnet")
extract_embeddings_and_save("openface", openface, dataset="fgnet")

# ---------- CACD ----------
extract_embeddings_and_save("facenet",  facenet,  dataset="cacd")
extract_embeddings_and_save("arcface",  arcface,  dataset="cacd")
extract_embeddings_and_save("vggface",  vggface,  dataset="cacd")
extract_embeddings_and_save("openface", openface, dataset="cacd")
