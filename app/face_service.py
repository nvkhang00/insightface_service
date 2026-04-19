import insightface
import numpy as np
import cv2

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # CPU (GPU = 0)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = model.get(img)

    if len(faces) == 0:
        return None

    return faces[0].embedding

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)