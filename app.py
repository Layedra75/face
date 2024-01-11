import requests
import cv2
import numpy as np
from flask import Flask, request, jsonify
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inicializar el modelo FaceNet
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# Inicializar MTCNN para detección de rostros
mtcnn = MTCNN(keep_all=True, device=device)

def get_frame_from_url(url):
    response = requests.get(url)
    return np.asarray(bytearray(response.content), dtype="uint8")

def extract_and_match_frames(img_url, vid_url, max_frames=50):
    image = cv2.imdecode(get_frame_from_url(img_url), cv2.IMREAD_COLOR)
    video = cv2.VideoCapture(vid_url)
    
    if not video.isOpened():
        return "Error: Cannot open video file."

    frames_processed = 0

    # Obtener embedding facial de la imagen
    faces_image = mtcnn(image)
    if faces_image is None or len(faces_image) == 0:
        return {"match_status": "No se detectó un rostro en la imagen"}
    embeddings_image = model(faces_image.to(device)).detach().cpu().numpy()

    while frames_processed < max_frames:
        ret, frame = video.read()

        if not ret:
            break

        # Obtener embedding facial del video
        faces_video = mtcnn(frame)
        if faces_video is None or len(faces_video) == 0:
            continue
        embeddings_video = model(faces_video.to(device)).detach().cpu().numpy()

        # Calcular la distancia coseno entre los embeddings faciales
        distances = np.linalg.norm(embeddings_image - embeddings_video, axis=1)

        # Puedes ajustar este umbral según tus necesidades
        similarity_threshold = 0.7

        if np.min(distances) < similarity_threshold:
            # No llamar a video.release() aquí para permitir procesar más frames
            result = {"match_status": "Coinciden", "image_url": img_url, "video_url": vid_url}
            return result

        frames_processed += 1

    video.release()
    result = {"match_status": "No coinciden", "image_url": img_url, "video_url": vid_url}
    return result

@app.route('/', methods=['POST'])
def process_request():
    img_url = request.json['image_url']
    vid_url = request.json['video_url']
    return jsonify(extract_and_match_frames(img_url, vid_url))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
