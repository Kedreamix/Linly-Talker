import os
import requests

# Set the host and port of the service
talker_service_host = os.environ.get("TALKER_SERVICE_HOST", "localhost")
talker_service_port = os.environ.get("TALKER_SERVICE_PORT", 8003)

# API endpoint URLs
CHANGE_MODEL_URL = f"http://{talker_service_host}:{talker_service_port}/talker_change_model/"
TALKER_RESPONSE_URL = f"http://{talker_service_host}:{talker_service_port}/talker_response/"

def change_talker_model(model_name):
    """Request to change the Talker model."""
    params = {"model_name": model_name}
    try:
        response = requests.post(CHANGE_MODEL_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Model change successful: {response.json()}")
    except requests.RequestException as e:
        print(f"Model change failed: {e}")

def request_talker_response(source_image_path, driven_audio_path, payload, output_video_path):
    """Request Talker to generate video."""
    
    # Prepare payload as form data (ensure all values are strings)
    data = {
        'preprocess_type': payload.get('preprocess_type', 'crop'),
        'is_still_mode': str(payload.get('is_still_mode', False)),
        'enhancer': str(payload.get('enhancer', False)),
        'batch_size': str(payload.get('batch_size', 4)),
        'size_of_image': str(payload.get('size_of_image', 256)),
        'pose_style': str(payload.get('pose_style', 0)),
        'facerender': payload.get('facerender', 'facevid2vid'),
        'exp_weight': str(payload.get('exp_weight', 1.0)),
        'blink_every': str(payload.get('blink_every', True)),
        'talker_method': payload.get('talker_method', 'SadTalker'),
        'fps': str(payload.get('fps', 30)),
    }

    # Prepare files for upload
    with open(source_image_path, 'rb') as image_file, open(driven_audio_path, 'rb') as audio_file:
        files = {
            'source_image': (os.path.basename(source_image_path), image_file, 'image/jpeg'),  # Adjust MIME type if necessary
            'driven_audio': (os.path.basename(driven_audio_path), audio_file, 'audio/wav'),  # Adjust MIME type if necessary
        }

        try:
            # Sending POST request to the server with form data and files
            response = requests.post(TALKER_RESPONSE_URL, data=data, files=files)
            response.raise_for_status()  # Raise an exception for HTTP errors

            with open(output_video_path, 'wb') as video_file:
                video_file.write(response.content)
            print(f"Talker response successful, video saved as: {output_video_path}")
        except requests.RequestException as e:
            print(f"Talker response failed: {e}")

if __name__ == "__main__":
    # Models to test
    models = [
        "SadTalker",
        "Wav2Lip",
        "Wav2Lipv2",
        "NeRFTalk",
    ]
    result_dir = "outputs"
    os.makedirs(result_dir, exist_ok=True)
    # Loop to change model and generate Talker response
    for model_name in models:
        print(f"Switching to model: {model_name}")
        change_talker_model(model_name)

        # Request Talker to generate video
        payload = {
            "preprocess_type": "crop",
            "is_still_mode": False,
            "enhancer": False,
            "batch_size": 4,
            "size_of_image": 256,
            "pose_style": 0,
            "facerender": "facevid2vid",
            "exp_weight": 1.0,
            "blink_every": True,
            "talker_method": model_name,
            "fps": 30,
        }
        request_talker_response(
            "inputs/example.png",
            "answer.wav",  # 确保音频文件路径正确
            payload, os.path.join(result_dir, f"output_video_{model_name}.mp4")
        )
        print("\n" + "-" * 50 + "\n")
