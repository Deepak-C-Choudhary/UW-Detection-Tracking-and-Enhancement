#-----------------------previous code means separate backend and react ----------------------


# from flask import Flask, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# from flask import render_template
# import os
# import cv2
# from ultralytics import YOLO
# from flask_cors import CORS
# import logging
# import traceback
# from colorama import init
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import supervision as sv
# from supervision.config import CLASS_NAME_DATA_FIELD
# import random

# # Initialize colorama
# init()

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Flask app initialization
# app = Flask(__name__)
# CORS(app)


# @app.route('/')
# def index():
#     return render_template('index.html')


# # Directory Configuration
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
# MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

# # File Configuration
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'webm'}
# MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

# app.config.update(
#     UPLOAD_FOLDER=UPLOAD_FOLDER,
#     PROCESSED_FOLDER=PROCESSED_FOLDER,
#     MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH
# )

# # Load YOLOv8 Model
# yolo_model = YOLO(MODEL_PATH)

# # DeepSort Tracker Setup
# tracker = DeepSort(max_age=30)  # Adjust max_age for better tracking

# # Ensure directories exist
# def ensure_directory(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#         logger.info(f"Created directory: {path}")

# # Check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Assign random colors to each class
# COLOR_MAP = {}

# def get_color_for_class(class_id):
#     if class_id not in COLOR_MAP:
#         COLOR_MAP[class_id] = tuple(random.choices(range(256), k=3))
#     return COLOR_MAP[class_id]

# # Draw detection boxes with labels
# def draw_detection_boxes(image, detections, class_names):
#     for box in detections.boxes:
#         if len(box.xyxy) > 0:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             confidence = float(box.conf[0]) if len(box.conf) > 0 else 0
#             class_id = int(box.cls[0]) if len(box.cls) > 0 else -1
#             class_name = class_names.get(class_id, f"Class {class_id}")

#             if confidence > 0.5:
#                 color = get_color_for_class(class_id)
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#                 label = f"{class_name}: {confidence:.2f}"
#                 cv2.putText(
#                     image,
#                     label,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     color,
#                     2
#                 )

# # CLAHE enhancement
# def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     l_clahe = clahe.apply(l)
#     lab_clahe = cv2.merge((l_clahe, a, b))
#     enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
#     return enhanced_img

# # Normalize coordinates
# def normalize_coordinates(box, image_shape):
#     height, width = image_shape[:2]
#     x1 = max(0, min(int(box[0]), width))
#     y1 = max(0, min(int(box[1]), height))
#     x2 = max(0, min(int(box[2]), width))
#     y2 = max(0, min(int(box[3]), height))

#     if x2 <= x1 or y2 <= y1:
#         return None

#     return [x1, y1, x2, y2]

# # Video processing with YOLOv8 and tracking
# def process_video_with_features(source_path, target_path):
#     video_info = sv.VideoInfo.from_video_path(source_path)
#     annotator = sv.BoundingBoxAnnotator(thickness=2)

#     cap = cv2.VideoCapture(source_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(target_path, fourcc, video_info.fps, (video_info.width, video_info.height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         enhanced_frame = apply_clahe(frame)
#         frame_rgb = enhanced_frame[..., ::-1]
#         results = yolo_model.predict(frame_rgb, conf=0.5)

#         detections = []
#         confidences = []
#         class_ids = []

#         for box in results[0].boxes:
#             bbox = box.xyxy[0].cpu().numpy()
#             normalized_bbox = normalize_coordinates(bbox, enhanced_frame.shape)
#             if normalized_bbox:
#                 detections.append(normalized_bbox)
#                 confidences.append(float(box.conf[0]))
#                 class_ids.append(int(box.cls[0]))

#         detections = np.array(detections)
#         confidences = np.array(confidences)
#         class_ids = np.array(class_ids)

#         tracks = tracker.update_tracks(detections, confidences, class_ids, enhanced_frame)

#         for track in tracks:
#             if track.is_confirmed():
#                 bbox = track.to_ltrb()
#                 normalized_bbox = normalize_coordinates(bbox, enhanced_frame.shape)
#                 if normalized_bbox:
#                     x1, y1, x2, y2 = normalized_bbox
#                     class_id = track.class_id
#                     track_id = track.track_id
#                     color = get_color_for_class(class_id)
#                     cv2.rectangle(
#                         enhanced_frame,
#                         (x1, y1),
#                         (x2, y2),
#                         color, 2
#                     )
#                     cv2.putText(
#                         enhanced_frame,
#                         f"ID: {track_id}, Class: {class_id}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         color, 2
#                     )

#         out.write(enhanced_frame)

#     cap.release()
#     out.release()

# # Additional folder configuration
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Ensure all required directories exist
# ensure_directory(DETECTION_FOLDER)
# ensure_directory(TRACKING_FOLDER)
# ensure_directory(ENHANCEMENT_FOLDER)
# ensure_directory(VIDEO_FOLDER)

# # Route for processing uploaded files
# @app.route('/api/process', methods=['POST'])
# def process_files():
#     try:
#         if 'images[]' not in request.files and 'videos[]' not in request.files:
#             return jsonify({"error": "No files provided"}), 400

#         processed_results = {'processedImages': [], 'processedVideos': []}

#         # Process images
#         if 'images[]' in request.files:
#             images = request.files.getlist('images[]')
#             for img in images:
#                 if img and allowed_file(img.filename):
#                     filename = secure_filename(img.filename)
#                     img_path = os.path.join(UPLOAD_FOLDER, filename)
#                     img.save(img_path)

#                     frame = cv2.imread(img_path)
#                     if frame is None:
#                         raise Exception(f"Error reading image file: {filename}")

#                     # Enhancement
#                     enhanced_frame = apply_clahe(frame)
#                     enhanced_filename = f"enhanced_{filename}"
#                     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, enhanced_filename)
#                     cv2.imwrite(enhancement_path, enhanced_frame)

#                     # Detection
#                     results = yolo_model.predict(enhanced_frame, conf=0.5)
#                     detection_frame = enhanced_frame.copy()
#                     draw_detection_boxes(detection_frame, results[0], yolo_model.names)

#                     detection_filename = f"detection_{filename}"
#                     detection_path = os.path.join(DETECTION_FOLDER, detection_filename)
#                     cv2.imwrite(detection_path, detection_frame)

#                     # Tracking
#                     tracking_frame = detection_frame.copy()
#                     detections = []
#                     confidences = []
#                     class_ids = []

#                     for box in results[0].boxes:
#                         bbox = box.xyxy[0].cpu().numpy()
#                         normalized_bbox = normalize_coordinates(bbox, tracking_frame.shape)
#                         if normalized_bbox:
#                             detections.append(normalized_bbox)
#                             confidences.append(float(box.conf[0]))
#                             class_ids.append(int(box.cls[0]))

#                     detections = np.array(detections)
#                     confidences = np.array(confidences)
#                     class_ids = np.array(class_ids)

#                     tracks = tracker.update_tracks(detections, confidences, class_ids, tracking_frame)

#                     for track in tracks:
#                         if track.is_confirmed():
#                             bbox = track.to_ltrb()
#                             normalized_bbox = normalize_coordinates(bbox, tracking_frame.shape)
#                             if normalized_bbox:
#                                 x1, y1, x2, y2 = normalized_bbox
#                                 class_id = track.class_id
#                                 track_id = track.track_id
#                                 color = get_color_for_class(class_id)
#                                 cv2.rectangle(
#                                     tracking_frame,
#                                     (x1, y1),
#                                     (x2, y2),
#                                     color, 2
#                                 )
#                                 cv2.putText(
#                                     tracking_frame,
#                                     f"ID: {track_id}, Class: {class_id}",
#                                     (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.5,
#                                     color, 2
#                                 )

#                     tracking_filename = f"tracking_{filename}"
#                     tracking_path = os.path.join(TRACKING_FOLDER, tracking_filename)
#                     cv2.imwrite(tracking_path, tracking_frame)

#                     processed_results['processedImages'].append({
#                         "enhancement": f'/processed/enhancement/{enhanced_filename}',
#                         "detection": f'/processed/detection/{detection_filename}',
#                         "tracking": f'/processed/tracking/{tracking_filename}'
#                     })

#         # Process videos
#         if 'videos[]' in request.files:
#             videos = request.files.getlist('videos[]')
#             for vid in videos:
#                 if vid and allowed_file(vid.filename):
#                     filename = secure_filename(vid.filename)
#                     vid_path = os.path.join(UPLOAD_FOLDER, filename)
#                     vid.save(vid_path)

#                     processed_filename = f"processed_{filename}"
#                     processed_path = os.path.join(VIDEO_FOLDER, processed_filename)
#                     process_video_with_features(vid_path, processed_path)

#                     processed_results['processedVideos'].append(f'/processed/video/{processed_filename}')

#         return jsonify(processed_results)

#     except Exception as e:
#         logger.error(f"Unexpected error in process_files: {str(e)}")
#         logger.error(traceback.format_exc())
#         return jsonify({"error": "An unexpected error occurred"}), 500

# # Route to serve processed files
# @app.route('/processed/<path:filename>', methods=['GET'])
# def serve_processed_file(filename):
#     filepath = os.path.join(PROCESSED_FOLDER, filename)
#     if os.path.exists(filepath):
#         return send_from_directory(PROCESSED_FOLDER, filename)
#     else:
#         logger.error(f"File not found: {filepath}")
#         return jsonify({"error": "File not found"}), 404

# if __name__ == '__main__':
#     ensure_directory(PROCESSED_FOLDER)
#     app.run(debug=True)








#######################working#######################################################################################


# from flask import Flask, request, jsonify, send_from_directory, render_template
# from werkzeug.utils import secure_filename
# import os
# import cv2
# from ultralytics import YOLO
# from flask_cors import CORS
# import logging
# import traceback
# from colorama import init
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import supervision as sv
# from supervision.config import CLASS_NAME_DATA_FIELD
# import random

# # Initialize colorama
# init()

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Flask app initialization
# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def index():
#     return render_template('index.html')

# # Directory Configuration
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
# MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

# # File Configuration
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'webm'}
# MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

# app.config.update(
#     UPLOAD_FOLDER=UPLOAD_FOLDER,
#     PROCESSED_FOLDER=PROCESSED_FOLDER,
#     MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH
# )

# # Load YOLOv8 Model
# yolo_model = YOLO(MODEL_PATH)

# # DeepSort Tracker Setup
# tracker = DeepSort(max_age=30)  # Adjust max_age for better tracking

# # Ensure directories exist
# def ensure_directory(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#         logger.info(f"Created directory: {path}")

# # Check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Assign random colors to each class
# COLOR_MAP = {}

# def get_color_for_class(class_id):
#     if class_id not in COLOR_MAP:
#         COLOR_MAP[class_id] = tuple(random.choices(range(256), k=3))
#     return COLOR_MAP[class_id]

# # Draw detection boxes with labels
# def draw_detection_boxes(image, detections, class_names):
#     for box in detections.boxes:
#         if len(box.xyxy) > 0:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             confidence = float(box.conf[0]) if len(box.conf) > 0 else 0
#             class_id = int(box.cls[0]) if len(box.cls) > 0 else -1
#             class_name = class_names.get(class_id, f"Class {class_id}")

#             if confidence > 0.5:
#                 color = get_color_for_class(class_id)
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#                 label = f"{class_name}: {confidence:.2f}"
#                 cv2.putText(
#                     image,
#                     label,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     color,
#                     2
#                 )

# # CLAHE enhancement
# def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     l_clahe = clahe.apply(l)
#     lab_clahe = cv2.merge((l_clahe, a, b))
#     enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
#     return enhanced_img

# # Normalize coordinates
# def normalize_coordinates(box, image_shape):
#     height, width = image_shape[:2]
#     x1 = max(0, min(int(box[0]), width))
#     y1 = max(0, min(int(box[1]), height))
#     x2 = max(0, min(int(box[2]), width))
#     y2 = max(0, min(int(box[3]), height))

#     if x2 <= x1 or y2 <= y1:
#         return None

#     return [x1, y1, x2, y2]

# def process_detections(results, frame):
#     """Helper function to process YOLO detections into the format expected by DeepSort"""
#     detections = []
#     confidences = []
#     class_ids = []
    
#     # Process each detection
#     for box in results[0].boxes:
#         if len(box.xyxy) > 0:
#             # Get the box coordinates
#             bbox = box.xyxy[0].cpu().numpy()
            
#             # Ensure bbox is in the correct format
#             normalized_bbox = normalize_coordinates(bbox, frame.shape)
#             if normalized_bbox:
#                 # Convert to [x1, y1, w, h] format for DeepSort
#                 x1, y1, x2, y2 = normalized_bbox
#                 w = x2 - x1
#                 h = y2 - y1
#                 detections.append([x1, y1, w, h])
                
#                 # Get confidence score
#                 confidence = float(box.conf[0]) if len(box.conf) > 0 else 0
#                 confidences.append(confidence)
                
#                 # Get class ID
#                 class_id = int(box.cls[0]) if len(box.cls) > 0 else -1
#                 class_ids.append(class_id)
    
#     # Convert to numpy arrays with proper shapes
#     if detections:
#         detections_array = np.array(detections, dtype=np.float64)
#         confidences_array = np.array(confidences, dtype=np.float64)
#         class_ids_array = np.array(class_ids, dtype=np.int64)
        
#         logger.debug(f"Detections shape: {detections_array.shape}")
#         logger.debug(f"Confidences shape: {confidences_array.shape}")
#         logger.debug(f"Class IDs shape: {class_ids_array.shape}")
        
#         return detections_array, confidences_array, class_ids_array
#     else:
#         # Return empty arrays if no detections
#         return np.empty((0, 4)), np.array([], dtype=np.float64), np.array([], dtype=np.int64)

# # Video processing with YOLOv8 and tracking
# def process_video_with_features(source_path, target_path):
#     video_info = sv.VideoInfo.from_video_path(source_path)
    
#     cap = cv2.VideoCapture(source_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(target_path, fourcc, video_info.fps, 
#                         (video_info.width, video_info.height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         enhanced_frame = apply_clahe(frame)
#         results = yolo_model.predict(enhanced_frame, conf=0.5)
        
#         # Process detections for DeepSort
#         detections, confidences, class_ids = process_detections(results, enhanced_frame)
        
#         # Only update tracks if we have detections
#         if len(detections) > 0:
#             tracks = tracker.update_tracks(detections, confidences, class_ids, enhanced_frame)
            
#             # Draw tracks
#             for track in tracks:
#                 if track.is_confirmed():
#                     bbox = track.to_ltrb()
#                     normalized_bbox = normalize_coordinates(bbox, enhanced_frame.shape)
#                     if normalized_bbox:
#                         x1, y1, x2, y2 = normalized_bbox
#                         class_id = track.class_id
#                         track_id = track.track_id
#                         color = get_color_for_class(class_id)
#                         cv2.rectangle(
#                             enhanced_frame,
#                             (x1, y1),
#                             (x2, y2),
#                             color, 2
#                         )
#                         cv2.putText(
#                             enhanced_frame,
#                             f"ID: {track_id}, Class: {class_id}",
#                             (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5,
#                             color, 2
#                         )

#         out.write(enhanced_frame)

#     cap.release()
#     out.release()

# # Additional folder configuration
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Ensure all required directories exist
# ensure_directory(DETECTION_FOLDER)
# ensure_directory(TRACKING_FOLDER)
# ensure_directory(ENHANCEMENT_FOLDER)
# ensure_directory(VIDEO_FOLDER)

# # Route for processing uploaded files
# @app.route('/api/process', methods=['POST'])
# def process_files():
#     try:
#         if 'images[]' not in request.files and 'videos[]' not in request.files:
#             return jsonify({"error": "No files provided"}), 400

#         processed_results = {'processedImages': [], 'processedVideos': []}

#         # Process images
#         if 'images[]' in request.files:
#             images = request.files.getlist('images[]')
#             for img in images:
#                 if img and allowed_file(img.filename):
#                     filename = secure_filename(img.filename)
#                     img_path = os.path.join(UPLOAD_FOLDER, filename)
#                     img.save(img_path)

#                     frame = cv2.imread(img_path)
#                     if frame is None:
#                         raise Exception(f"Error reading image file: {filename}")

#                     # Enhancement
#                     enhanced_frame = apply_clahe(frame)
#                     enhanced_filename = f"enhanced_{filename}"
#                     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, enhanced_filename)
#                     cv2.imwrite(enhancement_path, enhanced_frame)

#                     # Detection
#                     results = yolo_model.predict(enhanced_frame, conf=0.5)
#                     detection_frame = enhanced_frame.copy()
#                     draw_detection_boxes(detection_frame, results[0], yolo_model.names)

#                     detection_filename = f"detection_{filename}"
#                     detection_path = os.path.join(DETECTION_FOLDER, detection_filename)
#                     cv2.imwrite(detection_path, detection_frame)

#                     # Tracking
#                     tracking_frame = detection_frame.copy()
                    
#                     # Process detections for DeepSort
#                     detections, confidences, class_ids = process_detections(results, tracking_frame)
                    
#                     logger.debug(f"Detections: {detections}")
#                     logger.debug(f"Confidences: {confidences}")
#                     logger.debug(f"Class IDs: {class_ids}")
                    
#                     # Only update tracks if we have detections
#                     if len(detections) > 0:
#                         try:
#                             tracks = tracker.update_tracks(detections, confidences, class_ids, tracking_frame)
                            
#                             # Draw tracks
#                             for track in tracks:
#                                 if track.is_confirmed():
#                                     bbox = track.to_ltrb()
#                                     normalized_bbox = normalize_coordinates(bbox, tracking_frame.shape)
#                                     if normalized_bbox:
#                                         x1, y1, x2, y2 = normalized_bbox
#                                         class_id = track.class_id
#                                         track_id = track.track_id
#                                         color = get_color_for_class(class_id)
#                                         cv2.rectangle(
#                                             tracking_frame,
#                                             (x1, y1),
#                                             (x2, y2),
#                                             color, 2
#                                         )
#                                         cv2.putText(
#                                             tracking_frame,
#                                             f"ID: {track_id}, Class: {class_id}",
#                                             (x1, y1 - 10),
#                                             cv2.FONT_HERSHEY_SIMPLEX,
#                                             0.5,
#                                             color, 2
#                                         )
#                         except Exception as e:
#                             logger.error(f"Error in tracker.update_tracks: {str(e)}")
#                             logger.error(traceback.format_exc())
#                     else:
#                         logger.warning("No detections found in the image.")

#                     tracking_filename = f"tracking_{filename}"
#                     tracking_path = os.path.join(TRACKING_FOLDER, tracking_filename)
#                     cv2.imwrite(tracking_path, tracking_frame)

#                     processed_results['processedImages'].append({
#                         "enhancement": f'/processed/enhancement/{enhanced_filename}',
#                         "detection": f'/processed/detection/{detection_filename}',
#                         "tracking": f'/processed/tracking/{tracking_filename}'
#                     })

#         # Process videos
#         if 'videos[]' in request.files:
#             videos = request.files.getlist('videos[]')
#             for vid in videos:
#                 if vid and allowed_file(vid.filename):
#                     filename = secure_filename(vid.filename)
#                     vid_path = os.path.join(UPLOAD_FOLDER, filename)
#                     vid.save(vid_path)

#                     processed_filename = f"processed_{filename}"
#                     processed_path = os.path.join(VIDEO_FOLDER, processed_filename)
#                     process_video_with_features(vid_path, processed_path)

#                     processed_results['processedVideos'].append(f'/processed/video/{processed_filename}')

#         return jsonify(processed_results)

#     except Exception as e:
#         logger.error(f"Unexpected error in process_files: {str(e)}")
#         logger.error(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500

# # Route to serve processed files
# @app.route('/processed/<path:filename>', methods=['GET'])
# def serve_processed_file(filename):
#     # Split the path into directory and filename
#     directory = os.path.dirname(filename)
#     file = os.path.basename(filename)
    
#     # Construct the full directory path
#     full_dir = os.path.join(PROCESSED_FOLDER, directory)
    
#     if os.path.exists(os.path.join(full_dir, file)):
#         return send_from_directory(full_dir, file)
#     else:
#         logger.error(f"File not found: {os.path.join(full_dir, file)}")
#         return jsonify({"error": "File not found"}), 404

# if __name__ == '__main__':
#     ensure_directory(PROCESSED_FOLDER)
#     app.run(debug=True)







#aarohi org code ---------------------------------------------------------------------------------------


# import argparse
# import io
# from PIL import Image
# import datetime

# import torch
# import cv2
# import numpy as np
# import tensorflow as tf
# from re import DEBUG, sub
# from flask import Flask, render_template, request, redirect, send_file, url_for, Response
# from werkzeug.utils import secure_filename, send_from_directory
# import os
# import subprocess
# from subprocess import Popen
# import re
# import requests
# import shutil
# import time
# import glob


# from ultralytics import YOLO


# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return render_template('index.html')

    
# @app.route("/", methods=["GET", "POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath,'uploads',f.filename)
#             print("upload folder is ", filepath)
#             f.save(filepath)
#             global imgpath
#             predict_img.imgpath = f.filename
#             print("printing predict_img :::::: ", predict_img)
                                               
#             file_extension = f.filename.rsplit('.', 1)[1].lower() 
            
#             if file_extension == 'jpg':
#                 img = cv2.imread(filepath)

#                 # Perform the detection
#                 model = YOLO('best.pt')
#                 detections =  model(img, save=True) 
#                 return display(f.filename)
            
#             elif file_extension == 'mp4': 
#                 video_path = filepath  # replace with your video path
#                 cap = cv2.VideoCapture(video_path)

#                 # get video dimensions
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
#                 # Define the codec and create VideoWriter object
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
#                 # initialize the YOLOv8 model here
#                 model = YOLO('best.pt')
                
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break                                                      

#                     # do YOLOv9 detection on the frame here
#                     model = YOLO('best.pt')
#                     results = model(frame, save=True)  #working
#                     print(results)
#                     cv2.waitKey(1)

#                     res_plotted = results[0].plot()
#                     cv2.imshow("result", res_plotted)
                    
#                     # write the frame to the output video
#                     out.write(res_plotted)

#                     if cv2.waitKey(1) == ord('q'):
#                         break

#                 return video_feed()            


            
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
#     return render_template('index.html', image_path=image_path)
#     #return "done"



# # #The display function is used to serve the image or video from the folder_path directory.
# @app.route('/<path:filename>')
# def display(filename):
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     directory = folder_path+'/'+latest_subfolder    
#     print("printing directory: ",directory) 
#     files = os.listdir(directory)
#     latest_file = files[0]
    
#     print(latest_file)

#     filename = os.path.join(folder_path, latest_subfolder, latest_file)

#     file_extension = filename.rsplit('.', 1)[1].lower()

#     environ = request.environ
#     if file_extension == 'jpg':      
#         return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

#     else:
#         return "Invalid file format"
        
        
        

# def get_frame():
#     folder_path = os.getcwd()
#     mp4_files = 'output.mp4'
#     video = cv2.VideoCapture(mp4_files)  # detected video path
#     while True:
#         success, image = video.read()
#         if not success:
#             break
#         ret, jpeg = cv2.imencode('.jpg', image) 
      
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
#         time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# # function to display the detected objects video on html page
# @app.route("/video_feed")
# def video_feed():
#     print("function called")

#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     args = parser.parse_args()
#     model = YOLO('best.pt')
#     app.run(host="0.0.0.0", port=args.port) 








#working in video but not in images and track also but not display the video on ouput


# import argparse
# import os
# import cv2
# import time
# import json
# from flask import Flask, render_template, request, jsonify, Response
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# PROCESSED_FOLDER = "processed"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/api/process", methods=["POST"])
# def process_files():
#     if "images[]" not in request.files and "videos[]" not in request.files:
#         return jsonify({"error": "No files provided"}), 400

#     processed_images = []
#     processed_videos = []
#     model = YOLO("best.pt")  # Load YOLO model

#     # Process images
#     if "images[]" in request.files:
#         for file in request.files.getlist("images[]"):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(file_path)

#             # YOLO Detection
#             results = model(file_path, save=True)
#             output_path = os.path.join(PROCESSED_FOLDER, f"detection_{filename}")
#             results[0].save(save_dir=PROCESSED_FOLDER)

#             processed_images.append({
#                 "detection": f"/{PROCESSED_FOLDER}/detection_{filename}"
#             })

#     # Process videos
#     if "videos[]" in request.files:
#         for file in request.files.getlist("videos[]"):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(file_path)

#             # YOLO Video Processing
#             cap = cv2.VideoCapture(file_path)
#             frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             output_video_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
#             out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 results = model(frame)
#                 res_plotted = results[0].plot()
#                 out.write(res_plotted)

#             cap.release()
#             out.release()
#             processed_videos.append(f"/{output_video_path}")

#     response_data = {
#         "processedImages": processed_images,
#         "processedVideos": processed_videos,
#     }

#     return jsonify(response_data)

# @app.route(f"/{PROCESSED_FOLDER}/<path:filename>")
# def serve_processed_file(filename):
#     return send_from_directory(PROCESSED_FOLDER, filename)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
#     parser.add_argument("--port", default=5000, type=int, help="Port number")
#     args = parser.parse_args()

#     app.run(host="0.0.0.0", port=args.port, debug=True)











#generating id but with the same color 

# from flask import Flask, render_template, request, jsonify, Response, send_from_directory
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO
# import argparse
# import os
# import cv2
# import time
# import json
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# PROCESSED_FOLDER = "processed"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# # Initialize the tracker
# tracker = DeepSort(max_age=30)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/api/process", methods=["POST"])
# def process_files():
#     if "images[]" not in request.files and "videos[]" not in request.files:
#         return jsonify({"error": "No files provided"}), 400
    
#     processed_images = []
#     processed_videos = []
#     model = YOLO("best.pt")  # Load YOLO model
    
#     # Process images
#     if "images[]" in request.files:
#         for file in request.files.getlist("images[]"):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(file_path)
            
#             # YOLO Detection
#             results = model(file_path, save=True)
#             output_path = os.path.join(PROCESSED_FOLDER, f"detection_{filename}")
#             results[0].save(save_dir=PROCESSED_FOLDER)
            
#             processed_images.append({
#                 "detection": f"/{PROCESSED_FOLDER}/detection_{filename}"
#             })
    
#     # Process videos
#     if "videos[]" in request.files:
#         for file in request.files.getlist("videos[]"):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(file_path)
            
#             # YOLO Video Processing with Tracking
#             cap = cv2.VideoCapture(file_path)
#             frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             output_video_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
#             out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))
            
#             frame_count = 0
            
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 frame_count += 1
                
#                 # Run YOLO detection
#                 results = model(frame)
                
#                 # Get bounding boxes, scores, and class IDs
#                 boxes = results[0].boxes.xyxy.cpu().numpy()
#                 scores = results[0].boxes.conf.cpu().numpy()
#                 class_ids = results[0].boxes.cls.cpu().numpy()
                
#                 # Prepare detections for DeepSORT
#                 detections = []
#                 for box, score, class_id in zip(boxes, scores, class_ids):
#                     x1, y1, x2, y2 = box
#                     w = x2 - x1
#                     h = y2 - y1
#                     detections.append(([x1, y1, w, h], score, int(class_id)))
                
#                 # Update tracks
#                 tracks = tracker.update_tracks(detections, frame=frame)
                
#                 # Draw tracks on frame
#                 for track in tracks:
#                     if not track.is_confirmed():
#                         continue
                    
#                     track_id = track.track_id
#                     ltrb = track.to_ltrb()
                    
#                     # Draw bounding box
#                     cv2.rectangle(frame, 
#                                 (int(ltrb[0]), int(ltrb[1])), 
#                                 (int(ltrb[2]), int(ltrb[3])), 
#                                 (0, 255, 0), 2)
                    
#                     # Draw ID
#                     cv2.putText(frame, 
#                               f"ID: {track_id}", 
#                               (int(ltrb[0]), int(ltrb[1] - 10)), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 
#                               0.9, 
#                               (0, 255, 0), 
#                               2)
                
#                 out.write(frame)
            
#             cap.release()
#             out.release()
#             processed_videos.append(f"/{PROCESSED_FOLDER}/processed_{filename}")
    
#     response_data = {
#         "processedImages": processed_images,
#         "processedVideos": processed_videos,
#     }
    
#     return jsonify(response_data)

# @app.route(f"/{PROCESSED_FOLDER}/<path:filename>")
# def serve_processed_file(filename):
#     return send_from_directory(PROCESSED_FOLDER, filename)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
#     parser.add_argument("--port", default=5000, type=int, help="Port number")
#     args = parser.parse_args()
    
#     app.run(host="0.0.0.0", port=args.port, debug=True)









#working on images also on videos tracking is also done complete video but the ouput not displayed on the screen

# from flask import Flask, render_template, request, jsonify, send_file
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# import time
# from pathlib import Path
# import shutil

# app = Flask(__name__)

# # Configure directories based on the shown file structure
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Create all required directories
# for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     """Process a single image and return paths to processed outputs"""
#     img = cv2.imread(image_path)
#     filename = Path(image_path).stem
    
#     # Initialize YOLO model
#     model = YOLO('best.pt')
    
#     # Detection
#     detection_results = model(img, save=False)[0]
#     detection_img = detection_results.plot()
#     detection_path = os.path.join(DETECTION_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(detection_path, detection_img)
    
#     # Enhancement (in real application, implement actual enhancement)
#     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(enhancement_path, detection_img)
    
#     # Tracking (in real application, implement actual tracking)
#     tracking_path = os.path.join(TRACKING_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(tracking_path, detection_img)
    
#     return {
#         "detection": f"/processed/detection/{filename}.jpg",
#         "enhancement": f"/processed/enhancement/{filename}.jpg",
#         "tracking": f"/processed/tracking/{filename}.jpg"
#     }

# def process_video(video_path):
#     """Process a video and return path to processed output"""
#     filename = Path(video_path).stem
#     output_path = os.path.join(VIDEO_FOLDER, f"{filename}.mp4")
    
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
#     model = YOLO('best.pt')
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         results = model(frame, save=False)[0]
#         processed_frame = results.plot()
#         out.write(processed_frame)
    
#     cap.release()
#     out.release()
    
#     return f"/processed/video/{filename}.mp4"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/process', methods=['POST'])
# def process_files():
#     if 'images[]' not in request.files and 'videos[]' not in request.files:
#         return jsonify({'error': 'No files uploaded'}), 400
    
#     processed_images = []
#     processed_videos = []
    
#     # Process images
#     if 'images[]' in request.files:
#         image_files = request.files.getlist('images[]')
#         for file in image_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)
                
#                 try:
#                     processed_paths = process_image(filepath)
#                     processed_images.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing image {filename}: {str(e)}")
#                 finally:
#                     # Clean up upload after processing
#                     if os.path.exists(filepath):
#                         os.remove(filepath)
    
#     # Process videos
#     if 'videos[]' in request.files:
#         video_files = request.files.getlist('videos[]')
#         for file in video_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)
                
#                 try:
#                     processed_path = process_video(filepath)
#                     processed_videos.append(processed_path)
#                 except Exception as e:
#                     print(f"Error processing video {filename}: {str(e)}")
#                 finally:
#                     # Clean up upload after processing
#                     if os.path.exists(filepath):
#                         os.remove(filepath)
    
#     return jsonify({
#         'processedImages': processed_images,
#         'processedVideos': processed_videos
#     })

# # Add route to serve processed files
# @app.route('/processed/<path:filename>')
# def serve_processed_file(filename):
#     return send_file(os.path.join(PROCESSED_FOLDER, filename))

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)











#running save video in 3 form but in same video folder for full video

# from flask import Flask, render_template, request, jsonify, send_file
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# import time
# from pathlib import Path
# import shutil

# app = Flask(__name__)

# # Configure directories
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Create all required directories
# for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     """Process a single image and return paths to processed outputs"""
#     img = cv2.imread(image_path)
#     filename = Path(image_path).stem
    
#     # Initialize YOLO model
#     model = YOLO('best.pt')
    
#     # Detection
#     detection_results = model(img, save=False)[0]
#     detection_img = detection_results.plot()
#     detection_path = os.path.join(DETECTION_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(detection_path, detection_img)
    
#     # Enhancement (in real application, implement actual enhancement)
#     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(enhancement_path, detection_img)
    
#     # Tracking (in real application, implement actual tracking)
#     tracking_path = os.path.join(TRACKING_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(tracking_path, detection_img)
    
#     return {
#         "detection": f"/processed/detection/{filename}.jpg",
#         "enhancement": f"/processed/enhancement/{filename}.jpg",
#         "tracking": f"/processed/tracking/{filename}.jpg"
#     }

# def process_video(video_path):
#     """Process a video and return paths to processed outputs"""
#     filename = Path(video_path).stem
#     detection_path = os.path.join(VIDEO_FOLDER, f"{filename}_detection.mp4")
#     tracking_path = os.path.join(VIDEO_FOLDER, f"{filename}_tracking.mp4")
#     enhancement_path = os.path.join(VIDEO_FOLDER, f"{filename}_enhancement.mp4")
    
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_detection = cv2.VideoWriter(detection_path, fourcc, fps, (frame_width, frame_height))
#     out_tracking = cv2.VideoWriter(tracking_path, fourcc, fps, (frame_width, frame_height))
#     out_enhancement = cv2.VideoWriter(enhancement_path, fourcc, fps, (frame_width, frame_height))
    
#     model = YOLO('best.pt')
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         results = model(frame, save=False)[0]
#         processed_frame = results.plot()
        
#         # Write the same processed frame to all outputs (in real app, implement different processing)
#         out_detection.write(processed_frame)
#         out_tracking.write(processed_frame)
#         out_enhancement.write(processed_frame)
    
#     cap.release()
#     out_detection.release()
#     out_tracking.release()
#     out_enhancement.release()
    
#     return {
#         "detection": f"/processed/video/{filename}_detection.mp4",
#         "tracking": f"/processed/video/{filename}_tracking.mp4",
#         "enhancement": f"/processed/video/{filename}_enhancement.mp4"
#     }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/process', methods=['POST'])
# def process_files():
#     if 'images[]' not in request.files and 'videos[]' not in request.files:
#         return jsonify({'error': 'No files uploaded'}), 400
    
#     processed_images = []
#     processed_videos = []
    
#     # Process images
#     if 'images[]' in request.files:
#         image_files = request.files.getlist('images[]')
#         for file in image_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)
                
#                 try:
#                     processed_paths = process_image(filepath)
#                     processed_images.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing image {filename}: {str(e)}")
#                 finally:
#                     # Clean up upload after processing
#                     if os.path.exists(filepath):
#                         os.remove(filepath)
    
#     # Process videos
#     if 'videos[]' in request.files:
#         video_files = request.files.getlist('videos[]')
#         for file in video_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)
                
#                 try:
#                     processed_paths = process_video(filepath)
#                     processed_videos.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing video {filename}: {str(e)}")
#                 finally:
#                     # Clean up upload after processing
#                     if os.path.exists(filepath):
#                         os.remove(filepath)
    
#     return jsonify({
#         'processedImages': processed_images,
#         'processedVideos': processed_videos
#     })

# @app.route('/processed/<path:filename>')
# def serve_processed_file(filename):
#     return send_file(os.path.join(PROCESSED_FOLDER, filename))

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)














#working and displaying video on screen________________________________________________________________________________________________________________________________________________________________



# from flask import Flask, render_template, request, jsonify, send_file, url_for
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import shutil
# import mimetypes
# from werkzeug.utils import safe_join

# app = Flask(__name__)

# # Configure directories
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Create all required directories
# for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     """Process a single image and return paths to processed outputs"""
#     img = cv2.imread(image_path)
#     filename = Path(image_path).stem

#     # Initialize YOLO model
#     model = YOLO('best.pt')

#     # Detection
#     detection_results = model(img, save=False)[0]
#     detection_img = detection_results.plot()
#     detection_path = os.path.join(DETECTION_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(detection_path, detection_img)

#     # Enhancement (example: sharpening the image)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     enhanced_img = cv2.filter2D(detection_img, -1, kernel)
#     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(enhancement_path, enhanced_img)

#     # Tracking (copying detection output for simplicity, replace with tracking logic if needed)
#     tracking_path = os.path.join(TRACKING_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(tracking_path, detection_img)

#     return {
#         "detection": url_for('serve_processed_file', filename=f"detection/{filename}.jpg"),
#         "enhancement": url_for('serve_processed_file', filename=f"enhancement/{filename}.jpg"),
#         "tracking": url_for('serve_processed_file', filename=f"tracking/{filename}.jpg")
#     }

# def process_video(video_path):
#     """Process a video and return paths to processed outputs"""
#     filename = Path(video_path).stem
#     detection_path = os.path.join(VIDEO_FOLDER, f"{filename}_detection.mp4")
#     tracking_path = os.path.join(VIDEO_FOLDER, f"{filename}_tracking.mp4")
#     enhancement_path = os.path.join(VIDEO_FOLDER, f"{filename}_enhancement.mp4")

#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Make sure dimensions are even numbers
#     frame_width = frame_width if frame_width % 2 == 0 else frame_width - 1
#     frame_height = frame_height if frame_height % 2 == 0 else frame_height - 1

#     # Use avc1 codec instead of mp4v
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     out_detection = cv2.VideoWriter(detection_path, fourcc, fps, (frame_width, frame_height))
#     out_tracking = cv2.VideoWriter(tracking_path, fourcc, fps, (frame_width, frame_height))
#     out_enhancement = cv2.VideoWriter(enhancement_path, fourcc, fps, (frame_width, frame_height))

#     model = YOLO('best.pt')

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame, save=False)[0]
#         processed_frame = results.plot()

#         # Enhancement (example: increasing brightness)
#         enhanced_frame = cv2.convertScaleAbs(processed_frame, alpha=1.2, beta=30)

#         # Write the frames to respective outputs
#         out_detection.write(processed_frame)
#         out_tracking.write(processed_frame)  # Replace with tracking logic
#         out_enhancement.write(enhanced_frame)

#     cap.release()
#     out_detection.release()
#     out_tracking.release()
#     out_enhancement.release()

#     return {
#         "detection": url_for('serve_processed_file', filename=f"video/{filename}_detection.mp4"),
#         "tracking": url_for('serve_processed_file', filename=f"video/{filename}_tracking.mp4"),
#         "enhancement": url_for('serve_processed_file', filename=f"video/{filename}_enhancement.mp4")
#     }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/process', methods=['POST'])
# def process_files():
#     if 'images[]' not in request.files and 'videos[]' not in request.files:
#         return jsonify({'error': 'No files uploaded'}), 400

#     processed_images = []
#     processed_videos = []

#     # Process images
#     if 'images[]' in request.files:
#         image_files = request.files.getlist('images[]')
#         for file in image_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)

#                 try:
#                     processed_paths = process_image(filepath)
#                     processed_images.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing image {filename}: {str(e)}")
#                 finally:
#                     if os.path.exists(filepath):
#                         os.remove(filepath)

#     # Process videos
#     if 'videos[]' in request.files:
#         video_files = request.files.getlist('videos[]')
#         for file in video_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)

#                 try:
#                     processed_paths = process_video(filepath)
#                     processed_videos.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing video {filename}: {str(e)}")
#                 finally:
#                     if os.path.exists(filepath):
#                         os.remove(filepath)

#     return jsonify({
#         'processedImages': processed_images,
#         'processedVideos': processed_videos
#     })

# @app.route('/processed/<path:filename>')
# def serve_processed_file(filename):
#     try:
#         # Safely join the path to prevent directory traversal
#         file_path = safe_join(PROCESSED_FOLDER, filename)
        
#         if not file_path or not os.path.exists(file_path):
#             return jsonify({'error': 'File not found'}), 404

#         # Determine the correct MIME type
#         mime_type, _ = mimetypes.guess_type(file_path)
#         if not mime_type:
#             mime_type = 'application/octet-stream'
            
#         # Special handling for MP4 files
#         if filename.endswith('.mp4'):
#             mime_type = 'video/mp4'

#         # Send file with proper MIME type and enable partial content support
#         return send_file(
#             file_path,
#             mimetype=mime_type,
#             as_attachment=False,
#             conditional=True  # Enable partial content support
#         )

#     except Exception as e:
#         print(f"Error serving file {filename}: {str(e)}")
#         return jsonify({'error': 'Error serving file'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)





#working and displayin and also gives unique id to each object

# from flask import Flask, render_template, request, jsonify, send_file, url_for
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import shutil
# import mimetypes
# from werkzeug.utils import safe_join

# app = Flask(__name__)

# # Configure directories
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Create all required directories
# for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     """Process a single image and return paths to processed outputs"""
#     img = cv2.imread(image_path)
#     filename = Path(image_path).stem

#     # Initialize YOLO model
#     model = YOLO('best.pt')

#     # Detection
#     detection_results = model(img, save=False)[0]
#     detection_img = detection_results.plot()
#     detection_path = os.path.join(DETECTION_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(detection_path, detection_img)

#     # Enhancement (example: sharpening the image)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     enhanced_img = cv2.filter2D(detection_img, -1, kernel)
#     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(enhancement_path, enhanced_img)

#     # Tracking (copying detection output for simplicity, replace with tracking logic if needed)
#     tracking_path = os.path.join(TRACKING_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(tracking_path, detection_img)

#     return {
#         "detection": url_for('serve_processed_file', filename=f"detection/{filename}.jpg"),
#         "enhancement": url_for('serve_processed_file', filename=f"enhancement/{filename}.jpg"),
#         "tracking": url_for('serve_processed_file', filename=f"tracking/{filename}.jpg")
#     }

# def process_video(video_path):
#     """Process a video and return paths to processed outputs"""
#     filename = Path(video_path).stem
#     detection_path = os.path.join(VIDEO_FOLDER, f"{filename}_detection.mp4")
#     tracking_path = os.path.join(VIDEO_FOLDER, f"{filename}_tracking.mp4")
#     enhancement_path = os.path.join(VIDEO_FOLDER, f"{filename}_enhancement.mp4")

#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Make sure dimensions are even numbers
#     frame_width = frame_width if frame_width % 2 == 0 else frame_width - 1
#     frame_height = frame_height if frame_height % 2 == 0 else frame_height - 1

#     # Use avc1 codec instead of mp4v
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     out_detection = cv2.VideoWriter(detection_path, fourcc, fps, (frame_width, frame_height))
#     out_tracking = cv2.VideoWriter(tracking_path, fourcc, fps, (frame_width, frame_height))
#     out_enhancement = cv2.VideoWriter(enhancement_path, fourcc, fps, (frame_width, frame_height))

#     model = YOLO('best.pt')

#     # Dictionary to store object IDs and their frame counts
#     object_ids = {}
#     next_id = 1

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         results = model(frame, save=False)[0]
        
#         # Process detections and assign/update IDs
#         for det in results.boxes.data:
#             x1, y1, x2, y2, conf, cls = det[:6]
#             center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
#             matched_id = None
#             for obj_id, (obj_center, frame_num) in object_ids.items():
#                 distance = np.sqrt((center[0] - obj_center[0])**2 + (center[1] - obj_center[1])**2)
#                 if distance < 50:  # Adjust this threshold as needed
#                     matched_id = obj_id
#                     break
            
#             if matched_id is None:
#                 matched_id = next_id
#                 next_id += 1
            
#             object_ids[matched_id] = (center, frame_count)
        
#         # Remove IDs that haven't been seen for more than 10 frames
#         object_ids = {k: v for k, v in object_ids.items() if frame_count - v[1] <= 10}
        
#         # Plot results with IDs
#         processed_frame = results.plot()
#         for det in results.boxes.data:
#             x1, y1, x2, y2, conf, cls = det[:6]
#             center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
#             for obj_id, (obj_center, _) in object_ids.items():
#                 if np.sqrt((center[0] - obj_center[0])**2 + (center[1] - obj_center[1])**2) < 50:
#                     cv2.putText(processed_frame, f"ID: {obj_id}", (int(x1), int(y1) - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                     break

#         # Enhancement (example: increasing brightness)
#         enhanced_frame = cv2.convertScaleAbs(processed_frame, alpha=1.2, beta=30)

#         # Write the frames to respective outputs
#         out_detection.write(processed_frame)
#         out_tracking.write(processed_frame)
#         out_enhancement.write(enhanced_frame)

#     cap.release()
#     out_detection.release()
#     out_tracking.release()
#     out_enhancement.release()

#     return {
#         "detection": url_for('serve_processed_file', filename=f"video/{filename}_detection.mp4"),
#         "tracking": url_for('serve_processed_file', filename=f"video/{filename}_tracking.mp4"),
#         "enhancement": url_for('serve_processed_file', filename=f"video/{filename}_enhancement.mp4")
#     }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/process', methods=['POST'])
# def process_files():
#     if 'images[]' not in request.files and 'videos[]' not in request.files:
#         return jsonify({'error': 'No files uploaded'}), 400

#     processed_images = []
#     processed_videos = []

#     # Process images
#     if 'images[]' in request.files:
#         image_files = request.files.getlist('images[]')
#         for file in image_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)

#                 try:
#                     processed_paths = process_image(filepath)
#                     processed_images.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing image {filename}: {str(e)}")
#                 finally:
#                     if os.path.exists(filepath):
#                         os.remove(filepath)

#     # Process videos
#     if 'videos[]' in request.files:
#         video_files = request.files.getlist('videos[]')
#         for file in video_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)

#                 try:
#                     processed_paths = process_video(filepath)
#                     processed_videos.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing video {filename}: {str(e)}")
#                 finally:
#                     if os.path.exists(filepath):
#                         os.remove(filepath)

#     return jsonify({
#         'processedImages': processed_images,
#         'processedVideos': processed_videos
#     })

# @app.route('/processed/<path:filename>')
# def serve_processed_file(filename):
#     try:
#         # Safely join the path to prevent directory traversal
#         file_path = safe_join(PROCESSED_FOLDER, filename)
        
#         if not file_path or not os.path.exists(file_path):
#             return jsonify({'error': 'File not found'}), 404

#         # Determine the correct MIME type
#         mime_type, _ = mimetypes.guess_type(file_path)
#         if not mime_type:
#             mime_type = 'application/octet-stream'
            
#         # Special handling for MP4 files
#         if filename.endswith('.mp4'):
#             mime_type = 'video/mp4'

#         # Send file with proper MIME type and enable partial content support
#         return send_file(
#             file_path,
#             mimetype=mime_type,
#             as_attachment=False,
#             conditional=True  # Enable partial content support
#         )

#     except Exception as e:
#         print(f"Error serving file {filename}: {str(e)}")
#         return jsonify({'error': 'Error serving file'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)






#______________________Working everything fine______________________________________#

# from flask import Flask, render_template, request, jsonify, send_file, url_for
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import shutil
# import mimetypes
# from werkzeug.utils import safe_join

# app = Flask(__name__)

# # Configure directories
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Create all required directories
# for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     """Process a single image and return paths to processed outputs"""
#     img = cv2.imread(image_path)
#     filename = Path(image_path).stem

#     # Initialize YOLO model
#     model = YOLO('best.pt')

#     # Detection
#     detection_results = model(img, save=False)[0]
#     detection_img = detection_results.plot()
#     detection_path = os.path.join(DETECTION_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(detection_path, detection_img)

#     # Enhancement (example: sharpening the image)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     enhanced_img = cv2.filter2D(detection_img, -1, kernel)
#     enhancement_path = os.path.join(ENHANCEMENT_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(enhancement_path, enhanced_img)

#     # Tracking (copying detection output for simplicity, replace with tracking logic if needed)
#     tracking_path = os.path.join(TRACKING_FOLDER, f"{filename}.jpg")
#     cv2.imwrite(tracking_path, detection_img)

#     return {
#         "detection": url_for('serve_processed_file', filename=f"detection/{filename}.jpg"),
#         "enhancement": url_for('serve_processed_file', filename=f"enhancement/{filename}.jpg"),
#         "tracking": url_for('serve_processed_file', filename=f"tracking/{filename}.jpg")
#     }

# def process_video(video_path):
#     """Process a video and return paths to processed outputs"""
#     filename = Path(video_path).stem
#     detection_path = os.path.join(VIDEO_FOLDER, f"{filename}_detection.mp4")
#     tracking_path = os.path.join(VIDEO_FOLDER, f"{filename}_tracking.mp4")
#     enhancement_path = os.path.join(VIDEO_FOLDER, f"{filename}_enhancement.mp4")

#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Make sure dimensions are even numbers
#     frame_width = frame_width if frame_width % 2 == 0 else frame_width - 1
#     frame_height = frame_height if frame_height % 2 == 0 else frame_height - 1

#     # Use avc1 codec instead of mp4v
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     out_detection = cv2.VideoWriter(detection_path, fourcc, fps, (frame_width, frame_height))
#     out_tracking = cv2.VideoWriter(tracking_path, fourcc, fps, (frame_width, frame_height))
#     out_enhancement = cv2.VideoWriter(enhancement_path, fourcc, fps, (frame_width, frame_height))

#     model = YOLO('best.pt')

#     # Dictionary to store object IDs and their frame counts
#     object_ids = {}
#     next_id = 1

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         results = model(frame, save=False)[0]
        
#         # Process detections and assign/update IDs
#         for det in results.boxes.data:
#             x1, y1, x2, y2, conf, cls = det[:6]
#             center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
#             matched_id = None
#             for obj_id, (obj_center, frame_num) in object_ids.items():
#                 distance = np.sqrt((center[0] - obj_center[0])**2 + (center[1] - obj_center[1])**2)
#                 if distance < 50:  # Adjust this threshold as needed
#                     matched_id = obj_id
#                     break
            
#             if matched_id is None:
#                 matched_id = next_id
#                 next_id += 1
            
#             object_ids[matched_id] = (center, frame_count)
        
#         # Remove IDs that haven't been seen for more than 10 frames
#         object_ids = {k: v for k, v in object_ids.items() if frame_count - v[1] <= 10}
        
#         # Plot results with IDs
#         processed_frame = results.plot()
#         for det in results.boxes.data:
#             x1, y1, x2, y2, conf, cls = det[:6]
#             center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
#             for obj_id, (obj_center, _) in object_ids.items():
#                 if np.sqrt((center[0] - obj_center[0])**2 + (center[1] - obj_center[1])**2) < 50:
#                     cv2.putText(processed_frame, f"ID: {obj_id}", (int(x1), int(y1) - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                     break

#         # Add unique ID count to the left corner
#         unique_id_count = len(object_ids)
#         cv2.putText(processed_frame, f"Unique IDs: {unique_id_count}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Enhancement (example: increasing brightness)
#         enhanced_frame = cv2.convertScaleAbs(processed_frame, alpha=1.2, beta=30)

#         # Write the frames to respective outputs
#         out_detection.write(processed_frame)
#         out_tracking.write(processed_frame)
#         out_enhancement.write(enhanced_frame)

#     cap.release()
#     out_detection.release()
#     out_tracking.release()
#     out_enhancement.release()

#     return {
#         "detection": url_for('serve_processed_file', filename=f"video/{filename}_detection.mp4"),
#         "tracking": url_for('serve_processed_file', filename=f"video/{filename}_tracking.mp4"),
#         "enhancement": url_for('serve_processed_file', filename=f"video/{filename}_enhancement.mp4")
#     }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/process', methods=['POST'])
# def process_files():
#     if 'images[]' not in request.files and 'videos[]' not in request.files:
#         return jsonify({'error': 'No files uploaded'}), 400

#     processed_images = []
#     processed_videos = []

#     # Process images
#     if 'images[]' in request.files:
#         image_files = request.files.getlist('images[]')
#         for file in image_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)

#                 try:
#                     processed_paths = process_image(filepath)
#                     processed_images.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing image {filename}: {str(e)}")
#                 finally:
#                     if os.path.exists(filepath):
#                         os.remove(filepath)

#     # Process videos
#     if 'videos[]' in request.files:
#         video_files = request.files.getlist('videos[]')
#         for file in video_files:
#             if file and allowed_file(file.filename):
#                 filename = Path(file.filename).name
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)

#                 try:
#                     processed_paths = process_video(filepath)
#                     processed_videos.append(processed_paths)
#                 except Exception as e:
#                     print(f"Error processing video {filename}: {str(e)}")
#                 finally:
#                     if os.path.exists(filepath):
#                         os.remove(filepath)

#     return jsonify({
#         'processedImages': processed_images,
#         'processedVideos': processed_videos
#     })

# @app.route('/processed/<path:filename>')
# def serve_processed_file(filename):
#     try:
#         # Safely join the path to prevent directory traversal
#         file_path = safe_join(PROCESSED_FOLDER, filename)
        
#         if not file_path or not os.path.exists(file_path):
#             return jsonify({'error': 'File not found'}), 404

#         # Determine the correct MIME type
#         mime_type, _ = mimetypes.guess_type(file_path)
#         if not mime_type:
#             mime_type = 'application/octet-stream'
            
#         # Special handling for MP4 files
#         if filename.endswith('.mp4'):
#             mime_type = 'video/mp4'

#         # Send file with proper MIME type and enable partial content support
#         return send_file(
#             file_path,
#             mimetype=mime_type,
#             as_attachment=False,
#             conditional=True  # Enable partial content support
#         )

#     except Exception as e:
#         print(f"Error serving file {filename}: {str(e)}")
#         return jsonify({'error': 'Error serving file'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)









# _________________________Live stream working code detect, shows the output on UI as well______________

# from flask import Flask, render_template, request, jsonify, Response
# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import threading
# import queue
# import time

# app = Flask(__name__)

# # Configure directories
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
# ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
# TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
# VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# # Create directories if they don't exist
# for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# # Global variables for stream management
# stream_camera = None
# processing_thread = None
# stream_active = False
# frame_queues = {
#     'preview': queue.Queue(maxsize=10),
#     'detection': queue.Queue(maxsize=10),
#     'tracking': queue.Queue(maxsize=10),
#     'enhancement': queue.Queue(maxsize=10)
# }

# # Initialize models
# yolo_model = YOLO('best.pt')
# tracker = DeepSort(
#     max_age=30,
#     n_init=3,
#     nms_max_overlap=1.0,
#     max_cosine_distance=0.3,
#     nn_budget=None,
#     embedder="mobilenet",
#     half=True,
#     bgr=True,
#     embedder_gpu=True
# )

# def apply_clahe(frame):
#     """Apply CLAHE enhancement to frame"""
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     enhanced = cv2.merge((cl,a,b))
#     return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# def process_frame(frame):
#     """Process a single frame with detection, tracking, and enhancement"""
#     try:
#         # Create copies for each output
#         detection_frame = frame.copy()
#         tracking_frame = frame.copy()
        
#         # Detection
#         results = yolo_model(frame)
#         detection_frame = results[0].plot()
        
#         # Prepare detections for tracking
#         detections = []
#         if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
#             boxes = results[0].boxes
#             for box in boxes:
#                 try:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = float(box.conf[0])
#                     cls = int(box.cls[0])
#                     detections.append(([x1, y1, x2, y2], conf, cls))
#                 except Exception as e:
#                     print(f"Error processing detection: {str(e)}")
#                     continue
        
#         # Update tracks
#         if detections:
#             tracks = tracker.update_tracks(detections, frame=frame)
            
#             # Draw tracks
#             for track in tracks:
#                 if not track.is_confirmed():
#                     continue
                
#                 try:
#                     track_id = track.track_id
#                     ltrb = track.to_ltrb()
                    
#                     if ltrb is not None and len(ltrb) == 4:
#                         x1, y1, x2, y2 = map(int, ltrb)
                        
#                         # Draw bounding box
#                         cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
#                         # Draw ID and class
#                         label = f"ID: {track_id}"
#                         cv2.putText(tracking_frame, label,
#                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
#                                   0.5, (0, 255, 0), 2)
#                 except Exception as e:
#                     print(f"Error drawing track: {str(e)}")
#                     continue
            
#             # Add object count
#             try:
#                 confirmed_tracks = [t for t in tracks if t.is_confirmed()]
#                 cv2.putText(tracking_frame, f"Objects: {len(confirmed_tracks)}",
#                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                            1, (0, 255, 0), 2)
#             except Exception as e:
#                 print(f"Error adding object count: {str(e)}")
        
#         # Enhancement
#         enhanced_frame = apply_clahe(frame.copy())
        
#         return {
#             'preview': frame,
#             'detection': detection_frame,
#             'tracking': tracking_frame,
#             'enhancement': enhanced_frame
#         }
#     except Exception as e:
#         print(f"Error in process_frame: {str(e)}")
#         return None

# def stream_processor():
#     """Process frames from the camera stream"""
#     global stream_active
#     while stream_active and stream_camera is not None:
#         ret, frame = stream_camera.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
            
#         try:
#             processed_frames = process_frame(frame)
#             if processed_frames:
#                 # Update each frame queue
#                 for frame_type, processed_frame in processed_frames.items():
#                     try:
#                         if not frame_queues[frame_type].full():
#                             frame_queues[frame_type].put(processed_frame)
#                     except Exception as e:
#                         print(f"Error updating {frame_type} queue: {str(e)}")
#                         continue
            
#         except Exception as e:
#             print(f"Error processing frame: {str(e)}")
#             continue
            
#         time.sleep(0.01)  # Small delay to prevent overload

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/start_stream', methods=['GET'])
# def start_stream():
#     global stream_camera, processing_thread, stream_active
    
#     ip_address = request.args.get('ip')
#     if not ip_address:
#         return jsonify({'error': 'IP address is required'}), 400
        
#     try:
#         # Try different common IP webcam URLs
#         urls = [
#             f"http://{ip_address}:8080/video",
#             f"http://{ip_address}:8080/videofeed",
#             f"http://{ip_address}/video",
#             f"http://{ip_address}/videofeed"
#         ]
        
#         connected = False
#         for url in urls:
#             print(f"Trying to connect to: {url}")
#             stream_camera = cv2.VideoCapture(url)
#             if stream_camera.isOpened():
#                 connected = True
#                 print(f"Successfully connected to: {url}")
#                 break
                
#         if not connected:
#             return jsonify({'error': 'Failed to connect to camera'}), 400
            
#         # Clear all frame queues
#         for queue in frame_queues.values():
#             while not queue.empty():
#                 queue.get()
                
#         stream_active = True
#         processing_thread = threading.Thread(target=stream_processor)
#         processing_thread.start()
        
#         return jsonify({'streams': {
#             'preview': '/video_feed/preview',
#             'detection': '/video_feed/detection',
#             'tracking': '/video_feed/tracking',
#             'enhancement': '/video_feed/enhancement'
#         }}), 200
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/stop_stream', methods=['GET'])
# def stop_stream():
#     global stream_camera, processing_thread, stream_active
    
#     try:
#         stream_active = False
        
#         if processing_thread:
#             processing_thread.join(timeout=1.0)
            
#         if stream_camera:
#             stream_camera.release()
            
#         # Clear all frame queues
#         for queue in frame_queues.values():
#             while not queue.empty():
#                 queue.get()
                
#         return jsonify({'message': 'Stream stopped successfully'}), 200
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/video_feed/<feed_type>')
# def video_feed(feed_type):
#     """Video streaming route for specific feed type"""
#     if feed_type not in frame_queues:
#         return "Invalid feed type", 400
        
#     def generate():
#         while True:
#             if not frame_queues[feed_type].empty():
#                 frame = frame_queues[feed_type].get()
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#             else:
#                 time.sleep(0.01)
                
#     return Response(
#         generate(),
#         mimetype='multipart/x-mixed-replace; boundary=frame'
#     )

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)




# _______________________integrating both the features of real time streaming and upload files___________

from flask import Flask, render_template, request, jsonify, Response, send_file, url_for
import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import queue
import time
from pathlib import Path
from werkzeug.utils import safe_join
import mimetypes

app = Flask(__name__)

# Configure directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
DETECTION_FOLDER = os.path.join(PROCESSED_FOLDER, 'detection')
ENHANCEMENT_FOLDER = os.path.join(PROCESSED_FOLDER, 'enhancement')
TRACKING_FOLDER = os.path.join(PROCESSED_FOLDER, 'tracking')
VIDEO_FOLDER = os.path.join(PROCESSED_FOLDER, 'video')

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, DETECTION_FOLDER, ENHANCEMENT_FOLDER, TRACKING_FOLDER, VIDEO_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for stream management
stream_camera = None
processing_thread = None
stream_active = False
frame_queues = {
    'preview': queue.Queue(maxsize=10),
    'detection': queue.Queue(maxsize=10),
    'tracking': queue.Queue(maxsize=10),
    'enhancement': queue.Queue(maxsize=10)
}

# Initialize models
yolo_model = YOLO('best.pt')
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_clahe(frame):
    """Apply CLAHE enhancement to frame"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def process_frame(frame):
    """Process a single frame with detection, tracking, and enhancement"""
    try:
        # Create copies for each output
        detection_frame = frame.copy()
        tracking_frame = frame.copy()
        
        # Detection
        results = yolo_model(frame)
        detection_frame = results[0].plot()
        
        # Prepare detections for tracking
        detections = []
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    detections.append(([x1, y1, x2, y2], conf, cls))
                except Exception as e:
                    print(f"Error processing detection: {str(e)}")
                    continue
        
        # Update tracks
        if detections:
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Draw tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                try:
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    
                    if ltrb is not None and len(ltrb) == 4:
                        x1, y1, x2, y2 = map(int, ltrb)
                        
                        # Draw bounding box
                        cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw ID and class
                        label = f"ID: {track_id}"
                        cv2.putText(tracking_frame, label,
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error drawing track: {str(e)}")
                    continue
            
            # Add object count
            try:
                confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                cv2.putText(tracking_frame, f"Objects: {len(confirmed_tracks)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error adding object count: {str(e)}")
        
        # Enhancement
        enhanced_frame = apply_clahe(frame.copy())
        
        return {
            'preview': frame,
            'detection': detection_frame,
            'tracking': tracking_frame,
            'enhancement': enhanced_frame
        }
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None
#
def process_image(image_path):
    """Process a single image and return paths to processed outputs"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
            
        filename = Path(image_path).stem
        processed_frames = process_frame(img)
        
        if processed_frames is None:
            raise ValueError("Failed to process image")
            
        # Save processed outputs
        detection_path = os.path.join(DETECTION_FOLDER, f"{filename}.jpg")
        tracking_path = os.path.join(TRACKING_FOLDER, f"{filename}.jpg")
        enhancement_path = os.path.join(ENHANCEMENT_FOLDER, f"{filename}.jpg")
        
        cv2.imwrite(detection_path, processed_frames['detection'])
        cv2.imwrite(tracking_path, processed_frames['tracking'])
        cv2.imwrite(enhancement_path, processed_frames['enhancement'])
        
        return {
            "detection": url_for('serve_processed_file', filename=f"detection/{filename}.jpg"),
            "tracking": url_for('serve_processed_file', filename=f"tracking/{filename}.jpg"),
            "enhancement": url_for('serve_processed_file', filename=f"enhancement/{filename}.jpg")
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_video(video_path):
    """Process a video and return paths to processed outputs"""
    try:
        filename = Path(video_path).stem
        detection_path = os.path.join(VIDEO_FOLDER, f"{filename}_detection.mp4")
        tracking_path = os.path.join(VIDEO_FOLDER, f"{filename}_tracking.mp4")
        enhancement_path = os.path.join(VIDEO_FOLDER, f"{filename}_enhancement.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Ensure dimensions are even
        frame_width = frame_width if frame_width % 2 == 0 else frame_width - 1
        frame_height = frame_height if frame_height % 2 == 0 else frame_height - 1

        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_detection = cv2.VideoWriter(detection_path, fourcc, fps, (frame_width, frame_height))
        out_tracking = cv2.VideoWriter(tracking_path, fourcc, fps, (frame_width, frame_height))
        out_enhancement = cv2.VideoWriter(enhancement_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames = process_frame(frame)
            if processed_frames:
                out_detection.write(processed_frames['detection'])
                out_tracking.write(processed_frames['tracking'])
                out_enhancement.write(processed_frames['enhancement'])

        # Release resources
        cap.release()
        out_detection.release()
        out_tracking.release()
        out_enhancement.release()

        return {
            "detection": url_for('serve_processed_file', filename=f"video/{filename}_detection.mp4"),
            "tracking": url_for('serve_processed_file', filename=f"video/{filename}_tracking.mp4"),
            "enhancement": url_for('serve_processed_file', filename=f"video/{filename}_enhancement.mp4")
        }
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def stream_processor():
    """Process frames from the camera stream"""
    global stream_active
    while stream_active and stream_camera is not None:
        ret, frame = stream_camera.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        try:
            processed_frames = process_frame(frame)
            if processed_frames:
                # Update each frame queue
                for frame_type, processed_frame in processed_frames.items():
                    try:
                        if not frame_queues[frame_type].full():
                            frame_queues[frame_type].put(processed_frame)
                    except Exception as e:
                        print(f"Error updating {frame_type} queue: {str(e)}")
                        continue
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
            
        time.sleep(0.01)  # Small delay to prevent overload

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_stream', methods=['GET'])
def start_stream():
    global stream_camera, processing_thread, stream_active
    
    ip_address = request.args.get('ip')
    if not ip_address:
        return jsonify({'error': 'IP address is required'}), 400
        
    try:
        # Try different common IP webcam URLs
        urls = [
            f"http://{ip_address}:8080/video",
            f"http://{ip_address}:8080/videofeed",
            f"http://{ip_address}/video",
            f"http://{ip_address}/videofeed"
        ]
        
        connected = False
        for url in urls:
            print(f"Trying to connect to: {url}")
            stream_camera = cv2.VideoCapture(url)
            if stream_camera.isOpened():
                connected = True
                print(f"Successfully connected to: {url}")
                break
                
        if not connected:
            return jsonify({'error': 'Failed to connect to camera'}), 400
            
        # Clear all frame queues
        for queue in frame_queues.values():
            while not queue.empty():
                queue.get()
                
        stream_active = True
        processing_thread = threading.Thread(target=stream_processor)
        processing_thread.start()
        
        return jsonify({'streams': {
            'preview': '/video_feed/preview',
            'detection': '/video_feed/detection',
            'tracking': '/video_feed/tracking',
            'enhancement': '/video_feed/enhancement'
        }}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_stream', methods=['GET'])
def stop_stream():
    global stream_camera, processing_thread, stream_active
    
    try:
        stream_active = False
        
        if processing_thread:
            processing_thread.join(timeout=1.0)
            
        if stream_camera:
            stream_camera.release()
            
        # Clear all frame queues
        for queue in frame_queues.values():
            while not queue.empty():
                queue.get()
                
        return jsonify({'message': 'Stream stopped successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    """Video streaming route for specific feed type"""
    if feed_type not in frame_queues:
        return "Invalid feed type", 400
        
    def generate():
        while True:
            if not frame_queues[feed_type].empty():
                frame = frame_queues[feed_type].get()
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
                
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/process', methods=['POST'])
def process_files():
    """Handle file upload and processing"""
    if 'images[]' not in request.files and 'videos[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    processed_images = []
    processed_videos = []

    # Process images
    if 'images[]' in request.files:
        image_files = request.files.getlist('images[]')
        for file in image_files:
            if file and allowed_file(file.filename):
                filename = Path(file.filename).name
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                try:
                    processed_paths = process_image(filepath)
                    if processed_paths:
                        processed_images.append(processed_paths)
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

    # Process videos
    if 'videos[]' in request.files:
        video_files = request.files.getlist('videos[]')
        for file in video_files:
            if file and allowed_file(file.filename):
                filename = Path(file.filename).name
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                try:
                    processed_paths = process_video(filepath)
                    if processed_paths:
                        processed_videos.append(processed_paths)
                except Exception as e:
                    print(f"Error processing video {filename}: {str(e)}")
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

    return jsonify({
        'processedImages': processed_images,
        'processedVideos': processed_videos
    })

@app.route('/processed/<path:filename>')
def serve_processed_file(filename):
    """Serve processed files"""
    try:
        file_path = safe_join(PROCESSED_FOLDER, filename)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
            
        if filename.endswith('.mp4'):
            mime_type = 'video/mp4'

        return send_file(
            file_path,
            mimetype=mime_type,
            as_attachment=False,
            conditional=True
        )

    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'Error serving file'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)








