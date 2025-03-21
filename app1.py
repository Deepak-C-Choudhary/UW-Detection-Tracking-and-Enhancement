
#____________________________________working with the DIAT camera and live streaming well________________________________________


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
import requests
from urllib.parse import quote

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
current_rtsp_url = None  # Store the RTSP URL as a global variable
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
    global stream_active, stream_camera, current_rtsp_url
    frame_count = 0
    last_frame_time = time.time()
    
    while stream_active and stream_camera is not None:
        try:
            ret, frame = stream_camera.read()
            
            # Check if frame was successfully read
            if not ret or frame is None:
                print("Failed to read frame")
                # If we haven't received frames for 5 seconds, try to reconnect
                if time.time() - last_frame_time > 5 and current_rtsp_url is not None:
                    print("No frames received for 5 seconds, attempting to reconnect...")
                    # Close and reopen the camera connection
                    stream_camera.release()
                    time.sleep(1)
                    # Reconnect using the stored URL
                    stream_camera = cv2.VideoCapture(current_rtsp_url)
                    print(f"Reconnected to {current_rtsp_url}")
                    last_frame_time = time.time()
                time.sleep(0.1)
                continue
                
            # Update last successful frame time
            last_frame_time = time.time()
            
            # Process every 2nd frame to reduce CPU load
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            processed_frames = process_frame(frame)
            if processed_frames:
                # Update each frame queue
                for frame_type, processed_frame in processed_frames.items():
                    try:
                        # Clear queue if full
                        if frame_queues[frame_type].full():
                            try:
                                frame_queues[frame_type].get_nowait()
                            except queue.Empty:
                                pass
                        frame_queues[frame_type].put(processed_frame)
                    except Exception as e:
                        print(f"Error updating {frame_type} queue: {str(e)}")
                        continue
            
        except Exception as e:
            print(f"Error in stream processor: {str(e)}")
            time.sleep(0.5)
            continue
            
        time.sleep(0.01)  # Small delay to prevent overload

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_stream', methods=['GET'])
def start_stream():
    global stream_camera, processing_thread, stream_active, current_rtsp_url
    
    ip_address = request.args.get('ip')
    username = request.args.get('username', 'admin')
    password = request.args.get('password', 'admin123456')
    
    if not ip_address:
        return jsonify({'error': 'IP address is required'}), 400
        
    try:
        # Stop any existing stream
        if stream_active:
            stream_active = False
            if processing_thread:
                processing_thread.join(timeout=1.0)
            if stream_camera:
                stream_camera.release()
        
        # Clear all frame queues
        for queue_obj in frame_queues.values():
            while not queue_obj.empty():
                queue_obj.get()

        # Build the RTSP URL that works in VLC
        rtsp_url = f"rtsp://{username}:{password}@{ip_address}"
        print(f"Connecting to camera using: {rtsp_url}")
        
        # Store the URL globally
        current_rtsp_url = rtsp_url
        
        # Set up OpenCV VideoCapture with the RTSP URL
        stream_camera = cv2.VideoCapture(rtsp_url)
        
        # Configure capture properties
        stream_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
        # Check if connection was successful
        if not stream_camera.isOpened():
            # Try alternative RTSP URL formats if the simple one fails
            alternative_urls = [
                f"rtsp://{username}:{password}@{ip_address}/h264/ch1/main/av_stream",
                f"rtsp://{username}:{password}@{ip_address}/h264/ch1/sub/av_stream",
                f"rtsp://{username}:{password}@{ip_address}/Streaming/Channels/101",
                f"rtsp://{username}:{password}@{ip_address}/Streaming/Channels/102"
            ]
            
            for alt_url in alternative_urls:
                print(f"Trying alternative URL: {alt_url}")
                stream_camera.release()
                stream_camera = cv2.VideoCapture(alt_url)
                
                if stream_camera.isOpened():
                    # Try to read a test frame
                    ret, frame = stream_camera.read()
                    if ret and frame is not None:
                        print(f"Successfully connected using: {alt_url}")
                        current_rtsp_url = alt_url  # Update the global URL
                        break
            
            # If still not connected, return error
            if not stream_camera.isOpened():
                return jsonify({
                    'error': 'Failed to connect to camera. Please check:\n' +
                            '1. IP address is correct\n' +
                            '2. Username and password are correct\n' +
                            '3. Camera is powered on and connected to network\n' +
                            '4. No firewall is blocking the connection'
                }), 400
        
        # Start processing thread
        stream_active = True
        processing_thread = threading.Thread(target=stream_processor)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({'streams': {
            'preview': '/video_feed/preview',
            'detection': '/video_feed/detection',
            'tracking': '/video_feed/tracking',
            'enhancement': '/video_feed/enhancement'
        }}), 200
        
    except Exception as e:
        print(f"Error starting stream: {str(e)}")
        if stream_camera:
            stream_camera.release()
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
            if stream_active:
                if not frame_queues[feed_type].empty():
                    frame = frame_queues[feed_type].get()
                    # Ensure we have a valid frame
                    if frame is None:
                        time.sleep(0.01)
                        continue
                        
                    # Encode frame to JPEG
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_bytes = buffer.tobytes()
                        
                        # Yield properly formatted multipart response
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except Exception as e:
                        print(f"Error encoding frame: {str(e)}")
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)
            else:
                # If stream is not active, yield a placeholder or empty frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Stream not active", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)  # Longer delay for inactive stream
                
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

@app.route('/api/camera_status', methods=['GET'])
def camera_status():
    """Check if camera is active"""
    return jsonify({
        'active': stream_active,
        'connected': stream_camera is not None and stream_camera.isOpened() if stream_camera else False
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)