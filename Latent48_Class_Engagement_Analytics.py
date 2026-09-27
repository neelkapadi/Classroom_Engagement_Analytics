import cv2
import numpy as np
import pandas as pd
import librosa
import os
import math
import time
from datetime import datetime, timedelta
import urllib.request

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_VIDEO_PATH = "class3.mp4"
OUTPUT_VIDEO_PATH = "class3_output.mp4"
OUTPUT_PARQUET_PATH = "class3_data.parquet"
MODEL_PATH = "pose_landmarker.task"

ALERT_DELAY_SECONDS = 5.0    
WINDOW_SECONDS = 60          
FRAME_SKIP = 3               
MAX_STUDENTS = 20            # Maximum number of students to track simultaneously

if not os.path.exists(MODEL_PATH):
    print("[+] Downloading MediaPipe pose landmarker model...")
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(model_url, MODEL_PATH)

def extract_audio_metrics(video_path, window_sec):
    print("[+] Extracting and processing audio for Parquet dataset...")
    try:
        from moviepy import VideoFileClip
        
        # Extract audio track to a temporary WAV file
        temp_wav = "temp_audio.wav"
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            print("[-] No audio track found in this video.")
            return {}
            
        print("[+] Ripping audio track from MP4...")
        video.audio.write_audiofile(temp_wav, logger=None)
        
        # Load the temporary WAV file natively with librosa
        y, sr = librosa.load(temp_wav, sr=22050)
        
        # Clean up the temporary file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
    except ImportError:
        print("[-] Missing library. Run: pip install moviepy")
        return {}
    except Exception as e:
        print(f"[-] Audio extraction failed: {e}")
        return {}

    # Process the audio array
    _, y_percussive = librosa.effects.hpss(y)
    samples_per_window = int(sr * window_sec)
    num_windows = math.ceil(len(y_percussive) / samples_per_window)
    
    audio_energy_map = {}
    for i in range(num_windows):
        start_idx = i * samples_per_window
        end_idx = min((i + 1) * samples_per_window, len(y_percussive))
        window_samples = y_percussive[start_idx:end_idx]
        
        if len(window_samples) > 0:
            rms = np.sqrt(np.mean(window_samples**2))
            db = 20 * np.log10(rms + 1e-6)
            audio_energy_map[i] = round(float(db), 2)
        else:
            audio_energy_map[i] = -60.0
            
    return audio_energy_map

def process_classroom_data():
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"[-] Error: '{INPUT_VIDEO_PATH}' not found.")
        return

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=MAX_STUDENTS,
        min_pose_detection_confidence=0.15,
        min_pose_presence_confidence=0.15,
        min_tracking_confidence=0.15
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    audio_metrics = extract_audio_metrics(INPUT_VIDEO_PATH, WINDOW_SECONDS)

    frames_required_for_alert = int(fps * ALERT_DELAY_SECONDS)
    consecutive_slump_frames = 0

    frames_per_window = int(fps * WINDOW_SECONDS)
    frame_count = 0
    window_index = 0
    window_slump_angles = []
    window_fidget_velocities = []
    prev_shoulder_centroid = None
    records = []
    base_time = datetime.now().replace(microsecond=0)

    # --- TIMER SETUP VARIABLES ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    print(f"[+] Processing multi-person tracking (Max {MAX_STUDENTS} students)...")
    
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (7, 8), (0, 7), (0, 8)
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- REMAINING TIME (ETA) TRACKER ---
        if frame_count % (30 * FRAME_SKIP) == 0:
            elapsed_time = time.time() - start_time
            processing_fps = frame_count / elapsed_time
            remaining_frames = total_frames - frame_count
            
            if processing_fps > 0:
                eta_seconds = remaining_frames / processing_fps
                eta_mins = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                percent_done = (frame_count / total_frames) * 100
                
                print(f"[+] Progress: {percent_done:.1f}% | Speed: {processing_fps:.1f} FPS | ETA: {eta_mins}m {eta_secs}s remaining")

        if frame_count % FRAME_SKIP != 0:
            continue
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

        frame_slump_angles = []
        current_frame_shoulders = []

        if results.pose_landmarks:
            for landmarks in results.pose_landmarks:
                
                # Face Blurring
                face_landmarks = landmarks[0:11]
                x_coords = [int(lm.x * width) for lm in face_landmarks]
                y_coords = [int(lm.y * height) for lm in face_landmarks]
                
                if x_coords and y_coords:
                    pad_x = int(abs(max(x_coords) - min(x_coords)) * 0.5)
                    pad_y = int(abs(max(y_coords) - min(y_coords)) * 0.5)
                    x1 = max(0, min(x_coords) - pad_x)
                    y1 = max(0, min(y_coords) - pad_y - int(pad_y * 1.5))
                    x2 = min(width, max(x_coords) + pad_x)
                    y2 = min(height, max(y_coords) + pad_y)

                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        blurred_face = cv2.GaussianBlur(face_roi, (71, 71), 0)
                        frame[y1:y2, x1:x2] = blurred_face

                # Skeleton Drawing
                for connection in POSE_CONNECTIONS:
                    idx1, idx2 = connection
                    if idx1 < len(landmarks) and idx2 < len(landmarks):
                        p1, p2 = landmarks[idx1], landmarks[idx2]
                        cv2.line(frame, (int(p1.x * width), int(p1.y * height)), 
                                 (int(p2.x * width), int(p2.y * height)), (0, 255, 0), 2)

                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, (0, 0, 255), -1)

                # Slump Angle for this specific student
                l_shoulder, r_shoulder = landmarks[11], landmarks[12]
                l_ear, r_ear = landmarks[7], landmarks[8]

                shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2.0
                shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2.0
                ear_mid_x = (l_ear.x + r_ear.x) / 2.0
                ear_mid_y = (l_ear.y + r_ear.y) / 2.0

                dx = ear_mid_x - shoulder_mid_x
                dy = ear_mid_y - shoulder_mid_y
                student_slump = abs(math.degrees(math.atan2(dx, abs(dy))))
                
                frame_slump_angles.append(student_slump)
                current_frame_shoulders.append((shoulder_mid_x, shoulder_mid_y))

        # --- CALCULATE CLASSROOM AGGREGATES FOR THIS FRAME ---
        classroom_avg_slump = float(np.mean(frame_slump_angles)) if frame_slump_angles else 0.0
        window_slump_angles.append(classroom_avg_slump)

        if current_frame_shoulders:
            centroid_x = np.mean([s[0] for s in current_frame_shoulders])
            centroid_y = np.mean([s[1] for s in current_frame_shoulders])
            
            if prev_shoulder_centroid is not None:
                velocity = math.sqrt((centroid_x - prev_shoulder_centroid[0])**2 + 
                                     (centroid_y - prev_shoulder_centroid[1])**2)
                window_fidget_velocities.append(velocity)
            prev_shoulder_centroid = (centroid_x, centroid_y)

        # --- PARQUET WINDOW AGGREGATION LOGIC ---
        if frame_count >= (window_index + 1) * frames_per_window:
            avg_slump = float(np.mean(window_slump_angles[-frames_per_window:])) if window_slump_angles else 0.0
            avg_fidget = float(np.mean(window_fidget_velocities[-frames_per_window:])) if window_fidget_velocities else 0.0
            percussive_db = audio_metrics.get(window_index, -60.0)

            fatigue_score = min(100.0, (avg_slump * 2.2) + (avg_fidget * 150.0) + max(0, percussive_db + 40))
            alert_flag = fatigue_score > 65.0
            recommendation_text = "High Restlessness: Trigger Break" if alert_flag else (
                "Moderate Slumping: Interactive Q&A" if fatigue_score > 45.0 else "Optimal Engagement")

            window_time = base_time + timedelta(seconds=window_index * WINDOW_SECONDS)

            records.append({
                "timestamp": window_time,
                "window_id": window_index,
                "students_tracked": len(frame_slump_angles),
                "avg_slump_angle_deg": round(avg_slump, 2),
                "fidget_velocity_idx": round(avg_fidget, 4),
                "percussive_noise_db": percussive_db,
                "fatigue_score": round(fatigue_score, 2),
                "alert_triggered": alert_flag,
                "recommended_action": recommendation_text
            })
            window_index += 1

        # --- VISUAL UI TEMPORAL LOGIC GATE ---
        if classroom_avg_slump > 18.0:
            consecutive_slump_frames += 1
        else:
            consecutive_slump_frames = 0 

        if consecutive_slump_frames >= frames_required_for_alert:
            if classroom_avg_slump > 25.0:
                status_text, recommendation, banner_color = "ALERT: Sustained Classroom Slump!", "Action: Trigger 2-min Stretch Break", (0, 0, 255)  
            else:
                status_text, recommendation, banner_color = "WARNING: Sustained Mild Slacking", "Action: Switch to Interactive Q&A", (0, 140, 255) 
        elif consecutive_slump_frames > 0:
            time_left = round((frames_required_for_alert - consecutive_slump_frames) / fps, 1)
            status_text, recommendation, banner_color = f"Monitoring Posture... Alert in {time_left}s", "Action: Observe", (0, 200, 255) 
        else:
            status_text, recommendation, banner_color = "STATUS: Optimal Class Engagement", "Action: Continue Current Lesson", (0, 200, 0)  

        # --- DRAW UI OVERLAY ---
        cv2.rectangle(frame, (0, 0), (width, 95), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, 95), (width, 100), banner_color, -1)
        cv2.putText(frame, f"Class Avg Slump: {round(classroom_avg_slump, 1)} deg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)
        cv2.putText(frame, recommendation, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(OUTPUT_PARQUET_PATH, engine="pyarrow", index=False)
        print(f"\n[+] Success! Parquet dataset exported to: {OUTPUT_PARQUET_PATH}")
    
    print(f"[+] Success! Annotated video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_classroom_data()