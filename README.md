# Classroom Engagement Analytics

## 🎯 Project Goal
Classroom Engagement Analytics is a computer vision pipeline designed for volunteer educators at the Disha Foundation, KV School, IIT Guwahati. It passively quantifies primary school classroom engagement and physical posture fatigue without requiring teachers to manage complex hardware or monitor live dashboards. By converting physical signals into a structured time-series dataset, educators can identify chronic disengagement trends, optimize lesson pacing and identify student who need special attention.

## ⚙️ How It Works (Code Explanation)
The pipeline (`stickvideoblur_parquetfile.py`) processes raw classroom video through three main stages:
1. **Multi-Person Pose Tracking:** Utilizes Google's MediaPipe Tasks API (`pose_landmarker_heavy.task`) to extract 33-point kinematic keypoints for up to 20 students simultaneously.
2. **Audio Fidget Analysis:** Uses `moviepy` to extract the audio track and `librosa` (Harmonic-Percussive Source Separation) to isolate transient percussive noises (e.g., desk tapping, shuffling) as a restlessness metric.
3. **Privacy & Export:** Dynamically applies a Gaussian blur to all facial landmarks (`landmarks[0:11]`) before outputting the annotated video, and exports rolling 60-second engagement averages to a `.parquet` dataset.

## 🚀 Requirements
* Python 3.9+ - basic code language 
* OpenCV (`cv2`) - video I/O and drawing
* MediaPipe (`mediapipe`) - pose landmark detection
* Pandas & PyArrow (`pandas`, `pyarrow`) - parquet export
* Librosa & MoviePy (`librosa`, `moviepy`) - audio feature extraction

## 📊 Data Documentation
As per the Latent48 submission requirements, here is the profile of our submitted dataset:

* **Sources & Collection Timeline (Hackathon Authenticity):** Data was exclusively collected during the Latent48 competition window on **September 26, 2026, between 3:00 PM and 5:00 PM** at KV School. We recorded three raw classroom sessions (12-minute, 19-minute, and 9-minute) using a Nothing 4 PRO smartphone mounted on a tripod.
  
<img width="630" height="974" alt="WhatsApp Image 2026-09-27 at 15 05 37 (1)" src="https://github.com/user-attachments/assets/1f49aec2-0e28-4c8e-801d-19542a9f93c6" />
<img width="630" height="851" alt="WhatsApp Image 2026-09-27 at 15 05 36" src="https://github.com/user-attachments/assets/47905662-8a14-4074-838e-69da62a82dd6" />

* **Note on Raw Data Hosting:** The original `.mp4` recordings exceed 500MB each, bypassing GitHub's file size limits. To prove our physical-to-digital pipeline while adhering to platform constraints, we have uploaded the processed `.parquet` dataset, our codebase, and a compressed demo clip. The full raw footage is retained locally and available to the judging panel upon request. (Also uploaded in Drive submission link)
* **Row Count:** 19 rows (representing a 19-minute classroom sample).
* **Collection Window:** 60-second rolling aggregation windows (`WINDOW_SECONDS = 60`). Pipeline "collects" all the slumping, fidgeting, and audio data for a full 60 seconds. It then calculates the average for that entire minute and logs just one clean row in output Parquet file. Similarly it moves further for next 60-sec.
* **Sources:** A single smartphone camera (Nothing 4 PRO) mounted on a tripod in front corner of classroom capturing video and percussive room audio.
* **Observed vs Inferred vs Synthetic:** 
  * *Observed:* Raw spatial coordinates (X, Y) of student shoulders/ears and raw audio decibels.
  * *Inferred:* The `avg_slump_angle_deg` (calculated via trigonometric displacement), `fidget_velocity_idx` (calculated via centroid pixel tracking), and the composite `fatigue_score`.
  * *Synthetic:* 0%. All data represents actual physical movements and audio from the recorded classroom.
* **How AI Was Used:** We utilized the Google MediaPipe API to extract human kinematic keypoints. The AI strictly processes spatial coordinates, and dynamic facial blurring is applied in-memory to protect student privacy before any visual output is saved.

## 🗄️ Schema Description
The generated `classroom_engagement_data.parquet` file contains the following schema:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `timestamp` | datetime64[us] | The exact time the 60-second aggregation window completed. |
| `window_id` | int64 | Sequential ID of the 60-second processing block. |
| `students_tracked` | int64 | The number of students successfully tracked by MediaPipe in that window. |
| `avg_slump_angle_deg` | float64 | The mean forward-leaning angle of the classroom (in degrees). |
| `fidget_velocity_idx` | float64 | The calculated rate of movement based on shoulder centroid shifts. |
| `percussive_noise_db` | float64 | Background room noise/fidgeting extracted via Librosa HPSS (in decibels). |
| `fatigue_score` | float64 | A composite weighted metric (Max: 100.0) combining slump, fidgeting, and noise. |
| `alert_triggered` | bool | Boolean flag evaluating `true` if `fatigue_score` > 65.0. |
| `recommended_action` | str | Actionable UI string (e.g., "High Restlessness: Trigger Break"). |

## 📸 Project Gallery & Demo

### AI Pose Tracking in Action
<!-- Replace the link below with your actual image filename or GitHub drag-and-drop link -->
<img width="372" height="495" alt="Screenshot 2026-09-27 022159" src="https://github.com/user-attachments/assets/52abd055-c02b-4123-9ff7-1fd7fc86df2f" />

Attentive Pose --> Low Fatigue Score

<img width="512" height="482" alt="Screenshot 2026-09-27 022329" src="https://github.com/user-attachments/assets/16f271d3-a57d-41ea-9bc3-2bfb7ac85d29" />

Slouch Pose --> High Fatigue Score

### Generated Parquet Dataset (.parquet)
* Showing only first 5 rows 
<!-- Drag and drop your Parquet table screenshot here -->
<img width="1456" height="651" alt="Screenshot 2026-09-27 020416" src="https://github.com/user-attachments/assets/89725858-b20c-433e-a098-f85a58c386b8" />

### Short Sample Annotated Clip generated by code 
<!-- If you use the GitHub web editor, just drag your MP4 here and it will generate the video player automatically -->
https://github.com/user-attachments/assets/431b4534-654f-4478-8f31-1591350eadab

* As you can see in video, the child face and various body points are tracked, distance between them are calculated and the fatigue score is calculated and exported in parquet file.

* fatigue_score = min(100.0, (avg_slump * 2.2) + (avg_fidget * 150.0) + max(0, percussive_db + 40))

* **Slump Penalty** (avg_slump * 2.2): The average forward-leaning angle of the classroom is multiplied by a weight of 2.2.
* **Fidget Penalty** (avg_fidget * 150.0): The pixel displacement rate of the students' shoulder centroids is multiplied by 150.0 to scale the tiny physical movements into a readable metric.
* **Audio Noise Penalty** (max(0, percussive_db + 40)): A baseline of -40 dB is established; any sudden percussive classroom noise (like desk tapping or shuffling) above this threshold adds to the restlessness score.
* **Score Ceiling** (min(100.0, ...)): The entire composite score is capped at a maximum of 100.0 so the data remains within a clean 0–100 percentage scale. 

## 🚧 Known Limitations & Future Work
* **Desk Occlusion:** Desks and physical distance reduce MediaPipe detection accuracy for back-row students. Future iterations will explore overhead wide-angle cameras.
* **Macro-Only Aggregation:** The current model calculates classroom-wide averages. Future work involves integrating ByteTrack or DeepSORT to lock persistent IDs to specific seating zones, enabling individualized, post-class student attention diagnostics.
* **Face-Masking Issue:** Currently the face is blurred/masked only for front students in camera detection range and the students face at back row are still seen. Need high-resolution camera for better face tracking.

## ⚖️ Disclaimer
This tool is a hackathon prototype designed for generalized educational analytics. It is not a clinical diagnostic tool for ADHD or other learning disabilities.

## 👥 Team
* **[TEAM NAD]** - Latent48 Hackathon Submission
* Neel M. Kapadi & Dhanush V.
* M.Tech - Computational Mechanics
* IIT Guwahati
