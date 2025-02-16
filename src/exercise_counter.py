import sys
import mediapipe as mp
import cv2
import numpy as np
from db_utils import init_db, add_log, get_logs, add_dummy_data
from flask import Flask, jsonify
import time  # For tracking total exercise duration
import pyttsx3  # For Text-to-Speech



# Assuming calculate_angle is defined here or imported
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arccos(
        np.dot(b - a, c - b) / (np.linalg.norm(b - a) * np.linalg.norm(c - b))
    )
    angle = np.degrees(radians)

    return angle

# # Exercise functions
# def count_bicep_curls():
#     import time  # To calculate elapsed time for exercise session

#     # Initialize pose estimation tools
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils
#     cap = cv2.VideoCapture(0)
    
#     # UI settings
#     card_width, card_height = 150, 80
#     card_x, card_y = 10, 10
#     card_color = (255, 153, 13)

#     # Initialize counters and timers
#     curl_count = 0
#     is_curling = False
#     prev_angle = None
#     prev_angle1 = None
#     start_time = time.time()  # Start the timer for the session

#     # Pose detection and rep counting
#     with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while True:
#             ret, image = cap.read()  # Capture frame
#             if not ret:
#                 break

#             # Flip and preprocess image
#             image = cv2.flip(image, 1)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = pose.process(image_rgb)

#             # Draw pose landmarks
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             if results.pose_landmarks is not None:
#                 # Extract keypoints for left and right arm
#                 left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#                 left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
#                 left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#                 right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#                 right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
#                 right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

#                 # Calculate angles for both arms
#                 angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
#                 angle1 = calculate_angle(right_shoulder, right_elbow, right_wrist)

#                 # Check for motion thresholds
#                 if prev_angle and prev_angle1 is not None:
#                     delta = abs(angle - prev_angle)
#                     delta1 = abs(angle1 - prev_angle1)
#                     if not is_curling and delta > 20 and delta1 > 20:
#                         is_curling = True
#                     elif is_curling and delta < 10 and delta1 < 10:
#                         is_curling = False

#                 prev_angle = angle
#                 prev_angle1 = angle1

#                 # Count the curl
#                 if is_curling and angle > 160 and angle1 > 160:
#                     curl_count += 1
#                     is_curling = False  # Reset the curling state

#             # Display rep count on screen
#             cv2.rectangle(image, (card_x, card_y), (card_x + card_width, card_y + card_height), card_color, -1)
#             text_x, text_y = card_x + 10, card_y + 30
#             text_line1 = 'Bicep Curl'
#             text_line2 = f'Count: {curl_count}'
#             line_spacing = 30
#             font_scale = 0.7
#             font_color = (255, 255, 255)

#             cv2.putText(image, text_line1, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
#             cv2.putText(image, text_line2, (text_x, text_y + line_spacing), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)

#             # Show the window
#             cv2.imshow('Bicep Curl Counter', image)

#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) == ord('q'):
#                 break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

#     # Calculate total time and time per rep
#     end_time = time.time()  # End the timer
#     total_time = end_time - start_time
#     time_per_count = total_time / curl_count if curl_count > 0 else 0

#     # Log data to the database (newly added integration)
#     add_log(
#         user_id="user_001",  # Replace with dynamic user management if needed
#         exercise="bicep_curl",
#         count=curl_count,
#         time_per_count=time_per_count,
#         total_time=total_time
#     )

#     return curl_count


def count_bicep_curls():
    # Initialize TTS engine
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 200)  # Set speaking speed
    tts_engine.setProperty('volume', 0.9)  # Set volume level

    # Initialize pose estimation tools
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # UI settings
    card_width, card_height = 150, 80
    card_x, card_y = 10, 10
    card_color = (255, 153, 13)

    # Initialize counters and timers
    curl_count = 0
    is_curling = False
    prev_angle = None
    prev_angle1 = None
    start_time = time.time()  # Start the timer for the session

    # Pose detection and rep counting
    with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, image = cap.read()  # Capture frame
            if not ret:
                break

            # Flip and preprocess image
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                # Extract keypoints for left and right arm
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                # Calculate angles for both arms
                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle1 = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Check for motion thresholds
                if prev_angle and prev_angle1 is not None:
                    delta = abs(angle - prev_angle)
                    delta1 = abs(angle1 - prev_angle1)
                    if not is_curling and delta > 20 and delta1 > 20:
                        is_curling = True
                    elif is_curling and delta < 10 and delta1 < 10:
                        is_curling = False

                prev_angle = angle
                prev_angle1 = angle1

                # Count the curl
                if is_curling and angle > 160 and angle1 > 160:
                    curl_count += 1
                    is_curling = False  # Reset the curling state

                    # Announce the rep count using TTS
                    tts_engine.say(f"{curl_count}")
                    tts_engine.runAndWait()

            # Display rep count on screen
            cv2.rectangle(image, (card_x, card_y), (card_x + card_width, card_y + card_height), card_color, -1)
            text_x, text_y = card_x + 10, card_y + 30
            text_line1 = 'Bicep Curl'
            text_line2 = f'Count: {curl_count}'
            line_spacing = 30
            font_scale = 0.7
            font_color = (255, 255, 255)

            cv2.putText(image, text_line1, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(image, text_line2, (text_x, text_y + line_spacing), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)

            # Show the window
            cv2.imshow('Bicep Curl Counter', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Calculate total time and time per rep
    end_time = time.time()  # End the timer
    total_time = end_time - start_time
    time_per_count = total_time / curl_count if curl_count > 0 else 0

    # Log data to the database (newly added integration)
    add_log(
        user_id="user_001",  # Replace with dynamic user management if needed
        exercise="bicep_curl",
        count=curl_count,
        time_per_count=time_per_count,
        total_time=total_time
    )

    return curl_count


def count_bench_press():

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    bench_press_count = 0
    is_pressing = False

    prev_shoulder_angle = None
    prev_elbow_angle = None
    angle_threshold = 130  # Adjust this value based on your setup and preferences

    while True:
        ret, image = cap.read()

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(image_rgb)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

                shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                elbow_angle = calculate_angle(left_elbow, left_shoulder, left_wrist)

                if prev_shoulder_angle is not None and prev_elbow_angle is not None:
                    if not is_pressing and shoulder_angle < angle_threshold and elbow_angle < angle_threshold:
                        is_pressing = True
                    elif is_pressing and shoulder_angle > 160 and elbow_angle > 160:
                        is_pressing = False

                prev_shoulder_angle = shoulder_angle
                prev_elbow_angle = elbow_angle

                if is_pressing:
                    cv2.putText(image, "Pressing", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(image, "Not Pressing", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if is_pressing and shoulder_angle > 160 and elbow_angle > 160:
                    bench_press_count += 1
                    is_pressing = False

            cv2.imshow("Bench Press Counter", image)

            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return bench_press_count

def count_lateral_raises():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # cards constraints
    card_width, card_height = 150, 80
    card_x, card_y = 10, 10
    card_color = (255, 153, 13)

    with mp_pose.Pose(model_complexity =2 ,min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        raise_count = 0
        is_raising = False
        prev_angle = None
        prev_angle1 = None
        delta=None
        delta1=None

        while True:
            ret, image = cap.read()

            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                angle = calculate_angle(left_shoulder, left_elbow, left_hip)
            
                # cv2.putText(image, f"Right Angle: {angle:.2f} degrees", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                angle1 = calculate_angle(right_shoulder, right_elbow, right_hip)
                # cv2.putText(image, f"left Angle: {angle1:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # For lateral raise, we check the angle between the arms
                if prev_angle is not None:
                    delta = abs(angle - prev_angle)
                    print(delta)
                    delta1 = abs(angle1 - prev_angle1)
                    if not is_raising and delta > 15 and delta1 > 15:
                        is_raising = True
                    elif is_raising and delta < 7 and delta1 < 7:
                        is_raising = False

                prev_angle = angle
                prev_angle1 = angle1

                if is_raising and angle1>120 and angle > 120:
                    raise_count += 1

            cv2.rectangle(image, (card_x, card_y), (card_x + card_width, card_y + card_height), card_color, -1)

            # Display the text inside the card
            text_x, text_y = card_x + 10, card_y + 30
            text_line1 = 'Lateral Raise'
            text_line2 = f'Count: {raise_count}'
            line_spacing = 30
            font_scale = 0.7
            font_color = (255, 255, 255)

            cv2.putText(image, text_line1, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(image, text_line2, (text_x, text_y + line_spacing), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)

            cv2.imshow('Lateral Raise counter', image)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return raise_count
 

def count_shoulder_presses():
    """
    Count the number of shoulder presses performed in real-time using Mediapipe and log results to the database.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    card_width, card_height = 180, 100
    card_x, card_y = 10, 10
    card_color = (255, 153, 13)

    press_count = 0
    is_pressing = False

    # Thresholds for range of motion
    top_threshold = 160  # Arms fully extended above the head
    bottom_threshold = 80  # Arms lowered to starting position
    asymmetry_tolerance = 15  # Allowable angle difference between arms

    # Start time for the session
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.7) as pose:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks is not None:
                # Extract keypoints
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                # Calculate angles for both arms
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Allow slight asymmetry between arms
                average_angle = (left_arm_angle + right_arm_angle) / 2
                angle_difference = abs(left_arm_angle - right_arm_angle)

                if angle_difference <= asymmetry_tolerance:  # Both arms reasonably aligned
                    # Check for range of motion
                    if not is_pressing and average_angle > top_threshold:
                        is_pressing = True  # Start of a rep
                    elif is_pressing and average_angle < bottom_threshold:
                        press_count += 1  # Increment count when arms return to starting position
                        is_pressing = False  # Reset the state

            # Display rep count on the screen
            cv2.rectangle(image, (card_x, card_y), (card_x + card_width, card_y + card_height), card_color, -1)

            text_x, text_y = card_x + 10, card_y + 30
            text_line1 = '  Shoulder  '
            text_line3 = '   Press  '
            text_line2 = f'  Count: {press_count} '
            line_spacing = 30
            font_scale = 0.7
            font_color = (255, 255, 255)

            cv2.putText(image, text_line1, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(image, text_line3, (text_x, text_y + line_spacing), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
            cv2.putText(image, text_line2, (text_x, text_y + 2 * line_spacing), cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
            
            cv2.imshow('Shoulder Press Counter', image)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate total session time and time per rep
    end_time = time.time()
    total_time = end_time - start_time
    time_per_rep = total_time / press_count if press_count > 0 else 0

    # Log the results to the database
    add_log(
        user_id="user_001",  # Replace with dynamic user management later
        exercise="shoulder_press",
        count=press_count,
        time_per_count=time_per_rep,
        total_time=total_time
    )

    return press_count

# Function to route based on exercise ID
def perform_exercise(exercise_id):
    if exercise_id == 1:
        print("Starting Bicep Curls...")
        count = count_bicep_curls()
        print(f"Total Bicep Curls: {count}")
    elif exercise_id == 2:
        print("Starting Bench Press...")
        count = count_bench_press()
        print(f"Total Bench Presses: {count}")
    elif exercise_id == 3:
        print("Starting Lateral Raises...")
        count = count_lateral_raises()
        print(f"Total Lateral Raises: {count}")
    elif exercise_id == 4:
        print("Starting Shoulder Presses...")
        count = count_shoulder_presses()
        print(f"Total Shoulder Presses: {count}")
    else:
        print("Invalid Exercise ID! Please choose between 1 and 4.")
