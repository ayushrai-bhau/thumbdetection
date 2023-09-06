import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)


image = cv2.imread('WhatsApp Image 2023-05-31 at 11.34.14 AM.jpeg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results_hand = mp_hands.process(image_rgb)

results_face = mp_face.process(image_rgb)


thumbs_up_detected = False
face_detected = False
# Check if any hands are detected
if results_hand.multi_hand_landmarks:
    for hand_landmarks in results_hand.multi_hand_landmarks:
        if hand_landmarks.landmark[4]:
            thumb_x = hand_landmarks.landmark[4].x
            thumb_y = hand_landmarks.landmark[4].y
            print(thumb_x,thumb_y,"============" )

            # Check if the thumb is above the middle finger (thumbs-up gesture)
            middle_finger_y = hand_landmarks.landmark[12].y
            print(middle_finger_y,"===============")
            if thumb_y < middle_finger_y:
                thumbs_up_detected = True
                break 

#face detection
if results_face.detections:
    num_faces = len(results_face.detections)
    if num_faces == 1:
        face_detected = True
    else:
        print("Error: Multiple faces detected!")



# Draw the hand landmarks on the image
if results_hand.multi_hand_landmarks:
    for hand_landmarks in results_hand.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

# if results_face.detections:
#     for detection in results_face.detections:
#         mp.solutions.drawing_utils.draw_detection(image, detection)

if results_face.detections:
    for detection in results_face.detections:
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


# Display the image with hand landmarks
cv2.imshow('Thumb and Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if thumbs_up_detected and face_detected:
    print(f"Thumb'sup Detected: {thumbs_up_detected}")
    print(f"Face Detected: {face_detected}")

else:
     print("sorry it is not detecting")
