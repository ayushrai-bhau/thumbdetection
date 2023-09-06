from django.shortcuts import render
import cv2
import mediapipe as mp
import numpy as np 
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse

# Create your views here.

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

class Thumb_face_detection(APIView):
    def post(self, request):

        # picture = request.data.get('image')
        image_file = request.FILES['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_c = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        # image = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)
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
                return Response({"Error": "Multiple faces detected!"})
        if thumbs_up_detected and face_detected:
            return Response(
                {
                    "Thumb's up": "Thumb'sup detected",
                    "Face": "Face detected",
                },
            )
        else:
            return Response({"message":"No face or thumb detected"})


        
            



       