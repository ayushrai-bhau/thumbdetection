
import cv2
hand_cascade = cv2.CascadeClassifier('hand.xml')
image = cv2.imread('1.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
thumb_detected = False
for (x, y, w, h) in hands:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    if thumb_detected:
        thumb_detected = True
cv2.imshow('Hand Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"Thumb Detected: {thumb_detected}")
