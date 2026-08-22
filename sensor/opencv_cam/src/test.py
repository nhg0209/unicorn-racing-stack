# test_cam.py
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

for i in range(10):
    t = time.time()
    ret, frame = cap.read()
    print(f"frame {i}: {time.time()-t:.3f}s, ret={ret}")

cap.release()