import cv2
import time
import os
import requests

RTSP_URL = os.getenv('RTSP_URL')
SHOW_URL = os.getenv('SHOW_URL')
HIDE_URL = os.getenv('HIDE_URL')
THRESHOLD = int(os.getenv('THRESHOLD', '300'))
IDLE_TIME = int(os.getenv('IDLE_TIME', '10'))
DETECTION_DELAY_FRAMES = int(os.getenv('DETECTION_DELAY_FRAMES', '100'))

def main():
    cap = cv2.VideoCapture(RTSP_URL)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    is_showing = False
    last_motion_time = 0
    frame_count = 0

    print(f"RTSP-Watchman 啟動，開始取樣...")

    while True:
        for _ in range(5): cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("串流中斷，5秒後重試...")
            time.sleep(5)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        small_frame = cv2.resize(frame, (320, 180))
        fg_mask = back_sub.apply(small_frame)
        motion_score = cv2.countNonZero(fg_mask)
        current_time = time.time()

        if frame_count < DETECTION_DELAY_FRAMES:
            frame_count += 1
            if frame_count % 20 == 0:
                print(f"取樣中... ({frame_count}/{DETECTION_DELAY_FRAMES}) 當前噪點分數: {motion_score}")
            continue

        if motion_score > THRESHOLD:
            last_motion_time = current_time
            if not is_showing:
                print(f"[{time.strftime('%X')}] 偵測到移動 (Score: {motion_score}) -> SHOW")
                try:
                    requests.get(SHOW_URL, timeout=1)
                    is_showing = True
                except Exception as e:
                    print(f"SHOW 失敗: {e}")
        
        elif is_showing and (current_time - last_motion_time > IDLE_TIME):
            print(f"[{time.strftime('%X')}] 已安靜超過 {IDLE_TIME} 秒 -> HIDE")
            try:
                requests.get(HIDE_URL, timeout=1)
                is_showing = False
            except Exception as e:
                print(f"HIDE 失敗: {e}")

    cap.release()

if __name__ == "__main__":
    main()