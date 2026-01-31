import cv2
import time
import os

# 這裡可以直接填入你的 URL 方便測試
RTSP_URL = os.getenv('RTSP_URL', 'rtsp://admin:8-Qbbu-KXyer%40__R@192.168.94.201:554/ch01/0')
THRESHOLD = int(os.getenv('THRESHOLD', '300'))
IDLE_TIME = 10
WARMUP_COUNT = 100 # GUI 測試時熱身可以短一點

def detect_gui():
    cap = cv2.VideoCapture(RTSP_URL)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    is_showing = False
    last_motion_time = 0
    frame_count = 0

    print("GUI 測試模式啟動。按 'q' 鍵退出。")

    while True:
        # 為了 GUI 順暢度，我們在筆電測試時可以不跳幀，或者只跳 2 幀
        ret, frame = cap.read()
        
        if not ret:
            print("無法讀取串流...")
            break

        small_frame = cv2.resize(frame, (320, 180)) # 顯示大一點方便看
        fg_mask = back_sub.apply(small_frame)
        motion_score = cv2.countNonZero(fg_mask)
        current_time = time.time()

        # 狀態邏輯 (不發送 Request，僅改變狀態文字)
        status_text = "IDLE"
        text_color = (0, 255, 0) # 綠色

        if frame_count < WARMUP_COUNT:
            frame_count += 1
            status_text = f"WARMING UP ({frame_count}/{WARMUP_COUNT})"
            text_color = (0, 255, 255) # 黃色
        else:
            if motion_score > THRESHOLD:
                last_motion_time = current_time
                is_showing = True
            elif is_showing and (current_time - last_motion_time > IDLE_TIME):
                is_showing = False
            
            if is_showing:
                status_text = "MOTION DETECTED (SHOW)"
                text_color = (0, 0, 255) # 紅色

        # --- GUI 繪製資訊 ---
        # 在畫面上印出數值
        cv2.putText(small_frame, f"Score: {motion_score}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(small_frame, f"Thresh: {THRESHOLD}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(small_frame, f"Status: {status_text}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # 顯示兩個視窗：原始畫面與二值化遮罩
        cv2.imshow('CCTV Motion Monitor', small_frame)
        cv2.imshow('Motion Mask (Background Subtraction)', fg_mask)

        # 按 'q' 退出，'r' 重置背景模型
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
            frame_count = 0
            print("背景模型已手動重置")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_gui()