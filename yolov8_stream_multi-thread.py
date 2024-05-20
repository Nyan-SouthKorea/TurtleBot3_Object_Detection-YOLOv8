from utils.yolov8 import YOLOv8
import cv2

# 웹캠에서 비디오 캡처 시작
yolov8 = YOLOv8(model_path = './yolov8n.pt')
cap = cv2.VideoCapture('http://192.168.0.23:5000/video')

# 비디오 캡처가 정상적으로 시작되었는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    cap.release()
    exit()

# 웹캠으로부터 프레임을 계속 읽어와서 화면에 표시
while True:
    # 프레임별로 캡처
    ret, img = cap.read()
    yolov8.run(img)
    img = yolov8.draw()

    # 프레임을 읽는 데 실패하면 중지
    if not ret:
        print("프레임을 받아올 수 없습니다. 종료합니다.")
        break

    # 이미지를 윈도우 창에 표시
    cv2.imshow('Webcam Streaming', img)

    # 'q' 키가 눌리면 반복문에서 빠져나옴
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후, 캡처 장치와 윈도우를 해제
cap.release()
cv2.destroyAllWindows()
