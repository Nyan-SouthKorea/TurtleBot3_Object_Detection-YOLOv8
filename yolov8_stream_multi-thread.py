from utils.yolov8 import YOLOv8
import cv2

# 웹캠에서 비디오 캡처 시작
model_1 = YOLOv8(model_path = './yolov8_best.pt')
model_2 = YOLOv8(model_path = './yolov8_best.pt')
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
    # 합쳐져 있는 이미지를 두개의 이미지로 나누기
    h, w, c = img.shape
    img1 = img[:, :int(w/2)]
    img2 = img[:, int(w/2):w]
    img2 = cv2.rotate(img2, cv2.ROTATE_180)
    
    # 모델 1 인퍼런스
    model_1.img = img1
    img1 = model_1.draw(img1)

    # 모델 2 인퍼런스
    model_2.img = img2
    img2 = model_2.draw(img2)

    # 프레임을 읽는 데 실패하면 중지
    if not ret:
        print("프레임을 받아올 수 없습니다. 종료합니다.")
        break

    # 이미지를 윈도우 창에 표시
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    # 'q' 키가 눌리면 반복문에서 빠져나옴
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후, 캡처 장치와 윈도우를 해제
cap.release()
cv2.destroyAllWindows()
