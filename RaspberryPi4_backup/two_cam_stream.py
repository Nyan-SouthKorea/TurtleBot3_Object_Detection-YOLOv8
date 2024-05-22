from flask import Flask, Response, render_template_string
import cv2

# 캠 연결 시도
cap_list = [0, 2]
'''
cap_list = []
for i in range(1000):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'cap 찾음: {i}')
        cap_list.append(i)
    else:
        print(f'연결된 cap 아님: {i}')
print(f'cap 모두 검색 완료: {cap_list}')
'''
app = Flask(__name__)

def generate_frames():
    # 웹캠 설정
    cap1 = cv2.VideoCapture(cap_list[0])
    cap2 = cv2.VideoCapture(cap_list[1])
    print('웹캠 연결 완료')

    # 스트리밍 시작
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == False or ret2 == False:
            print(f'현재 ret 상태: [ret1:{ret1}], [ret2:{ret2}]')
            break
        else:
            # 이미지 2장 가로 Concat
            frame = cv2.hconcat([frame1, frame2])
            # 프레임을 JPEG 형식으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # 멀티파트 메시지 형식으로 프레임 생성
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # HTML 페이지에서 비디오 스트리밍을 보여주는 `<img>` 태그 포함
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Video Streaming</title>
    </head>
    <body>
    <h1>Video Streaming</h1>
    <img src="/video" width="640" height="480">
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video')
def video():
    # 비디오 스트리밍을 위한 Response 객체 생성
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
