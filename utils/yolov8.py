from ultralytics import YOLO
import numpy as np
import cv2
import time
import threading
from copy import deepcopy

# YOLOv8 연동
class YOLOv8:
    def __init__(self, model_path):
        '''
        YOLOv8모델 로드 및 인퍼런스 준비
        '''
        self.model = YOLO(model_path)
        print('YOLOv8 모델 로드 완료')
        self.cls_list = None
        self.img = []
        self.dic_list = []
        self.thread = threading.Thread(target = self.run)
        self.thread.start()

    def run(self):
        '''
        img: YOLOv8로 추론할 이미지 입력
        multi-thread로 self.img를 받아서, self.dic_list를 지속적으로 업데이트 함
        추론 결과인 dic_list 반환(bbox[x1, y1, x2, y2], conf, class_name 포함)
        '''
        print('YOLOv8 인퍼런스 Multi-Thread 구동 시작')
        while True:
            try:
                results = self.model.predict(source = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), verbose = False, conf = 0.5)[0]
            except:
                time.sleep(0.1)
                continue
            if self.cls_list == None:
                self.cls_list = results.names
                print(self.cls_list)
            results = results.boxes.data.tolist()
            dic_list = []
            for result in results:
                x1, y1, x2, y2, conf, cls = int(result[0]), int(result[1]), int(result[2]), int(result[3]), float(result[4]), int(result[5])
                # depth 추출
                dic_list.append({'bbox':[x1, y1, x2, y2], 'conf':conf, 'class_name':self.cls_list[cls]})
            self.dic_list = deepcopy(dic_list)

    def draw(self, img):
        '''
        입력된 최신 이미지에, 가장 최신의 dic_list로 그려주는 함수
        복잡하게 설명한 이유: 영상은 실시간 스트리밍 되고, 인퍼런스가 multi-thread로 작동하기 때문
        (이렇게 하지 않으면 영상에 딜레이가 생겨서 과거의 영상이 스트리밍 되는 현상이 있음)
        '''
        for dic in self.dic_list:
            cv2.rectangle(img, (dic['bbox'][0], dic['bbox'][1]), (dic['bbox'][2], dic['bbox'][3]), (0,0,255), 2)
            text = f'{dic["class_name"]}:{round(dic["conf"], 2)}'
            cv2.putText(img, text, (dic['bbox'][0], dic['bbox'][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img