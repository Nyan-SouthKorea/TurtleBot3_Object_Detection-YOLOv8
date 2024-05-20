import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense:
    def __init__(self):
        '''
        Real Sense 카메라 연결하여 파이썬으로 RGB 이미지와 Depth Map을 불러오는 Class.
        공식 문서를 참고하여 사용하기 쉽게 Class의 형태로 제작함
        '''
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
        print('Real Sense 카메라 수신 설정 완료')

    def get_cam(self):
        '''
        RealSense 카메라에서 frame들을 가져옴.
        이 frame들에서 RGB이미지, Depth맵 등 다양한 이미지가 추출됨
        frame이 return되지는 않고, 내부 변수로 활용됨
        '''
        self.frames = self.pipeline.wait_for_frames()
    
    def get_color_img(self):
        '''
        get_cam 실행 이후에 실행되어야 에러가 발생하지 않음
        self.frames라는 변수가 존재해야 실행 가능한 로직이기 때문임
        '''
        color_frame = self.frames.get_color_frame()
        if color_frame:
            self.color_image = np.asanyarray(color_frame.get_data())
            return self.color_image
        else:
            print('get_color_img 에러 발생!')
        
    def get_depth_img(self):
        '''
        get_cam 실행 이후에 실행되어야 에러가 발생하지 않음
        self.frames라는 변수가 존재해야 실행 가능한 로직이기 때문임
        '''
        depth_frame = self.frames.get_depth_frame()
        if depth_frame:
            self.depth_image = np.asanyarray(depth_frame.get_data())
            return self.depth_image
        else:
            print('get_depth_img 에러 발생!')

    def get_depth_color_map(self):
        '''
        self.depth_image로부터 color_map을 만들어서 반환함
        반드시 get_depth_img 함수가 실행되어야 에러가 발생하지 않음
        self.depth_image가 존재해야 실행 가능한 로직이기 때문임
        '''
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return self.depth_colormap
    
    def concat_all(self):
        '''
        위에서 만든 color_img, depth_map, depth_color_map 3개를 가로 Concat하여 반환
        위 3개 이미지가 모두 존재해야 에러가 발생하지 않음
        위 3개 함수를 선언하지 않고 코드를 실행하면 이전 frame의 이미지가 concat될 것임(주의 필요)
        '''
        # 깊이 이미지의 최대값을 확인하고, 8비트 형식으로 스케일 조정
        scaled_depth_image = cv2.convertScaleAbs(self.depth_image, alpha=(255.0/np.max(self.depth_image)))
        depth_3d = cv2.cvtColor(scaled_depth_image, cv2.COLOR_GRAY2BGR) # 1차원인 depth 이미지를 3차원으로 변경함
        concatenated_image = cv2.hconcat([self.color_image, depth_3d, self.depth_colormap])
        return concatenated_image