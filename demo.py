import cv2 as cv
import numpy as np
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

from tkinter import *
import PIL.Image, PIL.ImageTk

from symtable import Symbol
import mediapipe as mp

GUI_WIDTH = 600
GUI_HEIGHT = 750
CAM_WIDTH = 500
CAM_HEIGHT = 500
SAVE_DIR = "./img/"
SAVE_IMG = "test.png"

# 이미지 전처리
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
ORIGIN_MODEL = "best_model_origin.pth"
ESENTIAL_MODEL = "./faceshape_model.pth"
ESENTIAL_MODEL_GRAY = "./faceshape_model_gray.pth"

class EffNet(nn.Module):
    def __init__(self, num_classes=5):
        super(EffNet, self).__init__()
        self.eff = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes, in_channels=1)
    def forward(self, x):
        x = self.eff(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EffNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load(ESENTIAL_MODEL, map_location=device), strict=True)

model.eval()  # eval 모드로 설정

shape_class = {0: "heart", 1: "oblong", 2: "oval", 3: "round", 4: "square"}

# 긴 중안부, 긴 턱, 긴 인중
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 표현되는 랜드마크의 굵기와 반경
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
mean = 0
oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109]
cheek_left = [123, 50, 36, 137, 205, 206, 177, 147, 187, 207, 213, 216, 215, 192, 138,
        214, 212, 135]
cheek_right = [266, 280, 352, 366, 425, 426, 411, 427, 376, 401, 436, 433, 435, 416,
        434, 367, 364, 432]

face_whole = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109, 123, 50, 36, 137, 205, 206, 177, 147, 187, 207, 213, 216, 215, 192, 138,
        214, 212, 135, 266, 280, 352, 366, 425, 426, 411, 427, 376, 401, 436, 433, 435, 416,
        434, 367, 364, 432]

x_list = np.linspace(0, 0, len(face_whole))
y_list = np.linspace(0, 0, len(face_whole))
z_list = np.linspace(0, 0, len(face_whole))

class SampleApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self._frame = None
        self.switch_frame(MainPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class MainPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.pack(side='bottom')
        Button(self, text="Start", command=lambda: master.switch_frame(GetImagePage), width=7, height=2).pack(side='bottom', pady=10)
        
class GetImagePage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.IsCamStop = False
        self.cam_frame = Frame(self, bg='white', width=CAM_WIDTH, height=CAM_HEIGHT)
        self.cam_frame.pack(side='top', pady=10)
        Button(self, text="Capture", width=20, height=10, command=lambda: [self.stop_cam(), master.switch_frame(AnalysisPage)]).pack(side='bottom', pady=10)
        
        self.cap = cv.VideoCapture(cv.CAP_DSHOW+0) # VideoCapture 객체 정의
        # cap = cv.VideoCapture('http://192.168.0.8:4747/video')
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", 0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.canvas = Canvas(self.cam_frame, width=CAM_WIDTH, height=CAM_HEIGHT)
        self.canvas.pack()
        self.update()

    def update(self):
        if self.IsCamStop == False:
            ret, frame = self.cap.read()
            self.frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame))
            self.canvas.create_image([0,0], anchor=NW, image=self.photo)
            self.cam_frame.after(30, self.update)

    def stop_cam(self):
        cv.imwrite(SAVE_DIR+SAVE_IMG, self.frame)
        self.IsCamStop = True
        self.cap.release()

        
class AnalysisPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.preprocess_image()
        self.result()
        self.face_analysis()
        Button(self, text="Restart", command=lambda: [self.clear(), master.switch_frame(GetImagePage)], width=7, height=2).pack(side='bottom', pady=10)

    def clear(self):
        self.result_label.destroy()

    def preprocess_image(self):
        target_img = cv.imread(SAVE_DIR+SAVE_IMG)
        # target_img = cv.imread('./data_set/Oblong/oblong_eb.jpg')
        # 이미지 전처리
        gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)  # gray scale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 찾기
        for (x, y, w, h) in faces:
            cv.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped = gray[y: y + h, x: x + w]
            equalized = cv.equalizeHist(cropped)
            # edge enhancement
            kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
            self.pre_processed_img = cv.filter2D(src=equalized, ddepth=-1, kernel=kernel)
        # cv.imwrite("./img/pre_processed_img.jpg", pre_processed_img)
        # 텐서화
        convert_tensor = transforms.ToTensor()
        processed_img = convert_tensor(self.pre_processed_img)

        with torch.no_grad():
            inputs = torch.FloatTensor(processed_img.unsqueeze(0))
            output = model(inputs)

            print(output)
            self.pred_output = output.argmax(dim=1, keepdim=True)
            print(self.pred_output)
            print(shape_class[int(self.pred_output)])
    
    def result(self):
        # face_shape = Label(self, text="Face Shape: "+shape_class[int(self.pred_output)]).pack(side='left', pady=5)
        
        IMAGE_FILES=SAVE_DIR+SAVE_IMG
        # IMAGE_FILES='./data_set/Oblong/oblong_eb.jpg'
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
                
            # 이미지 불러오기
            image = cv.imread(IMAGE_FILES)
            # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

            # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
            # if not results.multi_face_landmarks:
            #     continue
            annotated_image = image.copy()
            ih, iw, ic = annotated_image.shape
            for face_landmarks in results.multi_face_landmarks:

                # 각 랜드마크를 image에 overlay 시켜줌
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec)
                    # connection_drawing_spec=mp_drawing_styles     <---- 이 부분, 눈썹과 눈, 오른쪽 왼쪽 색깔(초록색, 빨강색)
                    # .get_default_face_mesh_contours_style())

                # 랜드마크의 좌표 정보 확인
            for id, lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = annotated_image.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # print(id,x,y)
                # print(face_landmarks.landmark[id].x, face_landmarks.landmark[id].y, face_landmarks.landmark[id].z)
                if id == 105:  # 왼쪽 눈썹 위
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                elif id == 334:  # 오른쪽 눈썹 위
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                elif id == 94:  # 코 끝
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                elif id == 152:  # 턱 끝
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                elif id == 263 : # 오른눈 오른쪽 끝
                    cv.putText(annotated_image,str(id),(x,y), cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                elif id == 33 : # 왼눈 왼쪽 끝
                    cv.putText(annotated_image,str(id),(x,y), cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                elif id == 61:  # 왼입술 끝
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                elif id == 291:  # 오른입술 끝
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                elif id == 0:  # 입술 위
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                elif id == 17:  # 입술 아래
                    cv.putText(annotated_image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            cv.imshow("Image_ESEntial", annotated_image)

            whole_area = 0
            # 얼굴 전체의 크기 측정, 얼굴을 한 점을 공유하는 여러 개의 삼각형으로 나누어 삼각형의 넓이를 더함으로써 얼굴 넓이 측정
            for i, idx in enumerate(oval) :
                    if idx == 109 :
                            x_gap = face_landmarks.landmark[oval[i]].x - face_landmarks.landmark[oval[0]].x
                            y_gap = face_landmarks.landmark[oval[i]].y - face_landmarks.landmark[oval[0]].y
                            A = np.array([[y_gap/x_gap, -1], [-x_gap/y_gap, -1]])
                            B = np.array([y_gap/x_gap*face_landmarks.landmark[oval[i]].x-face_landmarks.landmark[oval[i]].y, -x_gap/y_gap*face_landmarks.landmark[5].x-face_landmarks.landmark[5].y])
                            x,y = np.linalg.solve(A,B)
                    else :
                            x_gap = face_landmarks.landmark[oval[i]].x - face_landmarks.landmark[oval[i+1]].x
                            y_gap = face_landmarks.landmark[oval[i]].y - face_landmarks.landmark[oval[i+1]].y
                            A = np.array([[y_gap/x_gap, -1], [-x_gap/y_gap, -1]])
                            B = np.array([y_gap/x_gap*face_landmarks.landmark[oval[i]].x-face_landmarks.landmark[oval[i]].y, -x_gap/y_gap*face_landmarks.landmark[5].x-face_landmarks.landmark[5].y])
                            x,y = np.linalg.solve(A,B)
                    vertical_x = face_landmarks.landmark[5].x - x
                    vertical_y = face_landmarks.landmark[5].y - y
                    temp = (np.sqrt(x_gap**2 + y_gap**2) * np.sqrt(vertical_x**2 + vertical_y**2)) / 2
                    whole_area = whole_area + temp

            # 눈 양 끝, 아랫입술 가운데의 landmark를 이용해서 삼각형을 그리고 이목구비를 구분해준다.
            # 이목구비/전체얼굴 비율을 구한다. 
            eye_x = face_landmarks.landmark[226].x - face_landmarks.landmark[446].x
            eye_y = face_landmarks.landmark[226].y - face_landmarks.landmark[446].y
            A = np.array([[eye_y/eye_x, -1], [-eye_x/eye_y, -1]])
            B = np.array([eye_y/eye_x*face_landmarks.landmark[226].x-face_landmarks.landmark[226].y, -eye_x/eye_y*face_landmarks.landmark[17].x-face_landmarks.landmark[17].y])
            x,y = np.linalg.solve(A,B)
            vertical_x = face_landmarks.landmark[17].x - x
            vertical_y = face_landmarks.landmark[17].y - y
            face_area = (np.sqrt(eye_x**2 + eye_y**2) * np.sqrt(vertical_x**2 + vertical_y**2)) / 2

            face_ratio = whole_area/face_area
            print("face ratio : ", face_ratio)

            # 결과값이 4.6 이상이면 여백이 많은 얼굴
            if (face_ratio > 4.6) :
                self.is_wide_margin = True
            else :
                self.is_wide_margin = False
            print("is_wide_margin = ", self.is_wide_margin)


            # 눈 양 끝, 아랫입술 가운데의 landmark를 이용해서 삼각형을 그리고 이목구비/전체얼굴 비율을 구한다.
            eye_x = face_landmarks.landmark[33].x - face_landmarks.landmark[263].x
            eye_y = face_landmarks.landmark[33].y - face_landmarks.landmark[263].y
            
            ## 얼굴 비율 측정 1, 긴 중안부 판단
            # 두 눈의 길이와 눈-코 길이 비교
            A = np.array([[eye_y/eye_x, -1], [-eye_x/eye_y, -1]])
            B = np.array([eye_y/eye_x*face_landmarks.landmark[33].x-face_landmarks.landmark[33].y, -eye_x/eye_y*face_landmarks.landmark[94].x-face_landmarks.landmark[94].y])
            x,y = np.linalg.solve(A,B)
            EtN_vertical_x = face_landmarks.landmark[94].x - x
            EtN_vertical_y = face_landmarks.landmark[94].y - y

            # Eye to Nose length
            EtN_len = np.sqrt(EtN_vertical_x**2 + EtN_vertical_y**2)  
            Eyes_len = np.sqrt(eye_x**2 + eye_y**2)
            
            # 결과값이 4.7 이상이면 긴 중안부
            if ((EtN_len/Eyes_len*10) > 4.7) :
                self.is_long_mid = True
            else :
                self.is_long_mid = False
            print("is_long_mid = ", self.is_long_mid)

            ## 얼굴 비율 측정 3, 긴 턱 판단 (중안부와 하안부의 비율)
            eyebrow_x = face_landmarks.landmark[105].x - face_landmarks.landmark[334].x
            eyebrow_y = face_landmarks.landmark[105].y - face_landmarks.landmark[334].y
            
            # 중안부 길이 구하기(눈썹 중간 - 코 끝)
            A = np.array([[eyebrow_y/eyebrow_x, -1], [-eyebrow_x/eyebrow_y, -1]])
            B = np.array([eyebrow_y/eyebrow_x*face_landmarks.landmark[105].x-face_landmarks.landmark[105].y, -eyebrow_x/eyebrow_y*face_landmarks.landmark[94].x-face_landmarks.landmark[94].y])
            x,y = np.linalg.solve(A,B)
            middle_face_x = face_landmarks.landmark[94].x - x
            middle_face_y = face_landmarks.landmark[94].y - y

            # Brow to Nose length
            BtN_len = np.sqrt(middle_face_x**2 + middle_face_y**2)
            
            # 하안부 길이 구하는 방법, 중안부의 길이를 빼줌
            A = np.array([[eyebrow_y/eyebrow_x, -1], [-eyebrow_x/eyebrow_y, -1]])
            B = np.array([eyebrow_y/eyebrow_x*face_landmarks.landmark[105].x-face_landmarks.landmark[105].y, -eyebrow_x/eyebrow_y*face_landmarks.landmark[152].x-face_landmarks.landmark[152].y])
            x,y = np.linalg.solve(A,B)
            middle_lower_face_x = face_landmarks.landmark[152].x - x
            middle_lower_face_y = face_landmarks.landmark[152].y - y

            # Eyebrow to Chin length
            BtC_len = np.sqrt(middle_lower_face_x**2 + middle_lower_face_y**2)

            middle_lower_length_ratio = BtN_len/(BtC_len-BtN_len)

            # 결과값이 1.1보다 작으면, 긴 턱
            if middle_lower_length_ratio < 1.1 :
                self.is_long_chin = True

            ## 얼굴 비율 측정 3, 긴 턱 판단 (인중 길이 대비 턱의 길이가 2배보다 길때)  
            else :
                # 코끝 - 윗 입술
                injung_x = face_landmarks.landmark[94].x - face_landmarks.landmark[0].x
                injung_y = face_landmarks.landmark[94].y - face_landmarks.landmark[0].y

                InJung_len = np.sqrt(injung_x**2 + injung_y**2)
            
                # 아랫 입술 - 턱 끝
                chin_x = face_landmarks.landmark[17].x - face_landmarks.landmark[152].x
                chin_y = face_landmarks.landmark[17].y - face_landmarks.landmark[152].y

                Chin_len = np.sqrt(chin_x**2 + chin_y**2)

                # 결과값이 1보다 크면, 긴 턱
                if Chin_len/(2*InJung_len) > 1:
                    self.is_long_chin = True
                else :
                    self.is_long_chin = False
            print("is_long_chin : ", self.is_long_chin)

            ## 얼굴 비율 측정 2, 하안부 중 긴 인중 판단
            nose2lip_x = face_landmarks.landmark[94].x - face_landmarks.landmark[17].x
            nose2lip_y = face_landmarks.landmark[94].y - face_landmarks.landmark[17].y
            lip2chin_x = face_landmarks.landmark[17].x - face_landmarks.landmark[152].x
            lip2chin_y = face_landmarks.landmark[17].y - face_landmarks.landmark[152].y
            
            # Nose to Under-Lip length
            NtL_len = np.sqrt(nose2lip_x**2 + nose2lip_y**2)

            # Under-Lip to Chin length
            LtC_len = np.sqrt(lip2chin_x**2 + lip2chin_y**2)

            length_ratio = (NtL_len*0.8)/LtC_len

            # 결과값이 0.9 이상이면 긴 인중
            if length_ratio > 0.9:
                self.is_long_philtrum = True
            else :
                self.is_long_philtrum = False
            print("is_long_philtrum = ", self.is_long_philtrum)

            # 이목구비 분석결과 출력
            # margin = Label(self, text="Is long margin: "+str(self.is_wide_margin)).pack(side='left', pady=5)
            # mid = Label(self, text="Is long mid: "+str(self.is_long_mid)).pack(side='left', pady=5)
            # philtrum = Label(self, text="Is long chin: "+str(self.is_long_chin)).pack(side='left',pady=5)
            # philtrum = Label(self, text="Is long philtrum: "+str(self.is_long_philtrum)).pack(side='left',pady=5)

    def print_image(self, file_name):
        image = PhotoImage(file='./UI_img/'+file_name+'.png').subsample(6)
        self.result_label = Label(image=image)
        self.result_label.image = image
        self.result_label.pack()


    def face_analysis(self):
        if int(int(self.pred_output)) == 0: # HEART
            print('Heart')
            if (self.is_wide_margin==True):
                print('Heart1')
                self.print_image("heart1qr")
            # if (self.is_wide_margin==False):
            #     print('Heart2')
            #     self.print_image("heart2qr")
            # else:
            #     print('Face feture Error')
            else:
                print('Heart2')
                self.print_image("heart2qr")

        elif int(int(self.pred_output)) == 1: # OBLONG
            print('Oblong')
            if (self.is_long_mid==True) or (self.is_long_philtrum==True):
                print('Oblong1')
                self.print_image("oblong1qr")
            elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==True):
                print('Oblong2')
                self.print_image("oblong2qr")
            # elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==False):
            #     print('Oblong3')
            #     self.print_image("oblong3qr")
            # else:
            #     print('Face feture Error')
            else:
                print('Oblong3')
                self.print_image("oblong3qr")


        elif int(int(self.pred_output)) == 2:
            print('Oval')
            if (self.is_long_mid==True) and (self.is_long_philtrum==False) and (self.is_wide_margin==False):
                print('Oval1')
                self.print_image("oval1qr")
            elif (self.is_long_mid==True) and (self.is_long_philtrum==False) and (self.is_wide_margin==False):
                print('Oval2')
                self.print_image("oval2qr")
            elif (self.is_long_mid==False) and (self.is_long_philtrum==True) and (self.is_long_chin==False) and (self.is_wide_margin==False):
                print('Oval3')
                self.print_image("oval3qr")
            elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==True) and (self.is_wide_margin==False):
                print('Oval4')
                self.print_image("oval4qr")
            # elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==False) and (self.is_wide_margin==True):
            #     print('Oval5')
            #     self.print_image("oval5qr")
            # else:
            #     print('Face feture Error')
            else:
                print('Oval5')
                self.print_image("oval5qr")

        elif int(int(self.pred_output)) == 3:
            print('Round')
            if (self.is_long_mid==True) or (self.is_long_philtrum==True) and (self.is_wide_margin==False):
                print('Round1')
                self.print_image("round1qr")
            elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==True) and (self.is_wide_margin==True):
                print('Round2')
                self.print_image("round2qr")
            # elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==False) and (self.is_wide_margin==False):
            #     print('Round3')
            #     self.print_image("round3qr")
            # else:
            #     print('Face feture Error')
            else:
                print('Round3')
                self.print_image("round3qr")

        elif int(int(self.pred_output)) == 4:
            print('Square')
            if (self.is_long_mid==True) or (self.is_long_philtrum==True) and (self.is_wide_margin==False):
                print('Square1')
                self.print_image("square1qr")
            elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==True) and (self.is_wide_margin==False):
                print('Square2')
                self.print_image("square2qr")
            elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==False) and (self.is_wide_margin==True):
                print('Square3')
                self.print_image("square3qr")
            # elif (self.is_long_mid==False) and (self.is_long_philtrum==False) and (self.is_long_chin==False) and (self.is_wide_margin==False):
            #     print('Square4')
            #     self.print_image("square4qr")
            # else:
            #     print('Face feture Error')
            #     self.print_image("square3qr")
            else:
                print('Square4')
                self.print_image("square4qr")

        else:
            print("Face Shape Error")


# self.is_wide_margin # 여백
# self.is_long_mid # 긴 중안부
# self.is_long_chin # 긴 턱
# self.is_long_philtrum # 긴 인중

        
        
if __name__ == "__main__":
    app = SampleApp()
    app.title('demo')
    app.geometry(str(GUI_WIDTH)+'x'+str(GUI_HEIGHT))
    app.mainloop()
