import cv2 as cv
import numpy as np
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

from tkinter import *
import PIL.Image, PIL.ImageTk

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
        self.pack(side="bottom")
        Button(self, text="Start", command=lambda: master.switch_frame(GetImagePage)).pack(pady=10)
        

class GetImagePage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.cam_frame = Frame(self, bg='white', width=400, height=400)
        self.cam_frame.pack(side='top', pady=10)
        Button(self, text="Capture", command=lambda: [master.switch_frame(AnalysisPage), self.stop_cam()]).pack(pady=10)
        
        self.cap = cv.VideoCapture(0) # VideoCapture 객체 정의
        # cap = cv.VideoCapture('http://192.168.0.8:4747/video')
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", 0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 400)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 400)
        self.canvas = Canvas(self.cam_frame, width=400, height=400)
        self.canvas.pack()
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        self.frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # if self.canvas_on_down == True:
        #     frame = cv.rectangle(frame, (self.canvas_start_x, self.canvas_start_y), (self.canvas_move_x, self.canvas_move_y), (0, 0, 255), 2)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame))
        self.canvas.create_image([0,0], anchor=NW, image=self.photo)
        self.cam_frame.after(30, self.update)

    def stop_cam(self):
        cv.imwrite('./img/test.jpg', self.frame)
        self.cap.release()


        # cam_frame.update()

        
class AnalysisPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.preprocess_image()
        self.result()

    def preprocess_image(self):
        target_img = cv.imread('./data_set/Heart/heart_mina.jpg')
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
            pre_processed_img = cv.filter2D(src=equalized, ddepth=-1, kernel=kernel)
        # 텐서화
        convert_tensor = transforms.ToTensor()
        processed_img = convert_tensor(pre_processed_img)

        with torch.no_grad():
            inputs = torch.FloatTensor(processed_img.unsqueeze(0))
            output = model(inputs)

            print(output)
            self.pred_output = output.argmax(dim=1, keepdim=True)
            print(self.pred_output)
            print(shape_class[int(self.pred_output)])
    
    def result(self):
        face_shape = Label(self, text="Face Shape: "+shape_class[int(self.pred_output)]).pack(pady=10)
        
        
if __name__ == "__main__":
    app = SampleApp()
    app.title('demo')
    app.geometry('500x500')
    app.mainloop()


# tk = Tk()
# tk.title("demo")
# start_button = Button(tk, text="실행", command=e_start)
# start_button.pack(padx=10, pady=10)
# tk.mainloop()