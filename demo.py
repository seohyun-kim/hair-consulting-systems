import cv2 as cv
import numpy as np
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

from tkinter import *
import PIL.Image, PIL.ImageTk

# 실행 버튼 이벤트
def e_start():
    print("event start")

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
        
    def update(self):
        ret, frame = self.cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if self.canvas_on_down == True:
            frame = cv.rectangle(frame, (self.canvas_start_x, self.canvas_start_y), (self.canvas_move_x, self.canvas_move_y), (0, 0, 255), 2)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
        self.window.after(self.delay, self.update)




class MainPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.pack(side="bottom")
        Button(self, text="Start", command=lambda: master.switch_frame(GetImagePage)).pack(pady=10)
        

class GetImagePage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        cam_frame = Frame(self, bg='white', width=400, height=400)
        cam_frame.pack(side='top', pady=10)
        Button(self, text="Capture", command=lambda: master.switch_frame(AnalysisPage)).pack(pady=10)
        
        # cap = cv.VideoCapture(0) # VideoCapture 객체 정의
        cap = cv.VideoCapture('http://192.168.0.8:4747/video')
        if not cap.isOpened():
            raise ValueError("Unable to open video source", 0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 400)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 400)
        cam_frame.update()

        
class AnalysisPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        
        
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