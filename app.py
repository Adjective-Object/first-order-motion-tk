import os.path
from typing import DefaultDict
import cv2
import torch
import zipfile
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gdown
import warnings
from PIL import ImageTk, Image

warnings.filterwarnings("ignore")

from skimage.transform import resize
from animate import normalize_kp
from demo import load_checkpoints

import tkinter as tk

class VideoCapture(tk.Widget):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.refresh_button = tk.Button(self)
        self.refresh_button["text"] = "Refresh Camera List"
        self.refresh_button["command"] = self.update_camera_list_and_repack


        self.reopen_button = tk.Button(self)
        self.reopen_button["text"] = "Reopen Current Camera"
        self.reopen_button["command"] = self.update_capture

        self.cam_dropdown = None
        self.cap = None
        self.img = None
        self.imgtk = None
        self.image_label = tk.Label(self)
        self.error_label = tk.Label(self, fg="red")
        self.after(10, self.show_frame)

        self.repack()
        self.selected_camera = tk.StringVar()
        self.selected_camera.trace('w', self.update_capture)
        self.update_camera_list_and_repack()
    
    def repack(self):
        self.refresh_button.pack(side="bottom")
        self.reopen_button.pack(side="bottom")
        if self.cam_dropdown is not None:
            self.cam_dropdown.pack(side="bottom")
        self.error_label.pack(side="top")
        self.image_label.pack(side="top")

    def update_capture(self, *argv):
        idx_str = self.selected_camera.get()[len("Camera "):]
        if len(idx_str):
            idx = int(idx_str)
            print('updating capture to idx', idx)
            self.cap = cv2.VideoCapture(idx)

            if not self.cap.isOpened():
                self.error_label["text"] = "Error Opening Camera %s" % idx
                self.cap = None
                self.imgtk = None
                self.img = None
                self.repack()
            else:
                self.error_label["text"] = ""
                self.repack()

        else:
            self.cap = None


    def show_frame(self):
        if self.cap is not None:
            _, frame = self.cap.read()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                self.img = Image.fromarray(cv2image)
                self.imgtk = ImageTk.PhotoImage(image=self.img)
                self.image_label.configure(image=self.imgtk)

        self.after(10, self.show_frame)


    def update_camera_list_and_repack(self):
        i = 0
        cameras = []
        while True:
            cap = cv2.VideoCapture(i)
            if i > 10 and (not cap.isOpened() or cap.read()[1] == None):
                break
            if cap.isOpened():
                cameras.append("Camera %s" % i)
            i += 1
        
        
        if len(cameras):
            self.video_capture = cv2.VideoCapture(len(cameras) - 1)

        print(self.selected_camera.get())
        if self.selected_camera.get() == "":
            print("setting")
            self.selected_camera.set(cameras[0])

        if self.cam_dropdown is not None:
            self.cam_dropdown.destroy()
        
        if not len(cameras):
            cameras.append(())  

        self.cam_dropdown = tk.OptionMenu(self, self.selected_camera, *cameras)

        self.repack()

    def get(self):
        return self.entry.get()

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.filename = "source_image_inputs/the_rock_colorkey.jpeg";

        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.pack_button()
        self.image_label = tk.Label(self)
        self.pack_preview_img()
        self.video_capture = VideoCapture(self)
        self.pack_videocapture()

    def pack_steal_button(self):
        self.steal_button.pack(side="bottom")


    def pack_videocapture(self):
        self.video_capture.pack(side="bottom")

    def pack_preview_img(self):
        print(self.filename)
        self.img = Image.open(self.filename).resize((255,255))
        self.photo_img = ImageTk.PhotoImage(self.img)
        self.image_label.configure(image=self.photo_img)
        self.image_label.image=self.photo_img
        self.image_label.pack(side="left")
    
    def pack_button(self):
        self.hi_there["text"] = os.path.basename(self.filename)
        self.hi_there["command"] = self.update_img
        self.hi_there.pack(side="left")

    def update_img(self):
        filename = tk.filedialog.askopenfilename(
            initialdir="./source_image_inputs",
            title="Select a File",
            filetypes=(("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("all files", "*.*")),
        )
        if filename is not None:
            self.filename = filename
            self.pack_preview_img()

            self.pack_button()
            self.pack_preview_img()

param_inputimg = None
pram_inputcap = None

root = tk.Tk()
app = Application(master=root)
app.mainloop()




# USE_CPU = False

# if not os.path.exists("temp"):
#     os.mkdir("temp")

# relative = True
# adapt_movement_scale = False


# model_checkpoint_exist = os.path.exists("extract/vox-cpk.pth.tar")
# if not model_checkpoint_exist:
#     url = "https://drive.google.com/uc?id=1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK"
#     output = "temp/checkpoints.zip"
#     gdown.download(url, output, quiet=False)
#     with zipfile.ZipFile(output, "r") as zip_ref:
#         zip_ref.extractall("extract")

# generator, kp_detector = load_checkpoints(
#     config_path="config/vox-256.yaml",
#     checkpoint_path="extract/vox-cpk.pth.tar",
#     cpu=USE_CPU,
# )

# source_image = resize(
#     imageio.imread("source_image_inputs/harry-potter-3-colorkey.png"), (256, 256)
# )[..., :3]

# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# # out1 = cv2.VideoWriter('temp/animation_from_webcam.avi', fourcc, 12, (256*3 , 256), True)
# cv2_source = cv2.cvtColor(source_image.astype("float32"), cv2.COLOR_BGR2RGB)
# cap = cv2.VideoCapture(1)

# count = 0
# try:
#     while True:

#         ret, frame = cap.read()
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#         with torch.no_grad():
#             predictions = []
#             source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(
#                 0, 3, 1, 2
#             )
#             if not USE_CPU:
#                 source = source.cuda()
#             kp_source = kp_detector(source)
#             ims = [source_image]
#             frame = cv2.flip(frame, 1)
#             frame = frame[y : y + h, x : x + w]
#             frame1 = resize(frame, (256, 256))[..., :3]

#             if count == 0:
#                 source_image1 = frame1
#                 source1 = torch.tensor(
#                     source_image1[np.newaxis].astype(np.float32)
#                 ).permute(0, 3, 1, 2)
#                 kp_driving_initial = kp_detector(source1)

#             if count % 2 == 0:
#                 frame_test = torch.tensor(
#                     frame1[np.newaxis].astype(np.float32)
#                 ).permute(0, 3, 1, 2)

#                 driving_frame = frame_test
#                 if not USE_CPU:
#                     driving_frame = driving_frame.cuda()
#                 kp_driving = kp_detector(driving_frame)
#                 kp_norm = normalize_kp(
#                     kp_source=kp_source,
#                     kp_driving=kp_driving,
#                     kp_driving_initial=kp_driving_initial,
#                     use_relative_movement=relative,
#                     use_relative_jacobian=relative,
#                     adapt_movement_scale=adapt_movement_scale,
#                 )
#                 out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
#                 #         predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
#                 im = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
#                 im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#             joinedFrame = np.concatenate((im, frame1), axis=1)

#             #         joinedFrameToSave = np.uint8(255 * joinedFrame)
#             #             out1.write(joinedFrameToSave)

#             cv2.imshow("Test", joinedFrame)

#             count += 1
# except:
#     raise
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
