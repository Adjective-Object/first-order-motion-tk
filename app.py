import os.path
from tkinter.constants import S
import cv2
import torch
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import gdown
import warnings
from PIL import ImageTk, Image
import concurrent.futures

warnings.filterwarnings("ignore")

from skimage.transform import resize
from animate import normalize_kp
from demo import load_checkpoints

import tkinter as tk

def prep_frame(frame):
    frame = cv2.flip(frame, 1)

    w = int(min(*frame.shape[0:2]) * 0.8)
    h = w
    x = frame.shape[1] // 2 - w // 2
    y = frame.shape[0] // 2 - h // 2

    frame = frame[y : y + h, x : x + w, :]

    return resize(frame, (256, 256))[..., :3]

class VideoDisplay(tk.Widget):
    def __init__(self, parent, cap, crop=False):
        tk.Frame.__init__(self, parent)
        self.cap = cap
        self.crop=crop

        if cap and not cap.isOpened():
            raise ValueError("Cap was not open?")

        self.img = None
        self.imgtk = None
        self.image_label = tk.Label(self)
        self.image_label.pack(side="top")
        self.after(10, self.show_frame)

    def set_cap(self, cap):
        self.cap = cap
        if cap is None:
            self.imgtk = None
            self.img = None

    def show_frame(self):
        print("SHOW FRAME", self.__hash__())
        if self.cap is not None:
            ret, frame = self.cap.read()
            if frame is not None:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                if self.crop:
                    cv2image = (prep_frame(cv2image) * 255).astype(np.uint8)
                self.img = Image.fromarray(cv2image)
                self.imgtk = ImageTk.PhotoImage(image=self.img)
                self.image_label.configure(image=self.imgtk)
            else:
                print("no frame?", ret)
        else:
            print("no cap?")

        self.after(10, self.show_frame)


class VideoCapture(tk.Widget):
    def __init__(self, parent, oncamloaded):
        tk.Frame.__init__(self, parent)

        self.refresh_button = tk.Button(self)
        self.refresh_button["text"] = "Refresh Camera List"
        self.refresh_button["command"] = self.update_camera_list_and_repack

        self.oncamloaded = oncamloaded

        self.reopen_button = tk.Button(self)
        self.reopen_button["text"] = "Reopen Current Camera"
        self.reopen_button["command"] = self.update_capture

        self.cam_dropdown = None
        self.video_display = VideoDisplay(self, None, crop=False)
        self.error_label = tk.Label(self, fg="red")

        self.repack()
        self.selected_camera = tk.StringVar()
        self.selected_camera.trace("w", self.update_capture)
        self.after(1, self.update_camera_list_and_repack)

    def repack(self):
        self.refresh_button.pack(side="bottom")
        self.reopen_button.pack(side="bottom")
        if self.cam_dropdown is not None:
            self.cam_dropdown.pack(side="bottom")
        self.error_label.pack(side="top")
        self.video_display.pack(side="top")

    def update_capture(self, *argv):
        idx_str = self.selected_camera.get()[len("Camera ") :]
        if len(idx_str):
            idx = int(idx_str)
            print("updating capture to idx", idx)
            cap = cv2.VideoCapture(idx)

            if not cap.isOpened():
                self.error_label["text"] = "Error Opening Camera %s" % idx
                self.video_display.set_cap(None)
                self.repack()
            else:
                self.video_display.set_cap(cap)
                self.error_label["text"] = ""
                self.repack()

        else:
            self.video_display.set_cap(None)

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


class GetInputsApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.filename = tk.StringVar(self, "source_image_inputs/the_rock_colorkey.jpeg")

        self.master = master
        self.pack()
        self.create_widgets()

        self.check_steal_button()

    def create_widgets(self):
        self.steal_button = tk.Button(self)
        self.steal_button["command"] = self.run_stolen_face

        self.loading_label = tk.Label(self, fg="red")
        self.loading_label["text"] = "Loading neural network.."
        self.loading_label.pack(side="bottom")

        model_future.add_done_callback(self.load_complete)

        self.pack_steal_button()
        self.hi_there = tk.Button(self)
        self.pack_button()
        self.image_label = tk.Label(self)
        self.pack_preview_img()
        self.video_capture = VideoCapture(self, self.check_steal_button)
        self.pack_videocapture()

    def load_complete(self, res):
        self.loading_label["text"] = "model loaded!"
        self.loading_label["fg"] = "green"
        self.check_steal_button()

    def pack_steal_button(self):
        self.steal_button["text"] = "Borrow that face!"
        self.steal_button["state"] = tk.DISABLED
        self.steal_button.pack(side="bottom")

    def pack_videocapture(self):
        self.video_capture.pack(side="bottom")

    def pack_preview_img(self):
        self.img = Image.open(self.filename.get()).resize((256, 256))
        self.photo_img = ImageTk.PhotoImage(self.img)
        self.image_label.configure(image=self.photo_img)
        self.image_label.image = self.photo_img
        self.image_label.pack(side="left")

    def pack_button(self):
        self.hi_there["text"] = os.path.basename(self.filename.get())
        self.hi_there["command"] = self.update_img
        self.hi_there.pack(side="left")

    def update_img(self):
        filename = tk.filedialog.askopenfilename(
            initialdir="./source_image_inputs",
            title="Select a File",
            filetypes=(
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("all files", "*.*"),
            ),
        )
        if filename is not None:
            self.filename.set(filename)
            self.pack_preview_img()

            self.pack_button()
            self.pack_preview_img()
            self.check_steal_button()

    def check_steal_button(self, *argv):
        if (
            self.img is not None
            and self.video_capture.video_display.cap is not None
            and self.video_capture.video_display.cap.isOpened()
            and model_future.done()
        ):
            self.steal_button["state"] = tk.ACTIVE
        else:
            self.steal_button["state"] = tk.DISABLED

    def run_stolen_face(self):
        global param_inputimg
        global param_inputcap
        if (
            self.img is not None
            and self.video_capture.video_display.cap is not None
            and self.video_capture.video_display.cap.isOpened()
        ):
            param_inputimg = self.img
            param_inputcap = self.video_capture.video_display.cap
            self.quit()


class RunSimulationApplication(tk.Frame):
    def __init__(
        self, source_image, video_capture, generator, kp_detector, master=None
    ):
        super().__init__(master)
        self.source_image = source_image
        self.video_capture = video_capture
        self.generator = generator
        self.kp_detector = kp_detector

        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.video_display = VideoDisplay(self, self.video_capture, crop=True)
        self.video_display.pack(side="right")


param_inputimg = None
param_inputcap = None


def download_and_load_model():
    USE_CPU = not torch.cuda.is_available()

    model_checkpoint_exist = os.path.exists("extract/vox-cpk.pth.tar")
    if not model_checkpoint_exist:
        url = "https://drive.google.com/uc?id=1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK"
        output = "temp/checkpoints.zip"
        gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall("extract")

    generator, kp_detector = load_checkpoints(
        config_path="config/vox-256.yaml",
        checkpoint_path="extract/vox-cpk.pth.tar",
        cpu=USE_CPU,
    )

    return generator, kp_detector


executor = concurrent.futures.ThreadPoolExecutor()
model_future = executor.submit(download_and_load_model)

root = tk.Tk()
app = GetInputsApplication(master=root)
app.mainloop()
app.destroy()
generator, kp_detector = model_future.result()
app2 = RunSimulationApplication(
    param_inputimg, param_inputcap, generator, kp_detector, master=root
)
app2.mainloop()
app2.destroy()
root.destroy()


if param_inputimg and param_inputcap:
    USE_CPU = not torch.cuda.is_available()

    if USE_CPU:
        print("CUDA IS NOT AVAILABLE: USING CPU RENDERING")

    relative = True
    adapt_movement_scale = False

    source_image = np.asarray(param_inputimg)
    print(source_image.shape)
    source_image = source_image[..., :3] / 255
    print(source_image.shape)
    cap = param_inputcap

    # render just the camera for a frame while we load
    ret, frame = cap.read()
    frame = prep_frame(frame)
    cv2.imshow("Face Borrower", frame)

    if not os.path.exists("temp"):
        os.mkdir("temp")

    count = 0
    try:
        while True:

            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            with torch.no_grad():
                predictions = []
                source = torch.tensor(
                    source_image[np.newaxis].astype(np.float32)
                ).permute(0, 3, 1, 2)
                if USE_CPU:
                    source = source.cuda()
                kp_source = kp_detector(source)
                ims = [source_image]
                frame = prep_frame(frame)

                if count == 0:
                    source_image1 = frame
                    source1 = torch.tensor(
                        source_image1[np.newaxis].astype(np.float32)
                    ).permute(0, 3, 1, 2)
                    kp_driving_initial = kp_detector(source1)

                if count % 1 == 0:
                    frame_test = torch.tensor(
                        frame[np.newaxis].astype(np.float32)
                    ).permute(0, 3, 1, 2)

                    driving_frame = frame_test
                    if not USE_CPU:
                        driving_frame = driving_frame.cuda()
                    kp_driving = kp_detector(driving_frame)
                    kp_norm = normalize_kp(
                        kp_source=kp_source,
                        kp_driving=kp_driving,
                        kp_driving_initial=kp_driving_initial,
                        use_relative_movement=relative,
                        use_relative_jacobian=relative,
                        adapt_movement_scale=adapt_movement_scale,
                    )
                    out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                    #         predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                    im = np.transpose(
                        out["prediction"].data.cpu().numpy(), [0, 2, 3, 1]
                    )[0]
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                joinedFrame = np.concatenate((im, frame), axis=1)

                #         joinedFrameToSave = np.uint8(256 * joinedFrame)
                #             out1.write(joinedFrameToSave)

                cv2.imshow("Face Borrower", joinedFrame)

                count += 1
    except:
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
