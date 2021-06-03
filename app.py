import os.path
import cv2
import torch
import zipfile
import numpy as np
import warnings
from PIL import ImageTk, Image
import concurrent.futures

try:
    from gdown import download as gdown_download
except:
    print("error importing gdown. Hopefully the archive is already downloaded..")

warnings.filterwarnings("ignore")

from skimage.transform import resize
from animate import normalize_kp
from demo import load_checkpoints

import tkinter as tk

USE_CPU = not torch.cuda.is_available()
INSTALLDIR = os.path.dirname(__file__)

print(
    "WARNING: CUDA not available, falling back to CPU. THIS WILL BE HORRIBLY SLOW"
    if USE_CPU
    else "Found CUDA, running the neural network with hardware acceleration."
)


def prep_frame(frame, zoom_factor=0.8):
    frame = cv2.flip(frame, 1)

    zoom_factor = max(min(zoom_factor if zoom_factor is not None else 0.8, 1), 0.25)

    w = int(min(*frame.shape[0:2]) * zoom_factor)
    h = w
    x = frame.shape[1] // 2 - w // 2
    y = frame.shape[0] // 2 - h // 2

    frame = frame[y : y + h, x : x + w, :]

    return resize(frame, (256, 256))[..., :3]


class VideoDisplay(tk.Widget):
    def __init__(self, parent, cap, oncamloaded=None, zoom_factor_var=None, crop=True):
        tk.Frame.__init__(self, parent)
        self.cap = cap
        self.crop = crop
        self.oncamloaded = oncamloaded
        self.zoom_factor_var = zoom_factor_var

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
        if self.oncamloaded is not None:
            self.oncamloaded()

    def show_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if frame is not None:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                if self.crop:
                    zoom_factor = (
                        self.zoom_factor_var.get() if self.zoom_factor_var else None
                    )
                    cv2image = (
                        prep_frame(cv2image, zoom_factor=zoom_factor) * 255
                    ).astype(np.uint8)
                self.img = Image.fromarray(cv2image)
                self.imgtk = ImageTk.PhotoImage(image=self.img)
                self.image_label.configure(image=self.imgtk)
            else:
                print("no frame?", ret)
        else:
            print("no cap?")

        self.after(10, self.show_frame)


class VideoCapture(tk.Widget):
    def __init__(self, parent, oncamloaded=None):
        tk.Frame.__init__(self, parent)

        self.refresh_button = tk.Button(self)
        self.refresh_button["text"] = "Refresh Camera List"
        self.refresh_button["command"] = self.update_camera_list_and_repack

        self.reopen_button = tk.Button(self)
        self.reopen_button["text"] = "Reopen Current Camera"
        self.reopen_button["command"] = self.update_capture

        self.cam_dropdown = None
        self.video_display = VideoDisplay(
            self, None, oncamloaded=oncamloaded, crop=True
        )
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
        self.filename = tk.StringVar(
            self,
            os.path.join(INSTALLDIR, "source_image_inputs", "the_rock_colorkey.jpeg"),
        )

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

        if USE_CPU:
            cuda_warning = tk.Label(self, fg="red")
            cuda_warning["text"] = "WARNING: Could not find CUDA. The neural network will be run on the CPU, and will not be realtime capable."
            cuda_warning.pack(side="bottom")

        model_future.add_done_callback(self.load_complete)

        self.pack_steal_button()
        left_frame = tk.Frame(self)
        left_frame.pack(side="left")
        self.select_image_button = tk.Button(left_frame)
        self.pack_button()
        self.image_label = tk.Label(left_frame)
        self.pack_preview_img()
        self.video_capture = VideoCapture(self, oncamloaded=self.check_steal_button)
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
        self.image_label.pack(side="top")

    def pack_button(self):
        self.select_image_button["text"] = os.path.basename(self.filename.get())
        self.select_image_button["command"] = self.update_img
        self.select_image_button.pack(side="bottom")

    def update_img(self):
        filename = tk.filedialog.askopenfilename(
            initialdir=os.path.join(INSTALLDIR, "source_image_inputs"),
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


class Distorter(tk.Frame):
    def __init__(
        self,
        parent,
        source_image,
        video_capture,
        generator,
        kp_detector,
        use_relative_movement_var=None,
        use_relative_jacobian_var=None,
        adapt_movement_scale_var=None,
        zoom_factor_var=None,
    ):
        tk.Frame.__init__(self, parent)
        self.source_image = source_image
        self.source_image_arr = np.asarray(self.source_image)[..., :3] / 255
        self.video_capture = video_capture
        self.generator = generator
        self.kp_detector = kp_detector

        self.zoom_factor_var = zoom_factor_var
        self.use_relative_movement = (
            use_relative_movement_var
            if use_relative_movement_var is not None
            else tk.BooleanVar(self, True)
        )
        self.use_relative_jacobian = (
            use_relative_jacobian_var
            if use_relative_jacobian_var is not None
            else tk.BooleanVar(self, True)
        )
        self.adapt_movement_scale = (
            adapt_movement_scale_var
            if adapt_movement_scale_var is not None
            else tk.BooleanVar(self, True)
        )

        self.pending_frame_promise = None
        self.kp_driving_initial = None
        self.kp_source = None
        self.source_tensor = None
        self.running = False

        self.init_network()

        self.pack()
        self.create_widgets()

        self.after(100, self.kickoff_mainthread_start)

    def get_prepped_frame(self):
        frame = self.video_capture.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        zoom_factor = self.zoom_factor_var.get() if self.zoom_factor_var else None
        frame = prep_frame(frame, zoom_factor=zoom_factor)
        return frame

    def init_network(self):
        frame = self.get_prepped_frame()

        self.source_tensor = torch.tensor(
            self.source_image_arr[np.newaxis].astype(np.float32)
        ).permute(0, 3, 1, 2)
        if not USE_CPU:
            self.source_tensor = self.source_tensor.cuda()

        kp_driving_initial_future = executor.submit(
            oneshot_run_kp, self.kp_detector, frame
        )
        kp_driving_initial_future.add_done_callback(self.set_kp_driving_initial)

        kp_source_future = executor.submit(
            oneshot_run_kp, self.kp_detector, self.source_image_arr
        )
        kp_source_future.add_done_callback(self.set_kp_source)

    def set_kp_driving_initial(self, kp_driving_initial_future):
        print("calculated kp_driving_initial")
        self.kp_driving_initial = kp_driving_initial_future.result()
        self.check_and_start_run()

    def set_kp_source(self, kp_source_future):
        print("calculated kp_source")
        self.kp_source = kp_source_future.result()
        self.check_and_start_run()

    def check_and_start_run(self):
        if self.running:
            return
        elif self.kp_driving_initial and self.kp_source:
            self.running = True

    def kickoff_mainthread_start(self):
        if self.running:
            self.show_frame()
        else:
            self.after(100, self.kickoff_mainthread_start)

    def show_frame(self):
        future = executor.submit(
            self.run_network_frame,
            self.get_prepped_frame(),
            self.source_tensor,
            self.kp_source,
            self.kp_driving_initial,
            self.use_relative_movement.get(),
            self.use_relative_jacobian.get(),
            self.adapt_movement_scale.get(),
            self.generator,
        )
        future.add_done_callback(self.render_and_run_again)

    def render_and_run_again(self, im_arr_future):
        try:
            self.img = Image.fromarray(
                cv2.cvtColor(
                    (im_arr_future.result() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
            )
            self.imgtk = ImageTk.PhotoImage(image=self.img)
            self.image_label.configure(image=self.imgtk)

            self.show_frame()
        except:
            self.after(1000, self.show_frame)

    def run_network_frame(
        self,
        frame,
        source_tensor,
        kp_source,
        kp_driving_initial,
        use_relative_movement,
        use_relative_jacobian,
        adapt_movement_scale,
        generator,
    ):
        driving_frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(
            0, 3, 1, 2
        )

        if not USE_CPU:
            driving_frame = driving_frame.cuda()

        kp_driving = kp_detector(driving_frame)
        kp_norm = normalize_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
            use_relative_movement=use_relative_movement,
            use_relative_jacobian=use_relative_jacobian,
            adapt_movement_scale=adapt_movement_scale,
        )

        out = generator(source_tensor, kp_source=kp_source, kp_driving=kp_norm)
        im = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        return im

    def create_widgets(self):
        self.img = self.source_image
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.image_label = tk.Label(self, image=self.imgtk)
        self.image_label.pack(side="left")

    def recalculate_initial_frame(self):
        frame = self.get_prepped_frame()
        kp_driving_initial_future = executor.submit(
            oneshot_run_kp, self.kp_detector, frame
        )
        kp_driving_initial_future.add_done_callback(self.set_kp_driving_initial)


def oneshot_run_kp(kp_detector, frame):
    source1 = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not USE_CPU:
        source1 = source1.cuda()

    return kp_detector(source1)


class RunSimulationApplication(tk.Frame):
    def __init__(
        self, source_image, video_capture, generator, kp_detector, master=None
    ):
        super().__init__(master)
        self.source_image = source_image
        self.video_capture = video_capture
        self.generator = generator
        self.kp_detector = kp_detector

        self.use_relative_movement_var = tk.BooleanVar(self, True)
        self.use_relative_jacobian_var = tk.BooleanVar(self, True)
        self.adapt_movement_scale_var = tk.BooleanVar(self, True)
        self.zoom_factor_var = tk.DoubleVar(self, 0.8)

        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(side="top")
        self.distorter = Distorter(
            top_frame,
            self.source_image,
            self.video_capture,
            self.generator,
            self.kp_detector,
            use_relative_movement_var=self.use_relative_movement_var,
            use_relative_jacobian_var=self.use_relative_jacobian_var,
            adapt_movement_scale_var=self.adapt_movement_scale_var,
            zoom_factor_var=self.zoom_factor_var,
        )
        self.distorter.pack(side="left")
        self.video_display = VideoDisplay(
            top_frame,
            self.video_capture,
            crop=True,
            zoom_factor_var=self.zoom_factor_var,
        )
        self.video_display.pack(side="right")

        slider_frame = tk.Frame(self)
        slider_frame.pack(side="bottom")
        slider_label = tk.Label(slider_frame)
        slider_label["text"] = "Camera Zoom:"
        slider_label.pack(side="left")
        slider = tk.Scale(
            slider_frame,
            from_=0.25,
            to=1.0,
            resolution=0.01,
            variable=self.zoom_factor_var,
            orient="horizontal",
            sliderlength=50,
        )
        slider.pack(side="left")
        resetbutton = tk.Button(slider_frame)
        resetbutton["text"] = "Reset Initial Frame"
        resetbutton["command"] = self.distorter.recalculate_initial_frame
        resetbutton.pack(side="right")

        checkbox_frame = tk.Frame(self)
        checkbox_frame.pack(side="bottom")

        checkbox_1 = tk.Checkbutton(
            checkbox_frame,
            selectcolor="red",
            text="relative_movement",
            onvalue=True,
            offvalue=False,
            variable=self.use_relative_movement_var,
        )
        checkbox_1.pack(side="left")
        checkbox_2 = tk.Checkbutton(
            checkbox_frame,
            selectcolor="red",
            text="relative_jacobian",
            onvalue=True,
            offvalue=False,
            variable=self.use_relative_jacobian_var,
        )
        checkbox_2.pack(side="left")
        checkbox_3 = tk.Checkbutton(
            checkbox_frame,
            selectcolor="red",
            text="adapt_movement_scale",
            onvalue=True,
            offvalue=False,
            variable=self.adapt_movement_scale_var,
        )
        checkbox_3.pack(side="left")


param_inputimg = None
param_inputcap = None


def download_and_load_model(use_advanced=False):
    USE_CPU = not torch.cuda.is_available()

    model_checkpoint_exist = os.path.exists("extract/vox-cpk.pth.tar")
    if not model_checkpoint_exist:
        url = "https://drive.google.com/uc?id=1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK"
        output = os.path.join("temp", "checkpoints.zip")
        gdown_download(url, output, quiet=False)
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall("extract")

    generator, kp_detector = load_checkpoints(
        config_path=("config/vox-adv-256.yaml" if use_advanced else "config/vox-256.yaml"),
        checkpoint_path=("extract/vox-adv-cpk.pth.tar" if use_advanced else "extract/vox-cpk.pth.tar"),
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


# if param_inputimg and param_inputcap:
#     USE_CPU = not torch.cuda.is_available()

#     if USE_CPU:
#         print("CUDA IS NOT AVAILABLE: USING CPU RENDERING")

#     relative = True
#     adapt_movement_scale = False

#     source_image = np.asarray(param_inputimg)
#     print(source_image.shape)
#     source_image = source_image[..., :3] / 255
#     print(source_image.shape)
#     cap = param_inputcap

#     # render just the camera for a frame while we load
#     ret, frame = cap.read()
#     frame = prep_frame(frame)
#     cv2.imshow("Face Borrower", frame)

#     if not os.path.exists("temp"):
#         os.mkdir("temp")

#     count = 0
#     try:
#         while True:

#             ret, frame = cap.read()

#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#             with torch.no_grad():
#                 predictions = []
#                 source = torch.tensor(
#                     source_image[np.newaxis].astype(np.float32)
#                 ).permute(0, 3, 1, 2)
#                 if not USE_CPU:
#                     source = source.cuda()
#                 kp_source = kp_detector(source)
#                 ims = [source_image]
#                 frame = prep_frame(frame)

#                 if count == 0:
#                     source_image1 = frame
#                     source1 = torch.tensor(
#                         source_image1[np.newaxis].astype(np.float32)
#                     ).permute(0, 3, 1, 2)
#                     kp_driving_initial = kp_detector(source1)

#                 if count % 1 == 0:
#                     frame_test = torch.tensor(
#                         frame[np.newaxis].astype(np.float32)
#                     ).permute(0, 3, 1, 2)

#                     driving_frame = frame_test
#                     if not USE_CPU:
#                         driving_frame = driving_frame.cuda()
#                     kp_driving = kp_detector(driving_frame)
#                     kp_norm = normalize_kp(
#                         kp_source=kp_source,
#                         kp_driving=kp_driving,
#                         kp_driving_initial=kp_driving_initial,
#                         use_relative_movement=relative,
#                         use_relative_jacobian=relative,
#                         adapt_movement_scale=adapt_movement_scale,
#                     )

#                     out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
#                     #         predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
#                     im = np.transpose(
#                         out["prediction"].data.cpu().numpy(), [0, 2, 3, 1]
#                     )[0]
#                     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#                 joinedFrame = np.concatenate((im, frame), axis=1)

#                 #         joinedFrameToSave = np.uint8(256 * joinedFrame)
#                 #             out1.write(joinedFrameToSave)

#                 cv2.imshow("Face Borrower", joinedFrame)

#                 count += 1
#     except:
#         raise
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
