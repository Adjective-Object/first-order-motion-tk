from asyncio.tasks import Task
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import pool
import os.path
from sys import exit, argv, stderr
from queue import Empty
from typing import List, Union
import cv2
import torch
import zipfile
import numpy as np
import warnings
from PIL import ImageTk, Image
import time
import webbrowser
import multiprocessing.shared_memory as shared_memory
import multiprocessing as mp
import traceback
import faulthandler
from normalize_kp import normalize_kp
from demo import load_checkpoints
import tkinter.filedialog
import tkinter.font
import tkinter as tk
import asyncio
# import gc
# gc.disable()

faulthandler.enable()
torch.backends.cudnn.benchmark = True
start_time = 0

try:
    from gdown import download as gdown_download
except:
    print("error importing gdown. Hopefully the archive is already downloaded..", file=sys.stderr)

warnings.filterwarnings("ignore")

USE_CPU = not torch.cuda.is_available()
INSTALLDIR = os.path.dirname(__file__)
SHARED_MEM_ID = os.getppid()
TRACE_MALLOC = '--trace' in argv
TRACE_GC = '--trace-gc' in argv
DEBUG = "-v" in argv or "--verbose" in argv
header = ("[%s]" if __name__ == "__main__" else "(%s)") % os.getpid()


def debug(*argv, **kwarg):
    if not DEBUG:
        return
    print(header, *argv, file=stderr, **kwarg)


debug(
    "WARNING: CUDA not available, falling back to CPU. THIS WILL BE HORRIBLY SLOW"
    if USE_CPU
    else "Found CUDA, the network will be run with hardware acceleration"
)


class DumbFutureLike:
    """
    Because the tkinter and asyncio event loops both want to live on the main thread,
    and because I don't want to restructure the app to use callbacks to switch between the active app,
    I'm adding this callback-container object to maintain future-like semantics and callback registration
    without actually getting the benefit of async/await syntax.

    This is kind of the worst of both worlds
    """

    def __init__(self):
        self._callbacks = []
        self._complete = False
        self._result = None

    def set_result(self, result):
        self._result = result
        self._complete = True
        while len(self._callbacks) != 0:
            cb = self._callbacks.pop()
            cb(self)

    def add_done_callback(self, cb):
        if self._complete:
            cb(self)
        else:
            self._callbacks.append(cb)

    def result(self):
        if not self._complete:
            raise Exception("blocked on result of incomplete DumbFutureLike")
        else:
            return self._result

DEFAULT_NO_GRAB_FRAME_TIMEOUT = 1000/60
class CV2VideoCaptureWrapper:
    def __init__(self, cv2_video_capture, no_grab_frame_timeout_s=DEFAULT_NO_GRAB_FRAME_TIMEOUT):
        self.cv2_video_capture = cv2_video_capture
        self.no_grab_frame_timeout_s = no_grab_frame_timeout_s
        self.last_frame_time = None
        self._captured_frame_buf = None

    def request_video_frame(self, force_new=False):
        now = time.time()
        if (force_new
            or self._captured_frame_buf is None
            or self.last_frame_time == None
            or (now - self.last_frame_time) > self.no_grab_frame_timeout_s):
            self._get_frame()

        return self._captured_frame_buf

    def _get_frame(self):
        if self._captured_frame_buf is None:
            self._captured_frame_buf = self.cv2_video_capture.read()[1]
        else:
            self.cv2_video_capture.read(self._captured_frame_buf)[1]
    
    def isOpened(self):
        return self.cv2_video_capture.isOpened()

in_place_swap_arr = np.empty(shape=(256*256), dtype=np.uint8)
def bgr2rgb(arr: np.ndarray):
    slice_size = arr.shape[0] * arr.shape[1]
    if (slice_size > in_place_swap_arr.shape[0]):
        in_place_swap_arr.resize((slice_size,))

    view = in_place_swap_arr[0:slice_size]
    view[:] = arr[:,:,0].reshape((slice_size,))
    arr[:,:,0] = arr[:,:,2]
    arr[:,:,2] = view[:].reshape((arr.shape[0], arr.shape[1]))
    return arr

rgb2bgr = bgr2rgb

# def rgb2bgr(arr):
#     in_place_swap_arr[:,:,:] = arr[:,:,:]
#     arr[:,:,0] = arr[:,:,2]
#     arr[:,:,1] = arr[:,:,1]
#     arr[:,:,2] = arr[:,:,0]
#     return arr
#     # return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def prep_frame(frame, zoom_factor=0.8) -> Image:
    frame = bgr2rgb(frame)
    frame = cv2.flip(frame, 1)

    zoom_factor = max(min(zoom_factor if zoom_factor is not None else 0.8, 1), 0.25)

    w = int(min(*frame.shape[0:2]) * zoom_factor)
    h = w
    x = frame.shape[1] // 2 - w // 2
    y = frame.shape[0] // 2 - h // 2

    frame = frame[y : y + h, x : x + w, :]

    return Image.fromarray(frame).resize((256, 256), resample=Image.NEAREST)


def download_model():
    if not os.path.exists("temp"):
        os.mkdir("temp")

    url = "https://drive.google.com/uc?id=1wCzJP1XJNB04vEORZvPjNz6drkXm5AUK"
    output = os.path.join("temp", "checkpoints.zip")
    gdown_download(url, output, quiet=False)
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall("extract")


def load_model(use_advanced=False):
    generator, kp_detector = load_checkpoints(
        config_path=(
            "config/vox-adv-256.yaml" if use_advanced else "config/vox-256.yaml"
        ),
        checkpoint_path=(
            "extract/vox-adv-cpk.pth.tar" if use_advanced else "extract/vox-cpk.pth.tar"
        ),
        cpu=USE_CPU,
    )

    return generator, kp_detector


was_loaded = None


def was_model_loaded_at_start():
    global was_loaded
    if was_loaded is None:
        was_loaded = os.path.exists(os.path.join("extract", "vox-cpk.pth.tar"))
    return was_loaded


def download_and_load_model(app):
    if not was_model_loaded_at_start():
        debug("downloading model")
        download_model()
        app.download_complete()
    # print("loading model")
    # return load_model()


def get_shared_input_name(i):
    return "shared_input_%s_%s" % (SHARED_MEM_ID, i)


def get_shared_output_name(i):
    return "shared_output_%s_%s" % (SHARED_MEM_ID, i)


def get_source_img_name():
    return "source_img_shared_%s" % SHARED_MEM_ID


def get_initial_driving_img_name():
    return "initial_driving_img_shared_%s" % SHARED_MEM_ID


# 256 x 256 array of 3 uint8s (RGB)
# hack: double the size?
DATA_SIZE = 256 * 256 * 3


class SharedData:
    shared_source: Union[shared_memory.SharedMemory, None] = None
    shared_source_arr: Union[np.ndarray, None] = None
    shared_initial_driving: Union[shared_memory.SharedMemory, None] = None
    shared_initial_driving: Union[np.ndarray, None] = None

    camera_frame_input_slots: List[shared_memory.SharedMemory] = []
    open_inputs: List[int] = []
    network_output_slots: List[shared_memory.SharedMemory] = []
    open_outputs: List[int] = []

    def __init__(self, pool_slots):
        self._pool_slots = pool_slots

    def start(self):
        self.shared_source = shared_memory.SharedMemory(
            name=get_source_img_name(), size=DATA_SIZE, create=True
        )
        self.shared_source_arr = SHMArray(get_source_img_name())
        self.shared_initial_driving = shared_memory.SharedMemory(
            name=get_initial_driving_img_name(), size=DATA_SIZE, create=True
        )
        self.shared_initial_driving_arr = SHMArray(get_initial_driving_img_name())

        self.camera_frame_input_slots = [
            shared_memory.SharedMemory(
                name=get_shared_input_name(i), size=DATA_SIZE, create=True
            )
            for i in range(self._pool_slots)
        ]
        self.open_inputs = list(range(self._pool_slots))
        self.network_output_slots = [
            shared_memory.SharedMemory(
                name=get_shared_output_name(i), size=DATA_SIZE, create=True
            )
            for i in range(self._pool_slots)
        ]
        self.open_outputs = list(range(self._pool_slots))

    def has_available_slot(self):
        return len(self.open_inputs) > 0 and len(self.open_outputs) > 0

    def try_get_available_in_out_pair(self):
        if not self.has_available_slot():
            return None

        in_slot = self.open_inputs.pop()
        out_slot = self.open_outputs.pop()

        return (in_slot, out_slot)

    def release_in_out_pair(self, in_slot, out_slot):
        self.open_inputs.append(in_slot)
        self.open_outputs.append(out_slot)

    def close(self):
        if self.shared_source:
            self.shared_source.close()
            self.shared_source.unlink()
        if self.shared_initial_driving:
            self.shared_initial_driving.close()
            self.shared_initial_driving.unlink()
        for frame in self.camera_frame_input_slots:
            frame.close()
            frame.unlink()
        for frame in self.network_output_slots:
            frame.close()
            frame.unlink()


class SHMArray:
    """
    workaround for the face that shared memory is cleaned up as soon as it falls out of scope,
    even if it is referred to by other processes.
    """

    def __init__(self, shared_memory_name, create=False):
        self.shared_mem = shared_memory.SharedMemory(
            name=shared_memory_name, size=DATA_SIZE, create=create
        )
        self.arr = np.ndarray(shape=(256, 256, 3), dtype=np.uint8, buffer=self.shared_mem.buf)  # type: ignore


class CUDARunRequest:
    """
    Represents a request to run CUDA on an input frame
    """

    def __init__(self, seq, camera_frame_input_slot, network_output_slot):
        self.seq = seq
        self.camera_frame_input_slot = camera_frame_input_slot
        self.network_output_slot = network_output_slot

    def get_input_as_shm_array(self) -> SHMArray:
        return SHMArray(get_shared_output_name(self.camera_frame_input_slot))

    def get_output_as_shm_array(self) -> SHMArray:
        return SHMArray(get_shared_output_name(self.network_output_slot))


class NotifyUpdatedDrivingImg:
    def __init__(self):
        pass


class NotifyUpdatedSettings:
    def __init__(
        self,
        use_relative_movement=True,
        use_relative_jacobian=True,
        adapt_movement_scale=True,
    ):
        self.use_relative_movement = use_relative_movement
        self.use_relative_jacobian = use_relative_jacobian
        self.adapt_movement_scale = adapt_movement_scale


class NotifyChildProcessCrashed:
    def __init__(self):
        pass

class NotifyWorkerReady:
    def __init__(self):
        pass

JOB_SLOTS_PER_WORKER = 2
class CUDAWorkerPool:
    def __init__(self, pool_size, on_ready=None):
        # we have to use spawn here rather than fork, because CUDA cannot be reinitialized
        # in forked processes
        self.ctx = mp.get_context("spawn")

        self.parent_to_child_message_queue = self.ctx.Queue()
        self.child_to_parent_message_queue = self.ctx.Queue()

        self._pool_size = pool_size
        self._ready_workers = 0
        self._on_ready = on_ready

        # allocate more shared_data slots than cuda pools.
        # we want to process the next frame before the shared memory slot
        # has been released, so keeping these slots open should allow
        # the subprocesses to consume the next result after posting back results.
        # without waiting on the consumer to read the data out and release the slot.
        self.shared_data = SharedData(pool_size * JOB_SLOTS_PER_WORKER)
        self.request_futures = dict()

    def step_loop(self):
        # debug('stepping event loop')
        if not self.child_to_parent_message_queue.empty():
            child_message = self.child_to_parent_message_queue.get_nowait()

            if child_message is not None:
                if isinstance(child_message, NotifyChildProcessCrashed):
                    debug("parent was notified of child process exception")
                    pass
                elif isinstance(child_message, CUDARunRequest):
                    debug("parent received network run result!")
                    future = self.request_futures[
                        (
                            child_message.camera_frame_input_slot,
                            child_message.network_output_slot,
                        )
                    ]
                    future.set_result(child_message)
                elif isinstance(child_message, NotifyWorkerReady):
                    print("ready workers", self._ready_workers)
                    self._ready_workers += 1
                    if self._ready_workers == self._pool_size and self._on_ready is not None:
                        self._on_ready()

    def release_request(self, child_message: CUDARunRequest):
        io_pair = (
            child_message.camera_frame_input_slot,
            child_message.network_output_slot,
        )
        del self.request_futures[io_pair]
        self.shared_data.release_in_out_pair(*io_pair)
        debug(
            "releasing request. Available slot counts: (%s,%s)"
            % (len(self.shared_data.open_inputs), len(self.shared_data.open_outputs))
        )

    def update_settings(
        self,
        use_relative_movement=True,
        use_relative_jacobian=True,
        adapt_movement_scale=True,
    ):
        debug("host update_settings")
        self.broadcast_channel_parent.send(
            NotifyUpdatedSettings(
                use_relative_movement=use_relative_movement,
                use_relative_jacobian=use_relative_jacobian,
                adapt_movement_scale=adapt_movement_scale,
            )
        )

    def update_initial_driving_img(self, new_driving_img):
        debug("host update initial_driving_img")
        initial_driving_img_shm_arr = SHMArray(get_initial_driving_img_name())
        initial_driving_img_shm_arr.arr[:, :, :] = new_driving_img
        self.broadcast_channel_parent.send(NotifyUpdatedDrivingImg())


    def has_open_slot(self):
        return self.shared_data.has_available_slot()

    def try_submit_job(self, seq: int, driving_arr: np.ndarray):
        io_slot = self.shared_data.try_get_available_in_out_pair()
        if io_slot is None:
            return io_slot
        in_slot, out_slot = io_slot
        # create the request and populate the driving array.
        request = CUDARunRequest(seq, in_slot, out_slot)

        shared_shm_arr = request.get_input_as_shm_array()
        debug("writing request image to shared memory")
        shared_shm_arr.arr[:, :, :] = driving_arr[:, :, :]
        self.parent_to_child_message_queue.put(request)
        # create a future, fulfill that future later.
        future = DumbFutureLike()
        self.request_futures[io_slot] = future
        return future

    def start(self, source_arr, initial_driving_img_arr):
        self.shared_data.start()

        self.shared_data.shared_initial_driving_arr.arr[
            :, :, :
        ] = initial_driving_img_arr
        self.shared_data.shared_source_arr.arr[:, :, :] = source_arr[:,:,0:3]

        self.broadcast_channel_parent, self.broadcast_channel_child = self.ctx.Pipe()

        self.processes = [
            self.ctx.Process(
                group=None,
                target=process_worker_entrypoint,
                args=(
                    self.parent_to_child_message_queue,
                    self.child_to_parent_message_queue,
                    self.broadcast_channel_child,
                ),
            )
            for _ in range(self._pool_size)
        ]

        for process in self.processes:
            process.start()

    def close(self):
        for process in self.processes:
            debug("killing worker process", process.pid)
            process.terminate()
            if process.is_alive:
                debug("process survied termination, joining", process.pid)
                process.join()
            process.close()
        self.shared_data.close()

    def entry_context(self, source_arr, initial_driving_img_arr):
        return CUDAWorkerPoolContext(self, source_arr, initial_driving_img_arr)


class CUDAWorkerPoolContext:
    def __init__(self, worker_pool, source_arr, initial_driving_img_arr):
        self.worker_pool = worker_pool
        self.source_arr = source_arr
        self.initial_driving_img_arr = initial_driving_img_arr

    def __enter__(self):
        self.worker_pool.start(self.source_arr, self.initial_driving_img_arr)

    def __exit__(self, extype, exvalue, tb):
        self.worker_pool.close()


def img_to_tensor(img_arr) -> torch.Tensor:
    tensor = torch.tensor(img_arr[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not USE_CPU:
        return tensor.cuda()
    else:
        return tensor


def process_worker_entrypoint(
    parent_to_child_message_queue: mp.Queue,
    child_to_parent_message_queue: mp.Queue,
    broadcast_channel_child,
):
    if TRACE_MALLOC:
        import tracemalloc
        tracemalloc.start(8)
        baseline_snapshot = tracemalloc.take_snapshot()
    if TRACE_GC:
        import gc
        gc.set_debug(
            gc.DEBUG_STATS | gc.DEBUG_LEAK
        )

    debug("worker starting up")

    # load the model before we start listening for requests
    debug("worker loading model")
    generator, kp_detector = load_model()

    debug("worker initializing tensors")

    source_shm_arr = SHMArray(get_source_img_name())
    source_tensor = img_to_tensor(prep_image_for_ml(source_shm_arr.arr))
    kp_source = kp_detector(source_tensor)
    del source_shm_arr

    initial_driving_img_shm_arr = SHMArray(get_initial_driving_img_name())
    initial_driving_img_tensor = img_to_tensor(
        prep_image_for_ml(initial_driving_img_shm_arr.arr)
    )
    kp_driving_initial = kp_detector(initial_driving_img_tensor)
    del initial_driving_img_shm_arr

    use_relative_movement = True
    use_relative_jacobian = True
    adapt_movement_scale = True

    debug("worker entering mainloop")
    child_to_parent_message_queue.put(NotifyWorkerReady())

    try:
        i = 0
        while True:
            if TRACE_GC and i%100 == 0:
                print("gc stats:", gc.get_stats())

            if TRACE_MALLOC and i % 100 == 0:
                loop_snapshot = tracemalloc.take_snapshot()
                top_stats = loop_snapshot.compare_to(baseline_snapshot, 'traceback')

                THRESHOLD = 40000
                if any(stat.size > THRESHOLD for stat in top_stats):
                    print("[ BIG ALLOCATORS ]")
                    for stat in top_stats:
                        if (stat.size > THRESHOLD):
                            print(stat)
                            print("Traceback (most recent call first):")
                            for line in stat.traceback:
                                print(line)

            parent_process_request: Union[
                CUDARunRequest, NotifyUpdatedDrivingImg, NotifyUpdatedSettings, None
            ] = None
            while parent_process_request is None:
                if broadcast_channel_child.poll():
                    debug("read from parent pipe")
                    parent_process_request = broadcast_channel_child.recv()
                else:
                    try:
                        parent_process_request = (
                            parent_to_child_message_queue.get_nowait()
                        )
                        debug("read from event queue ")
                    except Empty:
                        pass

            debug("child process got message", parent_process_request)

            if isinstance(parent_process_request, NotifyUpdatedSettings):
                use_relative_movement = parent_process_request.use_relative_movement
                use_relative_jacobian = parent_process_request.use_relative_jacobian
                adapt_movement_scale = parent_process_request.adapt_movement_scale
            elif isinstance(parent_process_request, NotifyUpdatedDrivingImg):
                initial_driving_img_shm_arr = SHMArray(get_initial_driving_img_name())
                initial_driving_img_tensor = img_to_tensor(
                    prep_image_for_ml(initial_driving_img_shm_arr.arr)
                )
                kp_driving_initial = kp_detector(initial_driving_img_tensor)
                del initial_driving_img_shm_arr
            elif isinstance(parent_process_request, CUDARunRequest):
                driving_frame_shm_arr = parent_process_request.get_input_as_shm_array()

                # TODO: update driving frame tensor's data array instead of reinitializing on each run?
                # could this be the source of the memory issues?
                driving_frame_tensor = img_to_tensor(
                    prep_image_for_ml(driving_frame_shm_arr.arr)
                )

                kp_driving = kp_detector(driving_frame_tensor)
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
                im *= 255
                # im = bgr2rgb(im)
                output_shm_arr = parent_process_request.get_output_as_shm_array()
                output_shm_arr.arr[:, :, :] = np.array(im, dtype=np.uint8)

                debug("worker process posting result back to parent")

                # send the request back to say the network has been completed and written
                # to the corresponding shared memory.
                child_to_parent_message_queue.put(parent_process_request)
            else:
                raise Exception("Received unknown request")
    except:
        debug("worker process encountered exception")
        traceback.print_exc()
        child_to_parent_message_queue.put(NotifyChildProcessCrashed())
        exit(1)


VIDEO_DISPLAY_FRAME_DELAY = int(1000 / 24)


class VideoDisplay(tk.Widget):
    frame_needs_update = False

    def __init__(self, parent, cap, oncamloaded=None, zoom_factor_var=None, crop=True):
        tk.Frame.__init__(self, parent)
        self.cap = cap
        self.crop = crop
        self.oncamloaded = oncamloaded
        self.zoom_factor_var = zoom_factor_var
        self.zoom_factor_val = self.zoom_factor_var.get() if self.zoom_factor_var else None

        if cap and not cap.isOpened():
            raise ValueError("Cap was not open?")

        self.img = None
        self.imgtk = None
        self.image_label = tk.Label(self)
        self.image_label.pack(side="top")
        self.request_frame()

        self.enabled = True

        # defer until tkinter has started
        self.after(0, lambda: asyncio.get_event_loop().create_task(asyncio.sleep(VIDEO_DISPLAY_FRAME_DELAY / 1000)).add_done_callback(self.poll_frame))

    def set_preview_enabled(self, preview_enabled):
        self.enabled = preview_enabled
        if (self.enabled):
            self.poll_frame()

    def set_cap(self, cap):
        self.cap = cap
        if cap is None:
            self.imgtk = None
            self.img = None
        if self.oncamloaded is not None:
            self.oncamloaded()

    def request_frame(self, *argv):
        on_captured = executor.submit(self.capture_frame)
        on_captured.add_done_callback(self.notify_needs_show_frame)

    def notify_needs_show_frame(self, frame_future):
        frame = frame_future.result()
        if frame is not None:
            self.img = frame
            self.frame_needs_update = True
        else:
            time.sleep(VIDEO_DISPLAY_FRAME_DELAY/ 1000)
            executor.submit(self.request_frame)

    def poll_frame(self, *argv):
        """
        Polls for an available frame (populated from the thread pool executor in request_frame -> capture_frame)
        """
        if self.enabled:
            try:
                if self.frame_needs_update:
                    self.frame_needs_update = False
                    self.imgtk = ImageTk.PhotoImage(image=self.img)
                    self.image_label.configure(image=self.imgtk)
                    self.zoom_factor_val = self.zoom_factor_var.get() if self.zoom_factor_val else None
                    self.request_frame()
            finally:
                asyncio.get_event_loop().create_task(asyncio.sleep(VIDEO_DISPLAY_FRAME_DELAY / 1000)).add_done_callback(self.poll_frame)

    def capture_frame(self):
        # print("capture_frame", os.getpid())
        if self.cap is not None:
            frame = self.cap.request_video_frame()
            if frame is not None:
                if self.crop:
                    zoom_factor = (
                        self.zoom_factor_val if self.zoom_factor_var else None
                    )
                    return prep_frame(frame, zoom_factor=zoom_factor)
                else:
                    return Image.fromarray(frame)
        return None


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
        asyncio.get_event_loop().create_task(asyncio.sleep(1 / 1000)).add_done_callback(self.update_camera_list_and_repack)

    def disable_ui(self):
        self.refresh_button['state'] = tk.DISABLED
        self.reopen_button['state'] = tk.DISABLED
        self.cam_dropdown['state'] = tk.DISABLED

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
            cap = CV2VideoCaptureWrapper(cv2.VideoCapture(idx))

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

    def update_camera_list_and_repack(self, *argv):
        i = 0
        cameras = []
        while True:
            cap = cv2.VideoCapture(i)
            if i > 10 and (not cap.isOpened() or cap.request_video_frame() == None):
                break
            if cap.isOpened():
                cameras.append("Camera %s" % i)
            i += 1

        if len(cameras):
            self.video_capture = CV2VideoCaptureWrapper(cv2.VideoCapture(len(cameras) - 1))

        if self.selected_camera.get() == "":
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
    def __init__(self, on_load_complete, master=None):
        super().__init__(master)
        self.filename = tk.StringVar(
            self,
            os.path.join(INSTALLDIR, "source_image_inputs", "shrek_pink_colorkey.jpg"),
        )

        self.master = master
        self.pack()
        self.create_widgets()
        self.model_ready = was_model_loaded_at_start()
        self.on_load_complete = on_load_complete

        self.check_steal_button()

    def create_widgets(self):
        left_frame = tk.Frame(self)
        self.select_image_button = tk.Button(left_frame)
        self.pack_button()
        self.image_label = tk.Label(left_frame)
        self.render_preview_img()

        self.video_capture = VideoCapture(self, oncamloaded=self.check_steal_button)
        self.worker_count_str_var = tk.StringVar(self,"1")

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side="bottom")

        self.steal_button = tk.Button(bottom_frame)
        self.steal_button["command"] = self.request_prewarm_workers
        self.steal_button["text"] = "Borrow that face!"
        self.steal_button["state"] = tk.DISABLED
        self.steal_button.pack(side="right")

        worker_count_dropdown_label = tk.Label(bottom_frame)
        worker_count_dropdown_label["text"] = "Worker Count"
        worker_count_dropdown_label.pack(side="left")
        self.worker_count_dropdown = tk.OptionMenu(bottom_frame, self.worker_count_str_var, 1,2,3,4,5)
        self.worker_count_dropdown.pack(side="left")

        self.loading_label = tk.Label(
            self, fg="green" if was_model_loaded_at_start() else "red"
        )
        self.loading_label["text"] = (
            "Model downloaded"
            if was_model_loaded_at_start()
            else "Downloading model, this might take a few minutes.."
        )
        self.loading_label.pack(side="bottom")

        if USE_CPU:
            warning_area = tk.Frame(self)
            cuda_link = tk.Label(warning_area, fg="red", cursor="hand2")
            cuda_link[
                "text"
            ] = "You can get the drivers at https://developer.nvidia.com/cuda-downloads"
            cuda_link.bind(
                "<Button-1>",
                lambda e: webbrowser.open(
                    "https://developer.nvidia.com/cuda-downloads"
                ),
            )
            f = tkinter.font.Font(cuda_link, cuda_link.cget("font"))
            f.configure(underline=True)
            cuda_link.configure(font=f)

            cuda_warning = tk.Label(warning_area, fg="red")
            cuda_warning[
                "text"
            ] = "WARNING: Could not find CUDA.\n The neural network will be run on the CPU, and will not be realtime capable."
            warning_area.pack(side="bottom")
            cuda_warning.pack(side="top")
            cuda_link.pack(side="top")

        left_frame.pack(side="left")
        self.image_label.pack(side="top")
        self.video_capture.pack(side="right")

    def download_complete(self):
        self.loading_label["text"] = "Download complete"
        self.loading_label["fg"] = "green"
        self.model_ready = True
        self.check_steal_button()


    def render_preview_img(self):
        self.img = Image.open(self.filename.get()).resize(
            (256, 256), resample=Image.NEAREST
        )
        self.photo_img = ImageTk.PhotoImage(self.img)
        self.image_label.configure(image=self.photo_img)
        self.image_label.image = self.photo_img

    def pack_button(self):
        self.select_image_button["text"] = os.path.basename(self.filename.get())
        self.select_image_button["command"] = self.update_img
        self.select_image_button.pack(side="bottom")

    def update_img(self):
        filename = tkinter.filedialog.askopenfilename(
            initialdir=os.path.join(INSTALLDIR, "source_image_inputs"),
            title="Select a File",
            filetypes=(
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("all files", "*.*"),
            ),
        )
        if filename is not None:
            self.filename.set(filename)
            self.render_preview_img()

            self.pack_button()
            self.render_preview_img()
            self.check_steal_button()

    def check_steal_button(self, *argv):
        if (
            self.img is not None
            and self.video_capture.video_display.cap is not None
            and self.video_capture.video_display.cap.isOpened()
            and self.model_ready
        ):
            self.steal_button["state"] = tk.ACTIVE
        else:
            self.steal_button["state"] = tk.DISABLED

    def request_prewarm_workers(self):
        self.steal_button["state"] = tk.DISABLED
        self.select_image_button["state"] = tk.DISABLED
        self.worker_count_dropdown["state"] = tk.DISABLED
        self.video_capture.disable_ui()

        self.loading_label["text"] = (
            "Warming worker pool.."
        )

        self.on_load_complete(
            self.img,
            self.video_capture.video_display.cap,
            int(self.worker_count_str_var.get())
        )

ML_FRAME_INTERVAL = 1000 // 24

def prep_image_for_ml(prepped_frame):
    return np.array(prepped_frame)[:, :, :3] / 255


FPS_COUNTER_FALLOFF_RATIO = 0.018


class Distorter(tk.Frame):
    last_frame_time = None
    last_frame_interval_rolling_delta = None
    seq = 0
    last_consumed_seq = 0

    def __init__(
        self,
        parent,
        initial_frame_img: Image,
        cuda_worker_pool: CUDAWorkerPool,
        video_capture: VideoCapture,
        zoom_factor_var=None,
        fps_var=None,
    ):
        tk.Frame.__init__(self, parent)
        self.cuda_worker_pool = cuda_worker_pool
        self.video_capture = video_capture
        self.fps_var = fps_var
        self.zoom_factor_var = zoom_factor_var

        # schedule after tkinter starts running
        self.after(0, lambda: asyncio.get_event_loop().create_task(asyncio.sleep(ML_FRAME_INTERVAL / 1000)).add_done_callback(self.request_frame))
        self.last_request_time = 0
        self.last_request_succ_time = 0

        # generate widgets
        self.create_widgets(initial_frame_img)

    def get_prepped_frame_arr(self):
        video_frame = self.video_capture.request_video_frame(True)
        if video_frame is not None:
            return np.array(
                prep_frame(
                    video_frame,
                    self.zoom_factor_var.get() if self.zoom_factor_var else 0.8,
                ),
                dtype=np.uint8,
            )
        else:
            return None

    def request_frame(self, *argv):
        t = time.time()
        delta = t - self.last_request_time
        # print("%02.04f, %02.04f,, " % (t - start_time, delta))
        self.last_request_time = t

        if (not self.cuda_worker_pool.has_open_slot()):
            debug("skipping schedule attempt -- no available slots!")
            # don't hold the CPU polling and prepping a frame
            # if there is no available job slot.
        else:
            prepped_frame_arr = self.get_prepped_frame_arr()
            if prepped_frame_arr is not None:
                future = self.cuda_worker_pool.try_submit_job(self.seq, prepped_frame_arr)
                self.seq += 1
                if future:
                    succ_delta = t - self.last_request_succ_time
                    # print("request_attempt_suc %02.04f" % (1/succ_delta), succ_delta)
                    # print("%02.04f, , %02.04f, " % (t - start_time, succ_delta))

                    self.last_request_succ_time = t
                    debug("waiting for job result")
                    future.add_done_callback(self.render_cuda_request_result)
                else:
                    debug("did not submit prepped frame to worker pool: no available workers")
                    pass
            else:
                debug("failed to get frame from video source")
                pass

        asyncio.get_event_loop().create_task(asyncio.sleep(ML_FRAME_INTERVAL / 1000)).add_done_callback(self.request_frame)

    def render_cuda_request_result(self, cuda_request_future):
        cuda_request: CUDARunRequest = cuda_request_future.result()
        if cuda_request.seq > self.last_consumed_seq:
            self.last_consumed_seq = cuda_request.seq
            debug("consuming result")
            # Read the image from the shared memory
            shm_array = cuda_request.get_output_as_shm_array()
            self.img = Image.fromarray(shm_array.arr)
            self.imgtk = ImageTk.PhotoImage(image=self.img)
            self.image_label.configure(image=self.imgtk)
            # print("delay to show", time.time() - self.last_frame_time_generated_time)
            self.tick_fps()
        else:
            debug("ignoring result due to out-of-order return")
        # Tell the worker pool that we've finished copying data out of the shared memory.
        # this releases the slot for future use
        self.cuda_worker_pool.release_request(cuda_request)

    def create_widgets(self, initial_frame_img):
        self.img = initial_frame_img
        # this should copy the image out of shared memory.
        self.imgtk = ImageTk.PhotoImage(image=initial_frame_img)
        self.image_label = tk.Label(self, image=self.imgtk)
        self.image_label.pack(side="left")

    def recalculate_initial_frame(self):
        prepped_frame_arr = self.get_prepped_frame_arr()
        if prepped_frame_arr is not None:
            self.cuda_worker_pool.update_initial_driving_img(prepped_frame_arr)
            debug("requested update to driving img")
        else:
            debug("failed to get frame to update the initial driving frame")

    def tick_fps(self):
        now = time.time()
        if self.fps_var is not None and self.last_frame_time is not None:
            delta = now - self.last_frame_time
            if self.last_frame_interval_rolling_delta == None:
                self.last_frame_interval_rolling_delta = delta
            else:
                self.last_frame_interval_rolling_delta = (
                    self.last_frame_interval_rolling_delta
                    * (1 - FPS_COUNTER_FALLOFF_RATIO)
                    + (FPS_COUNTER_FALLOFF_RATIO * delta)
                )
            # print("%02.04f, , , %02.04f (%02.04f fps)" % (now - start_time, delta, 1/delta))
            self.fps_var.set(
                "fps: %01.01f" % (1 / (self.last_frame_interval_rolling_delta))
            )
        self.last_frame_time = now


def oneshot_run_kp(kp_detector, frame):
    source1 = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not USE_CPU:
        source1 = source1.cuda()

    return kp_detector(source1)


class RunSimulationApplication(tk.Frame):
    def __init__(
        self,
        initial_frame_img: Image,
        video_capture: VideoCapture,
        cuda_worker_pool: CUDAWorkerPool,
        master=None,
    ):
        super().__init__(master)
        self.video_capture = video_capture
        self.cuda_worker_pool = cuda_worker_pool

        self.use_relative_movement_var = tk.BooleanVar(self, True)
        self.use_relative_movement_var.trace("w", self.on_settings_var_updated)
        self.use_relative_jacobian_var = tk.BooleanVar(self, True)
        self.use_relative_jacobian_var.trace("w", self.on_settings_var_updated)
        self.adapt_movement_scale_var = tk.BooleanVar(self, True)
        self.adapt_movement_scale_var.trace("w", self.on_settings_var_updated)

        self.camera_preview_enabled_var = tk.BooleanVar(self, True)
        self.camera_preview_enabled_var.trace("w", self.on_camera_preview_enabled_var_updated)

        self.zoom_factor_var = tk.DoubleVar(self, 0.8)
        self.fps_var = tk.StringVar(self, "fps: ")

        self.master = master
        self.pack()
        self.create_widgets(initial_frame_img)

    def on_camera_preview_enabled_var_updated(self, *argv):
        self.video_display.set_preview_enabled(self.camera_preview_enabled_var.get())

    def on_settings_var_updated(self, *argv):
        self.cuda_worker_pool.update_settings(
            use_relative_movement=self.use_relative_movement_var.get(),
            use_relative_jacobian=self.use_relative_jacobian_var.get(),
            adapt_movement_scale=self.adapt_movement_scale_var.get(),
        )

    def create_widgets(self, initial_frame_img):
        top_frame = tk.Frame(self)
        top_frame.pack(side="top")
        self.distorter = Distorter(
            top_frame,
            initial_frame_img=initial_frame_img,
            video_capture=self.video_capture,
            cuda_worker_pool=self.cuda_worker_pool,
            fps_var=self.fps_var,
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

        subwindow = tk.Toplevel(self)
        subwindow.title("Face Borrower Settings")

        slider_frame = tk.Frame(subwindow)
        slider_frame.pack(side="bottom")

        fps_label = tk.Label(slider_frame)
        fps_label["textvariable"] = self.fps_var
        fps_label.pack(side="left")

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

        checkbox_frame = tk.Frame(subwindow)
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
            text="scale_movement",
            onvalue=True,
            offvalue=False,
            variable=self.adapt_movement_scale_var,
        )
        checkbox_3.pack(side="left")
        checkbox_4 = tk.Checkbutton(
            checkbox_frame,
            selectcolor="red",
            text="show_camera",
            onvalue=True,
            offvalue=False,
            variable=self.camera_preview_enabled_var,
        )
        checkbox_4.pack(side="left")


executor = None

async def run_tk(root, condition, period = 0.05):
    '''
    Run a tkinter app in an asyncio event loop.
    '''
    try:
        while not condition():
            root.update()
            await asyncio.sleep(period / 1000)
    except tk.TclError as e:
        if "application has been destroyed" not in e.args[0]:
            raise

async def periodic(cb, period = 0.05):
    while True:
        cb()
        await asyncio.sleep(period / 1000)

def main():
    global SHARED_MEM_ID
    global executor
    executor = ThreadPoolExecutor(2)
    SHARED_MEM_ID = os.getpid()

    debug("main process started")

    root = tk.Tk()
    root.title("Face Borrower")
    inputs_set = False
    param_source_img = None
    param_video_cap = None
    param_worker_count = None

    def set_inputs(source_img, video_cap, worker_count):
        nonlocal param_source_img, param_video_cap, param_worker_count, inputs_set
        inputs_set = True
        param_source_img = source_img
        param_video_cap = video_cap
        param_worker_count = worker_count
    
    def is_initialized():
        return inputs_set

    get_inputs_app = GetInputsApplication(master=root, on_load_complete=set_inputs)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_tk(root, is_initialized))
    except KeyboardInterrupt:
        pass

    if param_source_img is None or param_video_cap is None:
        raise Exception(
            "failed to initialize? source_img=%s, param_video_cap=%s"
            % (param_source_img, param_video_cap)
        )

    source_img_arr = np.array(param_source_img, dtype=np.uint8)
    for i in range(10):
        frame = param_video_cap.request_video_frame()
        if frame is None:
            print("failed to get frame from video source. Trying again.")
            time.sleep(0.1)
        else:
            break
    if frame is None:
        raise Exception("could not get an initial frame from video source. crashing.")

    def release_app():
        get_inputs_app.destroy()
        root.geometry("512x256")
        app2 = RunSimulationApplication(
            initial_frame_img=param_source_img,
            video_capture=param_video_cap,
            cuda_worker_pool=cuda_worker_pool,
            master=root,
        )

    cuda_worker_pool = CUDAWorkerPool(param_worker_count, on_ready=release_app)
    initial_image_arr = np.array(prep_frame(frame), dtype=np.uint8)
    with cuda_worker_pool.entry_context(source_img_arr, initial_image_arr):
        global start_time
        start_time = time.time()

        loop.run_until_complete(
            asyncio.gather(
                run_tk(root, lambda: False),
                periodic(cuda_worker_pool.step_loop)
            )
        )

    root.destroy()


if __name__ == "__main__":
    # required so multiprocessing works with pyinstaller-bundled apps.
    # see https://stackoverflow.com/questions/33870542/pyinstaller-and-multiprocessing
    print("time, request_attempt_interval, request_succ_interval, frame_delta")
    mp.freeze_support()
    main()
