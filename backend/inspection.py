# inspection.py
import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import json
import re
import numpy as np
import cv2
import torch

from backend.engines import template_engine as template_engine
from backend.engines.patchcore_engine import PatchcoreEngine

# Resolve config.json absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

# ==================== CONFIGURATION  ====================

TARGET_DEVICE_USER_ID = _cfg["target_device_user_id"]
USE_EXTERNAL_TRIGGER  = _cfg["use_external_trigger"]
TRIGGER_SOURCE        = _cfg["trigger_source"]
TRIGGER_ACTIVATION    = _cfg["trigger_activation"]
PIXEL_FORMAT          = _cfg["pixel_format"]
FRAME_WIDTH           = _cfg["frame_width"]
FRAME_HEIGHT          = _cfg["frame_height"]
ROI_OFFSET_X          = _cfg["roi_offset_x"]
ROI_OFFSET_Y          = _cfg["roi_offset_y"]
EXPOSURE_TIME_US      = _cfg["exposure_time_us"]
GAIN_DB               = _cfg["gain_db"]
MAX_PARALLEL_FRAMES   = _cfg["max_parallel_frames"]
SAVE_DIRECTORY        = _cfg["save_directory"]
SAVE_FORMAT           = _cfg["save_format"]
JPEG_QUALITY          = _cfg["jpeg_quality"]
AUTO_EXPOSURE         = _cfg["auto_exposure"]
AUTO_GAIN             = _cfg["auto_gain"]
WHITE_BALANCE_AUTO    = _cfg["white_balance_auto"]
WB_RED_RATIO          = _cfg["wb_red_ratio"]
WB_GREEN_RATIO         = _cfg["wb_green_ratio"]
WB_BLUE_RATIO         = _cfg["wb_blue_ratio"]
TARGET_FORCE_IP       = _cfg["target_force_ip"]
TARGET_FORCE_SUBNET   = _cfg["target_force_subnet"]
TARGET_FORCE_GATEWAY  = _cfg["target_force_gateway"]
ROI_ENABLE            =_cfg["roi_enable"]
ENABLE_TEMPLATE_MATCHING = _cfg["enable_template_matching"]
ENABLE_PATCHCORE               = _cfg["enable_patchcore"]
PATCHCORE_THRESHOLD = _cfg["patchcore_threshold"]

# -------------------- CAMERA SELECTION (CONNECT BY DEVICE USER ID ONLY) --------------------
AUTO_CONNECT_FIRST_CAMERA = True       # If TARGET_DEVICE_USER_ID not found, connect first available camera

# Connection Check Interval
STATUS_CHECK_INTERVAL_SEC = 5.0

# Debug Output
ENABLE_DEBUG_LOGS = True
# ==================== END CONFIGURATION ====================

# ==================== INSPECTION CONFIG ====================

templates = [
    {
        "name": "Part_LH_T",
        "template_indices": list(range(0, 2)),   # 1.jpg, 2.jpg
        "template_crops": [
            (318, 767, 380, 807),
            (341, 773, 431, 812),
        ],
        "expected_roi": (250, 800, 350, 950),
        "threshold": 0.50
    },
    {
        "name": "Part_LH_B",
        "template_indices": list(range(2, 4)),   # 3.jpg, 4.jpg
        "template_crops": [
            (305, 753, 455, 852),
            (292, 750, 544, 938),
        ],
        "expected_roi": (250, 800, 350, 950),
        "threshold": 0.50
    },
    {
        "name": "Part_RH_T",
        "template_indices": list(range(4, 6)),   # 5.jpg, 6.jpg
        "template_crops": [
            (300, 757, 504, 892),
            (285, 761, 534, 904),
        ],
        "expected_roi": (250, 800, 350, 950),
        "threshold": 0.50
    },
    {
        "name": "Part_RH_B",
        "template_indices": list(range(6, 8)),   # 7.jpg, 8.jpg
        "template_crops": [
            (337, 781, 497, 875),
            (298, 753, 471, 846),
        ],
        "expected_roi": (250, 800, 350, 950),
        "threshold": 0.50
    }
]

TEMPLATE_DIR = os.path.join(BASE_DIR, "models", "templates")
TEMPLATE_IMAGES = [cv2.imread(os.path.join(TEMPLATE_DIR, f"{i}.jpg")) for i in range(1, 9)]
if any(t is None for t in TEMPLATE_IMAGES):
    raise RuntimeError(f"One or more template images missing — check {TEMPLATE_DIR} folder")

PREPPED_TEMPLATES = []
for tpl in templates:
    crops = []
    for local_i, img_i in enumerate(tpl["template_indices"]):
        ty1, ty2, tx1, tx2 = tpl["template_crops"][local_i]
        crops.append(TEMPLATE_IMAGES[img_i][ty1:ty2, tx1:tx2])
    PREPPED_TEMPLATES.append(crops)
print(f"[INFO] Template crops ready: {sum(len(c) for c in PREPPED_TEMPLATES)}")

# =========================
# PATCHCORE
# =========================

PATCHCORE_MODEL_PATH = _cfg["patchcore_model_path"]
# Resolve relative path for checkpoint against project root
abs_model_path = PATCHCORE_MODEL_PATH
if not os.path.isabs(abs_model_path):
    abs_model_path = os.path.join(BASE_DIR, abs_model_path)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize extracted PatchCore engine
PATCHCORE_ENGINE_INST = PatchcoreEngine(abs_model_path, PATCHCORE_THRESHOLD, DEVICE)

RAW_FRAME_QUEUE = queue.Queue(maxsize=MAX_PARALLEL_FRAMES * 2)
TEMPLATE_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL_FRAMES)
PATCHCORE_EXECUTOR     = ThreadPoolExecutor(max_workers=MAX_PARALLEL_FRAMES)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==================== END INSPECTION CONFIG ====================

# Shared state globals

JPEG_LOCK = threading.Lock()
DETECTION_LOCK = threading.Lock()
LATEST_JPEG_PROCESS = None
LATEST_DETECTION = None
SYSTEM_RUNNING = False
CAMERA_CONTROLLERS = []

# MVS SDK Import
from backend.camera.mvs_sdk_path import add_sdk_to_path
add_sdk_to_path()

from CameraParams_header import *
from MvCameraControl_class import *
from MvErrorDefine_const import *

def to_hex_str(num):
    cha = {10:'a',11:'b',12:'c',13:'d',14:'e',15:'f'}
    if num < 0:
        num += 2**32
    s = ""
    while num >= 16:
        d = num % 16
        s = cha.get(d, str(d)) + s
        num //= 16
    return cha.get(num, str(num)) + s

def decoding_char(c_ubyte_value):
    p = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        return p.value.decode('gbk')
    except:
        try:
            return p.value.decode('utf-8')
        except:
            return str(p.value)

def ensure_dir(directory):
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Save directory ready: {directory}")
    except Exception as e:
        print(f"[ERROR] Could not create directory {directory}: {e}")

def get_last_image_count(save_dir):
    max_count = 0

    if not os.path.exists(save_dir):
        return 0

    for filename in os.listdir(save_dir):

        name, ext = os.path.splitext(filename)

        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue

        match = re.search(r'(\d+)$', name)

        if match:
            count = int(match.group(1))

            if count > max_count:
                max_count = count

    return max_count
def timestamp_filename(ext):
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext

def log_debug(message):
    if ENABLE_DEBUG_LOGS:
        print(f"[DEBUG] {message}")

def _reload_config():
    """Re-read config.json into all global constants so UI changes take effect on START."""
    global TARGET_DEVICE_USER_ID, USE_EXTERNAL_TRIGGER, TRIGGER_SOURCE, TRIGGER_ACTIVATION
    global PIXEL_FORMAT, FRAME_WIDTH, FRAME_HEIGHT, ROI_ENABLE, ROI_OFFSET_X, ROI_OFFSET_Y
    global EXPOSURE_TIME_US, GAIN_DB, AUTO_EXPOSURE, AUTO_GAIN
    global WHITE_BALANCE_AUTO, WB_RED_RATIO, WB_GREEN_RATIO, WB_BLUE_RATIO
    global MAX_PARALLEL_FRAMES, SAVE_DIRECTORY, SAVE_FORMAT, JPEG_QUALITY
    global TARGET_FORCE_IP, TARGET_FORCE_SUBNET, TARGET_FORCE_GATEWAY
    global ENABLE_TEMPLATE_MATCHING, ENABLE_PATCHCORE, PATCHCORE_MODEL_PATH, PATCHCORE_THRESHOLD

    with open(CONFIG_PATH) as f:
        c = json.load(f)

    TARGET_DEVICE_USER_ID = c["target_device_user_id"]
    USE_EXTERNAL_TRIGGER  = c["use_external_trigger"]
    TRIGGER_SOURCE        = c["trigger_source"]
    TRIGGER_ACTIVATION    = c["trigger_activation"]
    PIXEL_FORMAT          = c["pixel_format"]
    FRAME_WIDTH           = c["frame_width"]
    FRAME_HEIGHT          = c["frame_height"]
    ROI_ENABLE            = c["roi_enable"]
    ROI_OFFSET_X          = c["roi_offset_x"]
    ROI_OFFSET_Y          = c["roi_offset_y"]
    EXPOSURE_TIME_US      = c["exposure_time_us"]
    GAIN_DB               = c["gain_db"]
    AUTO_EXPOSURE         = c["auto_exposure"]
    AUTO_GAIN             = c["auto_gain"]
    WHITE_BALANCE_AUTO    = c["white_balance_auto"]
    WB_RED_RATIO          = c["wb_red_ratio"]
    WB_GREEN_RATIO        = c["wb_green_ratio"]
    WB_BLUE_RATIO         = c["wb_blue_ratio"]
    MAX_PARALLEL_FRAMES   = c["max_parallel_frames"]
    SAVE_DIRECTORY        = c["save_directory"]
    SAVE_FORMAT           = c["save_format"]
    JPEG_QUALITY          = c["jpeg_quality"]
    TARGET_FORCE_IP       = c["target_force_ip"]
    TARGET_FORCE_SUBNET   = c["target_force_subnet"]
    TARGET_FORCE_GATEWAY  = c["target_force_gateway"]
    ENABLE_TEMPLATE_MATCHING = c["enable_template_matching"]
    ENABLE_PATCHCORE         = c["enable_patchcore"]
    PATCHCORE_MODEL_PATH     = c["patchcore_model_path"]
    PATCHCORE_THRESHOLD      = c["patchcore_threshold"]

def ip_to_int(ip_str):
    parts = ip_str.split('.')
    return (int(parts[0])<<24)+(int(parts[1])<<16)+(int(parts[2])<<8)+int(parts[3])

def int_to_ip(ip_int):
    return f"{(ip_int>>24)&0xFF}.{(ip_int>>16)&0xFF}.{(ip_int>>8)&0xFF}.{ip_int&0xFF}"

def force_ip_before_open(device_list):
    """
    If camera is found but on wrong subnet, use ForceIP to push it
    to a reachable IP before opening.
    Call this AFTER EnumDevices but BEFORE CreateHandle/OpenDevice.
    """

    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType not in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            continue

        current_ip = int_to_ip(info.SpecialInfo.stGigEInfo.nCurrentIp)
        if current_ip == TARGET_FORCE_IP:
            continue  # already correct

        print(f"[INFO] Camera at {current_ip}, forcing to {TARGET_FORCE_IP}...")

        tmp = MvCamera()
        ret = tmp.MV_CC_CreateHandle(info)
        if ret != MV_OK:
            print(f"[WARN] ForceIP CreateHandle failed: 0x{to_hex_str(ret)}")
            continue

        ip_int  = ip_to_int(TARGET_FORCE_IP)
        sub_int = ip_to_int(TARGET_FORCE_SUBNET)
        gw_int  = ip_to_int(TARGET_FORCE_GATEWAY)

        ret = tmp.MV_GIGE_ForceIpEx(ip_int, sub_int, gw_int)
        tmp.MV_CC_DestroyHandle()

        if ret == MV_OK:
            print(f"[OK] ForceIP success. Waiting 3s for camera to reconfigure...")
            time.sleep(3)
        else:
            print(f"[WARN] ForceIP failed: 0x{to_hex_str(ret)}")

def frame_to_bgr(frame_data, cam):
    """Convert raw camera buffer to BGR numpy array using ISP."""
    src = (ctypes.c_ubyte * frame_data['data_len']).from_buffer_copy(frame_data['data'])
    dst_len = frame_data['width'] * frame_data['height'] * 3
    dst = (ctypes.c_ubyte * dst_len)()

    convert_param = MV_CC_PIXEL_CONVERT_PARAM()
    memset(byref(convert_param), 0, sizeof(convert_param))
    convert_param.nWidth         = frame_data['width']
    convert_param.nHeight        = frame_data['height']
    convert_param.pSrcData       = ctypes.cast(src, POINTER(ctypes.c_ubyte))
    convert_param.nSrcDataLen    = frame_data['data_len']
    convert_param.enSrcPixelType = frame_data['pixel_type']
    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
    convert_param.pDstBuffer     = ctypes.cast(dst, POINTER(ctypes.c_ubyte))
    convert_param.nDstBufferSize = dst_len

    ret = cam.MV_CC_ConvertPixelType(convert_param)
    if ret != MV_OK:
        raise RuntimeError(f"ISP conversion failed: 0x{to_hex_str(ret)}")

    return np.frombuffer(dst, dtype=np.uint8).reshape(
        (frame_data['height'], frame_data['width'], 3)
    )

def get_device_user_id(device_info):
    """
    Read DeviceUserID / UserDefinedName from MV_CC_DEVICE_INFO (without opening device).
    Hikrobot MVS typically stores this in:
      - GigE: stGigEInfo.chUserDefinedName
      - USB : stUsb3VInfo.chUserDefinedName
    """
    try:
        if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            gige_info = device_info.SpecialInfo.stGigEInfo
            return decoding_char(gige_info.chUserDefinedName).strip()
        elif device_info.nTLayerType == MV_USB_DEVICE:
            usb_info = device_info.SpecialInfo.stUsb3VInfo
            return decoding_char(usb_info.chUserDefinedName).strip()
    except:
        pass
    return ""

def get_device_info_string(device_info):
    """Readable device info (includes DeviceUserID if available)"""
    uid = get_device_user_id(device_info)
    if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        gige_info = device_info.SpecialInfo.stGigEInfo
        serial = decoding_char(gige_info.chSerialNumber).strip()
        model = decoding_char(gige_info.chModelName).strip()
        # IP is not used for connection anymore, but can show as info
        ip_int = gige_info.nCurrentIp
        ip = f"{(ip_int>>24)&0xFF}.{(ip_int>>16)&0xFF}.{(ip_int>>8)&0xFF}.{ip_int&0xFF}"
        return f"UserID: {uid}, Model: {model}, S/N: {serial}, IP: {ip}"
    elif device_info.nTLayerType == MV_USB_DEVICE:
        usb_info = device_info.SpecialInfo.stUsb3VInfo
        serial = decoding_char(usb_info.chSerialNumber).strip()
        model = decoding_char(usb_info.chModelName).strip()
        return f"UserID: {uid}, Model: {model}, S/N: {serial}, USB"
    else:
        return f"UserID: {uid}, Other transport"

def find_device_by_user_id(device_list, target_user_id):
    """Find device index by Device User ID (UserDefinedName)"""
    if device_list is None or device_list.nDeviceNum == 0:
        return -1

    target = (target_user_id or "").strip()
    if not target:
        return -1

    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        uid = get_device_user_id(info)
        if uid == target:
            return i

    return -1

class CameraController:
    def __init__(self, cam_config):
        self.config = cam_config
        self.obj_cam = MvCamera()
        self.is_open = False
        self.is_grabbing = False

        self.grab_thread = None
        self.timer_thread = None
        self.status_thread = None
        self.process_threads = []
        self.stop_event = threading.Event()
        self._save_count_lock = threading.Lock()
        self._active_workers = 0

        self.last_save_time = 0
        self.saved_frame_count = 0
        self.frame_count = 0

        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def _set_enum_str(self, key, value, quiet=False):
        ret = self.obj_cam.MV_CC_SetEnumValueByString(key, value)
        if ret == MV_OK and not quiet:
            print(f"  [OK] {key} = {value}")
        elif not quiet:
            log_debug(f"{key} = {value} failed (0x{to_hex_str(ret)})")
        return ret

    def _set_int(self, key, value, quiet=False):
        ret = self.obj_cam.MV_CC_SetIntValue(key, int(value))
        if ret == MV_OK and not quiet:
            print(f"  [OK] {key} = {value}")
        elif not quiet:
            log_debug(f"{key} = {value} failed (0x{to_hex_str(ret)})")
        return ret

    def _set_float(self, key, value, quiet=False):
        ret = self.obj_cam.MV_CC_SetFloatValue(key, float(value))
        if ret == MV_OK and not quiet:
            print(f"  [OK] {key} = {value}")
        elif not quiet:
            log_debug(f"{key} = {value} failed (0x{to_hex_str(ret)})")
        return ret

    def connect(self, device_info):
        if self.is_open:
            print(f"[DeviceUserID: {self.config.get('user_id','?')}] Already connected")
            return False

        uid = self.config.get('user_id', 'Unknown')
        print(f"\n{'='*70}")
        print(f"[DeviceUserID: {uid}] Connecting...")
        print(f"  Device Info: {get_device_info_string(device_info)}")

        ret = self.obj_cam.MV_CC_CreateHandle(device_info)
        if ret != MV_OK:
            print(f"  [ERROR] Create handle failed: 0x{to_hex_str(ret)}")
            return False

        ret = self.obj_cam.MV_CC_OpenDevice()
        if ret != MV_OK:
            print(f"  [ERROR] Open device failed: 0x{to_hex_str(ret)}")
            self.obj_cam.MV_CC_DestroyHandle()
            return False

        if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            packet_size = self.obj_cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                log_debug(f"Set packet size: {packet_size}")

        self.is_open = True
        print(f"  [OK] Camera opened successfully")

        self._configure_camera()
        return True

    def _configure_camera(self):
        uid = self.config.get('user_id', 'Unknown')
        print(f"\n[DeviceUserID: {uid}] Configuring parameters...")

        if ROI_ENABLE:
            self._set_int("OffsetX", ROI_OFFSET_X)
            self._set_int("OffsetY", ROI_OFFSET_Y)
            self._set_int("Width", FRAME_WIDTH)
            self._set_int("Height", FRAME_HEIGHT)

        if not ROI_ENABLE :
            self._set_int("OffsetX", 0)
            self._set_int("OffsetY", 0)
            self._set_int("Width", 2448)
            self._set_int("Height", 2048)

        if PIXEL_FORMAT:
            self._set_enum_str("PixelFormat", PIXEL_FORMAT)

        if AUTO_EXPOSURE:
            self._set_enum_str("ExposureAuto", "Continuous")
        else:
            self._set_enum_str("ExposureAuto", "Off")
            self._set_float("ExposureTime", EXPOSURE_TIME_US)

        if AUTO_GAIN:
            self._set_enum_str("GainAuto", "Continuous")
        else:
            self._set_enum_str("GainAuto", "Off")
            self._set_float("Gain", GAIN_DB)

        if WHITE_BALANCE_AUTO:
            self._set_enum_str("BalanceWhiteAuto", "Continuous")
        else:
            self._set_enum_str("BalanceWhiteAuto", "Off")
            self._set_enum_str("BalanceRatioSelector", "Red", quiet=True)
            self._set_float("BalanceRatio", WB_RED_RATIO, quiet=True)
            self._set_enum_str("BalanceRatioSelector", "Green", quiet=True)
            self._set_float("BalanceRatio", WB_GREEN_RATIO, quiet=True)
            self._set_enum_str("BalanceRatioSelector", "Blue", quiet=True)
            self._set_float("BalanceRatio", WB_BLUE_RATIO, quiet=True)

        # self._set_enum_str("AcquisitionMode", "Continuous")
        if USE_EXTERNAL_TRIGGER:
            self._set_enum_str("TriggerMode", "On")
            self._set_enum_str("TriggerSelector", "FrameStart", quiet=True)
            self._set_enum_str("TriggerSource", TRIGGER_SOURCE)
            self._set_enum_str("TriggerActivation", TRIGGER_ACTIVATION)
            print(f"  [OK] External trigger enabled on {TRIGGER_SOURCE}")

        else:
            self._set_enum_str("TriggerMode", "Off")
            print(f"  [OK] Continuous (free-run) mode enabled")

        print(f"  [OK] Configuration complete")

    def start_grabbing(self):
        uid = self.config.get('user_id', 'Unknown')

        if not self.is_open:
            print(f"[DeviceUserID: {uid}] Camera not open")
            return False

        if self.is_grabbing:
            print(f"[DeviceUserID: {uid}] Already grabbing")
            return False

        save_dir = self.config.get('save_dir', SAVE_DIRECTORY)
        ensure_dir(save_dir)

        self.saved_frame_count = get_last_image_count(save_dir)

        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            print(f"[DeviceUserID: {uid}] Start grabbing failed: 0x{to_hex_str(ret)}")
            return False

        self.is_grabbing = True
        self.stop_event.clear()
        self.frame_count = 0
        self.last_save_time = time.time()

        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.grab_thread.start()

        self.process_threads = []
        for _ in range(MAX_PARALLEL_FRAMES):
            t = threading.Thread(target=self._process_worker, daemon=True)
            t.start()
            self.process_threads.append(t)

        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()

        print(f"\n{'='*70}")
        print(f"[DeviceUserID: {uid}] ✓ STARTED AUTOMATIC IMAGE CAPTURE")
        print(f"  Save Directory: {save_dir}")
        # print(f"  Timer Interval: {TIMER_SAVE_INTERVAL_SEC} seconds")
        print(f"  Format: {SAVE_FORMAT} (Quality: {JPEG_QUALITY if SAVE_FORMAT=='JPEG' else 'N/A'})")
        print(f"{'='*70}\n")

        return True

    def _grab_loop(self):
        stOutFrame = MV_FRAME_OUT()
        uid = self.config.get('user_id', 'Unknown')
        # save_dir = self.config.get('save_dir', SAVE_DIRECTORY)

        while not self.stop_event.is_set():
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 5000)

            if ret == MV_OK:
                self.frame_count += 1

                frame_data = {
                    'width':      stOutFrame.stFrameInfo.nWidth,
                    'height':     stOutFrame.stFrameInfo.nHeight,
                    'pixel_type': stOutFrame.stFrameInfo.enPixelType,
                    'data_len':   stOutFrame.stFrameInfo.nFrameLen,
                    'data':       string_at(stOutFrame.pBufAddr,
                                            stOutFrame.stFrameInfo.nFrameLen)
                }

                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)  # free immediately

                # Push to process queue (drop oldest if full)
                if RAW_FRAME_QUEUE.full():
                    try:
                        RAW_FRAME_QUEUE.get_nowait()
                    except queue.Empty:
                        pass
                RAW_FRAME_QUEUE.put(frame_data)
                log_debug(f"[{uid}] Frame {self.frame_count} queued for inspection")

            elif ret == MV_E_NODATA:
                log_debug(f"[{uid}] Waiting for trigger...")
            else:
                log_debug(f"[{uid}] Get frame failed: 0x{to_hex_str(ret)}")
    
    def _process_worker(self):
        """Dequeue frames, run template matching + patchcore (based on config flags), save result image."""
        uid = self.config.get('user_id', 'Unknown')
        save_dir = self.config.get('save_dir', SAVE_DIRECTORY)

        while not self.stop_event.is_set():
            try:
                frame_data = RAW_FRAME_QUEUE.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._save_count_lock:
                self._active_workers += 1

            try:
                t_total_start = perf_counter()

                # Step 1: Convert raw buffer → BGR image
                t0 = perf_counter()
                bgr = frame_to_bgr(frame_data, self.obj_cam)
                convert_ms = (perf_counter() - t0) * 1000

                # Step 2: Run enabled engines in parallel
                template_ms  = 0.0
                patchcore_ms  = 0.0
                parallel_ms  = 0.0
                results      = []
                patchcore_res      = None

                def _run_template(b):
                    t = perf_counter()
                    r = template_engine.inspect_frame(b, TEMPLATE_IMAGES, templates, PREPPED_TEMPLATES)
                    return r, (perf_counter() - t) * 1000

                def _run_patchcore_timed(b, roi):
                    t = perf_counter()
                    r = self._run_patchcore(b, roi)
                    return r, (perf_counter() - t) * 1000

                t_parallel_start = perf_counter()

                template_future = None
                patchcore_future      = None

                if ENABLE_TEMPLATE_MATCHING:
                    template_future = TEMPLATE_EXECUTOR.submit(_run_template, bgr)

                if ENABLE_PATCHCORE:
                    patchcore_future = PATCHCORE_EXECUTOR.submit(
                        _run_patchcore_timed, bgr, templates[0]["expected_roi"]
                    )

                if template_future is not None:
                    results, template_ms = template_future.result()

                if patchcore_future is not None:
                    patchcore_res, patchcore_ms = patchcore_future.result()

                parallel_ms = (perf_counter() - t_parallel_start) * 1000

                # Step 3: Find best template match (only if template matching enabled)
                matched_result   = None
                matched_template = None

                if ENABLE_TEMPLATE_MATCHING and results:
                    for res, tpl in zip(results, templates):
                        if res["ok"] and res["match_loc"] is not None:
                            if matched_result is None or res["score"] > matched_result["score"]:
                                matched_result   = res
                                matched_template = tpl

                # Step 4: Draw results on overlay
                t0 = perf_counter()
                overlay = bgr.copy()

                if ENABLE_TEMPLATE_MATCHING and matched_result is not None:
                    ey1, ey2, ex1, ex2 = matched_template["expected_roi"]
                    mx, my = matched_result["match_loc"]
                    h, w, _ = matched_result["match_shape"]
                    x, y = mx + ex1, my + ey1
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(overlay,
                                f"{matched_result['part_id']} | {matched_result['score']:.2f}",
                                (x, y - 10), FONT, 0.8, (0, 255, 0), 2)

                draw_ms = (perf_counter() - t0) * 1000

                # Step 5: QC decision based on enabled engines
                qc_fail = False

                if ENABLE_TEMPLATE_MATCHING and ENABLE_PATCHCORE:
                    if matched_result is not None and patchcore_res is not None and patchcore_res["is_ng"]:
                        qc_fail = True

                elif ENABLE_TEMPLATE_MATCHING and not ENABLE_PATCHCORE:
                    if matched_result is None:
                        qc_fail = True

                elif ENABLE_PATCHCORE and not ENABLE_TEMPLATE_MATCHING:
                    if patchcore_res is not None and patchcore_res["is_ng"]:
                        qc_fail = True

                # Draw PatchCore result on overlay
                if ENABLE_PATCHCORE and patchcore_res is not None:
                    roi_coords = matched_template["expected_roi"] if matched_template else templates[0]["expected_roi"]
                    ey1, ey2, ex1, ex2 = roi_coords
                    result_color = (0, 0, 255) if patchcore_res["is_ng"] else (0, 255, 0)
                    cv2.putText(overlay,
                                f"PatchCore: {'NG' if patchcore_res['is_ng'] else 'OK'} {patchcore_res['confidence']*100:.1f}%",
                                (ex1 + 10, ey1 + 30), FONT, 0.9, result_color, 2)

                # Step 6: Final status text
                if ENABLE_TEMPLATE_MATCHING and not ENABLE_PATCHCORE:
                    # Template only mode
                    if matched_result is None:
                        final_status = "NO_PART"
                        color = (0, 0, 255)
                    else:
                        final_status = "OK"
                        color = (0, 255, 0)

                elif ENABLE_PATCHCORE and not ENABLE_TEMPLATE_MATCHING:
                    # PatchCore only mode
                    if patchcore_res is None:
                        final_status = "NO_RESULT"
                        color = (0, 165, 255)
                    elif patchcore_res["is_ng"]:
                        final_status = "NG"
                        color = (0, 0, 255)
                    else:
                        final_status = "OK"
                        color = (0, 255, 0)

                elif ENABLE_TEMPLATE_MATCHING and ENABLE_PATCHCORE:
                    # Both enabled
                    if matched_result is None:
                        final_status = "NO_PART"
                        color = (0, 0, 255)
                    elif qc_fail:
                        final_status = "NG"
                        color = (0, 0, 255)
                    else:
                        final_status = "OK"
                        color = (0, 255, 0)

                else:
                    # Both disabled — should not happen, but handle gracefully
                    final_status = "DISABLED"
                    color = (0, 165, 255)

                cv2.putText(overlay, f"FINAL: {final_status}",
                            (30, 40), FONT, 1.2, color, 3)

                # Show active engine(s) on overlay for clarity
                engine_label = []
                if ENABLE_TEMPLATE_MATCHING:
                    engine_label.append("TM")
                if ENABLE_PATCHCORE:
                    engine_label.append("PATCHCORE")
                cv2.putText(overlay,
                            f"Engine: {'+'.join(engine_label) if engine_label else 'NONE'}",
                            (30, 80), FONT, 0.7, (255, 255, 0), 2)

                total_ms = (perf_counter() - t_total_start) * 1000

                with self._save_count_lock:
                    self.saved_frame_count += 1
                    count = self.saved_frame_count
                    active_now = self._active_workers

                # Save Anomalib
                if ENABLE_PATCHCORE and patchcore_res is not None and patchcore_res["vis_image"] is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    status_tag = "NG" if patchcore_res["is_ng"] else "OK"
                    vis_path = os.path.join(
                        save_dir,
                        f"patchcore_{status_tag}_{count :04d}.jpg"
                    )
                    cv2.imwrite(
                        vis_path,
                        patchcore_res["vis_image"],
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    log_debug(f"[{uid}] PatchCore vis saved → {vis_path}")
                
                # Encode overlay as JPEG for UI stream
                ok, jpeg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    with JPEG_LOCK:
                        global LATEST_JPEG_PROCESS
                        LATEST_JPEG_PROCESS = jpeg.tobytes()

                # Store latest detection for UI polling
                det = {
                    "part_id":        matched_result["part_id"] if matched_result else None,
                    "final_status":   final_status,
                    "qc_fail":        qc_fail,
                    "active_worker":  active_now,
                    "total_worker":   MAX_PARALLEL_FRAMES,
                    "frame_count":    count,
                    "score":          matched_result["score"] if matched_result else 0.0,
                    "patchcore_result":      patchcore_res["class"] if patchcore_res else None,
                    "patchcore_confidence": patchcore_res["confidence"] if patchcore_res else None,
                    "engine":         "+".join(engine_label) if engine_label else "NONE"
                }
                with DETECTION_LOCK:
                    global LATEST_DETECTION
                    LATEST_DETECTION = det
                
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{uid}] "
                    f"✓ #{count} → {final_status} | "
                    f"engine={det['engine']} | "
                    f"active={active_now}/{MAX_PARALLEL_FRAMES} | "
                    f"score={(matched_result['score'] if matched_result is not None else 0.0):.2f} | "
                    f"convert={convert_ms:.1f}ms | "
                    f"template={template_ms:.1f}ms | "
                    f"patchcore={patchcore_ms:.1f}ms | "
                    f"parallel={parallel_ms:.1f}ms | "
                    f"draw={draw_ms:.1f}ms | "
                    # f"save={save_ms:.1f}ms | "
                    f"TOTAL={total_ms:.1f}ms"
                )

            except Exception as e:
                print(f"[{uid}] [ERROR] Process worker: {e}")
                import traceback
                traceback.print_exc()
            finally:
                with self._save_count_lock:
                    self._active_workers -= 1

    def _run_patchcore(self, bgr, roi_coords):
        return PATCHCORE_ENGINE_INST.run_inference(bgr, roi_coords)

    def _status_monitor(self):
        uid = self.config.get('user_id', 'Unknown')
        while not self.stop_event.is_set():
            if self.is_open:
                ret = self.obj_cam.MV_CC_IsDeviceConnected()
                if not ret:
                    print(f"\n[DeviceUserID: {uid}] *** WARNING: Camera disconnected! ***\n")

            time.sleep(STATUS_CHECK_INTERVAL_SEC)

    def stop_grabbing(self):
        if not self.is_grabbing:
            return

        uid = self.config.get('user_id', 'Unknown')
        print(f"\n[DeviceUserID: {uid}] Stopping...")

        self.stop_event.set()

        if self.grab_thread:
            self.grab_thread.join(timeout=2.0)

        for t in self.process_threads:
            t.join(timeout=3.0)
        self.process_threads = []

        if self.is_open:
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != MV_OK:
                print(f"[DeviceUserID: {uid}] Stop failed: 0x{to_hex_str(ret)}")

        self.is_grabbing = False
        print(f"[DeviceUserID: {uid}] Stopped (Frames: {self.frame_count}, Saved: {self.saved_frame_count})")

    def disconnect(self):
        if self.is_grabbing:
            self.stop_grabbing()

        uid = self.config.get('user_id', 'Unknown')
        if self.is_open:
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret == MV_OK:
                print(f"[DeviceUserID: {uid}] Disconnected")
            self.obj_cam.MV_CC_DestroyHandle()
            self.is_open = False

def main():
    print("\n" + "=" * 80)
    print(" itek Camera - Device User ID Connection (NO IP)")
    print("=" * 80)

    if ENABLE_MULTIPLE_CAMERAS:
        print(" Mode: MULTI CAMERA")
        for c in CAMERA_LIST:
            print(f"  - UserID: {c.get('user_id')} -> Save: {c.get('save_dir')}")
    else:
        print(" Mode: SINGLE CAMERA")
        print(f" Target DeviceUserID: {TARGET_DEVICE_USER_ID}")
        print(f" Save Directory: {SAVE_DIRECTORY}")

    # print(f" Timer Interval: {TIMER_SAVE_INTERVAL_SEC} seconds")
    print("=" * 80 + "\n")

    MvCamera.MV_CC_Initialize()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    layers = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE |
              MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
    ret = MvCamera.MV_CC_EnumDevices(layers, deviceList)

    if ret != MV_OK:
        print(f"[ERROR] Enumerate failed: 0x{to_hex_str(ret)}")
        MvCamera.MV_CC_Finalize()
        return

    # set static ip
    force_ip_before_open(deviceList)

    # Then re-enumerate so SDK sees the new IP
    time.sleep(3)
    ret = MvCamera.MV_CC_EnumDevices(layers, deviceList)

    if deviceList.nDeviceNum == 0:
        print("[ERROR] No cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check camera power")
        print("  2. Check network cable / USB cable")
        print("  3. Open MVS software to verify camera")
        print("  4. Check firewall settings")
        MvCamera.MV_CC_Finalize()
        return

    print(f"Found {deviceList.nDeviceNum} camera(s)\n")
    print("Available cameras:")
    for i in range(deviceList.nDeviceNum):
        device_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        print(f"  [{i}] {get_device_info_string(device_info)}")
    print()

    cameras = []

    try:
        if ENABLE_MULTIPLE_CAMERAS:
            # Connect each camera by user_id from CAMERA_LIST
            for cam_cfg in CAMERA_LIST:
                uid = (cam_cfg.get("user_id") or "").strip()
                if not uid:
                    print("[WARN] Empty user_id in CAMERA_LIST, skipping...")
                    continue

                idx = find_device_by_user_id(deviceList, uid)
                if idx < 0:
                    print(f"[WARN] Camera UserID '{uid}' not found, skipping...")
                    continue

                device_info = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
                controller = CameraController(cam_cfg)

                if controller.connect(device_info):
                    controller.start_grabbing()
                    cameras.append(controller)

            if not cameras:
                print("[ERROR] No cameras connected (check DeviceUserID in MVS tool).")
                MvCamera.MV_CC_Finalize()
                return

        else:
            # Single camera by TARGET_DEVICE_USER_ID
            if not TARGET_DEVICE_USER_ID:
                print("[ERROR] TARGET_DEVICE_USER_ID not set!")
                MvCamera.MV_CC_Finalize()
                return

            idx = find_device_by_user_id(deviceList, TARGET_DEVICE_USER_ID)

            if idx < 0:
                print(f"[WARNING] Camera UserID '{TARGET_DEVICE_USER_ID}' not found!")
                if AUTO_CONNECT_FIRST_CAMERA and deviceList.nDeviceNum > 0:
                    print("[INFO] Auto-connecting to first camera...")
                    idx = 0
                    first_info = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
                    actual_uid = get_device_user_id(first_info)
                    config = {"user_id": actual_uid if actual_uid else "FIRST_CAMERA", "save_dir": SAVE_DIRECTORY}
                else:
                    print("\n[ERROR] Cannot connect!")
                    print("Manual fix required:")
                    print("  1. Open MVS software")
                    print("  2. Select your camera")
                    print(f"  3. Set DeviceUserID / UserDefinedName to: {TARGET_DEVICE_USER_ID}")
                    print("  4. Run this script again")
                    MvCamera.MV_CC_Finalize()
                    return
            else:
                config = {"user_id": TARGET_DEVICE_USER_ID, "save_dir": SAVE_DIRECTORY}

            device_info = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
            controller = CameraController(config)

            if controller.connect(device_info):
                controller.start_grabbing()
                cameras.append(controller)

            if not cameras:
                print("[ERROR] Failed to connect")
                MvCamera.MV_CC_Finalize()
                return

        print("=" * 80)
        print(" AUTOMATIC IMAGE CAPTURE ACTIVE")
        if ENABLE_MULTIPLE_CAMERAS:
            print(" Multiple cameras are saving automatically.")
        else:
            print(f" Saving every seconds to {SAVE_DIRECTORY}")
        print(" Press Ctrl+C to stop...")
        print("=" * 80 + "\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print(" SHUTDOWN REQUESTED")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        for cam in cameras:
            cam.disconnect()

        MvCamera.MV_CC_Finalize()
        print("\n" + "=" * 80)
        print(" Cleanup complete. Exited.")
        print("=" * 80 + "\n")


def start_system():
    global SYSTEM_RUNNING, CAMERA_CONTROLLERS

    if SYSTEM_RUNNING:
        return True

    _reload_config()
    SYSTEM_RUNNING = True
    CAMERA_CONTROLLERS.clear()

    MvCamera.MV_CC_Initialize()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    layers = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE |
              MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
    ret = MvCamera.MV_CC_EnumDevices(layers, deviceList)
    if ret != MV_OK or deviceList.nDeviceNum == 0:
        raise RuntimeError("No camera found")

    force_ip_before_open(deviceList)
    time.sleep(3)
    MvCamera.MV_CC_EnumDevices(layers, deviceList)

    idx = find_device_by_user_id(deviceList, TARGET_DEVICE_USER_ID)
    if idx < 0:
        idx = 0

    device_info = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
    controller = CameraController({"user_id": TARGET_DEVICE_USER_ID, "save_dir": SAVE_DIRECTORY})

    if controller.connect(device_info):
        controller.start_grabbing()
        CAMERA_CONTROLLERS.append(controller)

    return True

def stop_system():
    global SYSTEM_RUNNING, CAMERA_CONTROLLERS

    if not SYSTEM_RUNNING:
        return True

    for cam in CAMERA_CONTROLLERS:
        cam.disconnect()

    CAMERA_CONTROLLERS.clear()
    MvCamera.MV_CC_Finalize()
    SYSTEM_RUNNING = False
    return True