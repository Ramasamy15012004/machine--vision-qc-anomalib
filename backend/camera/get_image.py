# backend/camera/get_image.py
import os
import sys
import time
import ctypes
from ctypes import *
from datetime import datetime
import threading
import numpy as np
import cv2

# -------------------- CENTRAL PATH CONFIGURATION --------------------
from backend.camera.mvs_sdk_path import add_sdk_to_path
add_sdk_to_path()

from CameraParams_header import *
from MvCameraControl_class import *
from MvErrorDefine_const import *

# ==================== DEFAULT CONFIGURATION ====================
TARGET_DEVICE_USER_ID = "Cam1"
AUTO_CONNECT_FIRST_CAMERA = True
ENABLE_MULTIPLE_CAMERAS = False
CAMERA_LIST = [
    {"user_id": "Cam1", "save_dir": "data/captured"},
    {"user_id": "Cam2", "save_dir": "data/captured"},
]

SAVE_DIRECTORY = "data/captured"
TIMER_SAVE_INTERVAL_SEC = 2
SAVE_FORMAT = "JPEG"
JPEG_QUALITY = 90

EXPOSURE_TIME_US = 20000
GAIN_DB = 0.0
AUTO_EXPOSURE = False
AUTO_GAIN = False

WHITE_BALANCE_AUTO = False
WB_RED_RATIO = 1.5
WB_GREEN_RATIO = 1.0
WB_BLUE_RATIO = 1.8

ROI_ENABLE   = True
ROI_OFFSET_X = 330
ROI_OFFSET_Y = 1000
FRAME_WIDTH = 780               
FRAME_HEIGHT = 620    

PIXEL_FORMAT = "BayerRG8"
STATUS_CHECK_INTERVAL_SEC = 5.0
ENABLE_DEBUG_LOGS = True
FRAME_COUNT_LOG_INTERVAL = 50

TARGET_FORCE_IP       = "169.154.73.1"
TARGET_FORCE_SUBNET   = "255.255.0.0"
TARGET_FORCE_GATEWAY  = "169.154.0.1"
# ==================== END CONFIGURATION ====================

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
        if ext.lower() in ('.jpg', '.png', '.bmp'):
            if name.isdigit():
                count = int(name)
                if count > max_count:
                    max_count = count
    return max_count

def timestamp_filename(ext):
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext

def log_debug(message):
    if ENABLE_DEBUG_LOGS:
        print(f"[DEBUG] {message}")

def ip_to_int(ip_str):
    parts = ip_str.split('.')
    return (int(parts[0])<<24)+(int(parts[1])<<16)+(int(parts[2])<<8)+int(parts[3])

def int_to_ip(ip_int):
    return f"{(ip_int>>24)&0xFF}.{(ip_int>>16)&0xFF}.{(ip_int>>8)&0xFF}.{ip_int&0xFF}"

def force_ip_before_open(device_list):
    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType not in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            continue

        current_ip = int_to_ip(info.SpecialInfo.stGigEInfo.nCurrentIp)
        if current_ip == TARGET_FORCE_IP:
            continue

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

def preprocess_for_anomalib(bgr_image):
    x1, y1 = 350, 200
    x2, y2 = 950, 850
    cropped = bgr_image[y1:y2, x1:x2]

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    pseudo_rgb = cv2.merge([enhanced, enhanced, enhanced])
    resized = cv2.resize(pseudo_rgb, (256, 256))
    return gray

def get_device_user_id(device_info):
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
    uid = get_device_user_id(device_info)
    if device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        gige_info = device_info.SpecialInfo.stGigEInfo
        serial = decoding_char(gige_info.chSerialNumber).strip()
        model = decoding_char(gige_info.chModelName).strip()
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
        self.stop_event = threading.Event()

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

        self._set_enum_str("AcquisitionMode", "Continuous")
        self._set_enum_str("TriggerSelector", "FrameBurstStart")
        self._set_enum_str("TriggerMode", "On")
        self._set_enum_str("TriggerSource", "Line0")
        self._set_enum_str("TriggerActivation", "RisingEdge")
        print(f"  [OK] Hardware trigger mode enabled - Line0 (Pin 2)")
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

        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            print(f"[DeviceUserID: {uid}] Start grabbing failed: 0x{to_hex_str(ret)}")
            return False

        self.is_grabbing = True
        self.stop_event.clear()
        save_dir = self.config.get('save_dir', SAVE_DIRECTORY)
        self.saved_frame_count = get_last_image_count(save_dir)
        print(f"  [INFO] Continuing from image #{self.saved_frame_count}")
        self.frame_count = 0
        self.last_save_time = time.time()

        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.grab_thread.start()

        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()

        print(f"\n{'='*70}")
        print(f"[DeviceUserID: {uid}] ✓ STARTED AUTOMATIC IMAGE CAPTURE")
        print(f"  Save Directory: {save_dir}")
        print(f"  Timer Interval: {TIMER_SAVE_INTERVAL_SEC} seconds")
        print(f"  Format: {SAVE_FORMAT} (Quality: {JPEG_QUALITY if SAVE_FORMAT=='JPEG' else 'N/A'})")
        print(f"{'='*70}\n")

        return True

    def _grab_loop(self):
        stOutFrame = MV_FRAME_OUT()
        uid = self.config.get('user_id', 'Unknown')
        save_dir = self.config.get('save_dir', SAVE_DIRECTORY)

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

                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)

                bgr = frame_to_bgr(frame_data, self.obj_cam)
                processed = preprocess_for_anomalib(bgr)

                filepath = os.path.join(save_dir, f"{self.saved_frame_count + 1:04d}.jpg")
                cv2.imwrite(filepath, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                self.saved_frame_count += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"[{timestamp}] [{uid}] ✓ Saved #{self.saved_frame_count} (trigger {self.frame_count})")

            elif ret == MV_E_NODATA:
                log_debug(f"[{uid}] Waiting for trigger...")
            else:
                log_debug(f"[{uid}] Get frame failed: 0x{to_hex_str(ret)}")

    def _timer_save_loop(self):
        save_dir = self.config.get('save_dir', SAVE_DIRECTORY)
        uid = self.config.get('user_id', 'Unknown')

        print(f"[DeviceUserID: {uid}] Auto-save started - every {TIMER_SAVE_INTERVAL_SEC}s to {save_dir}\n")

        while not self.stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - self.last_save_time

            if elapsed >= TIMER_SAVE_INTERVAL_SEC:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        success = self._save_frame_from_buffer(self.latest_frame, save_dir)
                        if success:
                            self.saved_frame_count += 1
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            print(f"[{timestamp}] [DeviceUserID: {uid}] ✓ Saved image #{self.saved_frame_count}")
                        else:
                            print(f"[DeviceUserID: {uid}] ✗ Failed to save image")

                        self.last_save_time = current_time

            time.sleep(0.05)

    def _save_frame_from_buffer(self, frame_data, save_dir):
        try:
            if SAVE_FORMAT.upper() == "JPEG":
                ext = ".jpg"
                img_type = MV_Image_Jpeg
                quality = JPEG_QUALITY
            else:
                ext = ".bmp"
                img_type = MV_Image_Bmp
                quality = 0

            file_path = os.path.join(save_dir, timestamp_filename(ext))
            data_buffer = (ctypes.c_ubyte * frame_data['data_len']).from_buffer_copy(frame_data['data'])

            param = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
            memset(byref(param), 0, sizeof(param))
            param.enPixelType = frame_data['pixel_type']
            param.nWidth = frame_data['width']
            param.nHeight = frame_data['height']
            param.nDataLen = frame_data['data_len']
            param.pData = cast(data_buffer, POINTER(c_ubyte))
            param.enImageType = img_type
            param.pcImagePath = ctypes.create_string_buffer(file_path.encode('ascii'))
            param.iMethodValue = 1
            param.nQuality = quality

            ret = self.obj_cam.MV_CC_SaveImageToFileEx(param)
            if ret != MV_OK:
                log_debug(f"Save failed: 0x{to_hex_str(ret)}")
                return False
            return True
        except Exception as e:
            print(f"[ERROR] Exception during save: {e}")
            return False

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
