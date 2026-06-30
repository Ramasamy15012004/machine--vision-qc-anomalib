# tools/capture_dataset.py
import os
import sys
import time
from ctypes import *
import threading

# Ensure the project root is in the system path to allow importing the backend
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.camera.mvs_sdk_path import add_sdk_to_path
add_sdk_to_path()

from CameraParams_header import *
from MvCameraControl_class import *
from MvErrorDefine_const import *

from backend.camera.get_image import (
    CameraController,
    force_ip_before_open,
    find_device_by_user_id,
    get_device_user_id,
    get_device_info_string,
    to_hex_str,
    TARGET_DEVICE_USER_ID,
    AUTO_CONNECT_FIRST_CAMERA,
    ENABLE_MULTIPLE_CAMERAS,
    CAMERA_LIST,
    SAVE_DIRECTORY,
    TIMER_SAVE_INTERVAL_SEC
)

def main():
    print("\n" + "=" * 80)
    print(" Industrial Machine Vision System")
    print(" Camera Connection and Dataset Capture Utility")
    print("=" * 80)

    if ENABLE_MULTIPLE_CAMERAS:
        print(" Mode: MULTI CAMERA")
        for c in CAMERA_LIST:
            print(f"  - UserID: {c.get('user_id')} -> Save: {c.get('save_dir')}")
    else:
        print(" Mode: SINGLE CAMERA")
        print(f" Target DeviceUserID: {TARGET_DEVICE_USER_ID}")
        print(f" Save Directory: {SAVE_DIRECTORY}")

    print(f" Timer Interval: {TIMER_SAVE_INTERVAL_SEC} seconds")
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

    # Set static IP if required
    force_ip_before_open(deviceList)

    # Re-enumerate to ensure SDK registers static IP changes
    time.sleep(3)
    ret = MvCamera.MV_CC_EnumDevices(layers, deviceList)

    if deviceList.nDeviceNum == 0:
        print("[ERROR] No cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check camera power")
        print("  2. Check network cable / USB cable")
        print("  3. Open MVS software to verify camera connection")
        print("  4. Check network firewall settings")
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
                print("[ERROR] No cameras connected.")
                MvCamera.MV_CC_Finalize()
                return

        else:
            if not TARGET_DEVICE_USER_ID:
                print("[ERROR] TARGET_DEVICE_USER_ID not set!")
                MvCamera.MV_CC_Finalize()
                return

            idx = find_device_by_user_id(deviceList, TARGET_DEVICE_USER_ID)

            if idx < 0:
                print(f"[WARNING] Camera UserID '{TARGET_DEVICE_USER_ID}' not found!")
                if AUTO_CONNECT_FIRST_CAMERA and deviceList.nDeviceNum > 0:
                    print("[INFO] Auto-connecting to first available camera...")
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
                print("[ERROR] Failed to connect camera")
                MvCamera.MV_CC_Finalize()
                return

        print("=" * 80)
        print(" AUTOMATIC IMAGE CAPTURE ACTIVE")
        if ENABLE_MULTIPLE_CAMERAS:
            print(" Multiple cameras are saving automatically.")
        else:
            print(f" Saving every {TIMER_SAVE_INTERVAL_SEC} seconds to {SAVE_DIRECTORY}")
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

if __name__ == "__main__":
    main()
