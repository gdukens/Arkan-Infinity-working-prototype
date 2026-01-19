"""Create a HandLandmarker (VIDEO mode) and run detect_for_video on a captured frame."""
import time
import sys
import importlib

import cv2

# Import tasks hand_landmarker module
try:
    hand_mod = importlib.import_module('mediapipe.tasks.python.vision.hand_landmarker')
    core_mod = importlib.import_module('mediapipe.tasks.python.vision.core.image')
    core_base = importlib.import_module('mediapipe.tasks.python.core.base_options')
except Exception as e:
    print("Failed to import mediapipe modules:", e)
    sys.exit(1)

# Build options
HandLandmarker = getattr(hand_mod, 'HandLandmarker')
HandLandmarkerOptions = getattr(hand_mod, 'HandLandmarkerOptions')
BaseOptions = getattr(core_base, 'BaseOptions')
# Try to find running mode
try:
    vm = importlib.import_module('mediapipe.tasks.python.vision.core.vision_task_running_mode')
    RunningMode = getattr(vm, 'VisionTaskRunningMode')
except Exception:
    try:
        v = importlib.import_module('mediapipe.tasks.python.vision')
        RunningMode = getattr(v, 'VisionRunningMode')
    except Exception:
        RunningMode = None

running_mode_final = None
if RunningMode is not None:
    running_mode_final = getattr(RunningMode, 'VIDEO', getattr(RunningMode, 'VIDEO', None))

options_kwargs = {
    'base_options': BaseOptions(model_asset_path='hand_landmarker.task'),
    'num_hands': 2,
}
if running_mode_final is not None:
    options_kwargs['running_mode'] = running_mode_final

options = HandLandmarkerOptions(**options_kwargs)
print('Options created:', options)

try:
    landmarker = HandLandmarker.create_from_options(options)
    print('HandLandmarker created OK')
except Exception as e:
    print('Failed to create HandLandmarker:', e)
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Failed to open camera')
    sys.exit(1)

ret, frame = cap.read()
cap.release()
if not ret:
    print('Failed to capture frame')
    sys.exit(1)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print('Captured frame shape, dtype:', rgb.shape, rgb.dtype)

# Wrap in Image
Image = core_mod.Image
img = Image(core_mod.ImageFormat.SRGB, rgb)
print('Created Image wrapper: has ptr?', hasattr(img, '_image_ptr'))

# Call detect_for_video
ts = int(time.time() * 1000)
try:
    res = landmarker.detect_for_video(img, ts)
    print('Detection result:', res)
    if getattr(res, 'hand_landmarks', None):
        print('hands:', len(res.hand_landmarks))
except Exception as e:
    print('detect_for_video failed:', e)

landmarker.close()
print('Done')
