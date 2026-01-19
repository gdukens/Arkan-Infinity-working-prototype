import sys
sys.path.insert(0, '.')
import userInterface as ui
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera')
    raise SystemExit(1)
ret, frame = cap.read()
cap.release()
if not ret:
    print('No frame')
    raise SystemExit(2)

# Convert to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
if ui.hands is not None:
    res = ui.hands.process(rgb)
    print('solutions results:', type(res), dir(res)[:20])
    print('multi_hand_landmarks:', getattr(res, 'multi_hand_landmarks', None))
elif ui.has_mediapipe_tasks and ui.tasks_hand_landmarker is not None and ui.mp_tasks_vision is not None:
    ImageClass = getattr(ui.mp_tasks_vision, 'Image', None)
    if ImageClass is not None:
        try:
            mp_image = ImageClass.create_from_array(rgb)
            timestamp_ms = int(__import__('time').time() * 1000)
            task_result = ui.tasks_hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print('detect_for_video failed:', type(e), e)
            task_result = None
    else:
        # Try a numpy-based detection fallback (some builds accept raw arrays via `detect`)
        try:
            task_result = ui.tasks_hand_landmarker.detect(rgb)
        except Exception as e:
            print('numpy detect fallback failed:', type(e), e)
            task_result = None

    if task_result is not None:
        try:
            print('task_result attributes:', [a for a in dir(task_result) if not a.startswith('_')])
            print('hand_landmarks present?', getattr(task_result, 'hand_landmarks', None) is not None)
            if getattr(task_result, 'hand_landmarks', None):
                print('number of hands:', len(task_result.hand_landmarks))
                first = task_result.hand_landmarks[0]
                print('first landmarks length', len(getattr(first, 'landmark', first)))
        except Exception as e:
            print('Error reading task_result:', e)
    else:
        print('No task_result produced')
else:
    print('No detection backend available')
