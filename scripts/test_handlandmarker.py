import traceback
print('--- Environment Diagnostic ---')
# cv2
try:
    import cv2
    print('cv2 version:', cv2.__version__)
except Exception as e:
    print('cv2 import error:', type(e), e)

# mediapipe
try:
    import mediapipe as mp
    print('mediapipe version:', getattr(mp, '__version__', 'unknown'))
    print('mp.solutions available:', 'solutions' in dir(mp))
except Exception as e:
    print('mediapipe import error:', type(e), e)

# mediapipe.tasks
try:
    import mediapipe.tasks as tasks_pkg
    print('mediapipe.tasks present:', True)
    mpv = getattr(tasks_pkg, 'vision', None)
    print('mediapipe.tasks.vision present:', mpv is not None)
    if mpv is not None:
        keys = [n for n in dir(mpv) if not n.startswith('_')]
        print('vision symbols sample:', keys[:40])
except Exception as e:
    print('mediapipe.tasks error:', type(e), e)

# Attempt HandLandmarker creation
try:
    if mpv is None:
        raise RuntimeError('vision module missing')
    HandLandmarker = getattr(mpv, 'HandLandmarker')
    HandLandmarkerOptions = getattr(mpv, 'HandLandmarkerOptions')
    BaseOptions = getattr(tasks_pkg, 'BaseOptions')
    RunningMode = getattr(mpv, 'RunningMode')
    print('Found HandLandmarker symbols')
    opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='hand_landmarker.task'), running_mode=RunningMode.IMAGE, num_hands=2)
    hl = HandLandmarker.create_from_options(opts)
    print('HandLandmarker created OK')
    hl.close()
except Exception as e:
    print('HandLandmarker create error:', type(e), e)
    traceback.print_exc()

print('--- End Diagnostic ---')
