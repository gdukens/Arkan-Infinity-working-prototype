import sys
sys.path.insert(0, '.')
import userInterface as ui
mpv = ui.mp_tasks_vision
print('mp_tasks_vision module:', mpv)
print('available symbols sample:', [n for n in dir(mpv) if not n.startswith('_')][:120])
# Try to find Image or equivalent
candidates = [n for n in dir(mpv) if 'Image' in n or n.lower().startswith('image')]
print('Image-like candidates:', candidates)
# Show HandLandmarker API
hl = ui.tasks_hand_landmarker
print('HandLandmarker methods:', [n for n in dir(hl) if not n.startswith('_') and callable(getattr(hl,n))][:80])
try:
    import inspect
    print('detect_for_video signature:', inspect.signature(hl.detect_for_video))
except Exception as e:
    print('inspect error:', e)
