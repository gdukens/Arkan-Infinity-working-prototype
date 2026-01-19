import sys
sys.path.insert(0, '.')
import userInterface as ui
print('has_mediapipe_solutions=', ui.has_mediapipe_solutions)
print('has_mediapipe_tasks=', ui.has_mediapipe_tasks)
print('tasks_init_error=', ui.tasks_init_error)
print('tasks_hand_landmarker is None?', ui.tasks_hand_landmarker is None)
print('mp_tasks_vision:', ui.mp_tasks_vision)
import os
print('hand_landmarker.task exists?', os.path.exists('hand_landmarker.task'))
