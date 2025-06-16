import numpy as np

def vitpose25_to_coco17(keypoints):
    """
    Convert VitPose 25 keypoints to COCO 17 keypoints format.
    
    VitPose25 keypoint order:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle,
    17-24: additional face/hand keypoints
    
    COCO17 keypoint order:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    if isinstance(keypoints, np.ndarray):
        # Extract first 17 keypoints (which correspond to COCO17)
        coco17_kpts = keypoints[:, :17, :].copy()
        return coco17_kpts
    else:
        return keypoints

def convert_keypoints_for_json(keypoints, source_format='vitpose25'):
    """
    Convert keypoints to COCO17 format for JSON export.
    """
    if source_format == 'vitpose25':
        return vitpose25_to_coco17(keypoints)
    else:
        # Default: return as is
        return keypoints
