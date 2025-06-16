"""
Comprehensive joint format metadata for the PromptHMR pipeline.
"""

def get_smplx_joint_names():
    """Get SMPL-X joint names in order."""
    return [
        # Body joints (22)
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        # Hand joints (30 per hand = 60 total)
        # Left hand
        'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
        'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
        'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb_tip', 'left_index_tip', 'left_middle_tip',
        'left_ring_tip', 'left_pinky_tip',
        # Right hand
        'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3',
        'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
        'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb_tip', 'right_index_tip', 'right_middle_tip',
        'right_ring_tip', 'right_pinky_tip',
        # Face joints (variable, depends on model)
        # ... (face joints would be added here)
    ]

def get_vitpose25_joint_names():
    """Get VitPose25 joint names in order."""
    return [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel',
        'neck', 'mid_hip'
    ]

def get_coco17_joint_names():
    """Get COCO17 joint names in order."""
    return [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

def get_joint_format_metadata():
    """Get complete joint format metadata."""
    return {
        'vitpose25': {
            'count': 25,
            'structure': 'x,y,confidence triplets',
            'coordinate_system': 'image_pixels',
            'joints': get_vitpose25_joint_names()
        },
        'coco17': {
            'count': 17,
            'structure': 'x,y,confidence triplets',
            'coordinate_system': 'image_pixels',
            'joints': get_coco17_joint_names()
        },
        'smplx': {
            'count': 'variable',
            'structure': 'x,y,z coordinates',
            'coordinate_system': 'world_meters',
            'description': 'SMPL-X model joints including body, hands, and face',
            'joints': get_smplx_joint_names()
        },
        'bbox_center': {
            'count': 1,
            'structure': 'x,y,confidence',
            'coordinate_system': 'image_pixels',
            'joints': ['bbox_center']
        }
    }
