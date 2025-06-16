import os
import sys
import torch
import cv2
import json
import argparse
import numpy as np
from typing import Dict
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from pipeline import Pipeline

# BUGS:
# 1- Reusing the same pipeline causes major issues with GPU -> CPU tensor offloading. Data will not recover on 2nd batch
# 2- The result.pkl files have a /x2 bad data head. So the jsons in this script read directly from `results`, not cache

def __get_video_info(video_path: str) -> tuple:
    """Gets video information including total frames and FPS."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


def __extract_video_segment(video_path: str, output_path: str, start_time: float, duration: float) -> str:
    """Extracts a segment from video using ffmpeg."""
    import subprocess

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def __convert_tensor_to_list(item):
    """Recursively converts PyTorch tensors and numpy arrays to lists."""
    if isinstance(item, torch.Tensor):
        return item.cpu().detach().numpy().tolist()
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, np.generic):  # Handle NumPy scalar types
        return item.item()
    elif isinstance(item, dict):
        return {key: __convert_tensor_to_list(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [__convert_tensor_to_list(value) for value in item]
    else:
        return item


def __save_batch_poses_json(people: Dict, output_path: str, batch_num: int, frame_offset: int):
    """Save poses for a single batch in JSON format."""
    frame_poses = defaultdict(list)

    for pid, person_data in people.items():
        frames = person_data['frames']
        poses = person_data['smplx_world']['pose']
        shapes = person_data['smplx_world']['shape']
        trans = person_data['smplx_world']['trans']
        bboxes = person_data['bboxes']
        world_joints = person_data['smplx_world']['joints3d']
        
        # Get 2D keypoints data - ONLY use VitPose keypoints (preserve original format)
        keypoints_2d_json = person_data.get('keypoints_2d_json', None)
        keypoints_2d_format = person_data.get('keypoints_2d_format', 'vitpose25')
        detection_bbox = person_data.get('detection_bbox', None)

        # Debug print with proper handling of converted data
        if keypoints_2d_json is not None:
            if isinstance(keypoints_2d_json, list):
                print(f"Person {pid}: VitPose keypoints shape: {len(keypoints_2d_json)} frames, format: {keypoints_2d_format}")
            else:
                print(f"Person {pid}: VitPose keypoints shape: {keypoints_2d_json.shape}, format: {keypoints_2d_format}")
        else:
            print(f"Person {pid}: VitPose keypoints shape: None")

        for i, frame_num in enumerate(frames):
            # Adjust frame number with offset
            adjusted_frame_num = frame_num + frame_offset
            
            # Get ALL 3D joints without any truncation - preserve complete SMPL-X structure
            joints_3d = world_joints[i]
            joints_3d = np.array(joints_3d)
            
            # DEBUG: Print actual joint count for debugging SMPL-X truncation
            if i == 0 and pid == list(people.keys())[0]:  # Only print once per batch
                print(f"DEBUG: SMPL-X joints shape: {joints_3d.shape}")
                print(f"DEBUG: Total 3D joints found: {joints_3d.shape[0] if len(joints_3d.shape) > 1 else len(joints_3d) // 3}")
                
                # Check the smplx_world structure for truncation source
                smplx_world = person_data['smplx_world']
                if 'joints3d' in smplx_world:
                    joints3d_data = smplx_world['joints3d']
                    print(f"DEBUG: Source joints3d shape: {joints3d_data.shape if hasattr(joints3d_data, 'shape') else 'no shape'}")
                    print(f"DEBUG: Source joints3d format: {smplx_world.get('joints3d_format', 'unknown')}")
                
                # Additional debug: check the source data
                print(f"DEBUG: Raw world_joints type: {type(world_joints)}")
                print(f"DEBUG: Raw world_joints length: {len(world_joints) if isinstance(world_joints, (list, tuple)) else 'not list/tuple'}")
                if hasattr(world_joints, 'shape'):
                    print(f"DEBUG: Raw world_joints shape: {world_joints.shape}")
                
                # Check if truncation is happening in conversion
                raw_joints = world_joints[i]
                print(f"DEBUG: Raw joints[{i}] type: {type(raw_joints)}")
                if hasattr(raw_joints, 'shape'):
                    print(f"DEBUG: Raw joints[{i}] shape: {raw_joints.shape}")
                elif isinstance(raw_joints, (list, tuple)):
                    print(f"DEBUG: Raw joints[{i}] length: {len(raw_joints)}")
                
                # CRITICAL: Check the entire smplx_world structure
                smplx_world = person_data['smplx_world']
                print(f"DEBUG: smplx_world keys: {smplx_world.keys()}")
                if 'joints3d' in smplx_world:
                    joints3d_data = smplx_world['joints3d']
                    print(f"DEBUG: smplx_world['joints3d'] type: {type(joints3d_data)}")
                    if hasattr(joints3d_data, 'shape'):
                        print(f"DEBUG: smplx_world['joints3d'] shape: {joints3d_data.shape}")
                    elif isinstance(joints3d_data, (list, tuple)):
                        print(f"DEBUG: smplx_world['joints3d'] length: {len(joints3d_data)}")
                        if len(joints3d_data) > 0:
                            first_frame_joints = joints3d_data[0]
                            print(f"DEBUG: First frame joints type: {type(first_frame_joints)}")
                            if hasattr(first_frame_joints, 'shape'):
                                print(f"DEBUG: First frame joints shape: {first_frame_joints.shape}")
            
            # Count actual joints for metadata
            num_3d_joints = joints_3d.shape[0] if len(joints_3d.shape) > 1 else len(joints_3d) // 3
            # Flatten to 1D list preserving ALL joints - no truncation whatsoever
            joints_3d_flat = joints_3d.flatten().tolist()

            # Get other frame data
            bbox_data = bboxes[i] if isinstance(bboxes[i], list) else bboxes[i].tolist()
            pose_data = poses[i] if isinstance(poses[i], list) else poses[i]
            shape_data = shapes[i] if isinstance(shapes[i], list) else shapes[i]
            trans_data = trans[i] if isinstance(trans[i], list) else trans[i]

            # Get 2D keypoints - ONLY use VitPose25 keypoints (no COCO17 fallback)
            joints_2d = None
            joints_2d_format = 'unknown'
            
            # Use VitPose keypoints - preserve ALL 25 keypoints exactly as they are
            if keypoints_2d_json is not None and i < len(keypoints_2d_json):
                kp2d_frame = keypoints_2d_json[i]
                joints_2d_format = keypoints_2d_format  # This should always be 'vitpose25'
                if isinstance(kp2d_frame, (np.ndarray, list)):
                    # Convert to flat list preserving ALL data
                    kp2d_frame = np.array(kp2d_frame)
                    joints_2d = kp2d_frame.flatten().tolist()
            
            # Only use bbox center as absolute fallback (no COCO17 keypoints)
            if joints_2d is None:
                bbox_center_x = (bbox_data[0] + bbox_data[2]) / 2
                bbox_center_y = (bbox_data[1] + bbox_data[3]) / 2
                joints_2d = [bbox_center_x, bbox_center_y, 0.1]
                joints_2d_format = 'bbox_center'

            # Get detection bounding box
            if detection_bbox is not None:
                det_bbox = detection_bbox if isinstance(detection_bbox, list) else detection_bbox.tolist()
            else:
                det_bbox = bbox_data

            frame_data = {
                'joints3d': joints_3d_flat,  # ALL SMPL-X joints preserved - no truncation
                'joints2d': joints_2d,       # ALL VitPose25 keypoints preserved
                'joints2d_format': joints_2d_format,  # Should always be 'vitpose25' or 'bbox_center'
                'box': bbox_data,
                'detection_box': det_bbox,
                'betas': shape_data,  # Keep all shape parameters
                'translation': trans_data,
                'person_id': f"{batch_num}_{pid}"
            }
            frame_poses[adjusted_frame_num].append(frame_data)

    # Create comprehensive joint mapping metadata - declared ONCE at the top level
    poses_dict = {
        'metadata': {
            'batch_number': batch_num,
            'frame_offset': frame_offset,
            'batch_size': 60,
            'coordinate_system': 'floor_origin_xyz',
            'joints3d_format': 'smplx_full',
            'joints3d_structure': 'flattened [x,y,z] coordinates for ALL SMPL-X joints (body+hands+face)',
            'joints3d_total_joints': 'variable (depends on SMPL-X configuration)',
            'joints3d_joint_order': [
                # SMPL-X Body joints (indices 0-21: 22 joints)
                'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
                'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
                'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                # Face joints (indices 22-24: 3 joints) 
                'jaw', 'left_eye_smplhf', 'right_eye_smplhf',
                # Left hand joints (indices 25-39: 15 joints)
                'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
                'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
                'left_thumb1', 'left_thumb2', 'left_thumb3',
                # Right hand joints (indices 40-54: 15 joints)
                'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3',
                'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
                'right_thumb1', 'right_thumb2', 'right_thumb3',
                # Additional SMPL-X joints (if present - varies by model configuration)
                # Face landmarks and additional hand joints may extend beyond index 54
            ],
            'joints2d_formats': {
                'vitpose25': {
                    'joint_count': 25,
                    'data_structure': 'flattened [x0,y0,conf0, x1,y1,conf1, ...] for 25 keypoints',
                    'joint_order': [
                        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                        'left_big_toe', 'left_small_toe', 'left_heel',
                        'right_big_toe', 'right_small_toe', 'right_heel',
                        'neck', 'mid_hip'
                    ],
                    'coordinate_system': 'image_pixels_xy'
                },
                'coco17': {
                    'joint_count': 17,
                    'data_structure': 'flattened [x0,y0,conf0, x1,y1,conf1, ...] for 17 keypoints',
                    'joint_order': [
                        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                    ],
                    'coordinate_system': 'image_pixels_xy'
                },
                'bbox_center': {
                    'joint_count': 1,
                    'data_structure': '[x,y,conf]',
                    'joint_order': ['bbox_center'],
                    'coordinate_system': 'image_pixels_xy'
                }
            },
            'data_preservation_note': 'ALL keypoints preserved without truncation or format conversion. 3D joints include complete SMPL-X structure. 2D format varies per person/frame - check joints2d_format field.'
        },
        'frames': {str(k): v for k, v in sorted(frame_poses.items())}
    }

    with open(output_path, 'w') as f:
        json.dump(poses_dict, f, indent=2)


def __save_batch_camera_json(camera_data: Dict, output_path: str, batch_num: int, frame_offset: int):
    """Save camera trajectory for a single batch in JSON format."""
    camera_traj = {
        'metadata': {
            'batch_number': batch_num,
            'frame_offset': frame_offset,
            'coordinate_system': 'floor_origin'
        },
        'frames': {}
    }

    for frame_idx in range(len(camera_data.get('Rwc', []))):
        # Adjust frame number with offset
        adjusted_frame_idx = frame_idx + frame_offset
        
        rotation_matrix = camera_data['Rwc'][frame_idx]
        translation = camera_data['Twc'][frame_idx]

        camera_traj['frames'][str(adjusted_frame_idx)] = {
            'rotation_matrix': __convert_tensor_to_list(rotation_matrix),
            'translation': __convert_tensor_to_list(translation),
            'focal_length': float(camera_data['img_focal']),
            'principal_point': __convert_tensor_to_list(camera_data['img_center'])
        }

    with open(output_path, 'w') as f:
        json.dump(camera_traj, f, indent=2)


def main(input_video: str, output_dir: str, static_camera: bool = False):
    """Main function to process video in batches."""
    
    # MONKEY PATCH: Override any function that might truncate joints
    import numpy as np
    original_array_getitem = np.ndarray.__getitem__
    
    def debug_getitem(self, key):
        result = original_array_getitem(self, key)
        # Check if this looks like joint truncation
        if (hasattr(self, 'shape') and len(self.shape) >= 2 and 
            self.shape[-2] > 24 and isinstance(key, slice) and 
            key.start is None and key.stop == 24):
            print(f"WARNING: Potential joint truncation detected!")
            print(f"Array shape: {self.shape}, slice: {key}")
            print(f"Result shape: {result.shape}")
            import traceback
            traceback.print_stack()
        return result
    
    # np.ndarray.__getitem__ = debug_getitem
    
    os.makedirs(output_dir, exist_ok=True)

    # Get video info
    total_frames, fps = __get_video_info(input_video)
    duration_seconds = total_frames / fps
    batch_duration = 60.0

    print(f"Processing video: {duration_seconds:.1f}s ({total_frames} frames) at {fps:.1f} FPS")
    total_batches = int(np.ceil(duration_seconds / batch_duration))
    print(f"Will process in {total_batches} batches of {batch_duration}s each")

    # Process in batches - NO SHARED PIPELINE
    current_time = 0
    batch_num = 0
    successful_batches = 0
    failed_batches = []

    pbar = tqdm(total=total_batches, desc="Processing batches")

    try:
        while current_time < duration_seconds:
            batch_end_time = min(current_time + batch_duration, duration_seconds)
            actual_duration = batch_end_time - current_time

            if actual_duration <= 0.1:  # Skip very short segments
                break

            print(f"\n{'='*60}")
            print(f"Processing batch {batch_num + 1}/{total_batches}: {current_time:.1f}s - {batch_end_time:.1f}s")
            print(f"{'='*60}")

            # COMPLETE CLEANUP BEFORE EACH BATCH
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.set_device(0)
            
            import gc
            gc.collect()

            try:
                # Create temporary video segment
                temp_video = os.path.join(output_dir, f"temp_batch_{batch_num}.mp4")
                __extract_video_segment(input_video, temp_video, current_time, actual_duration)

                # Process batch
                batch_output_folder = os.path.join(output_dir, f"batch_{batch_num}")
                os.makedirs(batch_output_folder, exist_ok=True)
                
                print(f"Creating fresh pipeline for batch {batch_num}...")
                # CREATE COMPLETELY FRESH PIPELINE FOR EACH BATCH
                pipeline = Pipeline(static_cam=static_camera)
                
                print(f"Processing video segment...")
                results = pipeline.__call__(temp_video, batch_output_folder, save_only_essential=True)
                
                # Calculate frame offset
                frame_offset = int(current_time * fps)
                
                # Save individual JSON files for this batch
                if 'people' in results and results['people']:
                    print(f"Found {len(results['people'])} people in batch {batch_num}")
                    
                    # Convert results to proper format
                    converted_people = __convert_tensor_to_list(results['people'])
                    
                    # Save poses JSON for this batch
                    poses_json_path = os.path.join(batch_output_folder, f"poses_batch_{batch_num}.json")
                    __save_batch_poses_json(converted_people, poses_json_path, batch_num, frame_offset)
                    print(f"✓ Poses saved to: {poses_json_path}")
                else:
                    print(f"⚠ No people found in batch {batch_num}")

                # Save camera data for this batch
                if 'camera_world' in results and results['camera_world']:
                    converted_camera = __convert_tensor_to_list(results['camera_world'])
                    
                    camera_json_path = os.path.join(batch_output_folder, f"camera_batch_{batch_num}.json")
                    __save_batch_camera_json(converted_camera, camera_json_path, batch_num, frame_offset)
                    print(f"✓ Camera data saved to: {camera_json_path}")
                else:
                    print(f"⚠ No camera data found in batch {batch_num}")

                # COMPLETELY DESTROY PIPELINE AFTER EACH BATCH
                print(f"Destroying pipeline for batch {batch_num}...")
                del pipeline
                del results
                
                # Remove problematic pickle file if it exists
                results_pkl_path = os.path.join(batch_output_folder, "results.pkl")
                if os.path.exists(results_pkl_path):
                    try:
                        os.remove(results_pkl_path)
                        print(f"Removed problematic pickle file: {results_pkl_path}")
                    except:
                        pass

                # Clean up temporary files
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                    print(f"Temp video cleaned up: {temp_video}")

                # FORCE COMPLETE CLEANUP AFTER EACH BATCH
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                gc.collect()

                print(f"✓ Batch {batch_num} processed successfully")
                successful_batches += 1

            except KeyboardInterrupt:
                print(f"\nInterrupted during batch {batch_num}.")
                raise
            except Exception as e:
                print(f"✗ Error processing batch {batch_num}: {str(e)}")
                failed_batches.append(batch_num)
                import traceback
                traceback.print_exc()
                print("Continuing with next batch...")
                
                # CLEANUP EVEN ON ERROR
                try:
                    if 'pipeline' in locals():
                        del pipeline
                    if 'results' in locals():
                        del results
                except:
                    pass
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                gc.collect()

            current_time = batch_end_time
            batch_num += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        pbar.close()
        
        # FINAL CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()

        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed: {successful_batches}/{batch_num} batches")
        
        if failed_batches:
            print(f"Failed batches: {failed_batches}")
        else:
            print("All processed batches completed successfully!")

        # Now aggregate all the individual JSON files
        if successful_batches > 0:
            print(f"\nAggregating {successful_batches} batch JSON files...")
            try:
                aggregate_json_files(input_video, output_dir, total_frames, successful_batches)
            except Exception as agg_error:
                print(f"Error during aggregation: {agg_error}")
                import traceback
                traceback.print_exc()

        print("Processing complete!")


def aggregate_json_files(input_video: str, output_dir: str, total_frames: int, num_batches: int):
    """Aggregate all batch JSON files into final outputs."""
    print("Aggregating individual batch JSON files...")
    
    # Find all batch folders with JSON files
    all_frames = {}
    all_camera_frames = {}
    sample_metadata = None  # To extract metadata from first batch
    
    for batch_num in range(num_batches):
        batch_folder = os.path.join(output_dir, f"batch_{batch_num}")
        poses_json_path = os.path.join(batch_folder, f"poses_batch_{batch_num}.json")
        camera_json_path = os.path.join(batch_folder, f"camera_batch_{batch_num}.json")
        
        # Load poses
        if os.path.exists(poses_json_path):
            try:
                with open(poses_json_path, 'r') as f:
                    poses_data = json.load(f)
                    all_frames.update(poses_data['frames'])
                    # Capture metadata from first successful batch
                    if sample_metadata is None:
                        sample_metadata = poses_data.get('metadata', {})
                    print(f"✓ Loaded poses from batch {batch_num}: {len(poses_data['frames'])} frames")
            except Exception as e:
                print(f"✗ Error loading poses from batch {batch_num}: {e}")
        
        # Load camera
        if os.path.exists(camera_json_path):
            try:
                with open(camera_json_path, 'r') as f:
                    camera_data = json.load(f)
                    all_camera_frames.update(camera_data['frames'])
                    print(f"✓ Loaded camera from batch {batch_num}: {len(camera_data['frames'])} frames")
            except Exception as e:
                print(f"✗ Error loading camera from batch {batch_num}: {e}")

    # Create comprehensive final metadata with complete joint format information
    final_poses = {
        'metadata': {
            'video_path': input_video,
            'total_frames': total_frames,
            'batch_size': 60,
            'coordinate_system': 'floor_origin_xyz',
            'joints3d_format': 'smplx_full',
            'joints3d_structure': 'flattened [x,y,z] coordinates for ALL SMPL-X joints (body+hands+face)',
            'joints3d_total_joints': 'variable (depends on SMPL-X configuration)',
            'joints3d_joint_order': [
                # SMPL-X Body joints (indices 0-21: 22 joints)
                'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
                'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
                'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                # Face joints (indices 22-24: 3 joints) 
                'jaw', 'left_eye_smplhf', 'right_eye_smplhf',
                # Left hand joints (indices 25-39: 15 joints)
                'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
                'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
                'left_thumb1', 'left_thumb2', 'left_thumb3',
                # Right hand joints (indices 40-54: 15 joints)
                'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3',
                'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
                'right_thumb1', 'right_thumb2', 'right_thumb3',
                # Additional SMPL-X joints (if present - varies by model configuration)
                # Face landmarks and additional hand joints may extend beyond index 54
            ],
            'joints2d_formats': {
                'vitpose25': {
                    'joint_count': 25,
                    'data_structure': 'flattened [x0,y0,conf0, x1,y1,conf1, ...] for 25 keypoints',
                    'joint_order': [
                        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                        'left_big_toe', 'left_small_toe', 'left_heel',
                        'right_big_toe', 'right_small_toe', 'right_heel',
                        'neck', 'mid_hip'
                    ],
                    'coordinate_system': 'image_pixels_xy'
                },
                'coco17': {
                    'joint_count': 17,
                    'data_structure': 'flattened [x0,y0,conf0, x1,y1,conf1, ...] for 17 keypoints',
                    'joint_order': [
                        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                    ],
                    'coordinate_system': 'image_pixels_xy'
                },
                'bbox_center': {
                    'joint_count': 1,
                    'data_structure': '[x,y,conf]',
                    'joint_order': ['bbox_center'],
                    'coordinate_system': 'image_pixels_xy'
                }
            },
            'data_preservation_note': 'ALL keypoints preserved without truncation or format conversion. 3D joints include complete SMPL-X structure. 2D format varies per person/frame - check joints2d_format field.',
            'aggregation_info': {
                'total_batches_processed': num_batches,
                'batch_duration_seconds': 60
            }
        },
        'frames': all_frames
    }
    
    final_camera = {
        'metadata': {
            'video_path': input_video,
            'total_frames': total_frames,
            'coordinate_system': 'floor_origin'
        },
        'frames': all_camera_frames
    }
    
    # Write final files
    poses_output = os.path.join(output_dir, 'poses3d.json')
    camera_output = os.path.join(output_dir, 'camera_traj.json')
    
    if all_frames:
        with open(poses_output, 'w') as f:
            json.dump(final_poses, f, indent=2)
        print(f"✓ Final poses saved to: {poses_output}")
        print(f"  Total frames with poses: {len(all_frames)}")
    else:
        print("✗ No pose data to aggregate")
    
    if all_camera_frames:
        with open(camera_output, 'w') as f:
            json.dump(final_camera, f, indent=2)
        print(f"✓ Final camera trajectory saved to: {camera_output}")
        print(f"  Total camera frames: {len(all_camera_frames)}")
    else:
        print("✗ No camera data to aggregate")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--static_camera', action='store_true', help='Use static camera assumption')
    args = parser.parse_args()

    main(args.input_video, args.output_dir, static_camera=args.static_camera)
