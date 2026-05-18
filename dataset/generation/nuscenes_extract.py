"""
nuScenes 3-second clip extraction module.

Extracts fixed-length clips anchored at nuScenes keyframe samples,
looking backward 3.0 seconds to capture vehicle dynamics and camera frames.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)


@dataclass
class ClipFrameData:
    """Container for frame references within a clip."""
    tokens: List[str]
    paths: List[str]
    timestamps: List[float]


@dataclass
class EgoPoseData:
    """Container for ego pose at a single timestamp."""
    timestamp: float
    x: float
    y: float
    yaw: float  # radians
    rotation_quat: Optional[List[float]] = None  # [w, x, y, z]


@dataclass
class ClipData:
    """Container for all data associated with a single clip."""
    clip_id: str
    scene_token: str
    sample_token: str
    t_start: float
    t_end: float
    camera: str
    frames: ClipFrameData
    ego_poses: List[EgoPoseData]


class NuScenesClipExtractor:
    """Extracts 3-second clips from nuScenes dataset."""

    def __init__(
        self,
        nuscenes_root: str,
        version: str = "v1.0-trainval",
        clip_seconds: float = 3.0,
        min_frames: int = 20,
        camera: str = "CAM_FRONT",
    ):
        """
        Initialize the clip extractor.

        Args:
            nuscenes_root: Path to nuScenes dataset root
            version: nuScenes version (e.g., 'v1.0-trainval', 'v1.0-mini')
            clip_seconds: Length of each clip in seconds
            min_frames: Minimum number of frames required per clip
            camera: Camera sensor to use (default: CAM_FRONT)
        """
        self.nuscenes_root = Path(nuscenes_root)
        self.version = version
        self.clip_seconds = clip_seconds
        self.min_frames = min_frames
        self.camera = camera

        logger.info(f"Loading nuScenes {version} from {nuscenes_root}")
        self.nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=True)

    def extract_clips(
        self,
        max_clips: Optional[int] = None,
        scene_names: Optional[List[str]] = None,
    ) -> List[ClipData]:
        """
        Extract clips from nuScenes dataset.

        Args:
            max_clips: Maximum number of clips to extract (None = all)
            scene_names: List of scene names to process (None = all)

        Returns:
            List of ClipData objects
        """
        clips = []
        clip_count = 0

        # Filter scenes if requested
        scenes = self.nusc.scene
        if scene_names is not None:
            scenes = [s for s in scenes if s['name'] in scene_names]

        logger.info(f"Processing {len(scenes)} scenes")

        for scene_idx, scene in enumerate(scenes):
            logger.info(f"Scene {scene_idx + 1}/{len(scenes)}: {scene['name']}")

            # Get all samples in this scene
            sample_token = scene['first_sample_token']

            while sample_token:
                sample = self.nusc.get('sample', sample_token)

                # Try to extract clip ending at this sample
                clip_data = self._extract_clip_at_sample(sample, clip_count)

                if clip_data is not None:
                    clips.append(clip_data)
                    clip_count += 1

                    if max_clips is not None and clip_count >= max_clips:
                        logger.info(f"Reached max_clips limit: {max_clips}")
                        return clips

                # Move to next sample
                sample_token = sample['next']

        logger.info(f"Extracted {len(clips)} clips total")
        return clips

    def _extract_clip_at_sample(
        self,
        sample: dict,
        clip_id_num: int,
    ) -> Optional[ClipData]:
        """
        Extract a single clip ending at the given sample.

        Args:
            sample: nuScenes sample dict
            clip_id_num: Numeric clip ID for naming

        Returns:
            ClipData or None if clip doesn't meet requirements
        """
        # Define time window: [t_sample - clip_seconds, t_sample]
        t_end = sample['timestamp'] / 1e6  # Convert microseconds to seconds
        t_start = t_end - self.clip_seconds

        # Get camera sample_data token
        cam_token = sample['data'][self.camera]

        # Collect all camera frames in the time window
        frames = self._collect_camera_frames(cam_token, t_start, t_end)

        # Check minimum frame requirement
        if len(frames.tokens) < self.min_frames:
            logger.debug(
                f"Skipping sample {sample['token'][:8]}: "
                f"only {len(frames.tokens)} frames (min: {self.min_frames})"
            )
            return None

        # Check actual temporal coverage
        # The actual duration should be close to clip_seconds
        actual_duration = frames.timestamps[-1] - frames.timestamps[0]
        min_duration = self.clip_seconds * 0.9  # Allow 10% tolerance
        if actual_duration < min_duration:
            logger.debug(
                f"Skipping sample {sample['token'][:8]}: "
                f"actual duration {actual_duration:.2f}s < minimum {min_duration:.2f}s"
            )
            return None

        # Collect ego poses aligned with camera frames
        # Pass frame tokens directly for efficient lookup
        ego_poses = self._collect_ego_poses_from_tokens(frames.tokens)

        # Create clip data
        clip_id = f"clip_{clip_id_num:05d}"
        clip_data = ClipData(
            clip_id=clip_id,
            scene_token=sample['scene_token'],
            sample_token=sample['token'],
            t_start=t_start,
            t_end=t_end,
            camera=self.camera,
            frames=frames,
            ego_poses=ego_poses,
        )

        logger.debug(
            f"Extracted {clip_id}: {len(frames.tokens)} frames, "
            f"{len(ego_poses)} poses, duration={t_end - t_start:.3f}s"
        )

        return clip_data

    def _collect_camera_frames(
        self,
        start_token: str,
        t_start: float,
        t_end: float,
    ) -> ClipFrameData:
        """
        Collect all camera sample_data in the time window.

        Args:
            start_token: sample_data token to start searching from
            t_start: Start timestamp (seconds)
            t_end: End timestamp (seconds)

        Returns:
            ClipFrameData with tokens, paths, and timestamps
        """
        tokens = []
        paths = []
        timestamps = []

        # Walk backward from start_token to find all frames in window
        current_token = start_token

        # First, collect frames at/before t_end
        while current_token:
            sd = self.nusc.get('sample_data', current_token)
            timestamp = sd['timestamp'] / 1e6  # Convert to seconds

            if timestamp < t_start:
                break

            if t_start <= timestamp <= t_end:
                tokens.append(sd['token'])
                paths.append(sd['filename'])
                timestamps.append(timestamp)

            # Move to previous sample_data
            current_token = sd['prev']

        # Reverse to chronological order
        tokens.reverse()
        paths.reverse()
        timestamps.reverse()

        return ClipFrameData(tokens=tokens, paths=paths, timestamps=timestamps)

    def _collect_ego_poses_from_tokens(self, frame_tokens: List[str]) -> List[EgoPoseData]:
        """
        Collect ego poses for given frame tokens.

        Args:
            frame_tokens: List of sample_data tokens

        Returns:
            List of EgoPoseData aligned with input tokens
        """
        ego_poses = []

        for token in frame_tokens:
            # Get sample_data
            sd = self.nusc.get('sample_data', token)

            # Get associated ego_pose
            ego_pose = self.nusc.get('ego_pose', sd['ego_pose_token'])

            # Extract position
            x, y, z = ego_pose['translation']

            # Convert quaternion to yaw
            # nuScenes quaternion is [w, x, y, z]
            quat = ego_pose['rotation']
            rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
            yaw = rotation.as_euler('xyz', degrees=False)[2]  # Z-axis rotation is yaw

            ego_data = EgoPoseData(
                timestamp=sd['timestamp'] / 1e6,
                x=x,
                y=y,
                yaw=yaw,
                rotation_quat=quat,
            )
            ego_poses.append(ego_data)

        return ego_poses


def quaternion_to_yaw(quat: List[float]) -> float:
    """
    Convert quaternion to yaw angle.

    Args:
        quat: Quaternion in [w, x, y, z] format (nuScenes convention)

    Returns:
        Yaw angle in radians
    """
    # Convert to scipy format [x, y, z, w]
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    # Extract yaw (rotation around Z axis)
    euler = rotation.as_euler('xyz', degrees=False)
    return euler[2]
