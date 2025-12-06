import json
import os
import time

import imageio
import imageio_ffmpeg as imageio_ffmpeg  # Ensure ffmpeg is installed
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm, trange
from waymo_open_dataset.utils import frame_utils
parse_range_image_and_camera_projection = frame_utils.parse_range_image_and_camera_projection

from .visualization import depth_visualizer, scene_flow_to_rgb

from .utils import track_parallel_progress

ORIGINAL_SIZE = {
    "0": (1280, 1920),
    "1": (1280, 1920),
    "2": (1280, 1920),
    "3": (886, 1920),
    "4": (886, 1920),
}
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import box_utils, range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

##### Very important to set memory growth for GPU to avoid OOM errors
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def get_ground_np(pts):
    """
    This function performs ground removal on a point cloud.
    Modified from https://github.com/tusen-ai/LiDAR_SOT/blob/main/waymo_data/data_preprocessing/ground_removal.py

    Args:
        pts (numpy.ndarray): The input point cloud.

    Returns:
        numpy.ndarray: A boolean array indicating whether each point is ground or not.
    """
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    lpr = np.mean(pts_sort[:num_lpr_, 2])
    pts_g = pts_sort[pts_sort[:, 2] < lpr + th_seeds_, :]
    normal_ = np.zeros(3)
    for i in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=np.float32,
        )
        U, S, V = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3] @ normal_[..., np.newaxis]
        pts_g = pts[result.squeeze(-1) < th_dist_d_]
    ground_label = result < th_dist_d_
    return ground_label


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_origins = tf.zeros_like(range_image_points)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        range_image_origins = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )
            range_image_origins = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )
            range_image_origins = (
                tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins)
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_origins = tf.cast(range_image_origins, dtype=range_image_polar_dtype)
        return range_image_points, range_image_origins


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        (
            range_image_points_cartesian,
            range_image_origins_cartesian,
        ) = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_origins_cartesian, range_image_points_cartesian


def parse_range_image_flow_and_camera_projection(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if (
            len(laser.ri_return1.range_image_flow_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_flow_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, "ZLIB"
                )
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy())
                )

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if (
            len(laser.ri_return2.range_image_flow_compressed) > 0
        ):  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_flow_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, "ZLIB"
            )
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_point_cloud_flow(
    frame,
    range_images,
    range_images_flow,
    camera_projections,
    range_image_top_pose,
    ri_index=0,
):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Convert range images flow to scene flow.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_imaages_flow: A dict similar to range_images.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        points_flow: {[N, 3]} list of scene flow vector of each point.
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    origins, points, cp_points = [], [], []
    points_intensity = []
    points_elongation = []
    points_flow = []
    laser_ids = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_flow = range_images_flow[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_flow_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_flow.data), range_image_flow.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]

        flow_x = range_image_flow_tensor[..., 0]
        flow_y = range_image_flow_tensor[..., 1]
        flow_z = range_image_flow_tensor[..., 2]
        flow_class = range_image_flow_tensor[..., 3]

        mask_index = tf.where(range_image_mask)

        (origins_cartesian, points_cartesian,) = extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
        origins_cartesian = tf.squeeze(origins_cartesian, axis=0)
        points_cartesian = tf.squeeze(points_cartesian, axis=0)

        origins_tensor = tf.gather_nd(origins_cartesian, mask_index)
        points_tensor = tf.gather_nd(points_cartesian, mask_index)

        points_intensity_tensor = tf.gather_nd(range_image_intensity, mask_index)
        points_elongation_tensor = tf.gather_nd(range_image_elongation, mask_index)

        points_flow_x_tensor = tf.expand_dims(tf.gather_nd(flow_x, mask_index), axis=1)
        points_flow_y_tensor = tf.expand_dims(tf.gather_nd(flow_y, mask_index), axis=1)
        points_flow_z_tensor = tf.expand_dims(tf.gather_nd(flow_z, mask_index), axis=1)
        points_flow_class_tensor = tf.expand_dims(tf.gather_nd(flow_class, mask_index), axis=1)

        origins.append(origins_tensor.numpy())
        points.append(points_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())
        laser_ids.append(np.full_like(points_intensity_tensor.numpy(), c.name - 1))

        points_flow.append(
            tf.concat(
                [
                    points_flow_x_tensor,
                    points_flow_y_tensor,
                    points_flow_z_tensor,
                    points_flow_class_tensor,
                ],
                axis=-1,
            ).numpy()
        )

    return (
        origins,
        points,
        points_flow,
        cp_points,
        points_intensity,
        points_elongation,
        laser_ids,
    )


class WaymoProcessor:
    def __init__(
        self,
        load_dir,
        save_dir,
        scene_lists,
        prefix,
        downsample_factors=[4],
        process_keys=["images", "lidar", "calib", "pose", "ground", "dynamic_masks"],
        json_folder_to_save=None,
        num_workers=64,
        overwrite=False,
    ):
        self.overwrite = overwrite
        self.process_keys = process_keys
        self.json_folder_to_save = f"{json_folder_to_save}/{prefix}"
        os.makedirs(self.json_folder_to_save, exist_ok=True)
        print("will process keys: ", self.process_keys)

        self.load_dir = f"{load_dir}/{prefix}"
        self.save_dir = f"{save_dir}/{prefix}"
        self.prefix = prefix
        self.downsample_factors = downsample_factors
        self.num_workers = int(num_workers)
        self.scene_ids = [s[0] for s in scene_lists]
        self.scene_names = [s[1] for s in scene_lists]
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        track_parallel_progress(self.convert_one, range(len(self.scene_ids)), self.num_workers)
        print("\nFinished ...")

    def convert_one(self, file_id):
        """Convert action for single file.

        Args:
            file_id (int): Index of the file to be converted.
        """
        scene_id = self.scene_ids[file_id]
        scene_name = self.scene_names[file_id]
        tfrecord_path = f"{self.load_dir}/{scene_name}.tfrecord"
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
        num_frames = sum(1 for _ in dataset)

        for frame_id, data in enumerate(
            tqdm(dataset, desc=f"File {file_id}", total=num_frames, dynamic_ncols=True)
        ):
            frame = dataset_pb2.Frame()
            # frame.ParseFromString(bytearray(data.numpy()))
            frame.ParseFromString(data.numpy())
            if "images" in self.process_keys:
                self.save_image(frame, file_id, frame_id)
            if "calib" in self.process_keys:
                self.save_calib(frame, file_id, frame_id)
            if "pose" in self.process_keys:
                self.save_pose(frame, file_id, frame_id)
            if "lidar" in self.process_keys and "depth" not in self.process_keys:
                self.save_lidar(frame, file_id, frame_id)
            if "depth" in self.process_keys and "lidar" not in self.process_keys:
                self.save_depth(frame, file_id, frame_id)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(frame, file_id, frame_id)
            if "ground" in self.process_keys:
                self.save_ground(frame, file_id, frame_id)
        self.make_json(file_id)

    def make_json(self, file_id):
        scene_id = self.scene_ids[file_id]
        scene_name = self.scene_names[file_id]
        file_folder = os.path.join(self.save_dir, f"{scene_id:03d}")
        if os.path.exists(f"{self.json_folder_to_save}/{scene_name}.json") and not self.overwrite:
            print(
                f"Scene {scene_id} json already exists at {self.json_folder_to_save}/{scene_name}.json"
            )
            return
        else:
            print("start overwriting")
        num_timesteps = len(os.listdir(f"{file_folder}/ego_pose"))
        camera_list = ["0", "1", "2", "3", "4"]
        scene_dict = {
            "dataset": "waymo",
            "scene_id": int(scene_id),
            "scene_name": scene_name,
            "num_timesteps": num_timesteps,
            # 0: front, 1: left, 2: right, 3: front_left, 4: front_right
            "camera_list": camera_list,
            # list for synchronized timestamps, dict for unsynchronized, measured in seconds
            "normalized_time": [],
            # camera_name: [fx, fy, cx, cy] (real_fx = fx * width, real_fy = fy * height)
            "normalized_intrinsics": {cam: [] for cam in camera_list},
            # camera_name: 3x4 matrix
            "camera_to_ego": {cam: [] for cam in camera_list},
            # list of 4x4 matrices
            "ego_pose": [],
            "camera_to_world": {cam: [] for cam in camera_list},
            # camera_name: [height, width]
            "original_image_size": {
                "0": [1280, 1920],
                "1": [1280, 1920],
                "2": [1280, 1920],
                "3": [886, 1920],
                "4": [886, 1920],
            },
            # camera_name: relative path to the image
            "relative_image_path": {},
            "fps": 10,  # assume all cameras have the same fps
        }

        # Pre-load extrinsics for all cameras (used for computing cam_to_world)
        extrinsics_all = {}
        for cam_id in range(5):
            extrinsics_all[cam_id] = np.loadtxt(f"{file_folder}/extrinsics/{cam_id}.txt")

        for t in range(num_timesteps):
            ego_pose = np.loadtxt(f"{file_folder}/ego_pose/{t:03d}.txt")
            scene_dict["ego_pose"].append(ego_pose.tolist())
            scene_dict["normalized_time"].append(t / scene_dict["fps"])
            for cam_id in range(5):
                # Compute cam_to_world = ego_pose @ extrinsics (instead of reading from file)
                cam_to_world = ego_pose @ extrinsics_all[cam_id]
                scene_dict["camera_to_world"][str(cam_id)].append(cam_to_world.tolist())

        for cam_id in range(5):
            fx, fy, cx, cy = np.loadtxt(f"{file_folder}/intrinsics/{cam_id}.txt")[:4]
            extrinsics = extrinsics_all[cam_id]
            height = scene_dict["original_image_size"][str(cam_id)][0]
            width = scene_dict["original_image_size"][str(cam_id)][1]
            normalized_fx = fx / width
            normalized_fy = fy / height
            normalized_cx = cx / width
            normalized_cy = cy / height
            scene_dict["normalized_intrinsics"][str(cam_id)] = [
                normalized_fx,
                normalized_fy,
                normalized_cx,
                normalized_cy,
            ]
            scene_dict["camera_to_ego"][str(cam_id)] = extrinsics.tolist()

        for cam_id in range(5):
            scene_dict["relative_image_path"][str(cam_id)] = []
            for t in range(num_timesteps):
                scene_dict["relative_image_path"][str(cam_id)].append(
                    f"{self.prefix}/{scene_id:03d}/images/{t:03d}_{cam_id}.jpg"
                )
        with open(f"{self.json_folder_to_save}/{scene_name}.json", "w") as f:
            json.dump(scene_dict, f)

    def __len__(self):
        """Length of the filename list."""
        return len(self.scene_ids)

    def save_image(self, frame, file_id, frame_id):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (int): Current file index.
            frame_id (int): Current frame index.
        """
        scene_id = self.scene_ids[file_id]
        for img in frame.images:
            img_path = f"{self.save_dir}/{str(scene_id).zfill(3)}/images/{str(frame_id).zfill(3)}_{img.name - 1}.jpg"
            if not os.path.exists(img_path) or self.overwrite:
                with open(img_path, "wb") as f:
                    f.write(img.image)
            for downsample_factor in self.downsample_factors:
                if downsample_factor == 1:
                    continue
                else:
                    postfix = f"_{downsample_factor}"
                    downsampled_img_path = (
                        f"{self.save_dir}/{str(scene_id).zfill(3)}/images{postfix}/"
                        + f"{str(frame_id).zfill(3)}_{img.name - 1}.jpg"
                    )
                    if not os.path.exists(downsampled_img_path) or self.overwrite:
                        image = Image.open(img_path)
                        new_size = (
                            image.width // downsample_factor,
                            image.height // downsample_factor,
                        )
                        downsampled_image = image.resize(new_size)
                        downsampled_image.save(downsampled_img_path, format="JPEG")

    def save_calib(self, frame, file_id, frame_id):
        """Parse and save the calibration data.

        Args:
            file_id (int): Current file index.
            frame_id (int): Current frame index.
        """
        scene_id = self.scene_ids[file_id]
        if frame_id != 0:
            # only save the calibration data for the first frame
            # because the calibration data is the same for all frames
            return
        extrinsics = []
        intrinsics = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            np.savetxt(
                f"{self.save_dir}/{str(scene_id).zfill(3)}/extrinsics/" + f"{str(i)}.txt",
                extrinsics[i],
            )
            np.savetxt(
                f"{self.save_dir}/{str(scene_id).zfill(3)}/intrinsics/" + f"{str(i)}.txt",
                intrinsics[i],
            )

    def save_lidar(self, frame, file_id, frame_id):
        """Parse and save the lidar data in psd format.

        Args:
            file_id (int): Current file index.
            frame_id (int): Current frame index.
        """
        scene_id = self.scene_ids[file_id]
        scene_path = f"{self.save_dir}/{str(scene_id).zfill(3)}"
        pc_path = f"{scene_path}/lidar/{str(frame_id).zfill(3)}.bin"
        if not os.path.exists(pc_path) or self.overwrite:
            (
                range_images,
                camera_projections,
                seg_labels,
                range_image_top_pose,
            ) = parse_range_image_and_camera_projection(frame)
            # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto
            if range_image_top_pose is None:
                # the camera only split doesn't contain lidar points.
                return
            # collect first return only
            range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)
            (
                origins,
                points,
                flows,
                cp_points,
                intensity,
                elongation,
                laser_ids,
            ) = convert_range_image_to_point_cloud_flow(
                frame,
                range_images,
                range_images_flow,
                camera_projections,
                range_image_top_pose,
                ri_index=0,
            )
            origins = np.concatenate(origins, axis=0)
            points = np.concatenate(points, axis=0)
            #  -1: no-flow-label, the point has no flow information.
            #   0:  unlabeled or "background,", i.e., the point is not contained in a
            #       bounding box.
            #   1: vehicle, i.e., the point corresponds to a vehicle label box.
            #   2: pedestrian, i.e., the point corresponds to a pedestrian label box.
            #   3: sign, i.e., the point corresponds to a sign label box.
            #   4: cyclist, i.e., the point corresponds to a cyclist label box.
            flows = np.concatenate(flows, axis=0)
            point_cloud = np.column_stack((origins, points, flows))
            point_cloud.astype(np.float32).tofile(pc_path)
        else:
            point_cloud = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 10)

        for cam_id in range(5):
            world_to_cam = None

            for downsample_factor in self.downsample_factors:
                if downsample_factor == 1:
                    continue
                scale_postfix = f"_{downsample_factor}"
                depth_path = f"{scene_path}/depth_flows{scale_postfix}/{str(frame_id).zfill(3)}_{str(cam_id)}.npy"
                if not os.path.exists(depth_path) or self.overwrite:
                    points = point_cloud[:, 3:6].astype(np.float32)
                    flows = point_cloud[:, 6:9].astype(np.float32)
                    flow_class = point_cloud[:, 9].astype(np.int32)
                    if world_to_cam is None:
                        intrinsics = np.loadtxt(
                            f"{scene_path}/intrinsics/{str(cam_id)}.txt",
                            dtype=np.float32,
                        )
                        fx, fy, cx, cy = intrinsics[:4]
                        intrinsics = np.array(
                            [
                                [fx, 0, cx, 0],
                                [0, fy, cy, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ],
                            dtype=np.float32,
                        )
                        extrinsic = np.loadtxt(
                            f"{scene_path}/extrinsics/{str(cam_id)}.txt",
                            dtype=np.float32,
                        )
                        world_to_cam = np.linalg.inv(extrinsic @ OPENCV2DATASET)
                    target_size = (
                        ORIGINAL_SIZE[str(cam_id)][0] // downsample_factor,
                        ORIGINAL_SIZE[str(cam_id)][1] // downsample_factor,
                    )
                    _intrinsics = intrinsics.copy()
                    _intrinsics[0, 0] /= ORIGINAL_SIZE[str(cam_id)][1] / target_size[1]
                    _intrinsics[1, 1] /= ORIGINAL_SIZE[str(cam_id)][0] / target_size[0]
                    _intrinsics[0, 2] /= ORIGINAL_SIZE[str(cam_id)][1] / target_size[1]
                    _intrinsics[1, 2] /= ORIGINAL_SIZE[str(cam_id)][0] / target_size[0]
                    lidar2img = _intrinsics @ world_to_cam
                    points_2d = (np.dot(lidar2img[:3, :3], points.T) + lidar2img[:3, 3:4]).T
                    depth_2d = points_2d[:, 2]
                    cam_coords = points_2d[:, :2] / (depth_2d[:, None] + 1e-6)
                    valid_mask = (
                        (cam_coords[:, 0] >= 0)
                        & (cam_coords[:, 0] < target_size[1])
                        & (cam_coords[:, 1] >= 0)
                        & (cam_coords[:, 1] < target_size[0])
                        & (depth_2d > 0)
                    )
                    # Get valid depth points and corresponding coordinates
                    valid_depth_points = depth_2d[valid_mask]
                    valid_cam_coords = cam_coords[valid_mask]
                    # Convert coordinates to integer indices
                    x_indices = valid_cam_coords[:, 0].astype(np.int32)
                    y_indices = valid_cam_coords[:, 1].astype(np.int32)
                    # Initialize arrays to accumulate depth sums and counts
                    depth_sums = np.zeros(target_size)
                    depth_counts = np.zeros(target_size)

                    np.add.at(depth_sums, (y_indices, x_indices), valid_depth_points)
                    np.add.at(depth_counts, (y_indices, x_indices), 1)

                    depth_image = np.divide(depth_sums, depth_counts, where=depth_counts > 0)

                    depth_image[depth_counts == 0] = 0

                    valid_flow = flows[valid_mask] * (flow_class[valid_mask][:, None] >= 0)
                    flow_sums = np.zeros((target_size[0], target_size[1], 3))
                    flow_counts = np.zeros((target_size[0], target_size[1], 3))
                    np.add.at(flow_sums, (y_indices, x_indices), valid_flow)
                    np.add.at(flow_counts, (y_indices, x_indices), 1)
                    flow_image = np.divide(flow_sums, flow_counts, where=flow_counts > 0)
                    flow_image[flow_counts == 0] = 0
                    flow_image[np.linalg.norm(flow_image, axis=-1) < 0.5] = 0

                    concate_image = np.concatenate(
                        [depth_image[:, :, None], flow_image], axis=-1
                    ).astype(np.float32)
                    path = f"{scene_path}/depth_flows{scale_postfix}/{str(frame_id).zfill(3)}_{str(cam_id)}.npy"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    np.save(path, concate_image)

    def save_ground(self, frame, file_id, frame_id):
        """Parse and save the lidar data in psd format.

        Args:
            file_id (int): Current file index.
            frame_id (int): Current frame index.
        """
        scene_id = self.scene_ids[file_id]
        scene_path = f"{self.save_dir}/{str(scene_id).zfill(3)}"
        pc_path = f"{scene_path}/lidar/{str(frame_id).zfill(3)}.bin"
        point_cloud = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 10)

        for cam_id in range(5):
            world_to_cam = None

            for downsample_factor in self.downsample_factors:
                if downsample_factor == 1:
                    continue
                scale_postfix = f"_{downsample_factor}"
                ground_label_path = f"{scene_path}/ground_label{scale_postfix}/{str(frame_id).zfill(3)}_{str(cam_id)}.png"
                if not os.path.exists(ground_label_path) or self.overwrite:
                    points = point_cloud[:, 3:6].astype(np.float32)
                    ground_label = get_ground_np(points).reshape(-1)

                    if world_to_cam is None:
                        intrinsics = np.loadtxt(
                            f"{scene_path}/intrinsics/{str(cam_id)}.txt",
                            dtype=np.float32,
                        )
                        fx, fy, cx, cy = intrinsics[:4]
                        intrinsics = np.array(
                            [
                                [fx, 0, cx, 0],
                                [0, fy, cy, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ],
                            dtype=np.float32,
                        )
                        extrinsic = np.loadtxt(
                            f"{scene_path}/extrinsics/{str(cam_id)}.txt",
                            dtype=np.float32,
                        )
                        world_to_cam = np.linalg.inv(extrinsic @ OPENCV2DATASET)
                    target_size = (
                        ORIGINAL_SIZE[str(cam_id)][0] // downsample_factor,
                        ORIGINAL_SIZE[str(cam_id)][1] // downsample_factor,
                    )
                    _intrinsics = intrinsics.copy()
                    _intrinsics[0, 0] /= ORIGINAL_SIZE[str(cam_id)][1] / target_size[1]
                    _intrinsics[1, 1] /= ORIGINAL_SIZE[str(cam_id)][0] / target_size[0]
                    _intrinsics[0, 2] /= ORIGINAL_SIZE[str(cam_id)][1] / target_size[1]
                    _intrinsics[1, 2] /= ORIGINAL_SIZE[str(cam_id)][0] / target_size[0]
                    lidar2img = _intrinsics @ world_to_cam
                    points_2d = (np.dot(lidar2img[:3, :3], points.T) + lidar2img[:3, 3:4]).T
                    depth_2d = points_2d[:, 2]
                    cam_coords = points_2d[:, :2] / (depth_2d[:, None] + 1e-6)
                    valid_mask = (
                        (cam_coords[:, 0] >= 0)
                        & (cam_coords[:, 0] < target_size[1])
                        & (cam_coords[:, 1] >= 0)
                        & (cam_coords[:, 1] < target_size[0])
                        & (depth_2d > 0)
                    )
                    # Get valid depth points and corresponding coordinates
                    valid_cam_coords = cam_coords[valid_mask]
                    # Convert coordinates to integer indices
                    x_indices = valid_cam_coords[:, 0].astype(np.int32)
                    y_indices = valid_cam_coords[:, 1].astype(np.int32)
                    # Initialize arrays to accumulate depth sums and counts
                    depth_image = np.zeros(target_size)
                    depth_image[y_indices, x_indices] = ground_label[valid_mask]
                    imageio.imwrite(ground_label_path, (depth_image * 255).astype(np.uint8))

    def save_pose(self, frame, file_id, frame_id):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (int): Current file index.
            frame_id (int): Current frame index.
        """
        # Use the same (mature) logic as the other Waymo processor:
        # only save ego pose per frame; do not force saving cam_to_world here.
        scene_id = self.scene_ids[file_id]
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            f"{self.save_dir}/{str(scene_id).zfill(3)}/ego_pose/" + f"{str(frame_id).zfill(3)}.txt",
            pose,
        )

    def save_dynamic_mask(self, frame, file_id, frame_id):
        """Parse and save the segmentation data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        scene_id = self.scene_ids[file_id]
        scene_path = f"{self.save_dir}/{str(scene_id).zfill(3)}"
        for img in frame.images:
            # dynamic_mask
            img_path = f"{scene_path}/images/" + f"{str(frame_id).zfill(3)}_{str(img.name - 1)}.jpg"
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            filter_available = any(
                [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels]
            )
            calibration = next(
                cc for cc in frame.context.camera_calibrations if cc.name == img.name
            )
            for label in frame.laser_labels:
                # camera_synced_box is not available for the data with flow.
                # box = label.camera_synced_box
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                if not box.ByteSize():
                    continue  # Filter out labels that do not have a camera_synced_box.
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                    not filter_available and not label.num_lidar_points_in_box
                ):
                    continue  # Filter out likely occluded objects.

                # Retrieve upright 3D box corners.
                box_coords = np.array(
                    [
                        [
                            box.center_x,
                            box.center_y,
                            box.center_z,
                            box.length,
                            box.width,
                            box.height,
                            box.heading,
                        ]
                    ]
                )
                corners = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()  # [8, 3]

                # Project box corners from vehicle coordinates onto the image.
                projected_corners = project_vehicle_to_image(frame.pose, calibration, corners)
                u, v, ok = projected_corners.transpose()
                ok = ok.astype(bool)

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, calibration.width)
                v = np.clip(v, 0, calibration.height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask_path = (
                f"{scene_path}/dynamic_masks/" + f"{str(frame_id).zfill(3)}_{str(img.name - 1)}.png"
            )
            dynamic_mask.save(dynamic_mask_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        for i in self.scene_ids:
            scene_path = f"{self.save_dir}/{str(i).zfill(3)}"
            os.makedirs(f"{scene_path}/images", exist_ok=True)
            os.makedirs(f"{scene_path}/dynamic_masks", exist_ok=True)
            for downsample_factor in self.downsample_factors:
                postfix = "" if downsample_factor == 1 else f"_{downsample_factor}"
                os.makedirs(f"{scene_path}/images{postfix}", exist_ok=True)
                os.makedirs(f"{scene_path}/depth_flows{postfix}", exist_ok=True)
                os.makedirs(f"{scene_path}/ground_label{postfix}", exist_ok=True)
            os.makedirs(f"{scene_path}/ego_pose", exist_ok=True)
            os.makedirs(f"{scene_path}/extrinsics", exist_ok=True)
            os.makedirs(f"{scene_path}/intrinsics", exist_ok=True)
            os.makedirs(f"{scene_path}/lidar", exist_ok=True)
