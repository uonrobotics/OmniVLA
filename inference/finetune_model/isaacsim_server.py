# ===============================================================
# [ SIMULATOR SERVER ]
# 역할:
#   - stage/env/robot/camera 초기화
#   - 랜덤 spawn reset
#   - 현재 pose 반환
#   - 현재 RGB 반환
#   - 목표 도달/episode 종료 관리
# ===============================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "fast_shutdown": True,
    }
)

import base64
import io
import json
import math
import select
import socket
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import omni.timeline
import omni.usd
import yaml

from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension

ENV_USD_PATH = "/nas/sujinkim/data/goto/sim/goto_warehouse.usd"
ROBOT_REL_PATH = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"

ENV_PRIM_PATH = "/World/env"
ROBOT_ROOT_PRIM_PATH = "/World/Nova_Carter_ROS"
ROBOT_BODY_PRIM_PATH = "/World/Nova_Carter_ROS/chassis_link"
SPAWN_AREA_PRIM_PATH = "/World/env/spawn_area"

DEFAULT_Z = 0.0

FRONT_CAM_CFG = {
    "name": "cam_front",
    "camera_prim_path": "/World/replay_camera/front_camera",
    "resolution": (320, 240),
    "fov_deg": 90.0,
    "offset_xyz": [0.20, 0.0, 0.80],
    "rot_xyz_deg": [90.0, -90.0, 0.0],
}


def add_physics_scene(stage):
    scene_path = "/physicsScene"
    if stage.GetPrimAtPath(scene_path).IsValid():
        return
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")


def add_dome_light(stage):
    dome_path = "/World/DomeLight"
    if stage.GetPrimAtPath(dome_path).IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(1000)


def get_valid_prim(stage, prim_path: str, name: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"{name} prim not found: {prim_path}")
    return prim


def get_world_xy_yaw(stage, prim_path: str):
    prim = get_valid_prim(stage, prim_path, prim_path)
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    pos = mat.ExtractTranslation()
    world_forward = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    yaw = math.atan2(float(world_forward[1]), float(world_forward[0]))
    return float(pos[0]), float(pos[1]), float(yaw)


def get_world_bbox_xy(stage, prim_path: str):
    prim = get_valid_prim(stage, prim_path, prim_path)
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return float(min_pt[0]), float(max_pt[0]), float(min_pt[1]), float(max_pt[1])


def quat_wxyz_from_yaw(yaw_rad: float):
    half = yaw_rad * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def sample_xy_in_region(region, margin=0.3):
    xmin, xmax, ymin, ymax = region
    x = np.random.uniform(xmin + margin, xmax - margin)
    y = np.random.uniform(ymin + margin, ymax - margin)
    return float(x), float(y)


class OccupancyDistanceMap:
    def __init__(self, map_png_path, map_yaml_path):
        with open(map_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.resolution = float(cfg["resolution"])
        self.origin_x = float(cfg["origin"][0])
        self.origin_y = float(cfg["origin"][1])

        img = Image.open(map_png_path).convert("L")
        self.map_img = np.array(img)
        self.occupied = self.map_img < 128
        self.height, self.width = self.occupied.shape

        from scipy.ndimage import distance_transform_edt
        free_mask = ~self.occupied
        dist_pixels = distance_transform_edt(free_mask)
        self.dist_meters = dist_pixels * self.resolution

    def world_to_map_rc(self, x, y):
        mx = int((x - self.origin_x) / self.resolution)
        my = int((y - self.origin_y) / self.resolution)
        row = self.height - 1 - my
        col = mx
        return row, col

    def is_inside(self, x, y):
        row, col = self.world_to_map_rc(x, y)
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, x, y):
        if not self.is_inside(x, y):
            return False
        row, col = self.world_to_map_rc(x, y)
        return not self.occupied[row, col]

    def clearance(self, x, y):
        if not self.is_inside(x, y):
            return 0.0
        row, col = self.world_to_map_rc(x, y)
        return float(self.dist_meters[row, col])


def sample_conditioned_spawn(region, occ_map, min_clearance=1.5, max_trials=50):
    for _ in range(max_trials):
        x, y = sample_xy_in_region(region, margin=0.3)
        if not occ_map.is_inside(x, y):
            continue
        if not occ_map.is_free(x, y):
            continue
        if occ_map.clearance(x, y) < min_clearance:
            continue
        yaw = np.random.uniform(-math.pi, math.pi)
        return x, y, float(yaw)
    raise RuntimeError("Failed to sample valid spawn pose.")


class JsonSocketServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.client = None
        self.buffer = b""

    def poll_accept(self):
        if self.client is not None:
            return
        readable, _, _ = select.select([self.server], [], [], 0.0)
        if readable:
            conn, addr = self.server.accept()
            conn.setblocking(False)
            self.client = conn
            self.buffer = b""
            print(f"[IPC] connected from {addr}")

    def recv_message(self) -> Optional[dict]:
        self.poll_accept()
        if self.client is None:
            return None
        readable, _, _ = select.select([self.client], [], [], 0.0)
        if not readable:
            return None
        data = self.client.recv(10_000_000)
        if not data:
            self.client.close()
            self.client = None
            self.buffer = b""
            return None
        self.buffer += data
        if b"\n" not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def send_message(self, payload: dict):
        if self.client is None:
            raise RuntimeError("No client connected")
        self.client.sendall((json.dumps(payload) + "\n").encode("utf-8"))


class IsaacSimServer:
    def __init__(self):
        self.stage = None
        self.timeline = None
        self.robot = None
        self.camera = None
        self.spawn_region = None
        self.occ_map = OccupancyDistanceMap(
            "/nas/sujinkim/data/goto/sim/goto_warehouse.png",
            "/nas/sujinkim/data/goto/sim/goto_warehouse.yaml",
        )

    def setup(self):
        self._enable_extensions()
        self._create_stage_once()
        self._load_env_and_robot_once()
        self._start_simulation_once()
        self._initialize_articulation_once()
        self._setup_camera_once()

    def _enable_extensions(self):
        enable_extension("omni.physx")
        enable_extension("omni.physx.ui")
        enable_extension("omni.graph.nodes")
        enable_extension("isaacsim.core.nodes")
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.rtx")
        simulation_app.update()
        simulation_app.update()

    def _create_stage_once(self):
        create_new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        add_physics_scene(self.stage)
        add_dome_light(self.stage)

    def _load_env_and_robot_once(self):
        assets_root = get_assets_root_path()
        robot_usd = assets_root + ROBOT_REL_PATH
        add_reference_to_stage(usd_path=ENV_USD_PATH, prim_path=ENV_PRIM_PATH)
        add_reference_to_stage(usd_path=robot_usd, prim_path=ROBOT_ROOT_PRIM_PATH)
        simulation_app.update()
        simulation_app.update()
        self.spawn_region = get_world_bbox_xy(self.stage, SPAWN_AREA_PRIM_PATH)

    def _start_simulation_once(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        for _ in range(20):
            simulation_app.update()

    def _initialize_articulation_once(self):
        self.robot = Articulation(ROBOT_ROOT_PRIM_PATH)
        self.robot.initialize()
        for _ in range(10):
            simulation_app.update()

    @staticmethod
    def yaw_to_quat_xyzw(yaw: float):
        half = yaw * 0.5
        return [0.0, 0.0, math.sin(half), math.cos(half)]

    @staticmethod
    def quat_multiply_xyzw(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    @staticmethod
    def euler_xyz_deg_to_quat_xyzw(rx_deg, ry_deg, rz_deg):
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)
        cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
        cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
        cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
        return [qx, qy, qz, qw]

    @staticmethod
    def npquat_xyzw_to_gf(q):
        x, y, z, w = q
        return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))

    def set_xform_pose(self, prim, xyz, quat_xyzw):
        quatd = self.npquat_xyzw_to_gf(quat_xyzw)
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        translate_op = None
        orient_op = None
        for op in ordered_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        orient_op.Set(quatd)

    @staticmethod
    def fov_to_focal_length(fov_deg, aperture=20.955):
        fov_rad = math.radians(fov_deg)
        return aperture / (2.0 * math.tan(fov_rad / 2.0))

    def _setup_camera_once(self):
        stage = self.stage
        if not stage.GetPrimAtPath("/World/replay_camera").IsValid():
            UsdGeom.Xform.Define(stage, "/World/replay_camera")
        cam_prim_path = FRONT_CAM_CFG["camera_prim_path"]
        if not stage.GetPrimAtPath(cam_prim_path).IsValid():
            UsdGeom.Camera.Define(stage, cam_prim_path)

        self.camera = Camera(
            prim_path=cam_prim_path,
            name=FRONT_CAM_CFG["name"],
            frequency=30,
            resolution=FRONT_CAM_CFG["resolution"],
        )
        self.camera.initialize()

        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        cam_geom = UsdGeom.Camera(cam_prim)
        cam_geom.GetHorizontalApertureAttr().Set(20.955)
        cam_geom.GetVerticalApertureAttr().Set(15.2908)
        cam_geom.GetFocalLengthAttr().Set(self.fov_to_focal_length(FRONT_CAM_CFG["fov_deg"]))

        for _ in range(10):
            simulation_app.update()

    def compose_camera_world_pose(self, robot_pose_world: Tuple[float, float, float]):
        base_x, base_y, base_yaw = robot_pose_world
        dx, dy, dz = FRONT_CAM_CFG["offset_xyz"]
        r_deg, p_deg, y_deg = FRONT_CAM_CFG["rot_xyz_deg"]
        cam_x = base_x + math.cos(base_yaw) * dx - math.sin(base_yaw) * dy
        cam_y = base_y + math.sin(base_yaw) * dx + math.cos(base_yaw) * dy
        cam_z = dz
        q_base = self.yaw_to_quat_xyzw(base_yaw)
        q_cam_local = self.euler_xyz_deg_to_quat_xyzw(r_deg, p_deg, y_deg)
        q_cam_world = self.quat_multiply_xyzw(q_base, q_cam_local)
        return [cam_x, cam_y, cam_z], q_cam_world

    def reset_robot_random_pose(self):
        x, y, yaw = sample_conditioned_spawn(self.spawn_region, self.occ_map)
        quat_wxyz = quat_wxyz_from_yaw(yaw)
        self.robot.set_world_pose(
            position=np.array([x, y, DEFAULT_Z], dtype=np.float32),
            orientation=quat_wxyz,
        )
        self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for _ in range(120):
            simulation_app.update()
        return self.get_pose()

    def get_pose(self):
        x, y, yaw = get_world_xy_yaw(self.stage, ROBOT_BODY_PRIM_PATH)
        return {"x": x, "y": y, "yaw": yaw}

    def get_rgb_base64(self):
        pose = self.get_pose()
        cam_xyz, cam_quat = self.compose_camera_world_pose((pose["x"], pose["y"], pose["yaw"]))
        cam_prim = self.stage.GetPrimAtPath(FRONT_CAM_CFG["camera_prim_path"])
        self.set_xform_pose(cam_prim, cam_xyz, cam_quat)

        for _ in range(2):
            simulation_app.update()

        rgba = self.camera.get_rgba()
        rgb = np.asarray(rgba)[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    server = JsonSocketServer(host="0.0.0.0", port=8765)
    sim = IsaacSimServer()
    sim.setup()

    try:
        while simulation_app.is_running():
            simulation_app.update()
            msg = server.recv_message()
            if msg is None:
                continue

            cmd = msg.get("cmd")
            try:
                if cmd == "ping":
                    server.send_message({"ok": True, "msg": "pong"})
                elif cmd == "reset":
                    pose = sim.reset_robot_random_pose()
                    server.send_message({"ok": True, "pose": pose})
                elif cmd == "get_obs":
                    pose = sim.get_pose()
                    image_b64 = sim.get_rgb_base64()
                    server.send_message({"ok": True, "pose": pose, "image_b64": image_b64})
                else:
                    server.send_message({"ok": False, "error": f"unknown cmd: {cmd}"})
            except Exception as e:
                server.send_message({"ok": False, "error": str(e)})
    finally:
        if sim.timeline is not None:
            sim.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()