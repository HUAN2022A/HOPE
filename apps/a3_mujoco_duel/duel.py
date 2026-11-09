"""Two independent A3 ONNX controllers sharing one MuJoCo table and physical ball.

This is a local reference experiment, not the production ROS or hardware stack.
Only ball launch between points is scripted; within a point all ball motion and
all racket contacts are produced by MuJoCo. Robot state is never reset between points.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "a3_deploy/a3_deploy_example/reference"),
               str(ROOT / "hope_training/whole_body_tracking/scripts")]
from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES, HEAD_INDICES
from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
from a3_deploy_onnx_ref_pingpong.observation import RobotState, build_observation
from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
from a3_deploy_onnx_ref_pingpong.racket_command import RacketCommand
from mujoco_eval_onnx import _load_success_metric, _predict_command, _sample_serve
from mujoco_pingpong_scene import _dampratio_from_restitution


@dataclass
class PlayerFrame:
    index: int
    separation: float

    def position(self, world):
        p = np.asarray(world, dtype=float).copy()
        return p if self.index == 0 else np.array([self.separation-p[0], -p[1], p[2]])

    def vector(self, world):
        v = np.asarray(world, dtype=float).copy()
        return v if self.index == 0 else v * [-1., -1., 1.]

    def quaternion(self, world):
        q = np.asarray(world, dtype=float)
        if self.index == 0:
            return q.copy()
        # inverse yaw-pi * world quaternion, wxyz.
        return np.array([q[3], q[2], -q[1], -q[0]])

    # Both transforms are involutions (a 180-degree rotation plus translation).
    to_world_position = position
    to_world_vector = vector


class DuelScene:
    def __init__(self, cfg, ball_cfg, near_edge_x=.30):
        self.cfg, self.ball_cfg = cfg, ball_cfg
        self.near_edge_x = float(near_edge_x)
        self.length = float(ball_cfg["table"]["length"])
        self.width = float(ball_cfg["table"]["width"])
        self.table_height = float(ball_cfg["table"]["height"])
        self.ball_radius = float(ball_cfg["ball"]["radius"])
        self.ball_mass = float(ball_cfg["ball"]["mass"])
        self.net_x_table = self.length / 2
        self.offset = np.array([self.near_edge_x, self.width/2, self.table_height])
        self.table_restitution = float(ball_cfg["contact"]["table"]["restitution"])
        self.separation = self.length + 2*self.near_edge_x
        self.frames = [PlayerFrame(i, self.separation) for i in range(2)]
        self.reset_count = 0
        self._build()
        self.data = mujoco.MjData(self.model)
        self.players = [Player(self, i) for i in range(2)]
        self.ball_qadr = int(self.model.jnt_qposadr[self.jid("ball_free_joint")])
        self.ball_vadr = int(self.model.jnt_dofadr[self.jid("ball_free_joint")])
        self.ball_bid = self.bid("ball")
        self.ball_gid = self.gid("ball_geom")
        self.racket_gids = [self.gid(f"p{i+1}_right_racket_collision") for i in range(2)]
        self.table_gid = self.gid("table")
        self._table_contact = False
        self._last_bounce_time = -1.
        self.substeps = round(cfg.control_dt / self.model.opt.timestep)
        self.reset()

    def _build(self):
        source = mujoco.MjSpec.from_file(str(self.cfg.model_xml_path))
        source_model = source.compile()
        self.stand_base = source_model.key_qpos[0, :7].copy()
        self.stand_joints = {
            name: float(source_model.key_qpos[0, source_model.jnt_qposadr[
                mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
            for name in JOINT_NAMES}
        # Attach only the robot articulation. Old arena/ball/keyframes are not duplicated.
        for item in list(source.pairs) + list(source.keys):
            source.delete(item)
        for body in list(source.worldbody.bodies):
            if body.name != "pelvis_link":
                source.delete(body)
        spec = mujoco.MjSpec()
        spec.option.timestep = source.option.timestep
        spec.option.noslip_iterations = source.option.noslip_iterations
        spec.option.noslip_tolerance = source.option.noslip_tolerance
        spec.option.gravity = [0, 0, -float(self.ball_cfg["gravity"])]
        spec.nuserdata = source.nuserdata
        for i in range(2):
            child = source.copy()
            frame = spec.worldbody.add_frame(name=f"p{i+1}_frame",
                pos=[self.separation*i, 0, 0], quat=[1, 0, 0, 0] if i == 0 else [0, 0, 0, 1])
            frame.attach_body(child.body("pelvis_link"), prefix=f"p{i+1}_")
        spec.worldbody.add_light(pos=[self.separation/2,0,6], dir=[0,0,-1], ambient=[.3,.3,.3])
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[300,300,.125], contype=4, conaffinity=1, friction=[1.5,.005,.0001],
            rgba=[.21,.25,.29,1])
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[self.near_edge_x+self.length/2,0,self.table_height-.025],
            size=[self.length/2,self.width/2,.025], contype=8, conaffinity=8,
            rgba=[.06,.30,.52,1])
        net_h = float(self.ball_cfg["net"]["height"])
        spec.worldbody.add_geom(name="net", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[self.separation/2,0,self.table_height+net_h/2],
            size=[.005,self.width/2+.15,net_h/2], contype=8, conaffinity=8,
            rgba=[.9,.9,.9,.4])
        body = spec.worldbody.add_body(name="ball", pos=[self.separation/2,0,2])
        body.add_freejoint(name="ball_free_joint")
        body.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.ball_radius,0,0], mass=self.ball_mass,
            contype=8, conaffinity=15, rgba=[1,.5,0,1])
        materials = self.ball_cfg["contact"]
        for surface, kind in [("floor","floor"),("table","table"),("net","net"),
                              ("p1_right_racket_collision","paddle"),
                              ("p2_right_racket_collision","paddle")]:
            material = materials[kind]
            e = float(material["restitution"])
            pair = spec.add_pair(geomname1="ball_geom", geomname2=surface)
            pair.solref = [.03 if e>.8 else .02, _dampratio_from_restitution(e)]
            fr = float(material["dynamic_friction"])
            pair.friction = [fr,fr,.005,1e-4,1e-4]
            pair.condim = 3
        self.model = spec.compile()

    def jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def gid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def bid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def reset(self):
        self.reset_count += 1
        mujoco.mj_resetData(self.model, self.data)
        for player in self.players:
            a = player.base_qadr
            frame = player.frame
            self.data.qpos[a:a+3] = frame.to_world_position(self.stand_base[:3])
            # yaw-pi = inverse yaw-pi up to quaternion sign.
            self.data.qpos[a+3:a+7] = frame.quaternion(self.stand_base[3:7])
            self.data.qpos[player.qadr] = [self.stand_joints[n] for n in JOINT_NAMES]
        self.set_ball([self.separation+2,0,1], [0,0,0])

    def set_ball(self, position, velocity):
        self.data.qpos[self.ball_qadr:self.ball_qadr+7] = [*position,1,0,0,0]
        self.data.qvel[self.ball_vadr:self.ball_vadr+6] = [*velocity,0,0,0]
        mujoco.mj_forward(self.model, self.data)

    def ball_state(self):
        return (self.data.qpos[self.ball_qadr:self.ball_qadr+3].copy(),
                self.data.qvel[self.ball_vadr:self.ball_vadr+3].copy())

    def step(self):
        contacts = set()
        crossings = []
        bounces = []
        drag = float(self.ball_cfg["drag"]["k"])
        for _ in range(self.substeps):
            for player in self.players:
                tau = self.cfg.sim_kp*(player.q_des-self.data.qpos[player.qadr]) \
                    - self.cfg.sim_kd*self.data.qvel[player.vadr]
                bounds = self.model.actuator_ctrlrange[player.act]
                self.data.ctrl[player.act] = np.clip(tau,bounds[:,0],bounds[:,1])
            v = self.data.qvel[self.ball_vadr:self.ball_vadr+3]
            self.data.xfrc_applied[self.ball_bid,:3] = -self.ball_mass*drag*min(np.linalg.norm(v),50)*v
            before = self.data.qpos[self.ball_qadr:self.ball_qadr+3].copy()
            mujoco.mj_step(self.model,self.data)
            after = self.data.qpos[self.ball_qadr:self.ball_qadr+3]
            table_contact = False
            for contact in self.data.contact:
                pair = {contact.geom1,contact.geom2}
                if pair == {self.ball_gid,self.table_gid}:
                    table_contact = True
                for i,gid in enumerate(self.racket_gids):
                    if pair == {self.ball_gid,gid}:
                        contacts.add(i)
            if table_contact and not self._table_contact and self.data.time-self._last_bounce_time>.1:
                bounces.append({"half":1 if after[0]<self.separation/2 else 2,
                                "position":after.copy().tolist(),"time":float(self.data.time)})
                self._last_bounce_time=float(self.data.time)
            self._table_contact=table_contact
            if (before[0]-self.separation/2)*(after[0]-self.separation/2)<0:
                frac=(self.separation/2-before[0])/(after[0]-before[0])
                at_net=before+frac*(after-before)
                crossings.append({"direction":int(np.sign(after[0]-before[0])),
                                  "height":float(at_net[2]),"y":float(at_net[1])})
        self.data.xfrc_applied[self.ball_bid,:3] = 0
        return contacts,crossings,bounces


class Player:
    def __init__(self, scene, index):
        self.scene,self.index,self.frame = scene,index,scene.frames[index]
        m = scene.model
        prefix = f"p{index+1}_"
        joints = [scene.jid(prefix+n) for n in JOINT_NAMES]
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints],dtype=int)
        self.vadr = np.array([m.jnt_dofadr[j] for j in joints],dtype=int)
        self.act = np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in joints])
        self.base_qadr = int(m.jnt_qposadr[scene.jid(prefix+"pelvis_free_joint")])
        sensor = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SENSOR,prefix+"pelvis_imu_gyro")
        self.gyro_adr = int(m.sensor_adr[sensor])
        self.q_des = scene.cfg.action_adapter.default_q.copy()

    def state(self):
        d = self.scene.data
        a = self.base_qadr
        return RobotState(self.frame.position(d.qpos[a:a+3]),
            self.frame.quaternion(d.qpos[a+3:a+7]),d.sensordata[self.gyro_adr:self.gyro_adr+3].copy(),
            d.qpos[self.qadr].copy(),d.qvel[self.vadr].copy())


class Controller:
    def __init__(self, cfg, player, station_tracking=False):
        self.cfg,self.player=cfg,player
        self.policy = OnnxPolicy(cfg.onnx_path)
        self.life = SwingLifecycle(cfg.lifecycle)
        self.last_action = np.zeros(31)
        self.station = player.state().base_pos_w[:2].copy()
        self.home_station = self.station.copy()
        self.station_tracking = station_tracking
        if station_tracking:
            meta=self.policy._sess.get_modelmeta().custom_metadata_map
            self.reach=np.array([float(x) for x in meta["ref_reach_offset_xy"].split(",")]).reshape(2,2)
        self.engaged = set()
        self.cancelled = 0

    def tick(self, command, flight_id):
        if command is None:
            self.cancelled += int(self.life.cancel_active_task(flight_id))
        elif self.station_tracking:
            side=self.life.swing_sign if self.life.active_task_id==command.task_id else command.swing_sign
            clip=0 if side>0 else 1
            self.station[1]=np.clip(command.position[1]-self.reach[clip,1],
                                    self.home_station[1]-.35,self.home_station[1]+.35)
        target = self.life.update(command,self.player.state())
        if self.life.active_task_id is not None:
            self.engaged.add(self.life.active_task_id)
        obs = build_observation(self.player.state(),target,self.last_action,
                                self.cfg.action_adapter.default_q,self.station)
        action = self.policy.infer_target(obs,target.time_to_strike,self.life.swing_sign,self.cfg.control_dt)
        self.last_action = action.astype(float)
        self.last_action[list(HEAD_INDICES)] = 0
        self.player.q_des = self.cfg.action_adapter.decode(self.last_action)
        self.player.q_des[list(HEAD_INDICES)] = self.cfg.action_adapter.default_q[list(HEAD_INDICES)]
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(self.player.q_des)):
            raise FloatingPointError("Nonfinite controller values")


def load_scene(near_edge_x=.30):
    cfg=RuntimeConfig.load(ROOT/"a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml")
    metric=_load_success_metric(ROOT)
    ball_cfg=metric.load_ball_physics_config()
    return DuelScene(cfg,ball_cfg,near_edge_x),cfg,metric.BallPhysics.from_config(ball_cfg)


def incoming_at_strike(position, velocity, tts, scene, physics):
    """Recover the incoming velocity along the same reference prediction rollout."""
    p,v=np.asarray(position).copy(),np.asarray(velocity).copy()
    remaining=float(tts)
    surface=scene.table_height+scene.ball_radius
    while remaining>1e-9:
        dt=min(.002,remaining)
        a=physics.acceleration(v)
        pn,vn=p+v*dt+.5*a*dt*dt,v+a*dt
        if p[2]>surface>=pn[2] and scene.near_edge_x<=pn[0]<=scene.near_edge_x+scene.length \
                and abs(pn[1])<=scene.width/2:
            pn[2]=2*surface-pn[2]
            vn[2]*=-scene.table_restitution
        p,v=pn,vn
        remaining-=dt
    return v


def impact_velocity(incoming, outgoing, restitution):
    """No-spin moving-paddle normal impulse, as in the reference planner Stage 3."""
    normal=np.asarray(outgoing)-np.asarray(incoming)
    length=np.linalg.norm(normal)
    if length<1e-9:
        return np.zeros(3)
    normal=normal/length
    speed=np.dot(outgoing+restitution*incoming,normal)/(1+restitution)
    return speed*normal


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    scene,cfg,physics=load_scene(args.near_edge_x)
    controllers=[Controller(cfg,p,args.station_tracking) for p in scene.players]
    rng=np.random.default_rng(args.seed)
    geometry=scene
    sampler_args=argparse.Namespace(strike_plane_x=args.strike_plane_x)
    rows,states,points=[],[],[]
    flight_id=0
    start=time.perf_counter()
    min_height=[float("inf")]*2
    max_tilt=[0.]*2
    active_contact=set()
    for point in range(args.points):
        receiver=point%2
        side=1 if (point//2)%2==0 else -1
        _,position,velocity=_sample_serve(rng,side,geometry,physics,sampler_args)
        frame=scene.frames[receiver]
        scene.set_ball(frame.to_world_position(position),frame.to_world_vector(velocity))
        flight_id+=1
        record={"point":point+1,"initial_receiver":receiver+1,"contacts":[],"net_crossings":[],
                "table_bounces":[],"alternating_contacts":0,"ball_launches":1,
                "legal_contact_prefix":0,"first_rule_violation":None}
        last_hit=None
        last_contact_time=-1.
        leg_bounces=[]
        leg_net_crossings=[]
        point_start=scene.data.time
        record["start_time"]=float(point_start)
        while scene.data.time-point_start < args.max_point_seconds:
            p,v=scene.ball_state()
            for i,controller in enumerate(controllers):
                local_p,local_v=scene.frames[i].position(p),scene.frames[i].vector(v)
                command=None
                if i==receiver and local_v[0]<-.1:
                    y=local_p[1]
                    swing=1 if y<-.2 else -1
                    command=_predict_command(local_p,local_v,swing,geometry,physics,sampler_args,
                                             flight_id,len(rows),RacketCommand)
                    if command is not None and args.impact_model:
                        incoming=incoming_at_strike(local_p,local_v,command.time_to_strike,scene,physics)
                        outgoing=command.velocity.copy()  # legacy predictor emits the desired ball velocity
                        outgoing[1]=(args.landing_y-command.position[1])/.6
                        command.velocity=impact_velocity(incoming,outgoing,
                            float(scene.ball_cfg["contact"]["paddle"]["restitution"]))
                controller.tick(command,flight_id)
            contacts,crossings,bounces=scene.step()
            leg_bounces.extend(bounces)
            leg_net_crossings.extend(crossings)
            record["table_bounces"].extend(bounces)
            now=float(scene.data.time)
            for i in sorted(contacts-active_contact):
                if now-last_contact_time>.12:
                    desired_direction=-1 if i==0 else 1
                    net_clear=any(x["direction"]==desired_direction
                                  and x["height"]>scene.table_height+scene.ball_cfg["net"]["height"]+scene.ball_radius
                                  and abs(x["y"])<=scene.width/2+.15 for x in leg_net_crossings)
                    own_bounces=[x for x in leg_bounces if x["half"]==i+1]
                    valid_bounce=len(leg_bounces)==1 and len(own_bounces)==1
                    legal=last_hit!=i and valid_bounce and net_clear
                    record["contacts"].append({"player":i+1,"time":now,"legal":legal,
                                               "preceding_bounces":len(leg_bounces),"net_clear":net_clear})
                    if legal and record["first_rule_violation"] is None:
                        record["legal_contact_prefix"]+=1
                    elif record["first_rule_violation"] is None:
                        record["first_rule_violation"]={"player":i+1,"time":now,
                            "reason":"missing/surplus own-half bounce or net crossing, or repeated player"}
                    if last_hit!=i:
                        record["alternating_contacts"]+=1
                    last_hit=i
                    last_contact_time=now
                    receiver=1-i
                    flight_id+=1
                    leg_bounces=[]
                    leg_net_crossings=[]
            active_contact=contacts
            record["net_crossings"].extend({**x,"time":now} for x in crossings)
            status=[]
            for i,player in enumerate(scene.players):
                state=player.state()
                quat=state.base_quat_w/np.linalg.norm(state.base_quat_w)
                tilt=float(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1))))
                height=float(state.base_pos_w[2])
                min_height[i]=min(min_height[i],height)
                max_tilt[i]=max(max_tilt[i],tilt)
                status.append({"z":height,"tilt":tilt,"phase":controllers[i].life.phase.value})
            if not np.all(np.isfinite(scene.data.qpos)):
                raise FloatingPointError("Nonfinite plant state")
            states.append(scene.data.qpos.copy())
            rows.append({"time":now,"point":point+1,"ball":scene.ball_state()[0].tolist(),
                         "receiver":receiver+1,"players":status,"contacts":[i+1 for i in contacts]})
            p,v=scene.ball_state()
            if p[2]<.05 or p[0]<-1 or p[0]>scene.separation+1 or abs(p[1])>2.5:
                break
        record["end_time"]=float(scene.data.time)
        record["termination"]="time_limit" if scene.data.time-point_start>=args.max_point_seconds else "out_of_play"
        points.append(record)
        # Park only the ball while both controllers continue their recovery.
        scene.set_ball([scene.separation+2,0,1],[0,0,0])
        for _ in range(round(2/cfg.control_dt)):
            for controller in controllers:
                controller.tick(None,flight_id)
            scene.step()
            states.append(scene.data.qpos.copy())
            rows.append({"time":float(scene.data.time),"point":point+1,"ball":scene.ball_state()[0].tolist(),
                         "receiver":0,"players":[],"contacts":[]})
    state_array=np.stack(states)
    # Include recovery intervals in posture checks as well as ball-in-play time.
    for i,player in enumerate(scene.players):
        a=player.base_qadr
        quat=state_array[:,a+3:a+7]
        quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
        tilt=np.degrees(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
        max_tilt[i]=float(tilt.max())
        min_height[i]=float(state_array[:,a+2].min())
    report={"points":points,"simulated_seconds":float(scene.data.time),"wall_seconds":time.perf_counter()-start,
            "robots":2,"policies":2,"controlled_joints":int(scene.model.nu),"ball_free_joints":1,
            "robot_reset_count":scene.reset_count,"ball_launches":args.points,
            "raw_contacts_per_player":dict(Counter(e["player"] for r in points for e in r["contacts"])),
            "max_alternating_contacts_same_ball":max(r["alternating_contacts"] for r in points),
            "max_legal_alternating_contacts_same_ball":max(r["legal_contact_prefix"] for r in points),
            "max_pelvis_tilt_deg":max_tilt,"min_pelvis_z_m":min_height,
            "engaged_flights_per_player":[len(c.engaged) for c in controllers],
            "policy_sha256":hashlib.sha256(cfg.onnx_path.read_bytes()).hexdigest(),
            "seed":args.seed,"near_edge_x":args.near_edge_x,"strike_plane_x":args.strike_plane_x,
            "max_point_seconds":args.max_point_seconds,
            "impact_model":args.impact_model,"landing_y":args.landing_y,
            "station_tracking":args.station_tracking,
            "note":"Raw MuJoCo contact only. No proximity fallback, assisted ball return or within-point ball reset."}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    np.savez_compressed(args.output/"states.npz",qpos=state_array)
    with (args.output/"trace.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="points"},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--points",type=int,default=12)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--max-point-seconds",type=float,default=6.)
    p.add_argument("--near-edge-x",type=float,default=.30)
    p.add_argument("--strike-plane-x",type=float,default=0.)
    p.add_argument("--impact-model",action="store_true")
    p.add_argument("--landing-y",type=float,default=0.)
    p.add_argument("--station-tracking",action="store_true")
    run(p.parse_args())


if __name__=="__main__":
    main()
