"""Render a saved two-A3 run, with an optional uninterrupted point excerpt."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duel import load_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("record",type=Path)
    p.add_argument("--point",type=int)
    p.add_argument("--auto-frame",action="store_true",
                   help="Keep both robots and the table visible when they move away from the play area")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.record/"report.json").read_text())
    states=np.load(args.record/"states.npz")["qpos"]
    trace=[json.loads(x) for x in (args.record/"trace.jsonl").read_text().splitlines()]
    scene,_,_=load_scene(report["near_edge_x"])
    assert states.shape[1]==scene.model.nq
    scene.model.vis.global_.offwidth=1280
    scene.model.vis.global_.offheight=720
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[scene.separation/2,0,.9]
    camera.distance=5.8
    camera.azimuth=90
    camera.elevation=-18
    robot_bodies=np.array([i for i in range(1,scene.model.nbody)
        if scene.model.body(i).name.startswith(("p1_","p2_"))],dtype=int)
    table_points=np.array([[x,y,z] for x in (scene.near_edge_x,scene.near_edge_x+scene.length)
        for y in (-scene.width/2,scene.width/2) for z in (0.,scene.table_height+.16)])
    if args.auto_frame:
        scene.model.vis.global_.fovy=45
    indices=[i for i in range(0,len(states),2) if args.point is None
             or (trace[i]["point"]==args.point and trace[i]["receiver"]!=0)]
    if not indices:
        raise ValueError("No recorded frames for the selected point")
    font=ImageFont.load_default(size=23)
    small=ImageFont.load_default(size=20)
    start=time.perf_counter()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with mujoco.Renderer(scene.model,height=720,width=1280) as renderer, \
         imageio.get_writer(args.output,fps=25,codec="libx264",quality=7,
                            macro_block_size=1,pixelformat="yuv420p") as writer:
        for frame_no,i in enumerate(indices):
            scene.data.qpos[:]=states[i]
            mujoco.mj_forward(scene.model,scene.data)
            if args.auto_frame:
                cloud=np.vstack([scene.data.xpos[robot_bodies],table_points])
                center=(cloud.min(axis=0)+cloud.max(axis=0))/2
                center[2]=.9
                camera.lookat[:]+=.15*(center-camera.lookat)
                radius=float(np.linalg.norm(cloud-camera.lookat,axis=1).max())+.35
                distance=max(5.8,radius/np.sin(np.deg2rad(22.5)))
                # Widen promptly to retain fast-moving limbs; tighten gradually.
                camera.distance=max(distance,camera.distance+.05*(distance-camera.distance))
            renderer.update_scene(scene.data,camera=camera)
            image=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(image)
            row=trace[i]
            point=report["points"][row["point"]-1]
            context="ball in play" if row["receiver"] else "between points"
            hits=[x for x in point["contacts"] if x["time"]<=row["time"]+1e-9]
            legal=0
            for hit in hits:
                if not hit.get("legal",False):
                    break
                legal+=1
            draw.rectangle([0,0,1280,85],fill=(15,22,32))
            draw.text((15,9),"HOPE | two A3 robots | two independent ONNX controllers | one physical ball",
                       font=font,fill="white")
            draw.text((15,47),f"Point {row['point']:02d}/{len(report['points'])} | session {row['time']:.2f}s | "
                       f"P1 hits: {sum(x['player']==1 for x in hits)} | P2 hits: {sum(x['player']==2 for x in hits)} | "
                       f"legal chain: {legal} | {context} | 1x",font=small,fill=(185,220,255))
            draw.rectangle([0,684,1280,720],fill=(15,22,32))
            draw.text((15,691),"P1 (left) | physical ball within points; scripted serve between points; no robot resets | P2 (right)",
                       font=small,fill=(245,225,195))
            writer.append_data(np.asarray(image))
            if frame_no in {0,len(indices)//2,len(indices)-1}:
                image.save(args.output.with_name(args.output.stem+f"-frame-{frame_no:04d}.png"))
            if frame_no%300==0:
                print(f"render {frame_no}/{len(indices)} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    metadata={"source":str(args.record),"video":str(args.output),"point":args.point,
              "frames":len(indices),"fps":25,"duration_seconds":len(indices)/25,
              "resolution":[1280,720],"render_wall_seconds":time.perf_counter()-start,
              "auto_frame":args.auto_frame,
              "method":"mj_forward playback of saved qpos, original simulation speed, no policy/physics rerun"}
    args.output.with_suffix(".video.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
