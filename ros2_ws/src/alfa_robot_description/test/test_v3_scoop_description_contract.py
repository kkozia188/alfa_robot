#!/usr/bin/env python3
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
XACRO = PACKAGE / "urdf/alfa_robot.urdf.xacro"
EXPECTED = {
    "meshes/robot_v3_scoop/scoop_visual.stl": "7cc18d6ccf6ac9b4c5d231b45e5c71b4df0298190ff6e4321298d9ffec502d5c",
    "meshes/robot_v3_scoop/scoop_collision.stl": "633cc5f8e892e1ee0f05fa6079cacb7d9f1c61f584114838e00614a6d49c41e1",
}
for relative, expected in EXPECTED.items():
    assert sha256((PACKAGE / relative).read_bytes()).hexdigest() == expected

with tempfile.TemporaryDirectory() as tmp:
    roots = {}
    texts = {}
    for variant in ("suction", "scoop"):
        urdf = Path(tmp) / f"{variant}.urdf"
        with urdf.open("w") as output:
            subprocess.run(["xacro", str(XACRO), f"end_effector:={variant}"], check=True, stdout=output)
        subprocess.run(["check_urdf", str(urdf)], check=True, stdout=subprocess.DEVNULL)
        roots[variant] = ET.parse(urdf).getroot()
        texts[variant] = urdf.read_text()
    normalized = texts["scoop"].replace(
        "meshes/robot_v3_scoop/scoop_visual.stl", "meshes/robot_v3/suction_visual.stl").replace(
        "meshes/robot_v3_scoop/scoop_collision.stl", "meshes/robot_v3/suction_collision.stl")
    assert normalized == texts["suction"]
    for side in ("left", "right"):
        link = roots["scoop"].find(f"./link[@name='{side}_joint7']")
        assert link is not None
        assert link.find("./inertial/mass").attrib["value"] == "2.618"
        assert link.find("./collision/origin").attrib["xyz"] == "-0.158443333 -0.088374996 0"
        tool0 = roots["scoop"].find(f"./joint[@name='{side}_tool0_fixed']/origin")
        assert tool0 is not None and tool0.attrib["xyz"] == "0 0 0.13585"
print("PASS verified V3 scoop end-effector runtime contract")
