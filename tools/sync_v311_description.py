#!/usr/bin/env python3
"""Sync the active V3.1.1 demo model from the authoritative description repository."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws/src/alfa_robot_description"
DEFAULT_SOURCE_REPOSITORY = Path("/mnt/mydisk/ALFA/SevenovaHangzhou/robot_description")
DEFAULT_SOURCE_REF = "origin/robot_v3"
EXPECTED_MODEL_REVISION = "robot_v3.1.1"
ASSET_DIRECTORY = "robot_v3_1_1"


def git_output(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_blob(repository, source_ref, relative_path):
    return git_output(repository, "show", f"{source_ref}:{relative_path}")


def expected_snapshot(repository, source_ref):
    commit = git_output(repository, "rev-parse", f"{source_ref}^{{commit}}").decode().strip()
    manifest_bytes = source_blob(repository, source_ref, "config/model_manifest.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"{source_ref} provides {manifest.get('model_revision')}, expected {EXPECTED_MODEL_REVISION}"
        )
    source_xacro = source_blob(repository, source_ref, f"urdf/{ASSET_DIRECTORY}.xacro")
    active_xacro = source_xacro.replace(
        b"package://robot_description/meshes/",
        b"package://alfa_robot_description/meshes/",
    )
    movable_joint_names = sorted(set(manifest["joint_name_mapping"].values()))
    if len(movable_joint_names) != 27:
        raise ValueError(f"expected 27 movable joints, got {len(movable_joint_names)}")
    mesh_blobs = {
        relative_path: source_blob(repository, source_ref, relative_path)
        for relative_path in sorted(manifest["meshes"])
    }
    for relative_path, data in mesh_blobs.items():
        metadata = manifest["meshes"][relative_path]
        if digest(data) != metadata["sha256"] or len(data) != metadata["bytes"]:
            raise ValueError(f"upstream mesh does not match its manifest: {relative_path}")
    origin_url = git_output(repository, "remote", "get-url", "origin").decode().strip()
    lock = {
        "schema_version": 1,
        "model_revision": EXPECTED_MODEL_REVISION,
        "upstream_repository": origin_url,
        "upstream_branch": "robot_v3",
        "upstream_ref_used_for_sync": source_ref,
        "upstream_commit": commit,
        "upstream_manifest_sha256": digest(manifest_bytes),
        "active_xacro_sha256": digest(active_xacro),
        "package_uri_rewrite": {
            "from": "package://robot_description/meshes/",
            "to": "package://alfa_robot_description/meshes/",
        },
        "movable_joint_names": movable_joint_names,
        "mesh_count": len(mesh_blobs),
        "integration_status": "moveit_demo_consumer_snapshot",
    }
    return active_xacro, manifest_bytes, mesh_blobs, json.dumps(
        lock, ensure_ascii=False, indent=2, sort_keys=True
    ).encode() + b"\n"


def destination_files(snapshot):
    active_xacro, manifest, meshes, lock = snapshot
    files = {
        PACKAGE_ROOT / f"urdf/alfa_robot/{ASSET_DIRECTORY}.xacro": active_xacro,
        PACKAGE_ROOT / "config/upstream_description_manifest.json": manifest,
        PACKAGE_ROOT / "config/upstream_description.lock.json": lock,
    }
    for relative_path, data in meshes.items():
        filename = Path(relative_path).name
        files[PACKAGE_ROOT / "meshes" / ASSET_DIRECTORY / filename] = data
    return files


def check_snapshot(files):
    mismatches = []
    expected_paths = set(files)
    asset_root = PACKAGE_ROOT / "meshes" / ASSET_DIRECTORY
    actual_asset_paths = set(asset_root.glob("*.stl")) if asset_root.exists() else set()
    expected_asset_paths = {path for path in expected_paths if path.parent == asset_root}
    for path, expected in files.items():
        if not path.is_file():
            mismatches.append(f"missing: {path.relative_to(REPOSITORY_ROOT)}")
        elif path.read_bytes() != expected:
            mismatches.append(f"content differs: {path.relative_to(REPOSITORY_ROOT)}")
    for path in sorted(actual_asset_paths - expected_asset_paths):
        mismatches.append(f"unexpected: {path.relative_to(REPOSITORY_ROOT)}")
    if mismatches:
        raise SystemExit("V3.1.1 description snapshot mismatch:\n" + "\n".join(mismatches))


def check_local_snapshot():
    lock_path = PACKAGE_ROOT / "config/upstream_description.lock.json"
    manifest_path = PACKAGE_ROOT / "config/upstream_description_manifest.json"
    active_xacro = PACKAGE_ROOT / f"urdf/alfa_robot/{ASSET_DIRECTORY}.xacro"
    for path in (lock_path, manifest_path, active_xacro):
        if not path.is_file():
            raise SystemExit(f"V3.1.1 local snapshot file is missing: {path}")
    lock = json.loads(lock_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if lock.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise SystemExit("V3.1.1 local snapshot lock has the wrong model revision")
    if digest(manifest_path.read_bytes()) != lock["upstream_manifest_sha256"]:
        raise SystemExit("V3.1.1 local manifest does not match its lock")
    if digest(active_xacro.read_bytes()) != lock["active_xacro_sha256"]:
        raise SystemExit("V3.1.1 active Xacro does not match its lock")
    asset_root = PACKAGE_ROOT / "meshes" / ASSET_DIRECTORY
    expected_meshes = {Path(relative_path).name for relative_path in manifest["meshes"]}
    actual_meshes = {path.name for path in asset_root.glob("*.stl")}
    if actual_meshes != expected_meshes or len(actual_meshes) != lock["mesh_count"]:
        raise SystemExit("V3.1.1 local mesh inventory does not match its lock")
    for relative_path, metadata in manifest["meshes"].items():
        path = asset_root / Path(relative_path).name
        if path.stat().st_size != metadata["bytes"] or digest(path.read_bytes()) != metadata["sha256"]:
            raise SystemExit(f"V3.1.1 local mesh differs from its manifest: {path.name}")
    print(
        f"Local consumer snapshot matches {EXPECTED_MODEL_REVISION} at "
        f"{lock['upstream_commit'][:10]}."
    )


def write_snapshot(files):
    asset_root = PACKAGE_ROOT / "meshes" / ASSET_DIRECTORY
    if asset_root.exists():
        shutil.rmtree(asset_root)
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", type=Path, default=DEFAULT_SOURCE_REPOSITORY)
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare against the source repository")
    mode.add_argument("--check-local", action="store_true", help="verify the pinned local snapshot only")
    arguments = parser.parse_args()
    if arguments.check_local:
        check_local_snapshot()
        return
    source_repository = arguments.source_repository.resolve()
    snapshot = expected_snapshot(source_repository, arguments.source_ref)
    files = destination_files(snapshot)
    if arguments.check:
        check_snapshot(files)
        print("V3.1.1 consumer snapshot matches the authoritative description commit.")
        return
    write_snapshot(files)
    check_snapshot(files)
    lock = json.loads(files[PACKAGE_ROOT / "config/upstream_description.lock.json"])
    print(
        f"Synced {lock['model_revision']} at {lock['upstream_commit']} "
        f"with {lock['mesh_count']} meshes."
    )


if __name__ == "__main__":
    main()
