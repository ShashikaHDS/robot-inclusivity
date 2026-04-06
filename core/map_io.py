"""Map I/O utilities — PGM/YAML parsing and path helpers."""

import os
import re
import numpy as np


def parse_pgm(path):
    """Returns (width, height, pixels_uint8_array) — raw PGM row order."""
    with open(path, 'rb') as f:
        data = f.read()
    i = 0
    def skip_ws_comments():
        nonlocal i
        while i < len(data):
            if data[i] == 35:  # '#'
                while i < len(data) and data[i] != 10: i += 1
                i += 1
            elif data[i] <= 32: i += 1
            else: break
    def read_token():
        nonlocal i
        skip_ws_comments()
        t = bytearray()
        while i < len(data) and data[i] > 32:
            t.append(data[i]); i += 1
        return t.decode()
    magic = read_token()
    width = int(read_token())
    height = int(read_token())
    maxval = int(read_token())
    i += 1  # skip single whitespace after maxval
    if magic == 'P5':
        if maxval <= 255:
            pixels = np.frombuffer(data, dtype=np.uint8, offset=i, count=width * height).copy()
        else:
            raw = np.frombuffer(data, dtype=np.uint8, offset=i, count=width * height * 2)
            pixels = np.zeros(width * height, dtype=np.uint8)
            for j in range(width * height):
                pixels[j] = round((raw[j*2] * 256 + raw[j*2+1]) / maxval * 255)
    else:
        vals = data[i:].decode().split()
        pixels = np.array([round(int(v) / maxval * 255) for v in vals[:width*height]], dtype=np.uint8)
    return width, height, pixels


def parse_yaml(path):
    """Returns dict matching HTML parseYAML."""
    with open(path) as f: text = f.read()
    def get(k, default):
        m = re.search(rf'^{k}\s*:\s*(.+)', text, re.MULTILINE)
        return m.group(1).strip() if m else default
    origin_s = get('origin', '[0,0,0]').replace('[','').replace(']','')
    return dict(
        resolution=float(get('resolution', '0.05')),
        origin=[float(x) for x in origin_s.split(',')],
        free_thresh=float(get('free_thresh', '0.196')),
        negate=int(get('negate', '0'))
    )


def resolve_point_cloud_path(directory, stem_candidates):
    """Resolve the first existing `.pcd` or `.ply` file for the requested stem(s)."""
    for stem in stem_candidates:
        for ext in (".pcd", ".ply"):
            path = os.path.join(directory, stem + ext)
            if os.path.isfile(path):
                return path
    return os.path.join(directory, stem_candidates[0] + ".pcd")


def filtered_point_cloud_filename(raw_path):
    """Return the cleaned-point-cloud filename derived from the raw input name."""
    stem = os.path.splitext(os.path.basename(raw_path or ""))[0] or "GlobalMap"
    return f"filtered_{stem}.pcd"


def filtered_point_cloud_stem_candidates(raw_path):
    """Resolve new filtered filenames first, while still accepting legacy outputs."""
    new_stem = os.path.splitext(filtered_point_cloud_filename(raw_path))[0]
    candidates = [new_stem]
    if new_stem != "Filtered_GlobalMap":
        candidates.append("Filtered_GlobalMap")
    return candidates


def rewrite_nav2_yaml_image(yaml_text, image_name):
    """Update the YAML image entry so saved map bundles stay relocatable."""
    lines = []
    replaced = False
    for line in yaml_text.splitlines():
        if line.startswith("image:"):
            lines.append(f"image: {image_name}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f"image: {image_name}")
    return "\n".join(lines) + "\n"


def traversability_sidecar_path(pgm_path):
    stem, _ = os.path.splitext(pgm_path)
    return stem + "_traversable.pgm"


def floor_sidecar_path(pgm_path):
    stem, _ = os.path.splitext(pgm_path)
    return stem + "_floor.pgm"


def export_nav2_waypoints(waypoints_rc, yaml_data, output_path):
    """Export path planner waypoints as a Nav2-compatible YAML file.

    Args:
        waypoints_rc: list of (row, col) tuples from the planner
        yaml_data: dict with 'resolution' and 'origin' [x, y, ...]
        output_path: path to write the YAML file
    """
    res = float(yaml_data["resolution"])
    ox, oy = float(yaml_data["origin"][0]), float(yaml_data["origin"][1])
    h = yaml_data.get("height", 0)

    with open(output_path, "w") as f:
        f.write("waypoints:\n")
        for row, col in waypoints_rc:
            x = ox + col * res
            y = oy + (h - 1 - row) * res if h > 0 else oy + row * res
            f.write(f"  - position: {{x: {x:.4f}, y: {y:.4f}, z: 0.0}}\n")
            f.write(f"    orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}\n")
    return output_path


def export_inflated_costmap(inflated2d, yaml_data, output_prefix):
    """Export robot-inflated obstacle map as PGM+YAML for Nav2 costmap.

    Args:
        inflated2d: numpy array (h, w) with 0=free, 1=blocked (after inflation)
        yaml_data: dict with 'resolution' and 'origin'
        output_prefix: output path prefix (writes <prefix>.pgm and <prefix>.yaml)
    """
    from PIL import Image
    h, w = inflated2d.shape

    # Convert to Nav2 PGM: 0=occupied, 254=free, flipped vertically
    img = np.full((h, w), 254, dtype=np.uint8)
    img[inflated2d > 0] = 0
    img = np.flipud(img)

    pgm_path = output_prefix + ".pgm"
    Image.fromarray(img, mode='L').save(pgm_path)

    yaml_path = output_prefix + ".yaml"
    res = float(yaml_data["resolution"])
    ox, oy = float(yaml_data["origin"][0]), float(yaml_data["origin"][1])
    pgm_name = os.path.basename(pgm_path)
    with open(yaml_path, "w") as f:
        f.write(f"image: {pgm_name}\n")
        f.write(f"mode: trinary\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{ox}, {oy}, 0.0]\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: 0.65\n")
        f.write(f"free_thresh: 0.25\n")
    return pgm_path, yaml_path
