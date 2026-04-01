---
name: fix-3dgs-ply-export
description: How to diagnose and fix corrupted 3DGS Gaussian Splat PLY files for external viewer compatibility
---

# Fix 3DGS PLY Export for External Viewers

## When to Use This Skill

Use this skill when:
- A Gaussian Splatting app produces `.ply` files that appear corrupted in external viewers (SuperSplat, Luma, Antimatter15, etc.)
- PLY files open correctly in the app's own viewer but fail elsewhere
- File sizes don't match expectations based on vertex count × property count

## Diagnostic Steps

### 1. Analyze the PLY Header

PLY files have a text header followed by binary data. Read the header to understand the structure:

```python
from plyfile import PlyData

plydata = PlyData.read("path/to/file.ply")

# List all elements (standard PLY should have only "vertex")
for el in plydata.elements:
    print(f"{el.name}: {len(el.data)} entries")
    print(f"  Properties: {[p.name for p in el.properties]}")
```

### 2. Check for Non-Standard Elements

**Standard 3DGS PLY** has exactly **1 element**: `vertex` with these properties:
- Position: `x`, `y`, `z`
- Normals: `nx`, `ny`, `nz` (usually zeros, but required by many viewers)
- Spherical harmonics: `f_dc_0`, `f_dc_1`, `f_dc_2` (and optionally higher-order SH)
- Opacity: `opacity`
- Scale: `scale_0`, `scale_1`, `scale_2`
- Rotation: `rot_0`, `rot_1`, `rot_2`, `rot_3`

**If extra elements exist** (extrinsic, intrinsic, metadata, etc.), external viewers will misparse the binary data region, causing "corruption."

### 3. Verify Data Alignment

```python
import os

file_size = os.path.getsize("file.ply")
# After parsing header, compute:
header_size = ...  # bytes up to and including end_header\n
num_vertices = ...  # from "element vertex N"
bytes_per_vertex = sum(property_sizes)  # float=4, uchar=1, etc.

expected = header_size + (num_vertices * bytes_per_vertex) + supplementary_data
print(f"Expected: {expected}, Actual: {file_size}, Match: {expected == file_size}")
```

## The Fix: Export Standard PLY

### Core Conversion Function

```python
import numpy as np
from plyfile import PlyData, PlyElement

def save_standard_3dgs_ply(input_path, output_path):
    """Convert any 3DGS PLY to standard format compatible with all viewers."""
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    num_points = len(vertex.data)
    source_props = vertex.data.dtype.names

    # Standard property order with types
    standard_props = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    new_data = np.empty(num_points, dtype=standard_props)

    for name, _ in standard_props:
        if name in source_props:
            new_data[name] = vertex.data[name].astype(np.float32)
        else:
            new_data[name] = 0.0  # Normals default to zero

    # Write ONLY the vertex element — no supplementary elements
    vertex_element = PlyElement.describe(new_data, "vertex")
    PlyData([vertex_element], text=False).write(output_path)
```

### Key Rules

1. **Only ONE element**: The output must contain a single `vertex` element. Remove all others.
2. **Include normals**: Even if zeros, `nx/ny/nz` are expected by most viewers and the original 3DGS paper format.
3. **All float32**: Every property must be `float32` (`f4`). Some apps write `uchar` or `double` which breaks viewers.
4. **Property order matters**: Use the exact order listed above. Some viewers parse by index, not name.
5. **Binary little-endian**: Always use `binary_little_endian` format (default for `plyfile`).

### Integration Pattern for Pinokio Apps

When integrating with a Pinokio web UI:

```python
# In the app's file listing, prefer _standard.ply for downloads:
has_standard = any("_standard" in f and f.endswith(".ply") for f in files)
if has_standard:
    download_files = [f for f in files if not (f.endswith(".ply") and "_standard" not in f)]

# For the built-in 3D viewer, use the original (converted to Gradio format):
viewer_ply = next((f for f in files if f.endswith(".ply") and "_standard" not in f and "_gradio" not in f), None)
if viewer_ply:
    viewer_ply = convert_ply_for_gradio(viewer_ply)
```

## Common Root Causes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Viewer shows garbage/scrambled colors | Extra PLY elements shift binary offsets | Strip to vertex-only |
| Viewer refuses to load / "invalid file" | Missing normals (nx/ny/nz) | Add zero normals |
| File opens but looks wrong | Properties in non-standard order | Reorder to standard |
| Data type mismatch errors | Mixed float/uchar/double types | Convert all to float32 |
| Works in app A but not app B | App A handles multi-element, B doesn't | Export standard format |

## Verification

After converting, verify:
```python
plydata = PlyData.read(output_path)
assert len(plydata.elements) == 1, "Must have exactly 1 element"
assert plydata.elements[0].name == "vertex"
assert len(plydata.elements[0].properties) == 17  # xyz + normals + sh + opacity + scale + rot

# Size check
expected_data = len(plydata['vertex'].data) * 17 * 4  # 17 float32 props
header_end = ...  # find end_header offset
actual_data = os.path.getsize(output_path) - header_end
assert expected_data == actual_data, "Data size mismatch!"
```

## Real-World Example: ML-Sharp (Apple)

ML-Sharp writes PLY files with 8 elements (vertex + extrinsic, intrinsic, image_size, frame, disparity, color_space, version). These load fine in ML-Sharp's own renderer but break in SuperSplat, Luma, etc.

**Fix**: Added `save_standard_ply()` to `gaussians.py` that strips all supplementary elements and adds `nx/ny/nz`. The `_standard.ply` file is shown as the download, while the original is kept for ML-Sharp's internal renderer.
