import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import cv2

JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

CONNECTIONS = [
    ('head',           'left_shoulder'),
    ('head',           'right_shoulder'),
    ('left_shoulder',  'right_shoulder'),
    ('left_shoulder',  'left_elbow'),
    ('left_elbow',     'left_hand'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow',    'right_hand'),
    ('left_shoulder',  'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip',       'right_hip'),
    ('left_hip',       'left_knee'),
    ('left_knee',      'left_foot'),
    ('right_hip',      'right_knee'),
    ('right_knee',     'right_foot'),
]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} frames  |  "
          f"{len([c for c in df.columns if c.endswith('_x')])} joints")
    return df


def get_limits(df, margin=0.15):
    """
    Compute axis limits from data with a margin.
    Returns (x_min,x_max), (y_min,y_max), (z_min,z_max)

    Reference figure axes:
      X column → ax.set_xlim
      Y column → ax.set_ylim
      Z column → ax.set_zlim
    """
    def col_range(cols):
        vals = df[cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        c    = (vals.max() + vals.min()) / 2
        h    = (vals.max() - vals.min()) / 2 + margin
        return c - h, c + h

    x_cols = [f'{j}_x' for j in JOINTS if f'{j}_x' in df.columns]
    y_cols = [f'{j}_y' for j in JOINTS if f'{j}_y' in df.columns]
    z_cols = [f'{j}_z' for j in JOINTS if f'{j}_z' in df.columns]

    return col_range(x_cols), col_range(y_cols), col_range(z_cols)


def render_skeleton_video(csv_path, output_path=None, fps=30,
                          fig_size=(8, 8), dpi=100,
                          view_elev=20, view_azim=-60):
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.parent / (csv_path.stem + '_skeleton.mp4')
    output_path = Path(output_path)

    df       = load_data(csv_path)
    n_frames = len(df)

    # Axis limits — consistent across all frames
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_limits(df)

    print(f"\nAxis ranges:")
    print(f"  X (left/right): [{x_min:.3f}, {x_max:.3f}]")
    print(f"  Y (depth)     : [{y_min:.3f}, {y_max:.3f}]")
    print(f"  Z (up/down)   : [{z_min:.3f}, {z_max:.3f}]")

    frame_w = int(fig_size[0] * dpi)
    frame_h = int(fig_size[1] * dpi)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    print(f"\nRendering {n_frames} frames → {output_path.name}")

    fig = plt.figure(figsize=fig_size, dpi=dpi, facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')

    for frame_idx in range(n_frames):
        ax.cla()

        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        ax.grid(True, color='#dddddd', linewidth=0.5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X', fontsize=9, color='#333333')
        ax.set_ylabel('Y', fontsize=9, color='#333333')
        ax.set_zlabel('Z', fontsize=9, color='#333333')
        ax.tick_params(labelsize=7, colors='#555555')

        # View angle matching reference figure
        ax.view_init(elev=view_elev, azim=view_azim)

        row       = df.iloc[frame_idx]
        positions = {}
        for joint in JOINTS:
            xc = f'{joint}_x'
            yc = f'{joint}_y'
            zc = f'{joint}_z'
            if xc in df.columns and yc in df.columns and zc in df.columns:
                positions[joint] = (
                    float(row[xc]),   # Kinect x → plot X axis
                    float(row[zc]),   # Kinect y → plot Y axis
                    float(row[yc]),   # Kinect z → plot Z axis
                )

        # ax.plot(X, Y, Z) — direct: Kinect x→X, Kinect y→Y, Kinect z→Z
        for j_a, j_b in CONNECTIONS:
            if j_a in positions and j_b in positions:
                xa, ya, za = positions[j_a]
                xb, yb, zb = positions[j_b]
                ax.plot(
                    [xa, xb],   # X axis ← Kinect x
                    [ya, yb],   # Y axis ← Kinect y
                    [za, zb],   # Z axis ← Kinect z
                    color='steelblue', linewidth=2, alpha=0.8
                )

        for joint, (x, y, z) in positions.items():
            # Colour by body part — matching reference figure blue tones
            if joint == 'head':
                color, size = '#1f77b4', 80
            elif 'shoulder' in joint:
                color, size = '#2196F3', 60
            elif 'elbow' in joint:
                color, size = '#00BCD4', 50
            elif 'hand' in joint:
                color, size = '#26C6DA', 50
            elif 'hip' in joint:
                color, size = '#1565C0', 60
            elif 'knee' in joint:
                color, size = '#42A5F5', 50
            elif 'foot' in joint:
                color, size = '#90CAF9', 50
            else:
                color, size = '#1f77b4', 50

            ax.scatter(
                x, y, z,          # Kinect x→X, y→Y, z→Z
                color=color,
                s=size,
                zorder=5,
                depthshade=True
            )

        ax.set_title(
            f'3D Skeleton animation — frame {frame_idx + 1}/{n_frames}',
            fontsize=11, color='#222222', pad=10
        )

        fig.canvas.draw()
        try:
            buf      = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf      = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            buf      = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf      = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

        if frame_bgr.shape[:2] != (frame_h, frame_w):
            frame_bgr = cv2.resize(frame_bgr, (frame_w, frame_h))

        writer.write(frame_bgr)

        if (frame_idx + 1) % 30 == 0:
            print(f"  {frame_idx + 1}/{n_frames} frames done...")

    writer.release()
    plt.close(fig)
    print(f"\nSaved: {output_path}")
    return str(output_path)


def generate_from_pipeline_output(csv_path, fps=30):
    return render_skeleton_video(csv_path=csv_path, fps=fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='3D skeleton animation — axes match reference figure')
    parser.add_argument('--csv',    required=True,
                        help='Path to cut_3d_points.csv')
    parser.add_argument('--fps',    type=int,   default=30)
    parser.add_argument('--elev',   type=int,   default=20,
                        help='3D view elevation (default 20)')
    parser.add_argument('--azim',   type=int,   default=-60,
                        help='3D view azimuth (default -60)')
    parser.add_argument('--output', type=str,   default=None)
    args = parser.parse_args()

    render_skeleton_video(
        csv_path=args.csv,
        output_path=args.output,
        fps=args.fps,
        view_elev=args.elev,
        view_azim=args.azim,
    )