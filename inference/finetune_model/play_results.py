import argparse
import os
import subprocess

FPS = 30


def make_video(folder):
    img_dir = os.path.join("results", folder)

    if not os.path.exists(img_dir):
        raise RuntimeError(f"Folder not found: {img_dir}")

    pattern = os.path.join(img_dir, "%05d_ex.jpg")

    mp4_path = os.path.join(img_dir, f"{folder}.mp4")
    gif_path = os.path.join(img_dir, f"{folder}.gif")
    palette_path = os.path.join(img_dir, "palette.png")

    # -----------------------------
    # MP4 생성
    # -----------------------------
    cmd_mp4 = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        mp4_path,
    ]

    subprocess.run(cmd_mp4, check=True)

    # -----------------------------
    # GIF palette 생성
    # -----------------------------
    cmd_palette = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", pattern,
        "-vf", "palettegen",
        palette_path,
    ]

    subprocess.run(cmd_palette, check=True)

    # -----------------------------
    # GIF 생성
    # -----------------------------
    cmd_gif = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", pattern,
        "-i", palette_path,
        "-lavfi", "paletteuse",
        gif_path,
    ]

    subprocess.run(cmd_gif, check=True)

    os.remove(palette_path)

    print(f"MP4 saved: {mp4_path}")
    print(f"GIF saved: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder inside results/")
    args = parser.parse_args()

    make_video(args.folder)