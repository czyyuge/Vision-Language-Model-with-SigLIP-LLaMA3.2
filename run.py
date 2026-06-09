import argparse
import os
import sys
import time
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from extract_frames import extract_frames
from batch_inference import batch_inference
from generate_viewer import generate_viewer


def serve_and_open(port, root_dir, html_filename):
    """启动本地 HTTP 服务并在浏览器中打开索引页"""
    os.chdir(root_dir)                              # 确保服务器根目录正确
    server = HTTPServer(('', port), SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(1.0)                                 # 等待服务器就绪
    url = f"http://localhost:{port}/{html_filename}"
    print(f"\n🌐 正在浏览器中打开 {url}")
    webbrowser.open(url)

    print("✅ 服务已启动，按 Ctrl+C 停止")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🔌 关闭服务器...")
        server.shutdown()
        sys.exit(0)


def run(video_path, output_json, llama_path, checkpoint_path,
        frames_dir="frames", prompt=None,
        output_html="viewer.html", port=8000):
    # Step 1: extract frames
    print("=" * 50)
    print("Step 1/3: Extracting frames from video ...")
    print("=" * 50)
    extract_frames(video_path, frames_dir)

    # Step 2: batch inference
    print("\n" + "=" * 50)
    print("Step 2/3: Running VLM inference on frames ...")
    print("=" * 50)
    kwargs = {}
    if prompt:
        kwargs["prompt"] = prompt
    batch_inference(frames_dir, output_json, llama_path, checkpoint_path, **kwargs)

    # Step 3: generate searchable HTML viewer
    print("\n" + "=" * 50)
    print("Step 3/3: Generating interactive search page ...")
    print("=" * 50)
    generate_viewer(output_json, frames_dir, output_html)

    # 自动启动服务并打开浏览器
    serve_and_open(port, os.getcwd(), output_html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video → extract frames → VLM captions → interactive search page (one-click)"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON path")
    parser.add_argument("--llama", default="./Llama-3.2-3B",
                        help="Path to LLaMA-3.2-3B directory (default: ./Llama-3.2-3B)")
    parser.add_argument("--checkpoint", default="./checkpoints/checkpoint_epoch1_step28000.pth",
                        help="Path to model checkpoint .pth (default: ./checkpoints/checkpoint_epoch1_step28000.pth)")
    parser.add_argument("--frames-dir", default="frames", help="Temporary frames directory")
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Prompt for each frame")
    parser.add_argument("--output-html", default="viewer.html", help="Filename for the searchable HTML page")
    parser.add_argument("--port", type=int, default=8000, help="Local server port for preview")
    args = parser.parse_args()

    run(args.video, args.output, args.llama, args.checkpoint,
        args.frames_dir, args.prompt,
        args.output_html, args.port)