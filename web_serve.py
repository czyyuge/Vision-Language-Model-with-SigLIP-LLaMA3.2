from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import threading
import time

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ---- 全局任务状态（线程安全） ----
task_lock = threading.Lock()
task_status = {
    'running': False,
    'log': '',
    'result': None,
    'error': None,
    'html_page': None
}

def run_pipeline(cmd, output_path, html_page):
    """在后台线程中运行 run.py，实时捕获输出"""
    global task_status
    with task_lock:
        task_status['running'] = True
        task_status['log'] = ''
        task_status['result'] = None
        task_status['error'] = None
        task_status['html_page'] = html_page

    try:
        # 启动子进程，合并 stdout 和 stderr，使用行缓冲
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,          # 行缓冲
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}  # 强制子进程 utf-8 输出
        )
        # 逐行读取输出并更新全局日志
        for line in iter(proc.stdout.readline, ''):
            if line:  # 防止空行
                with task_lock:
                    task_status['log'] += line

        proc.stdout.close()
        return_code = proc.wait()

        with task_lock:
            task_status['running'] = False
            if return_code == 0:
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        task_status['result'] = json.load(f)
                except Exception as e:
                    task_status['error'] = f"读取结果文件失败: {str(e)}"
            else:
                task_status['error'] = task_status['log']  # 把完整日志当作错误信息
    except Exception as e:
        with task_lock:
            task_status['running'] = False
            task_status['error'] = str(e)


@app.route('/')
def index():
    return app.send_static_file('video_index_tool.html')

@app.route('/api/run', methods=['POST'])
def run_task():
    # ---- 防止重复启动 ----
    with task_lock:
        if task_status['running']:
            return jsonify({"status": "error", "msg": "当前已有任务在运行，请等待完成"}), 409

    if 'video' not in request.files:
        return jsonify({"status": "error", "msg": "未选择视频文件"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"status": "error", "msg": "文件名为空"}), 400

    # 收集表单参数（与之前一致）
    output_name = request.form.get('output', 'results.json')
    prompt = request.form.get('prompt', 'Describe this image in detail.')
    frames_dir = request.form.get('frames_dir', 'frames')
    output_html = request.form.get('output_html', 'viewer.html')
    port = request.form.get('port', '8000')
    llama_path = request.form.get('llama', './Llama-3.2-3B')
    checkpoint_path = request.form.get('checkpoint', './checkpoints/checkpoint_epoch0_step56000.pth')

    # 保存上传视频到 uploads/
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, video_file.filename)
    video_file.save(file_path)

    # 输出目录
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # 构建 run.py 命令行
    cmd = [
        "python", "-u", "run.py",   # -u 强制无缓冲输出，确保日志实时
        "--llama", llama_path,
        "--checkpoint", checkpoint_path,
        "-o", output_path,
        "--prompt", prompt,
        "--frames-dir", frames_dir,
        "--output-html", output_html,
        "--port", port,
        file_path
    ]

    # 启动后台线程
    thread = threading.Thread(
        target=run_pipeline,
        args=(cmd, output_path, output_html)
    )
    thread.start()

    return jsonify({
        "status": "started",
        "msg": "任务已开始，请等待..."
    })

@app.route('/api/status')
def get_status():
    """前端轮询此接口获取实时日志和任务状态"""
    with task_lock:
        return jsonify({
            "running": task_status['running'],
            "log": task_status['log'],
            "result": task_status['result'],
            "error": task_status['error'],
            "html_page": task_status['html_page']
        })

if __name__ == '__main__':
    # 启动后自动打开浏览器
    import webbrowser
    from threading import Timer
    Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)