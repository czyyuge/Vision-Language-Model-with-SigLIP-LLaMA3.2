import json
import os

def generate_viewer(results_path, frames_dir, output_path="viewer.html"):
    # 1. 读取 JSON 数据
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 将数据转为 JavaScript 字符串（注意特殊字符转义）
    data_json = json.dumps(data, ensure_ascii=False)

    # 3. 构建 HTML 模板（包含 CSS 和 JS 逻辑）
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频帧索引搜索</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ margin-bottom: 20px; color: #333; }}
        .search-box {{
            width: 100%; padding: 12px 16px; font-size: 16px;
            border: 2px solid #ddd; border-radius: 8px;
            margin-bottom: 20px; transition: 0.2s;
        }}
        .search-box:focus {{ border-color: #4a90d9; outline: none; }}
        .stats {{ color: #666; margin-bottom: 15px; }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden; transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-3px); }}
        .card img {{
            width: 100%; height: 180px; object-fit: cover;
            cursor: pointer; display: block;
        }}
        .card-body {{ padding: 12px; }}
        .time {{ font-size: 14px; color: #888; margin-bottom: 6px; }}
        .caption {{ font-size: 15px; color: #333; line-height: 1.4; }}
        .no-results {{ text-align: center; padding: 40px; color: #999; }}
        /* 全屏预览模态 */
        .modal {{
            display: none; position: fixed; z-index: 999; left: 0; top: 0;
            width: 100%; height: 100%; background: rgba(0,0,0,0.9);
            justify-content: center; align-items: center;
        }}
        .modal img {{ max-width: 90%; max-height: 90%; border-radius: 8px; }}
        .modal:target {{ display: flex; }}
        .close {{
            position: absolute; top: 20px; right: 40px;
            color: white; font-size: 40px; font-weight: bold;
            cursor: pointer; text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 视频帧索引检索</h1>
        <input type="text" class="search-box" id="search" placeholder="输入关键词搜索描述...">
        <div class="stats" id="stats"></div>
        <div class="gallery" id="gallery"></div>
        <div class="no-results" id="no-results" style="display:none;">没有匹配的结果</div>
        <div class="modal" id="modal">
            <a href="#" class="close">&times;</a>
            <img src="" id="modal-img" alt="放大图片">
        </div>
    </div>

    <script>
        // 所有数据（由 Python 生成时注入）
        const DATA = {data_json};
        const FRAMES_DIR = "{os.path.basename(frames_dir)}/"; // 图片目录名

        const searchInput = document.getElementById('search');
        const gallery = document.getElementById('gallery');
        const stats = document.getElementById('stats');
        const noResults = document.getElementById('no-results');
        const modal = document.getElementById('modal');
        const modalImg = document.getElementById('modal-img');

        function render(items) {{
            gallery.innerHTML = '';
            if (items.length === 0) {{
                noResults.style.display = 'block';
                stats.textContent = `共 0 条结果`;
                return;
            }}
            noResults.style.display = 'none';
            stats.textContent = `显示 ${{items.length}} / ${{DATA.length}} 条结果`;
            items.forEach(item => {{
                const card = document.createElement('div');
                card.className = 'card';
                const imgSrc = FRAMES_DIR + item.frame;  // 注意：frame 字段可能包含路径，需确保是纯文件名
                card.innerHTML = `
                    <img src="${{imgSrc}}" alt="Frame at ${{item.time_sec}}s" onclick="openModal('${{imgSrc}}')">
                    <div class="card-body">
                        <div class="time">🕒 ${{item.time_sec}} 秒</div>
                        <div class="caption">${{escapeHtml(item.caption)}}</div>
                    </div>
                `;
                gallery.appendChild(card);
            }});
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function openModal(src) {{
            modal.style.display = "flex";
            modalImg.src = src;
        }}

        // 点击关闭模态
        document.querySelector('.close').addEventListener('click', (e) => {{
            e.preventDefault();
            modal.style.display = "none";
        }});
        window.addEventListener('click', (e) => {{
            if (e.target === modal) modal.style.display = "none";
        }});

        // 搜索逻辑
        searchInput.addEventListener('input', (e) => {{
            const q = e.target.value.toLowerCase().trim();
            if (q === '') {{
                render(DATA);
            }} else {{
                const filtered = DATA.filter(item => item.caption.toLowerCase().includes(q));
                render(filtered);
            }}
        }});

        // 初始显示全部
        render(DATA);
    </script>
</body>
</html>"""

    # 4. 写入 HTML 文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ viewer.html 已生成，总帧数: {len(data)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成可搜索的视频帧索引页面")
    parser.add_argument("--results", default="results.json", help="results.json 路径")
    parser.add_argument("--frames-dir", default="frames", help="帧图片所在的目录")
    parser.add_argument("-o", "--output", default="viewer.html", help="输出的 HTML 文件")
    args = parser.parse_args()
    generate_viewer(args.results, args.frames_dir, args.output)