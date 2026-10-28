"""用原 PyTorch 解释器启动；Web 依赖仅安装到 API 项目本地。"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="启动 Titanic 预测 API")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--deps-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    required = ["app.py", "predictor.py", "model.py", "assets/titanic_model.pth", "assets/preprocessor.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        parser.error("项目文件不完整，请完整解压。缺失：" + ", ".join(missing))
    if any(importlib.util.find_spec(name) is None for name in ("torch", "numpy")):
        parser.error("请使用原 PyTorch 环境的解释器，例如 E:\\Anaconda\\envs\\pytorch\\python.exe")
    deps = args.deps_dir.resolve() if args.deps_dir else root / ".api-deps"
    web_files = ["fastapi/__init__.py", "uvicorn/__init__.py", "pydantic/__init__.py", "pydantic_core/__init__.py", "python_multipart/__init__.py"]
    if not all((deps / name).is_file() for name in web_files):
        deps.mkdir(parents=True, exist_ok=True)
        temp = root / ".api-temp"
        temp.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update({"TEMP": str(temp), "TMP": str(temp)})
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--only-binary=:all:", "--upgrade", "--target", str(deps),
             "-r", str(root / "requirements.txt")], check=True, env=env,
        )
    sys.path[:0] = [str(root), str(deps)]
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
