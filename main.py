#!/usr/bin/env python3
"""一鍵啟動：後端 FastAPI (port 8000) + 前端 Vite (port 5173)"""

import os
import sys
import subprocess
import time
import signal
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "web" / "backend"
FRONTEND_DIR = ROOT / "web" / "frontend"

procs = []

def stop_all(sig=None, frame=None):
    print("\n停止所有服務...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)


def find_node():
    """嘗試找到 node / npm（支援 nvm）"""
    nvm_dir = Path.home() / ".nvm"
    if nvm_dir.exists():
        # 找最新版本的 node
        versions = sorted((nvm_dir / "versions" / "node").glob("v*"), reverse=True)
        if versions:
            node_bin = versions[0] / "bin"
            os.environ["PATH"] = str(node_bin) + os.pathsep + os.environ["PATH"]
    # 確認 npm 可用
    result = subprocess.run(["npm", "--version"], capture_output=True)
    return result.returncode == 0


def install_frontend_deps():
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[前端] 安裝相依套件 (npm install)...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
        print("[前端] 安裝完成")


def start_backend():
    print("[後端] 啟動 FastAPI on http://localhost:8000")
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app",
         "--host", "0.0.0.0", "--port", "8000",
         "--loop", "asyncio", "--reload"],
        cwd=BACKEND_DIR,
    )
    procs.append(p)
    return p


def start_frontend():
    print("[前端] 啟動 Vite dev server on http://localhost:5173")
    p = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
    )
    procs.append(p)
    return p


def main():
    print("=" * 50)
    print("  台灣股票預測系統 — 一鍵啟動")
    print("=" * 50)

    if not find_node():
        print("[錯誤] 找不到 npm / node，請確認已安裝 Node.js")
        sys.exit(1)

    install_frontend_deps()

    start_backend()
    time.sleep(2)   # 等後端就緒
    start_frontend()
    time.sleep(3)   # 等前端就緒

    print()
    print("=" * 50)
    print("  前端網頁  →  http://localhost:5173")
    print("  後端 API  →  http://localhost:8000/api")
    print("  按 Ctrl+C 停止")
    print("=" * 50)

    webbrowser.open("http://localhost:5173")

    # 等待任一子程序結束
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"\n[警告] 某服務已意外結束 (exit code {p.returncode})")
                stop_all()
        time.sleep(1)


if __name__ == "__main__":
    main()
