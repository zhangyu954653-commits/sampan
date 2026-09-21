@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 中泰实时翻译

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没有检测到 Python。请先到 https://www.python.org/downloads/ 安装 Python 3.10+
  echo        安装时记得勾选 "Add Python to PATH"
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] 首次运行，正在创建环境并安装依赖，大约需要 3-10 分钟...
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip -q
  pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败，请截图上面的报错信息。
    pause
    exit /b 1
  )
  echo [2/2] 安装完成。
) else (
  call ".venv\Scripts\activate.bat"
)

echo.
python main.py %*
pause
