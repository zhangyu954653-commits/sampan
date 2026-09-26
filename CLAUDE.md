# Sampan：给 Claude 的项目规则

Sampan 是一个 Windows 桌面程序：抓取麦克风和系统声音（WASAPI 环回），做中泰语音识别，再用 Claude 翻译，把字幕显示在本地网页里。使用说明见 README.md，诊断脚本见 工具/说明.txt。

## 红线：公开仓库，任何时候都不能违反

- 不要提交 `config.json`（里面有三套密钥）、`会议记录/`，也不要提交任何会议内容。
- 代码、文档、示例里不要写真实密钥、真实公司名、客户名或人名。术语表示例只用通用词，比如 验厂、客诉。
- 推送前必须通过 `工具/开源前检查.py`。启用 `.githooks/pre-push` 后，推送时会自动运行这个检查。

## 改完代码怎么验证

- 本机（Windows）：`.venv\Scripts\python.exe 工具\自检.py`
- 云端或 Linux（没有声卡，也没有本地 Whisper 模型），按以下两步：
  1. 安装依赖：`pip install -r .github/ci/requirements.txt`
  2. 运行自检：`PYTHONPATH=.github/ci/stubs python 工具/自检.py`
- 每次推送和每个 PR，GitHub 都会自动跑同样的检查（`.github/workflows/check.yml`），另外还会在全新的 Windows 上跑一遍安装测试。
- 改到录音和回声相关的代码（`translator/audio_capture.py`、`translator/echo.py`）时，云端测不了真实声卡。PR 里要写明"需要在本机实测"。

## 容易踩的坑

- `translator/window.py` 的 `WINDOW_TITLE` 必须和 `translator/web/index.html` 的 `<title>` 一模一样：程序靠这个标题找到字幕窗口，并把它设为总在最前。改名要两处一起改。
- `启动.bat` 必须是 CRLF 换行，其余文件用 LF。`.gitattributes` 已经管好了，不要去改。
- `工具/自检.py` 只放纯逻辑测试：不联网，也不需要密钥。
- 图标有两份源文件：`assets/icon.svg` 用于 48 像素及以上，也是字幕窗口的图标；`assets/icon-small.svg` 用于 16 到 32 像素。改完运行 `.github/scripts/make_icon.py`，重新生成 `assets/sampan.ico`。改了 `icon.svg` 以后，要同步拷一份到 `translator/web/icon.svg`。

## 协作方式

- 提交信息用中文。所有改动都走 PR，不直接推到 main。
- 发版：在 Actions 页 → 发布 → Run workflow，填版本号（vX.Y.Z）。检查通过后，会自动打包 `Sampan-<版本>-Windows.zip` 并发布到 Releases。
- 面向用户的说明写在 README.md（中文）；README.en.md 只是英文简介，改功能时顺手同步。
