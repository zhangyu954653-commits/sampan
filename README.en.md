<div align="center">

# Sampan ⛵

**Live Chinese ⇄ Thai subtitles for video meetings**

[![Checks](https://github.com/zhangyu954653-commits/sampan/actions/workflows/check.yml/badge.svg)](../../actions/workflows/check.yml)
[![Release](https://img.shields.io/github/v/release/zhangyu954653-commits/sampan)](../../releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[Download](../../releases/latest)　·　[中文说明](README.md)

</div>

> **Sampan**: 舢板 in Chinese, เรือสำปั้น in Thai. The same word in all three languages, so it needs no translation.
>
> To *translate* literally means to carry across. A sampan is a small boat that carries meaning to the other bank.

Sampan runs alongside DingTalk, Tencent Meeting or Zoom and shows live subtitles: Chinese speech is translated into Thai, and Thai speech into Chinese. It is built for meetings between Chinese headquarters and Thai teams, where waiting for an interpreter slows down every exchange.

```
  your microphone ──┐                                     ┌─→ Chinese or Thai? (auto)
                    ├─→ segment ─→ speech-to-text ─→ translate ─→ big subtitles
  meeting audio ────┘   (VAD)                             └─→ ZH→TH / TH→ZH
```

## What it does

- **Hears both sides without extra hardware.** It records your microphone and captures the other side's voice directly from the speaker output (WASAPI loopback). You don't need a virtual audio cable or a phone held up to the mic.
- **Detects the language of each sentence** and translates it in the right direction.
- **Shows a large subtitle window** that you can screen-share. Alternatively, send a **private link** (protected by a random token that changes every session) so people can follow on their phones while you keep sharing slides.
- **Offers three speech recognition options:** Tencent Cloud, OpenAI Whisper, or local faster-whisper, which is free and works offline.
- **Translates with Claude** by default, with OpenAI as an option. A glossary and correction table handle company names and factory terms.
- **Keeps a transcript** and can draft bilingual meeting minutes (.md / .docx).
- **Costs about ¥6–7 (around US$1) per meeting hour** with the default model.

## Quick start

1. Use Windows 10 or 11 with Python 3.11 installed. Tick "Add python.exe to PATH" during installation.
2. Download `Sampan-vX.Y.Z-Windows.zip` from [Releases](../../releases/latest) and unzip it.
3. Double-click `启动.bat` ("start"). The first run sets up its environment (5–15 minutes) and creates `config.json`.
4. Add your Anthropic API key to `config.json`. Speech recognition runs locally without any key. If you want cloud recognition, also add your Tencent Cloud or OpenAI keys.

The interface and the full manual are in Chinese. See [README.md](README.md) for configuration, meeting setups and troubleshooting.

## Privacy

- **Audio:** with local recognition, audio never leaves your computer. With cloud recognition, audio segments are sent to the provider you chose (Tencent Cloud or OpenAI).
- **Text:** the recognised text is always sent to the translation API (Anthropic, or OpenAI if you switch).

For confidential meetings, use local recognition so that no audio leaves the machine. Your keys (`config.json`) and meeting records are excluded by `.gitignore`, so they are never added to this repository or to the release package.

## License

MIT
