"""从 assets/icon.svg 和 assets/icon-small.svg 重新生成 assets/sampan.ico。

    pip install playwright pillow && python -m playwright install chromium
    python .github/scripts/make_icon.py

两份 SVG 的分工：
  · icon-small.svg：16 / 24 / 32 像素用。去掉了字行和水波，
    帆和船身画得更大，否则缩到桌面小图标时会糊成一团。
  · icon.svg：48 像素及以上用，也是网页字幕窗口的图标（translator/web/icon.svg）。
改了 icon.svg，记得同步拷一份到 translator/web/icon.svg。
"""

from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
PLAN = {16: "icon-small", 24: "icon-small", 32: "icon-small",
        48: "icon", 64: "icon", 128: "icon", 256: "icon"}


def render() -> dict[int, Image.Image]:
    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for size, name in PLAN.items():
            svg = (ASSETS / f"{name}.svg").read_text(encoding="utf-8")
            svg = svg.replace('width="128" height="128"', f'width="{size}" height="{size}"')
            page = browser.new_page(viewport={"width": size, "height": size})
            page.set_content(f'<body style="margin:0;background:transparent">{svg}</body>')
            png = page.screenshot(omit_background=True,
                                  clip={"x": 0, "y": 0, "width": size, "height": size})
            page.close()
            out[size] = Image.open(BytesIO(png)).convert("RGBA")
        browser.close()
    return out


def main():
    imgs = render()
    sizes = sorted(imgs)
    target = ASSETS / "sampan.ico"
    imgs[256].save(target, format="ICO", sizes=[(s, s) for s in sizes],
                   append_images=[imgs[s] for s in sizes if s != 256])
    print(f"已生成 {target.relative_to(ROOT)}：{', '.join(str(s) for s in sizes)} 像素")


if __name__ == "__main__":
    main()
