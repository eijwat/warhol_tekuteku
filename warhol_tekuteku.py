"""
テクテク × ウォーホル風ポップアートポスター v2
白背景＋水色ラインの画像を色分解して再着色
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import numpy as np
import os
from datetime import datetime

# --- カラーパレット定義 ---
WARHOL_PALETTES = [
    ("Marilyn (Pink)", {
        "bg": (255, 115, 180),
        "line": (30, 30, 30),
        "fill": (255, 220, 50),
    }),
    ("Marilyn (Turquoise)", {
        "bg": (0, 200, 200),
        "line": (100, 0, 120),
        "fill": (255, 240, 100),
    }),
    ("Marilyn (Orange)", {
        "bg": (255, 140, 0),
        "line": (0, 60, 0),
        "fill": (255, 255, 150),
    }),
    ("Campbell's Soup", {
        "bg": (200, 30, 30),
        "line": (255, 255, 255),
        "fill": (240, 220, 180),
    }),
    ("Banana (Velvet)", {
        "bg": (240, 230, 200),
        "line": (50, 50, 50),
        "fill": (255, 220, 0),
    }),
    ("Electric Chair", {
        "bg": (200, 0, 50),
        "line": (0, 0, 0),
        "fill": (255, 100, 100),
    }),
]

HOKUSAI_PALETTES = [
    ("神奈川沖浪裏", {
        "bg": (30, 60, 120),
        "line": (255, 255, 255),
        "fill": (100, 160, 210),
    }),
    ("凱風快晴 (赤富士)", {
        "bg": (180, 60, 40),
        "line": (250, 240, 220),
        "fill": (220, 180, 120),
    }),
    ("山下白雨", {
        "bg": (40, 40, 60),
        "line": (255, 220, 50),
        "fill": (120, 140, 180),
    }),
    ("甲州石班澤", {
        "bg": (60, 100, 140),
        "line": (240, 230, 210),
        "fill": (180, 200, 180),
    }),
    ("尾州不二見原", {
        "bg": (180, 160, 120),
        "line": (40, 60, 100),
        "fill": (100, 160, 220),
    }),
    ("諸国瀧廻り", {
        "bg": (50, 120, 100),
        "line": (255, 255, 255),
        "fill": (80, 180, 160),
    }),
]


def extract_masks(img_path, target_size):
    """
    テクテク画像からライン・塗り領域のマスクを抽出
    元画像: 白背景 + 水色(~R80 G185 B225)のライン
    """
    img = Image.open(img_path).convert("RGB")
    
    # パネル内のパディング
    padding = int(target_size * 0.08)
    available = target_size - padding * 2
    
    w, h = img.size
    ratio = min(available / w, available / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    # ライン検出: 青チャンネルが高く、赤が低い → 水色部分
    # グラデーション（アンチエイリアス）を考慮して連続値マスク
    # 純白(255,255,255)からの距離でライン度を計算
    # 水色のコアカラー: 約(80, 185, 225)
    
    # 「白さ」を計算 (0=完全に白, 1=完全に色付き)
    whiteness = np.sqrt(((255 - r)**2 + (255 - g)**2 + (255 - b)**2)) / np.sqrt(3 * 255**2)
    
    # 水色度を計算
    cyan_ref = np.array([80, 185, 225])
    dist_to_cyan = np.sqrt((r - cyan_ref[0])**2 + (g - cyan_ref[1])**2 + (b - cyan_ref[2])**2)
    max_dist = np.sqrt(3 * 255**2)
    cyan_similarity = 1.0 - (dist_to_cyan / max_dist)
    
    # ラインマスク: 白くなく、水色に近い
    line_strength = np.clip(whiteness * 3, 0, 1)  # 白から離れるほど強い
    line_mask_arr = (line_strength * 255).astype(np.uint8)
    
    line_mask = Image.fromarray(line_mask_arr, mode='L')
    
    return line_mask, (new_w, new_h)


def colorize_tekuteku(panel_size, line_mask, char_size, palette):
    """
    テクテクを指定パレットで着色
    手順:
    1. 背景色でパネルを塗る
    2. キャラの内部をfill色で塗る
    3. ラインをline色で描く
    """
    panel = Image.new("RGB", (panel_size, panel_size), palette["bg"])
    
    # キャラクター中央配置
    x_off = (panel_size - char_size[0]) // 2
    y_off = (panel_size - char_size[1]) // 2
    
    # ラインマスクをリサイズ
    lm = line_mask.resize(char_size, Image.LANCZOS)
    
    # 内部領域マスクを作る（ラインで囲まれた領域をfloodfill）
    # 手法: ラインを閾値化→外側をfloodfill→反転で内部を得る
    lm_arr = np.array(lm)
    thresh = 40  # ラインと見なす閾値
    binary_line = (lm_arr > thresh).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary_line, mode='L')
    
    # 外部領域をフラッドフィルで検出
    # ラインで閉じた領域の外側=背景
    fill_detect = Image.new("L", char_size, 0)
    fill_arr = np.array(binary_img)
    
    # scipy floodfillの代わりにPILのImageDraw.floodfillを使う
    from PIL import ImageDraw as ID
    
    # バイナリラインを白線・黒背景に変換（floodfill用）
    inverted = Image.fromarray(255 - fill_arr, mode='L')
    # 外側（角）からフラッドフィルして外側を白にする
    ID.floodfill(inverted, (0, 0), 255, thresh=10)
    ID.floodfill(inverted, (char_size[0]-1, 0), 255, thresh=10)
    ID.floodfill(inverted, (0, char_size[1]-1), 255, thresh=10)
    ID.floodfill(inverted, (char_size[0]-1, char_size[1]-1), 255, thresh=10)
    
    inv_arr = np.array(inverted)
    # 内部 = フラッドフィルで白にならなかった部分 AND ラインでない部分
    interior = (inv_arr < 128) & (fill_arr < thresh)
    interior_mask_arr = (interior * 255).astype(np.uint8)
    
    # 少し膨張させてラインとの隙間を埋める
    interior_img = Image.fromarray(interior_mask_arr, mode='L')
    interior_img = interior_img.filter(ImageFilter.MaxFilter(5))
    
    # パネルサイズのフルマスクを作成
    # 1) fill色レイヤー
    full_interior = Image.new("L", (panel_size, panel_size), 0)
    full_interior.paste(interior_img, (x_off, y_off))
    
    fill_layer = Image.new("RGB", (panel_size, panel_size), palette["fill"])
    panel = Image.composite(fill_layer, panel, full_interior)
    
    # 2) line色レイヤー
    full_line = Image.new("L", (panel_size, panel_size), 0)
    full_line.paste(lm, (x_off, y_off))
    
    line_layer = Image.new("RGB", (panel_size, panel_size), palette["line"])
    panel = Image.composite(line_layer, panel, full_line)
    
    return panel


def create_poster(input_path="tekuteku.png", output_dir="."):
    tekuteku_path = input_path
    panel_size = 600
    
    line_mask, char_size = extract_masks(tekuteku_path, panel_size)
    
    cols, rows = 4, 3
    all_panels = WARHOL_PALETTES + HOKUSAI_PALETTES  # 6+6 = 12
    
    gap = 6
    title_h = 130
    label_h = 55
    poster_w = cols * panel_size + (cols + 1) * gap
    poster_h = title_h + rows * (panel_size + label_h) + (rows + 1) * gap + 20
    
    poster = Image.new("RGB", (poster_w, poster_h), (20, 20, 20))
    draw = ImageDraw.Draw(poster)
    
    # フォント（日本語対応）
    jp_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    jp_font_reg = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    en_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        title_font = ImageFont.truetype(en_bold, 64)
        label_font = ImageFont.truetype(jp_font_reg, 20)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # タイトル
    title = "TEKUTEKU  POP  ART"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((poster_w - tw) // 2, 20), title, fill=(255, 255, 255), font=title_font)
    
    sub = "Warhol × Hokusai Color Palettes"
    bbox2 = draw.textbbox((0, 0), sub, font=label_font)
    sw = bbox2[2] - bbox2[0]
    draw.text(((poster_w - sw) // 2, 90), sub, fill=(160, 160, 160), font=label_font)
    
    # パネル配置
    for idx, (name, palette) in enumerate(all_panels):
        row = idx // cols
        col = idx % cols
        
        x = gap + col * (panel_size + gap)
        y = title_h + gap + row * (panel_size + label_h + gap)
        
        panel = colorize_tekuteku(panel_size, line_mask, char_size, palette)
        poster.paste(panel, (x, y))
        
        # ラベル
        # シリーズ判定
        is_hokusai = idx >= 6
        label_color = (100, 180, 255) if is_hokusai else (255, 100, 150)
        
        bbox_l = draw.textbbox((0, 0), name, font=label_font)
        lw = bbox_l[2] - bbox_l[0]
        lx = x + (panel_size - lw) // 2
        ly = y + panel_size + 8
        draw.text((lx, ly), name, fill=label_color, font=label_font)
        
        # カラースウォッチ
        swatch_y = ly + 26
        swatch_sz = 14
        sg = 6
        colors = [palette["bg"], palette["line"], palette["fill"]]
        total = len(colors) * (swatch_sz + sg) - sg
        sx = x + (panel_size - total) // 2
        for c in colors:
            draw.rectangle([sx, swatch_y, sx + swatch_sz, swatch_y + swatch_sz], fill=c, outline=(80, 80, 80))
            sx += swatch_sz + sg
    
    # クレジット
    credit = "Art: Eiji | Palettes: Warhol + Hokusai"
    bbox_c = draw.textbbox((0, 0), credit, font=label_font)
    cw = bbox_c[2] - bbox_c[0]
    draw.text(((poster_w - cw) // 2, poster_h - 30), credit, fill=(100, 100, 100), font=label_font)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"tekuteku_warhol_poster_{ts}.png")
    poster.save(output_path, quality=95, dpi=(150, 150))
    print(f"Saved: {output_path} ({poster.size[0]}x{poster.size[1]})")
    return output_path


def create_2x2_popart(style="mix", input_path="tekuteku.png", output_dir="."):
    """
    2×2ポップアート生成
    style: "warhol" = 洋風ランダム4枚
           "hokusai" = 和風ランダム4枚
           "mix" = 洋風2枚 + 和風2枚
    """
    import random

    tekuteku_path = input_path
    panel_size = 800

    line_mask, char_size = extract_masks(tekuteku_path, panel_size)

    # パレット選択
    if style == "warhol":
        chosen = random.sample(WARHOL_PALETTES, 4)
        title = "TEKUTEKU × WARHOL"
        series_labels = ["Warhol"] * 4
    elif style == "hokusai":
        chosen = random.sample(HOKUSAI_PALETTES, 4)
        title = "テクテク × 北斎"
        series_labels = ["Hokusai"] * 4
    else:
        w2 = random.sample(WARHOL_PALETTES, 2)
        h2 = random.sample(HOKUSAI_PALETTES, 2)
        chosen = w2 + h2
        random.shuffle(chosen)
        series_labels = []
        for name, _ in chosen:
            is_w = any(name == wn for wn, _ in WARHOL_PALETTES)
            series_labels.append("Warhol" if is_w else "Hokusai")
        title = "TEKUTEKU POP ART"

    gap = 6
    label_h = 50
    title_h = 100
    poster_w = 2 * panel_size + 3 * gap
    poster_h = title_h + 2 * (panel_size + label_h) + 3 * gap

    poster = Image.new("RGB", (poster_w, poster_h), (20, 20, 20))
    draw = ImageDraw.Draw(poster)

    # フォント
    jp_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    jp_reg = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    en_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        title_font = ImageFont.truetype(jp_bold, 52)
        label_font = ImageFont.truetype(jp_reg, 22)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # タイトル
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((poster_w - tw) // 2, 20), title, fill=(255, 255, 255), font=title_font)

    # パネル配置
    for idx, ((name, palette), series) in enumerate(zip(chosen, series_labels)):
        row, col = divmod(idx, 2)
        x = gap + col * (panel_size + gap)
        y = title_h + gap + row * (panel_size + label_h + gap)

        panel = colorize_tekuteku(panel_size, line_mask, char_size, palette)
        poster.paste(panel, (x, y))

        # ラベル
        label_color = (255, 100, 150) if series == "Warhol" else (100, 180, 255)
        bbox_l = draw.textbbox((0, 0), name, font=label_font)
        lw = bbox_l[2] - bbox_l[0]
        lx = x + (panel_size - lw) // 2
        ly = y + panel_size + 8
        draw.text((lx, ly), name, fill=label_color, font=label_font)

        # スウォッチ
        swatch_y = ly + 28
        swatch_sz = 14
        sg = 6
        colors = [palette["bg"], palette["line"], palette["fill"]]
        total = len(colors) * (swatch_sz + sg) - sg
        sx = x + (panel_size - total) // 2
        for c in colors:
            draw.rectangle([sx, swatch_y, sx + swatch_sz, swatch_y + swatch_sz],
                           fill=c, outline=(80, 80, 80))
            sx += swatch_sz + sg

    suffix = style
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"tekuteku_2x2_{suffix}_{ts}.png")
    poster.save(output_path, quality=95, dpi=(150, 150))
    print(f"Saved: {output_path} ({poster.size[0]}x{poster.size[1]})")
    names = [n for n, _ in chosen]
    print(f"Palettes: {names}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="テクテク ポップアートポスター")
    parser.add_argument("--mode", choices=["full", "warhol", "hokusai", "mix"],
                        default="full",
                        help="full=4x3全パレット, warhol=2x2洋風, hokusai=2x2和風, mix=2x2混合")
    parser.add_argument("--input", "-i", default="tekuteku.png",
                        help="入力画像パス (デフォルト: tekuteku.png)")
    parser.add_argument("--output", "-o", default=".",
                        help="出力ディレクトリ (デフォルト: カレント)")
    args = parser.parse_args()

    if args.mode == "full":
        create_poster(args.input, args.output)
    else:
        create_2x2_popart(args.mode, args.input, args.output)
