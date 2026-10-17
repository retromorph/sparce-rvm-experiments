"""Catalog group ``misc``: Pillow, networkx, jinja2, pydantic and the Python
standard library.

Every :class:`ApiChange` below is a real, documented removal or rename.  Every
program was executed on both pins: ``old_code`` passes ``test_code`` on the old
pin and fails on the new pin, ``new_code`` passes on the new pin, and sibling
programs pass unchanged on both pins.

Code strings are plain strings; this module imports nothing third-party.
"""

from __future__ import annotations

import textwrap

from . import ApiChange, Program, SiblingProgram


def _d(code: str) -> str:
    return textwrap.dedent(code).strip("\n") + "\n"


def _fix(old: str, *pairs: tuple[str, str]) -> str:
    """Apply exact textual replacements to ``old``; every ``before`` must occur."""
    out = old
    for before, after in pairs:
        if before not in out:
            raise ValueError("fix text not found: %r" % (before,))
        out = out.replace(before, after)
    return out


def _p(name: str, old: str, new: str | None, test: str, description: str = "",
       fix: tuple[tuple[str, str], ...] = ()) -> Program:
    old_code = _d(old)
    new_code = _d(new) if new is not None else _fix(old_code, *fix)
    return Program(name=name, old_code=old_code, new_code=new_code,
                   test_code=_d(test), description=description)


def _s(name: str, code: str, test: str, description: str = "") -> SiblingProgram:
    return SiblingProgram(name=name, code=_d(code), test_code=_d(test),
                          description=description)


# ---------------------------------------------------------------------------
# Pillow 9.5.0 -> 10.4.0
# ---------------------------------------------------------------------------

_PIL_URL = "https://pillow.readthedocs.io/en/stable/releasenotes/10.0.0.html"
_PIL_DATE = "2023-07-01"
_PIL_OLD = "9.5.0"
_PIL_NEW = "10.4.0"

# Shared test preamble measuring text with APIs that exist on both pins.
_PIL_TEXT_TEST_HEAD = """
    import solution
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    def size_of(text):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    def msize_of(text, spacing=4):
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        return right - left, bottom - top
    def fsize_of(text):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
"""

_GRADIENT_HELPER = """
    def _gradient(width, height):
        im = Image.new("L", (width, height))
        im.putdata([(x * 255) // (width - 1) for _y in range(height) for x in range(width)])
        return im
"""

_PIL_ANTIALIAS = ApiChange(
    package="pillow",
    api_family="PIL.Image.ANTIALIAS",
    old_api="Image.ANTIALIAS",
    new_api="Image.LANCZOS",
    old_pin=_PIL_OLD,
    new_pin=_PIL_NEW,
    change_date=_PIL_DATE,
    change_url=_PIL_URL,
    summary="Pillow 10 removed the Image.ANTIALIAS constant; use Image.LANCZOS (Image.Resampling.LANCZOS).",
    programs=(
        _p(
            "thumbnail_size",
            """
            from PIL import Image

            def thumbnail_size(width, height, max_side, color=(200, 30, 30)):
                im = Image.new("RGB", (width, height), color)
                im.thumbnail((max_side, max_side), Image.ANTIALIAS)
                return im.size, im.getpixel((0, 0))
            """,
            None,
            """
            import solution
            assert solution.thumbnail_size(64, 32, 16) == ((16, 8), (200, 30, 30))
            assert solution.thumbnail_size(20, 40, 10, (0, 0, 255)) == ((5, 10), (0, 0, 255))
            assert solution.thumbnail_size(8, 8, 16) == ((8, 8), (200, 30, 30))
            """,
            "Thumbnail an in-memory image and report its size and colour.",
            fix=(("Image.ANTIALIAS", "Image.LANCZOS"),),
        ),
        _p(
            "downsample_gradient",
            """
            from PIL import Image
            """ + _GRADIENT_HELPER + """
            def downsample_gradient(width, height, new_width, new_height):
                small = _gradient(width, height).resize((new_width, new_height), Image.ANTIALIAS)
                return list(small.getdata())
            """,
            None,
            """
            import solution
            assert solution.downsample_gradient(16, 4, 4, 2) == [26, 92, 163, 229, 26, 92, 163, 229]
            assert solution.downsample_gradient(16, 4, 8, 8)[:8] == [9, 42, 77, 110, 144, 178, 213, 246]
            assert solution.downsample_gradient(16, 2, 16, 2)[:3] == [0, 17, 34]
            """,
            "Resample a horizontal gradient with the antialias filter.",
            fix=(("Image.ANTIALIAS", "Image.LANCZOS"),),
        ),
        _p(
            "icon_set",
            """
            from PIL import Image

            class IconSet:
                def __init__(self, size=8, ink=200):
                    self.base = Image.new("L", (size, size), 0)
                    for y in range(size):
                        for x in range(size // 2, size):
                            self.base.putpixel((x, y), ink)

                def render(self, side):
                    icon = self.base.resize((side, side), Image.ANTIALIAS)
                    return [icon.getpixel((x, 0)) for x in range(side)]
            """,
            None,
            """
            import solution
            icons = solution.IconSet()
            assert icons.render(4) == [0, 10, 190, 203]
            assert icons.render(8) == [0, 0, 0, 0, 200, 200, 200, 200]
            assert icons.render(2) == [10, 190]
            """,
            "Render a two-tone icon at several sizes.",
            fix=(("Image.ANTIALIAS", "Image.LANCZOS"),),
        ),
        _p(
            "fit_within",
            """
            from PIL import Image

            def fit_within(width, height, box_w, box_h):
                scale = min(box_w / width, box_h / height, 1.0)
                target = (max(1, round(width * scale)), max(1, round(height * scale)))
                im = Image.new("RGB", (width, height), (1, 2, 3))
                out = im.resize(target, resample=Image.ANTIALIAS)
                return out.size, out.mode, out.getpixel((0, 0))
            """,
            None,
            """
            import solution
            assert solution.fit_within(100, 50, 20, 20) == ((20, 10), "RGB", (1, 2, 3))
            assert solution.fit_within(10, 10, 50, 50) == ((10, 10), "RGB", (1, 2, 3))
            assert solution.fit_within(30, 90, 30, 30) == ((10, 30), "RGB", (1, 2, 3))
            """,
            "Shrink an image to fit inside a bounding box without upscaling.",
            fix=(("Image.ANTIALIAS", "Image.LANCZOS"),),
        ),
        _p(
            "resample_signal",
            """
            from PIL import Image

            def resample_signal(values, count):
                strip = Image.new("L", (len(values), 1))
                strip.putdata(list(values))
                return list(strip.resize((count, 1), Image.ANTIALIAS).getdata())
            """,
            None,
            """
            import solution
            ramp = [(x * 255) // 15 for x in range(16)]
            assert solution.resample_signal(ramp, 4) == [26, 92, 163, 229]
            assert solution.resample_signal([100] * 8, 2) == [100, 100]
            assert solution.resample_signal([0, 0, 0, 0, 200, 200, 200, 200], 4) == [0, 10, 190, 203]
            """,
            "Treat a 1-D list as a one-row image and resample it.",
            fix=(("Image.ANTIALIAS", "Image.LANCZOS"),),
        ),
        _p(
            "contact_sheet",
            """
            from PIL import Image

            def contact_sheet(colors, tile=4):
                sheet = Image.new("RGB", (tile * len(colors), tile), (0, 0, 0))
                for i, color in enumerate(colors):
                    img = Image.new("RGB", (tile * 3, tile * 2), color)
                    sheet.paste(img.resize((tile, tile), Image.ANTIALIAS), (i * tile, 0))
                return sheet.size, [sheet.getpixel((i * tile + 1, 1)) for i in range(len(colors))]
            """,
            None,
            """
            import solution
            size, px = solution.contact_sheet([(255, 0, 0), (0, 255, 0)])
            assert size == (8, 4)
            assert px == [(255, 0, 0), (0, 255, 0)]
            size, px = solution.contact_sheet([(9, 9, 9)], tile=2)
            assert size == (2, 2) and px == [(9, 9, 9)]
            """,
            "Paste resized colour tiles into a contact sheet.",
            fix=(("Image.ANTIALIAS", "Image.LANCZOS"),),
        ),
    ),
    siblings=(
        _s(
            "nearest_downsample",
            """
            from PIL import Image
            """ + _GRADIENT_HELPER + """
            def nearest_downsample(width, height, new_width, new_height):
                return list(_gradient(width, height).resize((new_width, new_height), Image.NEAREST).getdata())
            """,
            """
            import solution
            assert solution.nearest_downsample(16, 4, 4, 2) == [34, 102, 170, 238, 34, 102, 170, 238]
            assert solution.nearest_downsample(16, 4, 8, 8)[:8] == [17, 51, 85, 119, 153, 187, 221, 255]
            """,
            "Image.NEAREST is unchanged.",
        ),
        _s(
            "box_thumbnail",
            """
            from PIL import Image
            """ + _GRADIENT_HELPER + """
            def box_thumbnail(width, height, new_width, new_height):
                im = _gradient(width, height)
                return list(im.resize((new_width, new_height), Image.Resampling.BOX).getdata())
            """,
            """
            import solution
            assert solution.box_thumbnail(16, 4, 4, 2) == [26, 94, 162, 230, 26, 94, 162, 230]
            assert solution.box_thumbnail(16, 4, 8, 8)[:8] == [9, 43, 77, 111, 145, 179, 213, 247]
            """,
            "Image.Resampling.BOX exists on both pins.",
        ),
    ),
)

_PIL_LINEAR = ApiChange(
    package="pillow",
    api_family="PIL.Image.LINEAR",
    old_api="Image.LINEAR / Image.CUBIC",
    new_api="Image.BILINEAR / Image.BICUBIC",
    old_pin=_PIL_OLD,
    new_pin=_PIL_NEW,
    change_date=_PIL_DATE,
    change_url=_PIL_URL,
    summary="Pillow 10 removed the Image.LINEAR and Image.CUBIC aliases; use Image.BILINEAR and Image.BICUBIC.",
    programs=(
        _p(
            "upscale_linear",
            """
            from PIL import Image
            """ + _GRADIENT_HELPER + """
            def upscale_linear(width, height, factor):
                big = _gradient(width, height).resize((width * factor, height * factor), Image.LINEAR)
                return [big.getpixel((x, 0)) for x in range(width * factor)]
            """,
            None,
            """
            import solution
            assert solution.upscale_linear(16, 4, 2)[:8] == [12, 43, 77, 111, 145, 179, 213, 243]
            assert solution.upscale_linear(2, 1, 2) == [0, 64, 191, 255]
            """,
            "Enlarge a gradient with linear interpolation.",
            fix=(("Image.LINEAR", "Image.BILINEAR"),),
        ),
        _p(
            "smooth_downscale",
            """
            from PIL import Image

            def smooth_downscale(rows, new_width, new_height):
                im = Image.new("L", (len(rows[0]), len(rows)))
                im.putdata([v for row in rows for v in row])
                return list(im.resize((new_width, new_height), Image.CUBIC).getdata())
            """,
            None,
            """
            import solution
            rows = [[(x * 255) // 15 for x in range(16)] for _ in range(4)]
            assert solution.smooth_downscale(rows, 4, 2) == [27, 92, 163, 228, 27, 92, 163, 228]
            assert solution.smooth_downscale([[7] * 6] * 6, 3, 3) == [7] * 9
            """,
            "Shrink pixel rows with cubic interpolation.",
            fix=(("Image.CUBIC", "Image.BICUBIC"),),
        ),
        _p(
            "rotate_quadrants",
            """
            from PIL import Image

            def _quadrants(side):
                im = Image.new("L", (side, side))
                half = side // 2
                for y in range(side):
                    for x in range(side):
                        im.putpixel((x, y), 10 + 10 * (x >= half) + 20 * (y >= half))
                return im

            def rotate_quadrants(side, angle):
                out = _quadrants(side).rotate(angle, resample=Image.CUBIC)
                last = side - 1
                return [out.getpixel(p) for p in ((0, 0), (last, 0), (0, last), (last, last))]
            """,
            None,
            """
            import solution
            assert solution.rotate_quadrants(4, 0) == [10, 20, 30, 40]
            assert solution.rotate_quadrants(4, 90) == [20, 40, 10, 30]
            assert solution.rotate_quadrants(4, 180) == [40, 30, 20, 10]
            assert solution.rotate_quadrants(6, 270) == [30, 10, 40, 20]
            """,
            "Rotate a four-colour image and sample its corners.",
            fix=(("Image.CUBIC", "Image.BICUBIC"),),
        ),
        _p(
            "scale_half_linear",
            """
            from PIL import Image

            def scale_half_linear(side, ink):
                im = Image.new("L", (side, side), 0)
                for y in range(side):
                    for x in range(side // 2, side):
                        im.putpixel((x, y), ink)
                out = im.resize((side // 2, side // 2), resample=Image.LINEAR)
                return [out.getpixel((x, 0)) for x in range(side // 2)]
            """,
            None,
            """
            import solution
            assert solution.scale_half_linear(8, 200) == [0, 25, 175, 200]
            assert solution.scale_half_linear(4, 100) == [12, 88]
            """,
            "Halve a two-tone square with linear filtering.",
            fix=(("Image.LINEAR", "Image.BILINEAR"),),
        ),
        _p(
            "zoom_region",
            """
            from PIL import Image

            def zoom_region(side, box, out_side):
                im = Image.new("L", (side, side))
                half = side // 2
                for y in range(side):
                    for x in range(side):
                        im.putpixel((x, y), 10 + 10 * (x >= half) + 20 * (y >= half))
                zoomed = im.resize((out_side, out_side), Image.CUBIC, box=box)
                return sorted(set(zoomed.getdata())), zoomed.size
            """,
            None,
            """
            import solution
            assert solution.zoom_region(8, (0, 0, 4, 4), 4) == ([10], (4, 4))
            assert solution.zoom_region(8, (4, 4, 8, 8), 6) == ([40], (6, 6))
            assert solution.zoom_region(8, (4, 0, 8, 4), 2) == ([20], (2, 2))
            """,
            "Crop-and-zoom a quadrant via the resize box argument.",
            fix=(("Image.CUBIC", "Image.BICUBIC"),),
        ),
        _p(
            "resizer_quality",
            """
            from PIL import Image
            """ + _GRADIENT_HELPER + """
            class Resizer:
                def __init__(self, quality):
                    self.filter = Image.LINEAR if quality == "fast" else Image.CUBIC

                def apply(self, width, height, new_width, new_height):
                    out = _gradient(width, height).resize((new_width, new_height), self.filter)
                    return list(out.getdata())[:new_width]
            """,
            None,
            """
            import solution
            assert solution.Resizer("fast").apply(16, 4, 4, 2) == [32, 94, 162, 223]
            assert solution.Resizer("best").apply(16, 4, 4, 2) == [27, 92, 163, 228]
            """,
            "Pick a resampling filter from a quality setting.",
            fix=(("Image.LINEAR", "Image.BILINEAR"), ("Image.CUBIC", "Image.BICUBIC")),
        ),
    ),
    siblings=(
        _s(
            "hamming_downscale",
            """
            from PIL import Image
            """ + _GRADIENT_HELPER + """
            def hamming_downscale(width, height, new_width, new_height):
                return list(_gradient(width, height).resize((new_width, new_height), Image.HAMMING).getdata())
            """,
            """
            import solution
            assert solution.hamming_downscale(16, 4, 4, 2) == [28, 94, 162, 227, 28, 94, 162, 227]
            """,
            "Image.HAMMING is unchanged.",
        ),
        _s(
            "mirror_corners",
            """
            from PIL import Image

            def mirror_corners(side):
                im = Image.new("L", (side, side))
                half = side // 2
                for y in range(side):
                    for x in range(side):
                        im.putpixel((x, y), 10 + 10 * (x >= half) + 20 * (y >= half))
                out = im.transpose(Image.FLIP_LEFT_RIGHT)
                last = side - 1
                return [out.getpixel(p) for p in ((0, 0), (last, 0), (0, last), (last, last))]
            """,
            """
            import solution
            assert solution.mirror_corners(4) == [20, 10, 40, 30]
            assert solution.mirror_corners(6) == [20, 10, 40, 30]
            """,
            "Image.FLIP_LEFT_RIGHT is unchanged.",
        ),
    ),
)

_PIL_TEXTSIZE = ApiChange(
    package="pillow",
    api_family="PIL.ImageDraw.ImageDraw.textsize",
    old_api="ImageDraw.ImageDraw.textsize",
    new_api="ImageDraw.ImageDraw.textbbox",
    old_pin=_PIL_OLD,
    new_pin=_PIL_NEW,
    change_date=_PIL_DATE,
    change_url=_PIL_URL,
    summary="Pillow 10 removed ImageDraw.textsize; use textbbox (width = right - left, height = bottom - top).",
    programs=(
        _p(
            "center_x",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def center_x(canvas_width, text):
                draw = ImageDraw.Draw(Image.new("L", (canvas_width, 20)))
                text_w, _text_h = draw.textsize(text, font=FONT)
                return (canvas_width - text_w) // 2
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def center_x(canvas_width, text):
                draw = ImageDraw.Draw(Image.new("L", (canvas_width, 20)))
                left, _top, right, _bottom = draw.textbbox((0, 0), text, font=FONT)
                text_w = right - left
                return (canvas_width - text_w) // 2
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.center_x(200, "hello") == (200 - size_of("hello")[0]) // 2
            assert solution.center_x(100, "") == 50
            assert solution.center_x(300, "wide label") < solution.center_x(300, "hi")
            """,
            "Horizontal offset that centres a text string.",
        ),
        _p(
            "wrap_words",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def wrap_words(text, max_width):
                draw = ImageDraw.Draw(Image.new("1", (1, 1)))
                lines, current = [], ""
                for word in text.split():
                    candidate = (current + " " + word).strip()
                    if current and draw.textsize(candidate, font=FONT)[0] > max_width:
                        lines.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    lines.append(current)
                return lines
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def wrap_words(text, max_width):
                draw = ImageDraw.Draw(Image.new("1", (1, 1)))
                lines, current = [], ""
                for word in text.split():
                    candidate = (current + " " + word).strip()
                    if current and draw.textlength(candidate, font=FONT) > max_width:
                        lines.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    lines.append(current)
                return lines
            """,
            _PIL_TEXT_TEST_HEAD + """
            def reference(text, max_width):
                lines, current = [], ""
                for word in text.split():
                    candidate = (current + " " + word).strip()
                    if current and size_of(candidate)[0] > max_width:
                        lines.append(current)
                        current = word
                    else:
                        current = candidate
                if current:
                    lines.append(current)
                return lines
            text = "the quick brown fox jumps over the lazy dog"
            for width in (40, 60, 90):
                assert solution.wrap_words(text, width) == reference(text, width)
            assert solution.wrap_words("a b", 1000) == ["a b"]
            assert len(solution.wrap_words(text, 40)) > len(solution.wrap_words(text, 90))
            """,
            "Greedy word wrapping by rendered width.",
        ),
        _p(
            "label_box",
            """
            from PIL import Image, ImageDraw, ImageFont

            class Label:
                font = ImageFont.load_default()

                def __init__(self, text, padding=2):
                    self.text = text
                    self.padding = padding

                def box_size(self):
                    draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
                    w, h = draw.textsize(self.text, font=self.font)
                    return w + 2 * self.padding, h + 2 * self.padding
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            class Label:
                font = ImageFont.load_default()

                def __init__(self, text, padding=2):
                    self.text = text
                    self.padding = padding

                def box_size(self):
                    draw = ImageDraw.Draw(Image.new("RGB", (4, 4)))
                    left, top, right, bottom = draw.textbbox((0, 0), self.text, font=self.font)
                    w, h = right - left, bottom - top
                    return w + 2 * self.padding, h + 2 * self.padding
            """,
            _PIL_TEXT_TEST_HEAD + """
            w, h = size_of("Submit")
            assert solution.Label("Submit").box_size() == (w + 4, h + 4)
            w, h = size_of("OK")
            assert solution.Label("OK", padding=5).box_size() == (w + 10, h + 10)
            assert solution.Label("longer text").box_size()[0] > solution.Label("OK").box_size()[0]
            """,
            "Padded bounding box for a UI label.",
        ),
        _p(
            "truncate_to_width",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def truncate_to_width(text, max_width):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                while text and draw.textsize(text, font=FONT)[0] > max_width:
                    text = text[:-1]
                return text
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def truncate_to_width(text, max_width):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                while text and draw.textbbox((0, 0), text, font=FONT)[2] > max_width:
                    text = text[:-1]
                return text
            """,
            _PIL_TEXT_TEST_HEAD + """
            def reference(text, max_width):
                while text and size_of(text)[0] > max_width:
                    text = text[:-1]
                return text
            for text, width in (("abcdefghij", 30), ("hello world", 45), ("x", 0), ("short", 500)):
                assert solution.truncate_to_width(text, width) == reference(text, width)
            assert solution.truncate_to_width("short", 500) == "short"
            assert solution.truncate_to_width("abcdefghij", 30) != "abcdefghij"
            """,
            "Drop trailing characters until the text fits a width.",
        ),
        _p(
            "render_badge",
            """
            from PIL import Image, ImageDraw, ImageFont

            def render_badge(text, pad=3):
                font = ImageFont.load_default()
                probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                w, h = probe.textsize(text, font=font)
                badge = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (30, 30, 30))
                ImageDraw.Draw(badge).text((pad, pad), text, fill=(255, 255, 255), font=font)
                return badge.size, badge.getpixel((0, 0))
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            def render_badge(text, pad=3):
                font = ImageFont.load_default()
                probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
                w, h = right - left, bottom - top
                badge = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (30, 30, 30))
                ImageDraw.Draw(badge).text((pad, pad), text, fill=(255, 255, 255), font=font)
                return badge.size, badge.getpixel((0, 0))
            """,
            _PIL_TEXT_TEST_HEAD + """
            w, h = size_of("beta")
            assert solution.render_badge("beta") == ((w + 6, h + 6), (30, 30, 30))
            w, h = size_of("v2")
            assert solution.render_badge("v2", pad=1) == ((w + 2, h + 2), (30, 30, 30))
            """,
            "Draw a text badge sized to its content.",
        ),
        _p(
            "widest_label",
            """
            from PIL import Image, ImageDraw, ImageFont

            def widest_label(labels):
                font = ImageFont.load_default()
                draw = ImageDraw.Draw(Image.new("L", (2, 2)))
                widths = {label: draw.textsize(label, font=font)[0] for label in labels}
                best = max(labels, key=lambda label: (widths[label], label))
                return best, widths[best]
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            def widest_label(labels):
                font = ImageFont.load_default()
                draw = ImageDraw.Draw(Image.new("L", (2, 2)))
                widths = {label: draw.textbbox((0, 0), label, font=font)[2] for label in labels}
                best = max(labels, key=lambda label: (widths[label], label))
                return best, widths[best]
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.widest_label(["a", "abc", "ab"]) == ("abc", size_of("abc")[0])
            assert solution.widest_label(["name", "e-mail address", "id"]) == ("e-mail address", size_of("e-mail address")[0])
            """,
            "Find the label that renders widest.",
        ),
    ),
    siblings=(
        _s(
            "text_advance",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def text_advance(text):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                return int(draw.textlength(text, font=FONT))
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.text_advance("hello") == size_of("hello")[0]
            assert solution.text_advance("") == 0
            assert solution.text_advance("hello world") > solution.text_advance("hello")
            """,
            "ImageDraw.textlength exists on both pins.",
        ),
        _s(
            "text_box",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def text_box(text, x, y):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                return draw.textbbox((x, y), text, font=FONT)
            """,
            _PIL_TEXT_TEST_HEAD + """
            w, h = size_of("box")
            left, top, right, bottom = solution.text_box("box", 5, 7)
            assert right - left == w and bottom - top == h
            assert left == 5 + draw.textbbox((0, 0), "box", font=font)[0]
            """,
            "ImageDraw.textbbox is unchanged.",
        ),
    ),
)

_PIL_GETSIZE = ApiChange(
    package="pillow",
    api_family="PIL.ImageFont.FreeTypeFont.getsize",
    old_api="ImageFont.ImageFont.getsize / FreeTypeFont.getsize",
    new_api="ImageFont.FreeTypeFont.getbbox",
    old_pin=_PIL_OLD,
    new_pin=_PIL_NEW,
    change_date=_PIL_DATE,
    change_url=_PIL_URL,
    summary="Pillow 10 removed the font getsize() method; use getbbox() (or getlength()).",
    programs=(
        _p(
            "text_width",
            """
            from PIL import ImageFont

            FONT = ImageFont.load_default()

            def text_width(text):
                return FONT.getsize(text)[0]
            """,
            """
            from PIL import ImageFont

            FONT = ImageFont.load_default()

            def text_width(text):
                left, _top, right, _bottom = FONT.getbbox(text)
                return right - left
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.text_width("hello") == fsize_of("hello")[0]
            assert solution.text_width("") == 0
            assert solution.text_width("hello there") > solution.text_width("hello")
            """,
            "Pixel width of a string in the default font.",
        ),
        _p(
            "columns_that_fit",
            """
            from PIL import ImageFont

            def columns_that_fit(labels, total_width, gap=4):
                font = ImageFont.load_default()
                used, count = 0, 0
                for label in labels:
                    w = font.getsize(label)[0]
                    if used + w > total_width:
                        break
                    used += w + gap
                    count += 1
                return count
            """,
            """
            from PIL import ImageFont

            def columns_that_fit(labels, total_width, gap=4):
                font = ImageFont.load_default()
                used, count = 0, 0
                for label in labels:
                    left, _top, right, _bottom = font.getbbox(label)
                    w = right - left
                    if used + w > total_width:
                        break
                    used += w + gap
                    count += 1
                return count
            """,
            _PIL_TEXT_TEST_HEAD + """
            def reference(labels, total_width, gap=4):
                used, count = 0, 0
                for label in labels:
                    w = fsize_of(label)[0]
                    if used + w > total_width:
                        break
                    used += w + gap
                    count += 1
                return count
            labels = ["id", "name", "email", "created", "status"]
            for width in (10, 40, 80, 1000):
                assert solution.columns_that_fit(labels, width) == reference(labels, width)
            assert solution.columns_that_fit(labels, 1000) == 5
            assert solution.columns_that_fit(labels, 0) == 0
            """,
            "Count how many column headers fit in a row.",
        ),
        _p(
            "char_widths",
            """
            from PIL import ImageFont

            def char_widths(text):
                font = ImageFont.load_default()
                return [font.getsize(ch)[0] for ch in text]
            """,
            """
            from PIL import ImageFont

            def char_widths(text):
                font = ImageFont.load_default()
                return [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.char_widths("Wi") == [fsize_of("W")[0], fsize_of("i")[0]]
            assert solution.char_widths("") == []
            assert len(solution.char_widths("abc")) == 3
            """,
            "Per-character advance widths.",
        ),
        _p(
            "text_meter",
            """
            from PIL import ImageFont

            class TextMeter:
                def __init__(self):
                    self.font = ImageFont.load_default()

                def measure(self, text):
                    w, h = self.font.getsize(text)
                    return {"width": w, "height": h, "area": w * h}
            """,
            """
            from PIL import ImageFont

            class TextMeter:
                def __init__(self):
                    self.font = ImageFont.load_default()

                def measure(self, text):
                    left, top, right, bottom = self.font.getbbox(text)
                    w, h = right - left, bottom - top
                    return {"width": w, "height": h, "area": w * h}
            """,
            _PIL_TEXT_TEST_HEAD + """
            w, h = fsize_of("metric")
            assert solution.TextMeter().measure("metric") == {"width": w, "height": h, "area": w * h}
            w, h = fsize_of("A")
            assert solution.TextMeter().measure("A")["area"] == w * h
            """,
            "Measure width, height and area of text.",
        ),
        _p(
            "longest_prefix_fitting",
            """
            from PIL import ImageFont

            FONT = ImageFont.load_default()

            def longest_prefix_fitting(text, max_width):
                best = ""
                for end in range(1, len(text) + 1):
                    if FONT.getsize(text[:end])[0] <= max_width:
                        best = text[:end]
                    else:
                        break
                return best
            """,
            """
            from PIL import ImageFont

            FONT = ImageFont.load_default()

            def longest_prefix_fitting(text, max_width):
                best = ""
                for end in range(1, len(text) + 1):
                    if FONT.getbbox(text[:end])[2] <= max_width:
                        best = text[:end]
                    else:
                        break
                return best
            """,
            _PIL_TEXT_TEST_HEAD + """
            def reference(text, max_width):
                best = ""
                for end in range(1, len(text) + 1):
                    if fsize_of(text[:end])[0] <= max_width:
                        best = text[:end]
                    else:
                        break
                return best
            for text, width in (("typewriter", 35), ("hello", 1000), ("hello", 0), ("abcdef", 20)):
                assert solution.longest_prefix_fitting(text, width) == reference(text, width)
            assert solution.longest_prefix_fitting("hello", 1000) == "hello"
            assert solution.longest_prefix_fitting("hello", 0) == ""
            """,
            "Longest prefix whose width fits a limit.",
        ),
        _p(
            "stack_height",
            """
            from PIL import ImageFont

            def stack_height(lines, spacing=2):
                font = ImageFont.load_default()
                heights = [font.getsize(line)[1] for line in lines]
                return sum(heights) + spacing * max(0, len(lines) - 1)
            """,
            """
            from PIL import ImageFont

            def stack_height(lines, spacing=2):
                font = ImageFont.load_default()
                heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
                return sum(heights) + spacing * max(0, len(lines) - 1)
            """,
            _PIL_TEXT_TEST_HEAD + """
            lines = ["alpha", "beta", "gamma"]
            expected = sum(fsize_of(line)[1] for line in lines) + 4
            assert solution.stack_height(lines) == expected
            assert solution.stack_height([]) == 0
            assert solution.stack_height(["x"], spacing=10) == fsize_of("x")[1]
            """,
            "Total height of stacked text lines.",
        ),
    ),
    siblings=(
        _s(
            "font_length",
            """
            from PIL import ImageFont

            FONT = ImageFont.load_default()

            def font_length(text):
                return int(FONT.getlength(text))
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.font_length("hello") == fsize_of("hello")[0]
            assert solution.font_length("") == 0
            """,
            "ImageFont getlength exists on both pins.",
        ),
        _s(
            "font_bbox_height",
            """
            from PIL import ImageFont

            def font_bbox_height(text):
                font = ImageFont.load_default()
                left, top, right, bottom = font.getbbox(text)
                return bottom - top
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.font_bbox_height("Hg") == fsize_of("Hg")[1]
            assert solution.font_bbox_height("Hg") > 0
            """,
            "ImageFont getbbox is unchanged.",
        ),
    ),
)

_PIL_MULTILINE = ApiChange(
    package="pillow",
    api_family="PIL.ImageDraw.ImageDraw.multiline_textsize",
    old_api="ImageDraw.ImageDraw.multiline_textsize",
    new_api="ImageDraw.ImageDraw.multiline_textbbox",
    old_pin=_PIL_OLD,
    new_pin=_PIL_NEW,
    change_date=_PIL_DATE,
    change_url=_PIL_URL,
    summary="Pillow 10 removed ImageDraw.multiline_textsize; use multiline_textbbox.",
    programs=(
        _p(
            "paragraph_size",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def paragraph_size(lines):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                return draw.multiline_textsize("\\n".join(lines), font=FONT)
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def paragraph_size(lines):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                left, top, right, bottom = draw.multiline_textbbox((0, 0), "\\n".join(lines), font=FONT)
                return right - left, bottom - top
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert tuple(solution.paragraph_size(["line one", "line two", "three"])) == msize_of("line one\\nline two\\nthree")
            assert tuple(solution.paragraph_size(["solo"])) == msize_of("solo")
            assert solution.paragraph_size(["a", "b"])[1] > solution.paragraph_size(["a"])[1]
            """,
            "Size of a multi-line paragraph.",
        ),
        _p(
            "fits_in_box",
            """
            from PIL import Image, ImageDraw, ImageFont

            def fits_in_box(text, box_w, box_h):
                draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                w, h = draw.multiline_textsize(text, font=ImageFont.load_default())
                return w <= box_w and h <= box_h
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            def fits_in_box(text, box_w, box_h):
                draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=ImageFont.load_default())
                w, h = right - left, bottom - top
                return w <= box_w and h <= box_h
            """,
            _PIL_TEXT_TEST_HEAD + """
            text = "first line\\nsecond line"
            w, h = msize_of(text)
            assert solution.fits_in_box(text, w, h) is True
            assert solution.fits_in_box(text, w - 1, h) is False
            assert solution.fits_in_box(text, w, h - 1) is False
            assert solution.fits_in_box("tiny", 1000, 1000) is True
            """,
            "Check whether wrapped text fits a box.",
        ),
        _p(
            "caption_spacing",
            """
            from PIL import Image, ImageDraw, ImageFont

            class Caption:
                def __init__(self, spacing):
                    self.spacing = spacing
                    self.font = ImageFont.load_default()

                def size(self, text):
                    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                    return draw.multiline_textsize(text, font=self.font, spacing=self.spacing)
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            class Caption:
                def __init__(self, spacing):
                    self.spacing = spacing
                    self.font = ImageFont.load_default()

                def size(self, text):
                    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=self.font, spacing=self.spacing)
                    return right - left, bottom - top
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert tuple(solution.Caption(10).size("a\\nb")) == msize_of("a\\nb", spacing=10)
            assert tuple(solution.Caption(0).size("one\\ntwo\\nthree")) == msize_of("one\\ntwo\\nthree", spacing=0)
            assert solution.Caption(10).size("a\\nb")[1] - solution.Caption(0).size("a\\nb")[1] == 10
            """,
            "Caption size with configurable line spacing.",
        ),
        _p(
            "tallest_block",
            """
            from PIL import Image, ImageDraw, ImageFont

            def tallest_block(blocks):
                font = ImageFont.load_default()
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                heights = [draw.multiline_textsize(block, font=font)[1] for block in blocks]
                return heights.index(max(heights)), max(heights)
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            def tallest_block(blocks):
                font = ImageFont.load_default()
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                heights = []
                for block in blocks:
                    left, top, right, bottom = draw.multiline_textbbox((0, 0), block, font=font)
                    heights.append(bottom - top)
                return heights.index(max(heights)), max(heights)
            """,
            _PIL_TEXT_TEST_HEAD + """
            blocks = ["one", "one\\ntwo\\nthree", "one\\ntwo"]
            assert solution.tallest_block(blocks) == (1, msize_of(blocks[1])[1])
            assert solution.tallest_block(["x"]) == (0, msize_of("x")[1])
            """,
            "Index and height of the tallest text block.",
        ),
        _p(
            "render_block",
            """
            from PIL import Image, ImageDraw, ImageFont

            def render_block(text, margin=2):
                font = ImageFont.load_default()
                probe = ImageDraw.Draw(Image.new("L", (1, 1)))
                w, h = probe.multiline_textsize(text, font=font)
                canvas = Image.new("L", (w + 2 * margin, h + 2 * margin), 0)
                ImageDraw.Draw(canvas).multiline_text((margin, margin), text, fill=255, font=font)
                return canvas.size, canvas.getbbox() is not None
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            def render_block(text, margin=2):
                font = ImageFont.load_default()
                probe = ImageDraw.Draw(Image.new("L", (1, 1)))
                left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font)
                w, h = right - left, bottom - top
                canvas = Image.new("L", (w + 2 * margin, h + 2 * margin), 0)
                ImageDraw.Draw(canvas).multiline_text((margin, margin), text, fill=255, font=font)
                return canvas.size, canvas.getbbox() is not None
            """,
            _PIL_TEXT_TEST_HEAD + """
            w, h = msize_of("hello\\nworld")
            assert solution.render_block("hello\\nworld") == ((w + 4, h + 4), True)
            w, h = msize_of("hi")
            assert solution.render_block("hi", margin=0) == ((w, h), True)
            """,
            "Render multi-line text on a canvas sized to fit.",
        ),
        _p(
            "block_aspect",
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def block_aspect(text):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                w, h = draw.multiline_textsize(text, font=FONT)
                return round(w / h, 3) if h else 0.0
            """,
            """
            from PIL import Image, ImageDraw, ImageFont

            FONT = ImageFont.load_default()

            def block_aspect(text):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=FONT)
                w, h = right - left, bottom - top
                return round(w / h, 3) if h else 0.0
            """,
            _PIL_TEXT_TEST_HEAD + """
            w, h = msize_of("wide line of text\\nx")
            assert solution.block_aspect("wide line of text\\nx") == round(w / h, 3)
            assert solution.block_aspect("a\\nb\\nc\\nd") < solution.block_aspect("abcdefgh")
            """,
            "Width-to-height ratio of a text block.",
        ),
    ),
    siblings=(
        _s(
            "multiline_box",
            """
            from PIL import Image, ImageDraw, ImageFont

            def multiline_box(text, spacing=4):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                return draw.multiline_textbbox((0, 0), text, font=ImageFont.load_default(), spacing=spacing)
            """,
            _PIL_TEXT_TEST_HEAD + """
            left, top, right, bottom = solution.multiline_box("a\\nb")
            assert (right - left, bottom - top) == msize_of("a\\nb")
            assert solution.multiline_box("a\\nb", spacing=12)[3] - solution.multiline_box("a\\nb", spacing=2)[3] == 10
            """,
            "ImageDraw.multiline_textbbox is unchanged.",
        ),
        _s(
            "line_count_height",
            """
            from PIL import Image, ImageDraw, ImageFont

            def line_count_height(text):
                draw = ImageDraw.Draw(Image.new("L", (1, 1)))
                font = ImageFont.load_default()
                lines = text.split("\\n")
                return len(lines), max(int(draw.textlength(line, font=font)) for line in lines)
            """,
            _PIL_TEXT_TEST_HEAD + """
            assert solution.line_count_height("ab\\nabcd\\na") == (3, size_of("abcd")[0])
            assert solution.line_count_height("solo") == (1, size_of("solo")[0])
            """,
            "ImageDraw.textlength is unchanged.",
        ),
    ),
)


# ---------------------------------------------------------------------------
# networkx 2.8.8 -> 3.3
# ---------------------------------------------------------------------------

_NX_URL = "https://networkx.org/documentation/stable/release/release_3.0.html"
_NX_DATE = "2023-01-07"
_NX_OLD = "2.8.8"
_NX_NEW = "3.3"
_NX_EXTRA = ("numpy==1.26.4", "scipy==1.10.1")

# A small directed graph whose pagerank/HITS values are well conditioned and
# identical (to 1e-5) on both pins.
_NX_ARCS = "[(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0), (1, 3)]"

_NX_INFO = ApiChange(
    package="networkx",
    api_family="networkx.info",
    old_api="nx.info(G)",
    new_api="str(G)",
    old_pin=_NX_OLD,
    new_pin=_NX_NEW,
    change_date=_NX_DATE,
    change_url=_NX_URL,
    summary="networkx 3.0 removed nx.info(G); the same one-line summary is available as str(G).",
    programs=(
        _p(
            "describe_graph",
            """
            import networkx as nx

            def describe_graph(edges):
                G = nx.Graph()
                G.add_edges_from(edges)
                return nx.info(G)
            """,
            None,
            """
            import solution
            assert solution.describe_graph([(0, 1), (1, 2), (2, 3)]) == "Graph with 4 nodes and 3 edges"
            assert solution.describe_graph([]) == "Graph with 0 nodes and 0 edges"
            assert solution.describe_graph([("a", "b"), ("b", "a")]) == "Graph with 2 nodes and 1 edges"
            """,
            "One-line summary of an undirected graph.",
            fix=(("nx.info(G)", "str(G)"),),
        ),
        _p(
            "component_summaries",
            """
            import networkx as nx

            def component_summaries(edges):
                G = nx.Graph(edges)
                parts = [G.subgraph(c).copy() for c in nx.connected_components(G)]
                parts.sort(key=lambda H: (-H.number_of_nodes(), min(H.nodes)))
                return [nx.info(H) for H in parts]
            """,
            None,
            """
            import solution
            out = solution.component_summaries([(0, 1), (1, 2), (5, 6), (8, 9), (9, 10), (10, 8)])
            assert out == [
                "Graph with 3 nodes and 2 edges",
                "Graph with 3 nodes and 3 edges",
                "Graph with 2 nodes and 1 edges",
            ]
            assert solution.component_summaries([(1, 2)]) == ["Graph with 2 nodes and 1 edges"]
            """,
            "Summaries of connected components, largest first.",
            fix=(("nx.info(H)", "str(H)"),),
        ),
        _p(
            "graph_report",
            """
            import networkx as nx

            class GraphReport:
                def __init__(self, name):
                    self.G = nx.Graph(name=name)

                def add(self, u, v):
                    self.G.add_edge(u, v)

                def report(self):
                    return {"info": nx.info(self.G), "density": round(nx.density(self.G), 3)}
            """,
            None,
            """
            import solution
            r = solution.GraphReport("roads")
            r.add("a", "b")
            r.add("b", "c")
            r.add("c", "a")
            assert r.report() == {"info": "Graph named 'roads' with 3 nodes and 3 edges", "density": 1.0}
            r.add("c", "d")
            assert r.report()["info"] == "Graph named 'roads' with 4 nodes and 4 edges"
            assert r.report()["density"] == 0.667
            """,
            "Named graph report with info string and density.",
            fix=(("nx.info(self.G)", "str(self.G)"),),
        ),
        _p(
            "digraph_info",
            """
            import networkx as nx

            def digraph_info(arcs):
                D = nx.DiGraph()
                D.add_edges_from(arcs)
                return nx.info(D)
            """,
            None,
            """
            import solution
            assert solution.digraph_info([(0, 1), (1, 2), (2, 0)]) == "DiGraph with 3 nodes and 3 edges"
            assert solution.digraph_info([(1, 2), (2, 1)]) == "DiGraph with 2 nodes and 2 edges"
            """,
            "Summary string for a directed graph.",
            fix=(("nx.info(D)", "str(D)"),),
        ),
        _p(
            "parse_counts",
            """
            import networkx as nx

            def parse_counts(edges):
                G = nx.Graph(edges)
                words = nx.info(G).split()
                nodes = int(words[words.index("nodes") - 1])
                links = int(words[words.index("edges") - 1])
                return nodes, links
            """,
            None,
            """
            import solution
            assert solution.parse_counts([(0, 1), (1, 2), (2, 3), (3, 4)]) == (5, 4)
            assert solution.parse_counts([(0, 1), (1, 0)]) == (2, 1)
            assert solution.parse_counts([]) == (0, 0)
            """,
            "Parse node and edge counts back out of the summary.",
            fix=(("nx.info(G)", "str(G)"),),
        ),
        _p(
            "same_shape",
            """
            import networkx as nx

            def same_shape(edges_a, edges_b):
                A, B = nx.Graph(edges_a), nx.Graph(edges_b)
                if nx.info(A) == nx.info(B):
                    return "same"
                return nx.info(A) + " | " + nx.info(B)
            """,
            None,
            """
            import solution
            assert solution.same_shape([(0, 1), (1, 2)], [(5, 6), (6, 7)]) == "same"
            assert solution.same_shape([(0, 1)], [(0, 1), (1, 2)]) == "Graph with 2 nodes and 1 edges | Graph with 3 nodes and 2 edges"
            """,
            "Compare two graphs by their summary strings.",
            fix=(("nx.info(A)", "str(A)"), ("nx.info(B)", "str(B)")),
        ),
    ),
    siblings=(
        _s(
            "density_summary",
            """
            import networkx as nx

            def density_summary(edges):
                G = nx.Graph(edges)
                return G.number_of_nodes(), G.number_of_edges(), round(nx.density(G), 3)
            """,
            """
            import solution
            assert solution.density_summary([(0, 1), (1, 2), (2, 0)]) == (3, 3, 1.0)
            assert solution.density_summary([(0, 1), (1, 2), (2, 3)]) == (4, 3, 0.5)
            """,
            "nx.density and graph counters are unchanged.",
        ),
        _s(
            "degree_histogram_summary",
            """
            import networkx as nx

            def degree_histogram_summary(edges):
                G = nx.Graph(edges)
                return nx.degree_histogram(G), G.name
            """,
            """
            import solution
            assert solution.degree_histogram_summary([(0, 1), (1, 2), (2, 3)]) == ([0, 2, 2], "")
            assert solution.degree_histogram_summary([(0, 1), (0, 2), (0, 3)]) == ([0, 3, 0, 1], "")
            """,
            "nx.degree_histogram is unchanged.",
        ),
    ),
    extra_requirements=_NX_EXTRA,
)

_NX_NUMPY_MATRIX = ApiChange(
    package="networkx",
    api_family="networkx.to_numpy_matrix",
    old_api="nx.to_numpy_matrix / nx.from_numpy_matrix",
    new_api="nx.to_numpy_array / nx.from_numpy_array",
    old_pin=_NX_OLD,
    new_pin=_NX_NEW,
    change_date=_NX_DATE,
    change_url=_NX_URL,
    summary="networkx 3.0 removed to_numpy_matrix/from_numpy_matrix; use to_numpy_array/from_numpy_array.",
    programs=(
        _p(
            "adjacency_rows",
            """
            import networkx as nx

            def adjacency_rows(edges, order):
                G = nx.Graph(edges)
                return nx.to_numpy_matrix(G, nodelist=order).astype(int).tolist()
            """,
            None,
            """
            import solution
            assert solution.adjacency_rows([(0, 1), (1, 2)], [0, 1, 2]) == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            assert solution.adjacency_rows([(0, 1), (1, 2)], [2, 1, 0]) == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            assert solution.adjacency_rows([("a", "b")], ["b", "a"]) == [[0, 1], [1, 0]]
            """,
            "Adjacency matrix rows in a caller-chosen node order.",
            fix=(("nx.to_numpy_matrix", "nx.to_numpy_array"),),
        ),
        _p(
            "row_degrees",
            """
            import networkx as nx
            import numpy as np

            def row_degrees(edges):
                G = nx.Graph(edges)
                M = nx.to_numpy_matrix(G, nodelist=sorted(G))
                return [int(d) for d in np.asarray(M).sum(axis=1)]
            """,
            None,
            """
            import solution
            assert solution.row_degrees([(0, 1), (1, 2), (2, 0), (2, 3)]) == [2, 2, 3, 1]
            assert solution.row_degrees([(5, 6)]) == [1, 1]
            """,
            "Node degrees computed from the dense adjacency matrix.",
            fix=(("nx.to_numpy_matrix", "nx.to_numpy_array"),),
        ),
        _p(
            "graph_from_rows",
            """
            import networkx as nx
            import numpy as np

            def graph_from_rows(rows):
                G = nx.from_numpy_matrix(np.array(rows))
                return sorted((u, v, int(d["weight"])) for u, v, d in G.edges(data=True))
            """,
            None,
            """
            import solution
            assert solution.graph_from_rows([[0, 2], [2, 0]]) == [(0, 1, 2)]
            assert solution.graph_from_rows([[0, 1, 0], [1, 0, 3], [0, 3, 0]]) == [(0, 1, 1), (1, 2, 3)]
            assert solution.graph_from_rows([[0, 0], [0, 0]]) == []
            """,
            "Weighted edge list reconstructed from matrix rows.",
            fix=(("nx.from_numpy_matrix", "nx.from_numpy_array"),),
        ),
        _p(
            "symmetrize_arcs",
            """
            import networkx as nx
            import numpy as np

            def symmetrize_arcs(rows):
                D = nx.from_numpy_matrix(np.array(rows), create_using=nx.DiGraph)
                U = D.to_undirected()
                return nx.to_numpy_matrix(U, nodelist=sorted(U)).astype(int).tolist()
            """,
            None,
            """
            import solution
            assert solution.symmetrize_arcs([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            assert solution.symmetrize_arcs([[0, 0], [1, 0]]) == [[0, 1], [1, 0]]
            """,
            "Round-trip a directed matrix through an undirected graph.",
            fix=(("nx.from_numpy_matrix", "nx.from_numpy_array"), ("nx.to_numpy_matrix", "nx.to_numpy_array")),
        ),
        _p(
            "matrix_cache",
            """
            import networkx as nx

            class MatrixCache:
                def __init__(self):
                    self.store = {}

                def put(self, name, edges):
                    G = nx.Graph(edges)
                    self.store[name] = nx.to_numpy_matrix(G, nodelist=sorted(G))

                def edge_count(self, name):
                    return nx.from_numpy_matrix(self.store[name]).number_of_edges()

                def shape(self, name):
                    return tuple(int(n) for n in self.store[name].shape)
            """,
            None,
            """
            import solution
            cache = solution.MatrixCache()
            cache.put("tri", [(0, 1), (1, 2), (2, 0)])
            cache.put("path", [(0, 1), (1, 2), (2, 3)])
            assert cache.edge_count("tri") == 3
            assert cache.edge_count("path") == 3
            assert cache.shape("path") == (4, 4)
            """,
            "Cache dense matrices and rebuild graphs from them.",
            fix=(("nx.to_numpy_matrix", "nx.to_numpy_array"), ("nx.from_numpy_matrix", "nx.from_numpy_array")),
        ),
        _p(
            "weighted_total",
            """
            import networkx as nx

            def weighted_total(weighted_edges):
                G = nx.Graph()
                G.add_weighted_edges_from(weighted_edges)
                M = nx.to_numpy_matrix(G, weight="weight")
                return float(M.sum()) / 2.0
            """,
            None,
            """
            import solution
            assert solution.weighted_total([(0, 1, 2.5), (1, 2, 1.0)]) == 3.5
            assert solution.weighted_total([("a", "b", 4), ("b", "c", 6), ("c", "a", 10)]) == 20.0
            """,
            "Sum of edge weights via the dense weighted matrix.",
            fix=(("nx.to_numpy_matrix", "nx.to_numpy_array"),),
        ),
        _p(
            "two_hop_paths",
            """
            import networkx as nx
            import numpy as np

            def two_hop_paths(edges, source, target):
                G = nx.Graph(edges)
                nodes = sorted(G)
                A = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
                A2 = A @ A
                return int(A2[nodes.index(source), nodes.index(target)])
            """,
            None,
            """
            import solution
            square = [(0, 1), (1, 2), (2, 3), (3, 0)]
            assert solution.two_hop_paths(square, 0, 2) == 2
            assert solution.two_hop_paths(square, 0, 1) == 0
            assert solution.two_hop_paths(square, 0, 0) == 2
            """,
            "Number of length-two walks between two nodes.",
            fix=(("nx.to_numpy_matrix", "nx.to_numpy_array"),),
        ),
    ),
    siblings=(
        _s(
            "dense_from_array",
            """
            import networkx as nx
            import numpy as np

            def dense_from_array(rows):
                G = nx.from_numpy_array(np.array(rows))
                return G.number_of_edges(), nx.to_numpy_array(G).astype(int).tolist()
            """,
            """
            import solution
            assert solution.dense_from_array([[0, 1], [1, 0]]) == (1, [[0, 1], [1, 0]])
            assert solution.dense_from_array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]) == (2, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])
            """,
            "to_numpy_array/from_numpy_array exist on both pins.",
        ),
        _s(
            "dict_of_dicts_roundtrip",
            """
            import networkx as nx

            def dict_of_dicts_roundtrip(edges):
                G = nx.Graph(edges)
                dod = nx.to_dict_of_dicts(G)
                return sorted(dod), nx.from_dict_of_dicts(dod).number_of_edges()
            """,
            """
            import solution
            assert solution.dict_of_dicts_roundtrip([(0, 1), (1, 2)]) == ([0, 1, 2], 2)
            """,
            "to_dict_of_dicts/from_dict_of_dicts are unchanged.",
        ),
    ),
    extra_requirements=_NX_EXTRA,
)

_NX_GPICKLE = ApiChange(
    package="networkx",
    api_family="networkx.read_gpickle",
    old_api="nx.write_gpickle / nx.read_gpickle",
    new_api="pickle.dump / pickle.load",
    old_pin=_NX_OLD,
    new_pin=_NX_NEW,
    change_date=_NX_DATE,
    change_url=_NX_URL,
    summary="networkx 3.0 removed read_gpickle/write_gpickle; use the pickle module directly.",
    programs=(
        _p(
            "roundtrip_edges",
            """
            import io
            import networkx as nx

            def roundtrip_edges(edges):
                G = nx.Graph(edges)
                buf = io.BytesIO()
                nx.write_gpickle(G, buf)
                buf.seek(0)
                H = nx.read_gpickle(buf)
                return sorted(H.edges())
            """,
            """
            import io
            import pickle
            import networkx as nx

            def roundtrip_edges(edges):
                G = nx.Graph(edges)
                buf = io.BytesIO()
                pickle.dump(G, buf, pickle.HIGHEST_PROTOCOL)
                buf.seek(0)
                H = pickle.load(buf)
                return sorted(H.edges())
            """,
            """
            import solution
            assert solution.roundtrip_edges([(0, 1), (1, 2)]) == [(0, 1), (1, 2)]
            assert solution.roundtrip_edges([("b", "a")]) == [("b", "a")]
            assert solution.roundtrip_edges([]) == []
            """,
            "Serialise a graph to bytes and read it back.",
        ),
        _p(
            "graph_blob",
            """
            import io
            import networkx as nx

            def to_blob(arcs):
                D = nx.DiGraph(arcs)
                buf = io.BytesIO()
                nx.write_gpickle(D, buf)
                return buf.getvalue()

            def from_blob(blob):
                D = nx.read_gpickle(io.BytesIO(blob))
                return D.number_of_nodes(), D.number_of_edges(), D.is_directed()
            """,
            """
            import io
            import pickle
            import networkx as nx

            def to_blob(arcs):
                D = nx.DiGraph(arcs)
                buf = io.BytesIO()
                pickle.dump(D, buf, pickle.HIGHEST_PROTOCOL)
                return buf.getvalue()

            def from_blob(blob):
                D = pickle.load(io.BytesIO(blob))
                return D.number_of_nodes(), D.number_of_edges(), D.is_directed()
            """,
            """
            import solution
            blob = solution.to_blob([(0, 1), (1, 2), (2, 0)])
            assert isinstance(blob, bytes) and len(blob) > 0
            assert solution.from_blob(blob) == (3, 3, True)
            assert solution.from_blob(solution.to_blob([(5, 6)])) == (2, 1, True)
            """,
            "Bytes encoder/decoder pair for directed graphs.",
        ),
        _p(
            "graph_store",
            """
            import io
            import networkx as nx

            class GraphStore:
                def __init__(self):
                    self.blobs = {}

                def save(self, name, G):
                    buf = io.BytesIO()
                    nx.write_gpickle(G, buf)
                    self.blobs[name] = buf.getvalue()

                def load(self, name):
                    return nx.read_gpickle(io.BytesIO(self.blobs[name]))
            """,
            """
            import io
            import pickle
            import networkx as nx

            class GraphStore:
                def __init__(self):
                    self.blobs = {}

                def save(self, name, G):
                    buf = io.BytesIO()
                    pickle.dump(G, buf, pickle.HIGHEST_PROTOCOL)
                    self.blobs[name] = buf.getvalue()

                def load(self, name):
                    return pickle.load(io.BytesIO(self.blobs[name]))
            """,
            """
            import solution
            import networkx as nx
            store = solution.GraphStore()
            store.save("cycle", nx.cycle_graph(5))
            store.save("star", nx.star_graph(3))
            assert sorted(store.load("cycle").edges()) == [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]
            assert store.load("star").number_of_edges() == 3
            assert sorted(store.blobs) == ["cycle", "star"]
            """,
            "In-memory named store of pickled graphs.",
        ),
        _p(
            "clone_with_attrs",
            """
            import io
            import networkx as nx

            def clone_with_attrs(edges, labels):
                G = nx.Graph(edges)
                nx.set_node_attributes(G, labels, "label")
                buf = io.BytesIO()
                nx.write_gpickle(G, buf)
                buf.seek(0)
                H = nx.read_gpickle(buf)
                return dict(sorted(nx.get_node_attributes(H, "label").items()))
            """,
            """
            import io
            import pickle
            import networkx as nx

            def clone_with_attrs(edges, labels):
                G = nx.Graph(edges)
                nx.set_node_attributes(G, labels, "label")
                buf = io.BytesIO()
                pickle.dump(G, buf, pickle.HIGHEST_PROTOCOL)
                buf.seek(0)
                H = pickle.load(buf)
                return dict(sorted(nx.get_node_attributes(H, "label").items()))
            """,
            """
            import solution
            out = solution.clone_with_attrs([(1, 2), (2, 3)], {1: "one", 3: "three"})
            assert out == {1: "one", 3: "three"}
            assert solution.clone_with_attrs([(1, 2)], {}) == {}
            """,
            "Node attributes survive a pickle round trip.",
        ),
        _p(
            "save_local",
            """
            import os
            import networkx as nx

            def save_local(edges, filename):
                G = nx.Graph(edges)
                nx.write_gpickle(G, filename)
                H = nx.read_gpickle(filename)
                size = os.path.getsize(filename)
                os.remove(filename)
                return H.number_of_edges(), size > 0
            """,
            """
            import os
            import pickle
            import networkx as nx

            def save_local(edges, filename):
                G = nx.Graph(edges)
                with open(filename, "wb") as fh:
                    pickle.dump(G, fh, pickle.HIGHEST_PROTOCOL)
                with open(filename, "rb") as fh:
                    H = pickle.load(fh)
                size = os.path.getsize(filename)
                os.remove(filename)
                return H.number_of_edges(), size > 0
            """,
            """
            import solution
            import os
            assert solution.save_local([(0, 1), (1, 2)], "graph_tmp.gpickle") == (2, True)
            assert not os.path.exists("graph_tmp.gpickle")
            assert solution.save_local([], "empty_tmp.gpickle") == (0, True)
            """,
            "Write a pickle to the working directory and read it back.",
        ),
        _p(
            "graph_digest",
            """
            import hashlib
            import io
            import networkx as nx

            def graph_digest(edges):
                buf = io.BytesIO()
                nx.write_gpickle(nx.Graph(edges), buf)
                buf.seek(0)
                H = nx.read_gpickle(buf)
                canonical = repr(sorted(tuple(sorted(e)) for e in H.edges()))
                return hashlib.sha256(canonical.encode()).hexdigest()[:12]
            """,
            """
            import hashlib
            import io
            import pickle
            import networkx as nx

            def graph_digest(edges):
                buf = io.BytesIO()
                pickle.dump(nx.Graph(edges), buf, pickle.HIGHEST_PROTOCOL)
                buf.seek(0)
                H = pickle.load(buf)
                canonical = repr(sorted(tuple(sorted(e)) for e in H.edges()))
                return hashlib.sha256(canonical.encode()).hexdigest()[:12]
            """,
            """
            import solution
            import hashlib
            expected = hashlib.sha256(repr([(0, 1), (1, 2)]).encode()).hexdigest()[:12]
            assert solution.graph_digest([(1, 2), (1, 0)]) == expected
            assert solution.graph_digest([(2, 1), (0, 1)]) == expected
            assert solution.graph_digest([(0, 1)]) != expected
            """,
            "Stable digest of a graph after a pickle round trip.",
        ),
    ),
    siblings=(
        _s(
            "edgelist_roundtrip",
            """
            import io
            import networkx as nx

            def edgelist_roundtrip(edges):
                G = nx.Graph(edges)
                buf = io.BytesIO()
                nx.write_edgelist(G, buf, data=False)
                buf.seek(0)
                H = nx.read_edgelist(buf, nodetype=int)
                return sorted(H.edges())
            """,
            """
            import solution
            assert solution.edgelist_roundtrip([(0, 1), (1, 2)]) == [(0, 1), (1, 2)]
            assert solution.edgelist_roundtrip([(3, 4)]) == [(3, 4)]
            """,
            "write_edgelist/read_edgelist are unchanged.",
        ),
        _s(
            "node_link_roundtrip",
            """
            import json
            import networkx as nx

            def node_link_roundtrip(edges):
                G = nx.Graph(edges)
                payload = json.dumps(nx.node_link_data(G))
                H = nx.node_link_graph(json.loads(payload))
                return sorted(H.nodes()), H.number_of_edges()
            """,
            """
            import solution
            assert solution.node_link_roundtrip([(0, 1), (1, 2)]) == ([0, 1, 2], 2)
            """,
            "node_link_data/node_link_graph are unchanged.",
        ),
    ),
    extra_requirements=_NX_EXTRA,
)

_NX_SCIPY_SPARSE = ApiChange(
    package="networkx",
    api_family="networkx.to_scipy_sparse_matrix",
    old_api="nx.to_scipy_sparse_matrix / nx.from_scipy_sparse_matrix",
    new_api="nx.to_scipy_sparse_array / nx.from_scipy_sparse_array",
    old_pin=_NX_OLD,
    new_pin=_NX_NEW,
    change_date=_NX_DATE,
    change_url=_NX_URL,
    summary="networkx 3.0 removed to_scipy_sparse_matrix/from_scipy_sparse_matrix; use the *_array functions.",
    programs=(
        _p(
            "sparse_degrees",
            """
            import networkx as nx
            import numpy as np

            def sparse_degrees(edges):
                G = nx.Graph(edges)
                S = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G))
                return [int(d) for d in np.asarray(S.sum(axis=1)).ravel()]
            """,
            None,
            """
            import solution
            assert solution.sparse_degrees([(0, 1), (1, 2), (2, 0), (2, 3)]) == [2, 2, 3, 1]
            assert solution.sparse_degrees([(7, 8)]) == [1, 1]
            """,
            "Degrees from the sparse adjacency matrix.",
            fix=(("nx.to_scipy_sparse_matrix", "nx.to_scipy_sparse_array"),),
        ),
        _p(
            "sparse_stats",
            """
            import networkx as nx

            def sparse_stats(arcs):
                D = nx.DiGraph(arcs)
                S = nx.to_scipy_sparse_matrix(D, format="csr")
                return int(S.nnz), tuple(int(n) for n in S.shape), float(S.sum())
            """,
            None,
            """
            import solution
            assert solution.sparse_stats([(0, 1), (1, 2), (2, 0)]) == (3, (3, 3), 3.0)
            assert solution.sparse_stats([(0, 1), (1, 0), (1, 2)]) == (3, (3, 3), 3.0)
            """,
            "Non-zero count, shape and total of a sparse adjacency matrix.",
            fix=(("nx.to_scipy_sparse_matrix", "nx.to_scipy_sparse_array"),),
        ),
        _p(
            "graph_from_sparse",
            """
            import networkx as nx
            import scipy.sparse as sp

            def graph_from_sparse(rows):
                S = sp.csr_matrix(rows)
                G = nx.from_scipy_sparse_matrix(S)
                return sorted((u, v, int(d["weight"])) for u, v, d in G.edges(data=True))
            """,
            None,
            """
            import solution
            assert solution.graph_from_sparse([[0, 2], [2, 0]]) == [(0, 1, 2)]
            assert solution.graph_from_sparse([[0, 1, 0], [1, 0, 3], [0, 3, 0]]) == [(0, 1, 1), (1, 2, 3)]
            assert solution.graph_from_sparse([[0, 0], [0, 0]]) == []
            """,
            "Build a weighted graph from a scipy sparse matrix.",
            fix=(("nx.from_scipy_sparse_matrix", "nx.from_scipy_sparse_array"),),
        ),
        _p(
            "sparse_adjacency",
            """
            import networkx as nx
            import numpy as np

            class SparseAdjacency:
                def __init__(self, edges):
                    self.G = nx.Graph(edges)
                    self.nodes = sorted(self.G)

                def matrix(self):
                    return nx.to_scipy_sparse_matrix(self.G, nodelist=self.nodes, format="csc")

                def column_sums(self):
                    return [int(c) for c in np.asarray(self.matrix().sum(axis=0)).ravel()]
            """,
            None,
            """
            import solution
            adj = solution.SparseAdjacency([(0, 1), (0, 2), (0, 3)])
            assert adj.column_sums() == [3, 1, 1, 1]
            assert adj.matrix().shape == (4, 4)
            assert solution.SparseAdjacency([(1, 2)]).column_sums() == [1, 1]
            """,
            "Column sums of a CSC adjacency matrix.",
            fix=(("nx.to_scipy_sparse_matrix", "nx.to_scipy_sparse_array"),),
        ),
        _p(
            "weighted_sparse_total",
            """
            import networkx as nx

            def weighted_sparse_total(weighted_edges):
                G = nx.Graph()
                G.add_weighted_edges_from(weighted_edges)
                S = nx.to_scipy_sparse_matrix(G, weight="weight")
                return float(S.sum()) / 2.0
            """,
            None,
            """
            import solution
            assert solution.weighted_sparse_total([(0, 1, 2.5), (1, 2, 1.0)]) == 3.5
            assert solution.weighted_sparse_total([("a", "b", 4), ("b", "c", 6)]) == 10.0
            """,
            "Sum of weights through the sparse matrix.",
            fix=(("nx.to_scipy_sparse_matrix", "nx.to_scipy_sparse_array"),),
        ),
        _p(
            "laplacian_trace",
            """
            import networkx as nx
            import numpy as np
            import scipy.sparse as sp

            def laplacian_trace(edges):
                G = nx.Graph(edges)
                A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G), format="csr")
                degrees = np.asarray(A.sum(axis=1)).ravel()
                L = sp.diags(degrees) - A
                return int(L.diagonal().sum())
            """,
            None,
            """
            import solution
            assert solution.laplacian_trace([(0, 1), (1, 2), (2, 0)]) == 6
            assert solution.laplacian_trace([(0, 1), (1, 2), (2, 3)]) == 6
            assert solution.laplacian_trace([(0, 1)]) == 2
            """,
            "Trace of the graph Laplacian (twice the edge count).",
            fix=(("nx.to_scipy_sparse_matrix", "nx.to_scipy_sparse_array"),),
        ),
    ),
    siblings=(
        _s(
            "sparse_array_roundtrip",
            """
            import networkx as nx

            def sparse_array_roundtrip(edges):
                G = nx.Graph(edges)
                S = nx.to_scipy_sparse_array(G, nodelist=sorted(G))
                H = nx.from_scipy_sparse_array(S)
                return int(S.nnz), H.number_of_edges()
            """,
            """
            import solution
            assert solution.sparse_array_roundtrip([(0, 1), (1, 2)]) == (4, 2)
            assert solution.sparse_array_roundtrip([(0, 1)]) == (2, 1)
            """,
            "to_scipy_sparse_array/from_scipy_sparse_array exist on both pins.",
        ),
        _s(
            "adjacency_dense",
            """
            import networkx as nx

            def adjacency_dense(edges):
                G = nx.Graph(edges)
                return nx.adjacency_matrix(G, nodelist=sorted(G)).toarray().astype(int).tolist()
            """,
            """
            import solution
            assert solution.adjacency_dense([(0, 1), (1, 2)]) == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            """,
            "nx.adjacency_matrix(...).toarray() works on both pins.",
        ),
    ),
    extra_requirements=_NX_EXTRA,
)

_NX_PAGERANK = ApiChange(
    package="networkx",
    api_family="networkx.pagerank_numpy",
    old_api="nx.pagerank_numpy / nx.pagerank_scipy",
    new_api="nx.pagerank",
    old_pin=_NX_OLD,
    new_pin=_NX_NEW,
    change_date=_NX_DATE,
    change_url=_NX_URL,
    summary="networkx 3.0 removed pagerank_numpy and pagerank_scipy; use nx.pagerank.",
    programs=(
        _p(
            "rank_nodes",
            """
            import networkx as nx

            def rank_nodes(arcs):
                D = nx.DiGraph(arcs)
                return {n: round(score, 4) for n, score in nx.pagerank_numpy(D).items()}
            """,
            None,
            """
            import solution
            scores = solution.rank_nodes(""" + _NX_ARCS + """)
            expected = {0: 0.3374, 1: 0.1809, 2: 0.2578, 3: 0.2239}
            assert scores.keys() == expected.keys()
            assert all(abs(scores[k] - expected[k]) < 2e-4 for k in expected)
            """,
            "PageRank scores rounded to four places.",
            fix=(("nx.pagerank_numpy", "nx.pagerank"),),
        ),
        _p(
            "top_node",
            """
            import networkx as nx

            def top_node(arcs):
                D = nx.DiGraph(arcs)
                scores = nx.pagerank_scipy(D)
                return max(scores, key=lambda n: (scores[n], -n))
            """,
            None,
            """
            import solution
            assert solution.top_node(""" + _NX_ARCS + """) == 0
            assert solution.top_node([(1, 2), (3, 2), (2, 4)]) == 4
            """,
            "Node with the highest PageRank.",
            fix=(("nx.pagerank_scipy", "nx.pagerank"),),
        ),
        _p(
            "personalized_rank",
            """
            import networkx as nx

            def personalized_rank(arcs, focus, alpha=0.9):
                D = nx.DiGraph(arcs)
                scores = nx.pagerank_numpy(D, alpha=alpha, personalization={focus: 1.0})
                return {n: round(s, 4) for n, s in scores.items()}
            """,
            None,
            """
            import solution
            scores = solution.personalized_rank(""" + _NX_ARCS + """, 0)
            expected = {0: 0.3848, 1: 0.1732, 2: 0.2511, 3: 0.1909}
            assert all(abs(scores[k] - expected[k]) < 2e-4 for k in expected)
            assert len(scores) == 4
            """,
            "Personalised PageRank biased towards one node.",
            fix=(("nx.pagerank_numpy", "nx.pagerank"),),
        ),
        _p(
            "ranker_order",
            """
            import networkx as nx

            class Ranker:
                def __init__(self, arcs, alpha=0.85):
                    self.D = nx.DiGraph(arcs)
                    self.alpha = alpha

                def scores(self):
                    return nx.pagerank_scipy(self.D, alpha=self.alpha)

                def order(self):
                    scores = self.scores()
                    return sorted(scores, key=lambda n: (-scores[n], n))
            """,
            None,
            """
            import solution
            ranker = solution.Ranker(""" + _NX_ARCS + """)
            assert ranker.order() == [0, 2, 3, 1]
            assert abs(sum(ranker.scores().values()) - 1.0) < 1e-6
            """,
            "Nodes ordered by PageRank.",
            fix=(("nx.pagerank_scipy", "nx.pagerank"),),
        ),
        _p(
            "weighted_rank",
            """
            import networkx as nx

            def weighted_rank(weighted_edges):
                G = nx.Graph()
                G.add_weighted_edges_from(weighted_edges)
                scores = nx.pagerank_numpy(G, weight="weight")
                return {n: round(s, 5) for n, s in scores.items()}
            """,
            None,
            """
            import solution
            import networkx as nx
            edges = [(0, 1, 5.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 2.0)]
            G = nx.Graph()
            G.add_weighted_edges_from(edges)
            expected = nx.pagerank(G, weight="weight")
            scores = solution.weighted_rank(edges)
            assert all(abs(scores[k] - expected[k]) < 1e-4 for k in expected)
            assert scores[0] > scores[3]
            """,
            "Weighted PageRank on an undirected graph.",
            fix=(("nx.pagerank_numpy", "nx.pagerank"),),
        ),
        _p(
            "rank_spread",
            """
            import networkx as nx

            def rank_spread(arcs):
                scores = nx.pagerank_scipy(nx.DiGraph(arcs))
                return round(max(scores.values()) - min(scores.values()), 4)
            """,
            None,
            """
            import solution
            assert abs(solution.rank_spread(""" + _NX_ARCS + """) - 0.1565) < 2e-4
            assert solution.rank_spread([(0, 1), (1, 0)]) == 0.0
            """,
            "Difference between the best and worst PageRank.",
            fix=(("nx.pagerank_scipy", "nx.pagerank"),),
        ),
    ),
    siblings=(
        _s(
            "plain_pagerank",
            """
            import networkx as nx

            def plain_pagerank(arcs):
                return {n: round(s, 4) for n, s in nx.pagerank(nx.DiGraph(arcs)).items()}
            """,
            """
            import solution
            scores = solution.plain_pagerank(""" + _NX_ARCS + """)
            expected = {0: 0.3374, 1: 0.1809, 2: 0.2578, 3: 0.2239}
            assert all(abs(scores[k] - expected[k]) < 2e-4 for k in expected)
            """,
            "nx.pagerank is unchanged.",
        ),
        _s(
            "eigen_centrality",
            """
            import networkx as nx

            def eigen_centrality(edges):
                return {n: round(s, 3) for n, s in nx.eigenvector_centrality_numpy(nx.Graph(edges)).items()}
            """,
            """
            import solution
            assert solution.eigen_centrality([(0, 1), (1, 2), (2, 3)]) == {0: 0.372, 1: 0.602, 2: 0.602, 3: 0.372}
            """,
            "nx.eigenvector_centrality_numpy is unchanged.",
        ),
    ),
    extra_requirements=_NX_EXTRA,
)

_NX_HITS = ApiChange(
    package="networkx",
    api_family="networkx.hits_numpy",
    old_api="nx.hits_numpy / nx.hits_scipy",
    new_api="nx.hits",
    old_pin=_NX_OLD,
    new_pin=_NX_NEW,
    change_date=_NX_DATE,
    change_url=_NX_URL,
    summary="networkx 3.0 removed hits_numpy and hits_scipy; use nx.hits.",
    programs=(
        _p(
            "hub_scores",
            """
            import networkx as nx

            def hub_scores(arcs):
                hubs, _authorities = nx.hits_numpy(nx.DiGraph(arcs))
                return {n: round(h, 4) for n, h in hubs.items()}
            """,
            None,
            """
            import solution
            hubs = solution.hub_scores(""" + _NX_ARCS + """)
            expected = {0: 0.2267, 1: 0.3473, 2: 0.3054, 3: 0.1206}
            assert all(abs(hubs[k] - expected[k]) < 2e-4 for k in expected)
            """,
            "HITS hub scores.",
            fix=(("nx.hits_numpy", "nx.hits"),),
        ),
        _p(
            "best_authority",
            """
            import networkx as nx

            def best_authority(arcs):
                _hubs, authorities = nx.hits_scipy(nx.DiGraph(arcs))
                best = max(authorities, key=lambda n: (authorities[n], -n))
                return best, round(authorities[best], 4)
            """,
            None,
            """
            import solution
            node, score = solution.best_authority(""" + _NX_ARCS + """)
            assert node == 3 and abs(score - 0.3473) < 2e-4
            """,
            "Node with the highest authority score.",
            fix=(("nx.hits_scipy", "nx.hits"),),
        ),
        _p(
            "hits_analyzer",
            """
            import networkx as nx

            class HitsAnalyzer:
                def __init__(self, arcs):
                    self.D = nx.DiGraph(arcs)
                    self._hubs, self._auth = nx.hits_numpy(self.D, normalized=True)

                def hubs(self):
                    return {n: round(v, 4) for n, v in self._hubs.items()}

                def authorities(self):
                    return {n: round(v, 4) for n, v in self._auth.items()}
            """,
            None,
            """
            import solution
            an = solution.HitsAnalyzer(""" + _NX_ARCS + """)
            auth = an.authorities()
            expected = {0: 0.2267, 1: 0.1206, 2: 0.3054, 3: 0.3473}
            assert all(abs(auth[k] - expected[k]) < 2e-4 for k in expected)
            assert abs(sum(an.hubs().values()) - 1.0) < 1e-3
            """,
            "Normalised hub and authority scores.",
            fix=(("nx.hits_numpy", "nx.hits"),),
        ),
        _p(
            "hub_authority_gap",
            """
            import networkx as nx

            def hub_authority_gap(arcs):
                hubs, auth = nx.hits_scipy(nx.DiGraph(arcs))
                return round(sum(abs(hubs[n] - auth[n]) for n in hubs), 4)
            """,
            None,
            """
            import solution
            assert abs(solution.hub_authority_gap(""" + _NX_ARCS + """) - 0.4534) < 3e-4
            assert solution.hub_authority_gap([(0, 1), (1, 0)]) < 1e-6
            """,
            "Total absolute difference between hub and authority scores.",
            fix=(("nx.hits_scipy", "nx.hits"),),
        ),
        _p(
            "rank_by_hub",
            """
            import networkx as nx

            def rank_by_hub(arcs):
                hubs, _auth = nx.hits_numpy(nx.DiGraph(arcs))
                return sorted(hubs, key=lambda n: (-hubs[n], n))
            """,
            None,
            """
            import solution
            assert solution.rank_by_hub(""" + _NX_ARCS + """) == [1, 2, 0, 3]
            """,
            "Nodes ordered by hub score.",
            fix=(("nx.hits_numpy", "nx.hits"),),
        ),
        _p(
            "central_hub",
            """
            import networkx as nx

            def central_hub(edges):
                hubs, _auth = nx.hits_scipy(nx.Graph(edges))
                node = max(hubs, key=lambda n: (hubs[n], -n))
                return node, round(hubs[node], 4)
            """,
            None,
            """
            import solution
            node, score = solution.central_hub([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
            assert node == 2 and abs(score - 0.2808) < 2e-4
            """,
            "Most central node of an undirected graph by HITS.",
            fix=(("nx.hits_scipy", "nx.hits"),),
        ),
    ),
    siblings=(
        _s(
            "plain_hits",
            """
            import networkx as nx

            def plain_hits(arcs):
                hubs, auth = nx.hits(nx.DiGraph(arcs))
                return {n: round(v, 4) for n, v in hubs.items()}, {n: round(v, 4) for n, v in auth.items()}
            """,
            """
            import solution
            hubs, auth = solution.plain_hits(""" + _NX_ARCS + """)
            assert abs(hubs[1] - 0.3473) < 2e-4 and abs(auth[3] - 0.3473) < 2e-4
            """,
            "nx.hits is unchanged.",
        ),
        _s(
            "degree_connectivity",
            """
            import networkx as nx

            def degree_connectivity(edges):
                return nx.average_degree_connectivity(nx.Graph(edges))
            """,
            """
            import solution
            assert solution.degree_connectivity([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]) == {2: 3.0, 4: 2.0}
            """,
            "nx.average_degree_connectivity is unchanged.",
        ),
    ),
    extra_requirements=_NX_EXTRA,
)


# ---------------------------------------------------------------------------
# jinja2 2.11.3 -> 3.1.4  (the removals shipped in Jinja 3.1.0, 2022-03-24)
# ---------------------------------------------------------------------------

_J2_URL = "https://jinja.palletsprojects.com/en/stable/changes/#version-3-1-0"
_J2_DATE = "2022-03-24"
_J2_OLD = "2.11.3"
_J2_NEW = "3.1.4"
_J2_EXTRA = ("markupsafe==2.0.1",)

_J2_MARKUP = ApiChange(
    package="jinja2",
    api_family="jinja2.Markup",
    old_api="jinja2.Markup",
    new_api="markupsafe.Markup",
    old_pin=_J2_OLD,
    new_pin=_J2_NEW,
    change_date=_J2_DATE,
    change_url=_J2_URL,
    summary="Jinja 3.1 removed the jinja2.Markup alias (deprecated in 3.0); import Markup from markupsafe.",
    programs=(
        _p(
            "bold",
            """
            from jinja2 import Markup

            def bold(text):
                return str(Markup("<b>%s</b>") % text)
            """,
            None,
            """
            import solution
            assert solution.bold("hi") == "<b>hi</b>"
            assert solution.bold("<i>") == "<b>&lt;i&gt;</b>"
            assert solution.bold("a & b") == "<b>a &amp; b</b>"
            """,
            "Wrap escaped text in bold tags.",
            fix=(("from jinja2 import Markup", "from markupsafe import Markup"),),
        ),
        _p(
            "render_items",
            """
            from jinja2 import Environment, Markup

            ENV = Environment(autoescape=True)
            TEMPLATE = ENV.from_string("<ul>{% for it in items %}<li>{{ it }}</li>{% endfor %}</ul>")

            def render_items(items, trusted=()):
                safe = [Markup(it) if it in trusted else it for it in items]
                return TEMPLATE.render(items=safe)
            """,
            None,
            """
            import solution
            assert solution.render_items(["a", "<b>x</b>"]) == "<ul><li>a</li><li>&lt;b&gt;x&lt;/b&gt;</li></ul>"
            assert solution.render_items(["a", "<b>x</b>"], trusted={"<b>x</b>"}) == "<ul><li>a</li><li><b>x</b></li></ul>"
            assert solution.render_items([]) == "<ul></ul>"
            """,
            "Render a list, marking trusted entries as safe HTML.",
            fix=(("from jinja2 import Environment, Markup", "from jinja2 import Environment\nfrom markupsafe import Markup"),),
        ),
        _p(
            "snippet",
            """
            from jinja2 import Markup

            class Snippet:
                def __init__(self, tag, body):
                    self.tag = tag
                    self.body = body

                def html(self):
                    return Markup("<%s>" % self.tag) + self.body + Markup("</%s>" % self.tag)

                def __html__(self):
                    return self.html()
            """,
            None,
            """
            import solution
            s = solution.Snippet("em", "x < y")
            assert str(s.html()) == "<em>x &lt; y</em>"
            assert str(s.__html__()) == "<em>x &lt; y</em>"
            assert str(solution.Snippet("p", "plain").html()) == "<p>plain</p>"
            """,
            "HTML snippet object implementing __html__.",
            fix=(("from jinja2 import Markup", "from markupsafe import Markup"),),
        ),
        _p(
            "join_tags",
            """
            from jinja2 import Markup

            def join_tags(tags, cls="tag"):
                parts = [Markup('<span class="%s">{}</span>' % cls).format(t) for t in tags]
                return str(Markup(" ").join(parts))
            """,
            None,
            """
            import solution
            assert solution.join_tags(["a", "b"]) == '<span class="tag">a</span> <span class="tag">b</span>'
            assert solution.join_tags(["<x>"], cls="t") == '<span class="t">&lt;x&gt;</span>'
            assert solution.join_tags([]) == ""
            """,
            "Join tag names into escaped span elements.",
            fix=(("from jinja2 import Markup", "from markupsafe import Markup"),),
        ),
        _p(
            "escape_values",
            """
            import jinja2

            def escape_values(record):
                return {key: str(jinja2.Markup.escape(value)) for key, value in record.items()}
            """,
            None,
            """
            import solution
            assert solution.escape_values({"name": "Tom & Jerry", "n": 3}) == {"name": "Tom &amp; Jerry", "n": "3"}
            assert solution.escape_values({"q": '"quoted"'}) == {"q": "&#34;quoted&#34;"}
            assert solution.escape_values({}) == {}
            """,
            "Escape every value of a record.",
            fix=(("import jinja2\n", "import markupsafe\n"), ("jinja2.Markup.escape", "markupsafe.Markup.escape")),
        ),
        _p(
            "strip_html",
            """
            from jinja2 import Markup

            def strip_html(html):
                return Markup(html).striptags()
            """,
            None,
            """
            import solution
            assert solution.strip_html("<p>Hello <b>world</b></p>") == "Hello world"
            assert solution.strip_html("a &amp; b") == "a & b"
            assert solution.strip_html("plain") == "plain"
            """,
            "Remove tags and unescape entities.",
            fix=(("from jinja2 import Markup", "from markupsafe import Markup"),),
        ),
    ),
    siblings=(
        _s(
            "autoescaped_render",
            """
            from jinja2 import Environment

            ENV = Environment(autoescape=True)

            def autoescaped_render(template, **context):
                return ENV.from_string(template).render(**context)
            """,
            """
            import solution
            assert solution.autoescaped_render("<p>{{ x }}</p>", x="<b>") == "<p>&lt;b&gt;</p>"
            assert solution.autoescaped_render("{{ x | safe }}", x="<b>") == "<b>"
            """,
            "Environment(autoescape=True) and the safe filter are unchanged.",
        ),
        _s(
            "select_autoescape_env",
            """
            from jinja2 import Environment, select_autoescape

            def select_autoescape_env(name, value):
                env = Environment(autoescape=select_autoescape(["html"]))
                return env.autoescape(name), env.from_string("{{ v }}").render(v=value)
            """,
            """
            import solution
            assert solution.select_autoescape_env("page.html", "<x>") == (True, "<x>")
            assert solution.select_autoescape_env("page.txt", "<x>") == (False, "<x>")
            """,
            "jinja2.select_autoescape is unchanged.",
        ),
    ),
    extra_requirements=_J2_EXTRA,
)

_J2_ESCAPE = ApiChange(
    package="jinja2",
    api_family="jinja2.escape",
    old_api="jinja2.escape",
    new_api="markupsafe.escape",
    old_pin=_J2_OLD,
    new_pin=_J2_NEW,
    change_date=_J2_DATE,
    change_url=_J2_URL,
    summary="Jinja 3.1 removed the jinja2.escape alias (deprecated in 3.0); import escape from markupsafe.",
    programs=(
        _p(
            "escape_all",
            """
            from jinja2 import escape

            def escape_all(values):
                return [str(escape(v)) for v in values]
            """,
            None,
            """
            import solution
            assert solution.escape_all(["<a>", "b & c", "d"]) == ["&lt;a&gt;", "b &amp; c", "d"]
            assert solution.escape_all([]) == []
            assert solution.escape_all(["'"]) == ["&#39;"]
            """,
            "Escape a list of strings.",
            fix=(("from jinja2 import escape", "from markupsafe import escape"),),
        ),
        _p(
            "html_attr",
            """
            from jinja2 import escape

            def html_attr(name, value):
                return ' %s="%s"' % (name, escape(value))
            """,
            None,
            """
            import solution
            assert solution.html_attr("title", 'say "hi"') == ' title="say &#34;hi&#34;"'
            assert solution.html_attr("alt", "cats & dogs") == ' alt="cats &amp; dogs"'
            """,
            "Format an escaped HTML attribute.",
            fix=(("from jinja2 import escape", "from markupsafe import escape"),),
        ),
        _p(
            "table_row",
            """
            import jinja2

            def table_row(cells):
                return "<tr>" + "".join("<td>%s</td>" % jinja2.escape(c) for c in cells) + "</tr>"
            """,
            None,
            """
            import solution
            assert solution.table_row(["a", "<b>"]) == "<tr><td>a</td><td>&lt;b&gt;</td></tr>"
            assert solution.table_row([]) == "<tr></tr>"
            assert solution.table_row([1, 2]) == "<tr><td>1</td><td>2</td></tr>"
            """,
            "Render an escaped table row.",
            fix=(("import jinja2\n", "import markupsafe\n"), ("jinja2.escape", "markupsafe.escape")),
        ),
        _p(
            "escaper_cache",
            """
            from jinja2 import escape

            class Escaper:
                def __init__(self):
                    self.cache = {}

                def __call__(self, text):
                    if text not in self.cache:
                        self.cache[text] = str(escape(text))
                    return self.cache[text]
            """,
            None,
            """
            import solution
            esc = solution.Escaper()
            assert esc("<x>") == "&lt;x&gt;"
            assert esc("<x>") == "&lt;x&gt;"
            assert esc("ok") == "ok"
            assert sorted(esc.cache) == ["<x>", "ok"]
            """,
            "Memoising escaper callable.",
            fix=(("from jinja2 import escape", "from markupsafe import escape"),),
        ),
        _p(
            "html_comment",
            """
            from jinja2 import escape

            def html_comment(text):
                return "<!-- %s -->" % escape(text)
            """,
            None,
            """
            import solution
            assert solution.html_comment("note") == "<!-- note -->"
            assert solution.html_comment("--> <script>") == "<!-- --&gt; &lt;script&gt; -->"
            """,
            "Wrap escaped text in an HTML comment.",
            fix=(("from jinja2 import escape", "from markupsafe import escape"),),
        ),
        _p(
            "escape_unless_safe",
            """
            from jinja2 import escape

            def escape_unless_safe(values):
                return [str(escape(v)) for v in values]
            """,
            None,
            """
            import solution
            from markupsafe import Markup
            assert solution.escape_unless_safe(["<b>", Markup("<b>")]) == ["&lt;b&gt;", "<b>"]
            assert solution.escape_unless_safe([5]) == ["5"]
            """,
            "Escape plain strings but pass Markup objects through.",
            fix=(("from jinja2 import escape", "from markupsafe import escape"),),
        ),
    ),
    siblings=(
        _s(
            "escape_filter_render",
            """
            from jinja2 import Environment

            ENV = Environment()

            def escape_filter_render(value):
                return ENV.from_string("{{ v | e }}").render(v=value)
            """,
            """
            import solution
            assert solution.escape_filter_render("<b>") == "&lt;b&gt;"
            assert solution.escape_filter_render("x") == "x"
            """,
            "The built-in escape filter is unchanged.",
        ),
        _s(
            "forceescape_render",
            """
            from jinja2 import Environment

            def forceescape_render(value):
                return Environment().from_string("{{ v | forceescape }}").render(v=value)
            """,
            """
            import solution
            from markupsafe import Markup
            assert solution.forceescape_render(Markup("<b>")) == "&lt;b&gt;"
            """,
            "The forceescape filter is unchanged.",
        ),
    ),
    extra_requirements=_J2_EXTRA,
)

_J2_CONTEXTFUNCTION = ApiChange(
    package="jinja2",
    api_family="jinja2.contextfunction",
    old_api="@jinja2.contextfunction",
    new_api="@jinja2.pass_context",
    old_pin=_J2_OLD,
    new_pin=_J2_NEW,
    change_date=_J2_DATE,
    change_url=_J2_URL,
    summary="Jinja 3.1 removed the contextfunction decorator (deprecated in 3.0); use pass_context.",
    programs=(
        _p(
            "render_greeting",
            """
            from jinja2 import Environment, contextfunction

            @contextfunction
            def greet(ctx):
                return "Hello " + ctx["user"]

            def render_greeting(user):
                env = Environment()
                env.globals["greet"] = greet
                return env.from_string("{{ greet() }}!").render(user=user)
            """,
            None,
            """
            import solution
            assert solution.render_greeting("Ana") == "Hello Ana!"
            assert solution.render_greeting("Bob") == "Hello Bob!"
            """,
            "Global function reading the template context.",
            fix=(("contextfunction", "pass_context"),),
        ),
        _p(
            "render_item_count",
            """
            from jinja2 import Environment, contextfunction

            @contextfunction
            def item_count(ctx, label):
                return "%d %s" % (len(ctx["items"]), label)

            def render_item_count(items):
                env = Environment()
                env.globals["item_count"] = item_count
                return env.from_string("{{ item_count('rows') }}").render(items=items)
            """,
            None,
            """
            import solution
            assert solution.render_item_count([1, 2, 3]) == "3 rows"
            assert solution.render_item_count([]) == "0 rows"
            """,
            "Context-aware global with an extra argument.",
            fix=(("contextfunction", "pass_context"),),
        ),
        _p(
            "site_links",
            """
            from jinja2 import Environment, contextfunction

            class Site:
                def __init__(self, base):
                    self.base = base
                    self.env = Environment()
                    self.env.globals["url_for"] = self.url_for

                @staticmethod
                @contextfunction
                def url_for(ctx, path):
                    return ctx["base"].rstrip("/") + "/" + path.lstrip("/")

                def render(self, template):
                    return self.env.from_string(template).render(base=self.base)
            """,
            None,
            """
            import solution
            site = solution.Site("https://example.org/")
            assert site.render("<a href=\\"{{ url_for('/docs') }}\\">d</a>") == '<a href="https://example.org/docs">d</a>'
            assert solution.Site("http://h").render("{{ url_for('x/y') }}") == "http://h/x/y"
            """,
            "URL helper that reads the base URL from the context.",
            fix=(("contextfunction", "pass_context"),),
        ),
        _p(
            "render_with_prefix",
            """
            import jinja2

            @jinja2.contextfunction
            def titled(ctx, text):
                return ctx["prefix"] + " " + text.title()

            def render_with_prefix(prefix, name):
                env = jinja2.Environment()
                env.globals["titled"] = titled
                return env.from_string("{{ titled(name) }}").render(prefix=prefix, name=name)
            """,
            None,
            """
            import solution
            assert solution.render_with_prefix("Dr.", "jane doe") == "Dr. Jane Doe"
            assert solution.render_with_prefix("Ms", "ann") == "Ms Ann"
            """,
            "Title-case helper using a context prefix.",
            fix=(("jinja2.contextfunction", "jinja2.pass_context"),),
        ),
        _p(
            "translate",
            """
            from jinja2 import Environment, contextfunction

            @contextfunction
            def t(ctx, key):
                return ctx["messages"].get(key, key)

            def translate(template, messages):
                env = Environment()
                env.globals["t"] = t
                return env.from_string(template).render(messages=messages)
            """,
            None,
            """
            import solution
            msgs = {"hello": "hola", "bye": "adios"}
            assert solution.translate("{{ t('hello') }}, {{ t('bye') }}", msgs) == "hola, adios"
            assert solution.translate("{{ t('missing') }}", msgs) == "missing"
            """,
            "Message lookup through a context dictionary.",
            fix=(("contextfunction", "pass_context"),),
        ),
        _p(
            "render_flags",
            """
            from jinja2 import Environment, contextfunction

            @contextfunction
            def enabled(ctx, *names):
                return [n for n in names if ctx["flags"].get(n)]

            def render_flags(flags):
                env = Environment()
                env.globals["enabled"] = enabled
                return env.from_string("{{ enabled('a', 'b', 'c') | join(',') }}").render(flags=flags)
            """,
            None,
            """
            import solution
            assert solution.render_flags({"a": True, "b": False, "c": True}) == "a,c"
            assert solution.render_flags({}) == ""
            """,
            "Filter feature names by flags stored in the context.",
            fix=(("contextfunction", "pass_context"),),
        ),
    ),
    siblings=(
        _s(
            "plain_global",
            """
            from jinja2 import Environment

            def shout(text):
                return text.upper() + "!"

            def plain_global(text):
                env = Environment()
                env.globals["shout"] = shout
                return env.from_string("{{ shout(t) }}").render(t=text)
            """,
            """
            import solution
            assert solution.plain_global("hey") == "HEY!"
            """,
            "Registering plain callables in env.globals is unchanged.",
        ),
        _s(
            "namespace_global",
            """
            from jinja2 import Environment

            def namespace_global(values):
                tpl = "{% set ns = namespace(total=0) %}{% for v in values %}{% set ns.total = ns.total + v %}{% endfor %}{{ ns.total }}"
                return int(Environment().from_string(tpl).render(values=values))
            """,
            """
            import solution
            assert solution.namespace_global([1, 2, 3]) == 6
            assert solution.namespace_global([]) == 0
            """,
            "The namespace() global is unchanged.",
        ),
    ),
    extra_requirements=_J2_EXTRA,
)

_J2_CONTEXTFILTER = ApiChange(
    package="jinja2",
    api_family="jinja2.contextfilter",
    old_api="@jinja2.contextfilter",
    new_api="@jinja2.pass_context",
    old_pin=_J2_OLD,
    new_pin=_J2_NEW,
    change_date=_J2_DATE,
    change_url=_J2_URL,
    summary="Jinja 3.1 removed the contextfilter decorator (deprecated in 3.0); use pass_context.",
    programs=(
        _p(
            "currency_filter",
            """
            from jinja2 import Environment, contextfilter

            @contextfilter
            def currency(ctx, amount):
                return "%s%.2f" % (ctx["symbol"], amount)

            def render_price(amount, symbol):
                env = Environment()
                env.filters["currency"] = currency
                return env.from_string("{{ amount | currency }}").render(amount=amount, symbol=symbol)
            """,
            None,
            """
            import solution
            assert solution.render_price(3.5, "$") == "$3.50"
            assert solution.render_price(10, "EUR ") == "EUR 10.00"
            """,
            "Currency filter reading the symbol from the context.",
            fix=(("contextfilter", "pass_context"),),
        ),
        _p(
            "highlight_filter",
            """
            from jinja2 import Environment, contextfilter

            @contextfilter
            def highlight(ctx, text):
                query = ctx["query"]
                return text.replace(query, "[" + query + "]") if query else text

            def render_highlight(text, query):
                env = Environment()
                env.filters["highlight"] = highlight
                return env.from_string("{{ text | highlight }}").render(text=text, query=query)
            """,
            None,
            """
            import solution
            assert solution.render_highlight("the cat sat", "cat") == "the [cat] sat"
            assert solution.render_highlight("the cat sat", "") == "the cat sat"
            """,
            "Bracket matches of a context-supplied query.",
            fix=(("contextfilter", "pass_context"),),
        ),
        _p(
            "relative_url_filter",
            """
            import jinja2

            @jinja2.contextfilter
            def absolute(ctx, path):
                return ctx["base"].rstrip("/") + "/" + path.lstrip("/")

            def render_links(paths, base):
                env = jinja2.Environment()
                env.filters["absolute"] = absolute
                tpl = "{% for p in paths %}{{ p | absolute }}{% if not loop.last %} {% endif %}{% endfor %}"
                return env.from_string(tpl).render(paths=paths, base=base)
            """,
            None,
            """
            import solution
            assert solution.render_links(["/a", "b"], "http://x/") == "http://x/a http://x/b"
            assert solution.render_links([], "http://x") == ""
            """,
            "Filter resolving paths against a context base URL.",
            fix=(("jinja2.contextfilter", "jinja2.pass_context"),),
        ),
        _p(
            "precision_formatter",
            """
            from jinja2 import Environment, contextfilter

            class Formatter:
                def __init__(self):
                    self.env = Environment()
                    self.env.filters["num"] = self.num

                @staticmethod
                @contextfilter
                def num(ctx, value):
                    return ("%." + str(ctx["precision"]) + "f") % value

                def render(self, values, precision):
                    tpl = "{{ values | map('num') | join(';') }}"
                    return self.env.from_string(tpl).render(values=values, precision=precision)
            """,
            None,
            """
            import solution
            f = solution.Formatter()
            assert f.render([1, 2.5], 1) == "1.0;2.5"
            assert f.render([3.14159], 3) == "3.142"
            """,
            "Number formatting with context-controlled precision.",
            fix=(("contextfilter", "pass_context"),),
        ),
        _p(
            "pluralize_filter",
            """
            from jinja2 import Environment, contextfilter

            @contextfilter
            def plural(ctx, word, count):
                forms = ctx["plurals"]
                return word if count == 1 else forms.get(word, word + "s")

            def render_count(word, count, plurals):
                env = Environment()
                env.filters["plural"] = plural
                return env.from_string("{{ count }} {{ word | plural(count) }}").render(word=word, count=count, plurals=plurals)
            """,
            None,
            """
            import solution
            assert solution.render_count("mouse", 2, {"mouse": "mice"}) == "2 mice"
            assert solution.render_count("mouse", 1, {"mouse": "mice"}) == "1 mouse"
            assert solution.render_count("cat", 3, {}) == "3 cats"
            """,
            "Pluralisation using irregular forms from the context.",
            fix=(("contextfilter", "pass_context"),),
        ),
        _p(
            "mask_filter",
            """
            from jinja2 import Environment, contextfilter

            @contextfilter
            def mask(ctx, value):
                if ctx.get("show_secrets"):
                    return value
                return "*" * len(value)

            def render_secret(value, show_secrets):
                env = Environment()
                env.filters["mask"] = mask
                return env.from_string("{{ v | mask }}").render(v=value, show_secrets=show_secrets)
            """,
            None,
            """
            import solution
            assert solution.render_secret("hunter2", False) == "*******"
            assert solution.render_secret("hunter2", True) == "hunter2"
            """,
            "Mask values unless the context allows showing them.",
            fix=(("contextfilter", "pass_context"),),
        ),
    ),
    siblings=(
        _s(
            "plain_filter",
            """
            from jinja2 import Environment

            def shout(text):
                return text.upper()

            def plain_filter(text):
                env = Environment()
                env.filters["shout"] = shout
                return env.from_string("{{ t | shout }}").render(t=text)
            """,
            """
            import solution
            assert solution.plain_filter("hey") == "HEY"
            """,
            "Registering plain filters is unchanged.",
        ),
        _s(
            "builtin_filters",
            """
            from jinja2 import Environment

            def builtin_filters(values):
                tpl = "{{ values | sort | join(',') }}|{{ values | max }}|{{ values | length }}"
                return Environment().from_string(tpl).render(values=values)
            """,
            """
            import solution
            assert solution.builtin_filters([3, 1, 2]) == "1,2,3|3|3"
            """,
            "Built-in sort/join/max/length filters are unchanged.",
        ),
    ),
    extra_requirements=_J2_EXTRA,
)

_J2_ENVFILTER = ApiChange(
    package="jinja2",
    api_family="jinja2.environmentfilter",
    old_api="@jinja2.environmentfilter",
    new_api="@jinja2.pass_environment",
    old_pin=_J2_OLD,
    new_pin=_J2_NEW,
    change_date=_J2_DATE,
    change_url=_J2_URL,
    summary="Jinja 3.1 removed the environmentfilter decorator (deprecated in 3.0); use pass_environment.",
    programs=(
        _p(
            "chain_filter",
            """
            from jinja2 import Environment, environmentfilter

            @environmentfilter
            def chain(env, value, *names):
                for name in names:
                    value = env.filters[name](value)
                return value

            def render_chain(value, names):
                env = Environment()
                env.filters["chain"] = chain
                return env.from_string("{{ v | chain(*names) }}").render(v=value, names=names)
            """,
            None,
            """
            import solution
            assert solution.render_chain("abc", ["upper", "reverse"]) == "CBA"
            assert solution.render_chain("  x ", ["trim"]) == "x"
            assert solution.render_chain("q", []) == "q"
            """,
            "Apply a sequence of environment filters by name.",
            fix=(("environmentfilter", "pass_environment"),),
        ),
        _p(
            "suffix_filter",
            """
            from jinja2 import Environment, environmentfilter

            @environmentfilter
            def with_suffix(env, value):
                return value + env.globals.get("suffix", "")

            def render_suffixed(values, suffix):
                env = Environment()
                env.globals["suffix"] = suffix
                env.filters["with_suffix"] = with_suffix
                return env.from_string("{{ values | map('with_suffix') | join(' ') }}").render(values=values)
            """,
            None,
            """
            import solution
            assert solution.render_suffixed(["a", "b"], ".txt") == "a.txt b.txt"
            assert solution.render_suffixed(["a"], "") == "a"
            """,
            "Append a suffix stored in the environment globals.",
            fix=(("environmentfilter", "pass_environment"),),
        ),
        _p(
            "pluck_filter",
            """
            import jinja2

            @jinja2.environmentfilter
            def pluck(env, items, key):
                return [env.getitem(item, key) for item in items]

            def render_pluck(items, key):
                env = jinja2.Environment()
                env.filters["pluck"] = pluck
                return env.from_string("{{ items | pluck(key) | join(',') }}").render(items=items, key=key)
            """,
            None,
            """
            import solution
            rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
            assert solution.render_pluck(rows, "name") == "a,b"
            assert solution.render_pluck(rows, "id") == "1,2"
            assert solution.render_pluck([], "id") == ""
            """,
            "Pluck a key from each item using env.getitem.",
            fix=(("jinja2.environmentfilter", "jinja2.pass_environment"),),
        ),
        _p(
            "pipeline",
            """
            from jinja2 import Environment, environmentfilter

            class Pipeline:
                def __init__(self, steps):
                    self.steps = steps
                    self.env = Environment()
                    self.env.filters["run"] = self.run

                @staticmethod
                @environmentfilter
                def run(env, value):
                    for step in env.globals["steps"]:
                        value = env.filters[step](value)
                    return value

                def apply(self, value):
                    self.env.globals["steps"] = self.steps
                    return self.env.from_string("{{ v | run }}").render(v=value)
            """,
            None,
            """
            import solution
            assert solution.Pipeline(["trim", "upper"]).apply("  hi ") == "HI"
            assert solution.Pipeline(["capitalize"]).apply("hELLO") == "Hello"
            """,
            "Named filter pipeline resolved through the environment.",
            fix=(("environmentfilter", "pass_environment"),),
        ),
        _p(
            "nl2br_env",
            """
            from jinja2 import Environment, environmentfilter
            from markupsafe import Markup, escape

            @environmentfilter
            def nl2br(env, value):
                result = "<br>".join(str(escape(line)) for line in value.split("\\n"))
                return Markup(result) if env.autoescape else result

            def render_nl2br(text, autoescape):
                env = Environment(autoescape=autoescape)
                env.filters["nl2br"] = nl2br
                return env.from_string("{{ t | nl2br }}").render(t=text)
            """,
            None,
            """
            import solution
            assert solution.render_nl2br("a\\nb", True) == "a<br>b"
            assert solution.render_nl2br("a\\nb", False) == "a<br>b"
            assert solution.render_nl2br("<x>\\ny", True) == "&lt;x&gt;<br>y"
            """,
            "Newline-to-br filter that respects env.autoescape.",
            fix=(("environmentfilter", "pass_environment"),),
        ),
        _p(
            "default_filter",
            """
            from jinja2 import Environment, environmentfilter

            @environmentfilter
            def or_default(env, value, name):
                if value in (None, ""):
                    return env.globals["defaults"][name]
                return value

            def render_defaults(values, defaults):
                env = Environment()
                env.globals["defaults"] = defaults
                env.filters["or_default"] = or_default
                tpl = "{{ host | or_default('host') }}:{{ port | or_default('port') }}"
                return env.from_string(tpl).render(**values)
            """,
            None,
            """
            import solution
            defaults = {"host": "localhost", "port": 80}
            assert solution.render_defaults({"host": "", "port": None}, defaults) == "localhost:80"
            assert solution.render_defaults({"host": "h", "port": 8080}, defaults) == "h:8080"
            """,
            "Fallback values taken from environment globals.",
            fix=(("environmentfilter", "pass_environment"),),
        ),
    ),
    siblings=(
        _s(
            "env_filter_lookup",
            """
            from jinja2 import Environment

            def env_filter_lookup(value, names):
                env = Environment()
                for name in names:
                    value = env.filters[name](value)
                return value
            """,
            """
            import solution
            assert solution.env_filter_lookup("abc", ["upper", "reverse"]) == "CBA"
            assert solution.env_filter_lookup(" x ", ["trim"]) == "x"
            """,
            "Looking filters up in env.filters is unchanged.",
        ),
        _s(
            "env_getitem",
            """
            from jinja2 import Environment

            def env_getitem(items, key):
                env = Environment()
                return [env.getitem(item, key) for item in items]
            """,
            """
            import solution
            assert solution.env_getitem([{"a": 1}, {"a": 2}], "a") == [1, 2]
            """,
            "Environment.getitem is unchanged.",
        ),
    ),
    extra_requirements=_J2_EXTRA,
)


# ---------------------------------------------------------------------------
# pydantic 1.10.18 -> 2.9.2
# ---------------------------------------------------------------------------

_PYD_URL = "https://docs.pydantic.dev/latest/migration/"
_PYD_DATE = "2023-06-30"
_PYD_OLD = "1.10.18"
_PYD_NEW = "2.9.2"

_PYD_REGEX = ApiChange(
    package="pydantic",
    api_family="pydantic.Field(regex=)",
    old_api="Field(regex=...)",
    new_api="Field(pattern=...)",
    old_pin=_PYD_OLD,
    new_pin=_PYD_NEW,
    change_date=_PYD_DATE,
    change_url=_PYD_URL,
    summary="pydantic 2 removed the Field(regex=) keyword (raises PydanticUserError); use Field(pattern=).",
    programs=(
        _p(
            "parse_sku",
            """
            from pydantic import BaseModel, Field

            class Product(BaseModel):
                sku: str = Field(..., regex=r"^[A-Z]{2}-\\d{3}$")
                qty: int = 1

            def parse_sku(data):
                p = Product(**data)
                return p.sku, p.qty
            """,
            None,
            """
            import solution
            assert solution.parse_sku({"sku": "AB-123", "qty": "4"}) == ("AB-123", 4)
            assert solution.parse_sku({"sku": "ZZ-000"}) == ("ZZ-000", 1)
            try:
                solution.parse_sku({"sku": "ab-123"})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Validate a product SKU against a pattern.",
            fix=(("regex=", "pattern="),),
        ),
        _p(
            "make_user",
            """
            from pydantic import BaseModel, Field

            class User(BaseModel):
                username: str = Field(regex="^[a-z_]{3,16}$")
                email: str = Field(regex=r"^[^@\\s]+@[^@\\s]+\\.[a-z]+$")

            def make_user(username, email):
                user = User(username=username, email=email)
                return {"username": user.username, "domain": user.email.split("@")[1]}
            """,
            None,
            """
            import solution
            assert solution.make_user("alice_b", "alice@example.com") == {"username": "alice_b", "domain": "example.com"}
            for bad in (("Al", "a@b.com"), ("alice", "not-an-email")):
                try:
                    solution.make_user(*bad)
                    raise AssertionError("expected rejection")
                except ValueError:
                    pass
            """,
            "User record with username and e-mail patterns.",
            fix=(("regex=", "pattern="), ("regex=", "pattern=")),
        ),
        _p(
            "count_valid_colors",
            """
            from pydantic import BaseModel, Field, ValidationError

            class Swatch(BaseModel):
                hex_code: str = Field(regex="^#[0-9a-fA-F]{6}$")

            def count_valid_colors(codes):
                good, bad = 0, []
                for code in codes:
                    try:
                        Swatch(hex_code=code)
                        good += 1
                    except ValidationError:
                        bad.append(code)
                return good, bad
            """,
            None,
            """
            import solution
            assert solution.count_valid_colors(["#ffffff", "#12ab9F", "red", "#12345"]) == (2, ["red", "#12345"])
            assert solution.count_valid_colors([]) == (0, [])
            """,
            "Count hex colour codes that match the pattern.",
            fix=(("regex=", "pattern="),),
        ),
        _p(
            "normalize_path",
            """
            from pydantic import BaseModel, Field

            class Endpoint(BaseModel):
                path: str = Field("/", regex="^/")
                method: str = "GET"

            def normalize_path(path=None, method="get"):
                ep = Endpoint(method=method.upper()) if path is None else Endpoint(path=path, method=method.upper())
                return ep.method + " " + ep.path.rstrip("/") + "/"
            """,
            None,
            """
            import solution
            assert solution.normalize_path("/users/") == "GET /users/"
            assert solution.normalize_path() == "GET /"
            assert solution.normalize_path("/a", "post") == "POST /a/"
            try:
                solution.normalize_path("users")
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Endpoint path that must start with a slash.",
            fix=(("regex=", "pattern="),),
        ),
        _p(
            "phone_book",
            """
            from pydantic import BaseModel, Field

            class Entry(BaseModel):
                name: str
                phone: str = Field(regex=r"^\\+?\\d{7,15}$")

            class PhoneBook:
                def __init__(self):
                    self.entries = []

                def add(self, name, phone):
                    self.entries.append(Entry(name=name, phone=phone))
                    return len(self.entries)

                def lookup(self, name):
                    return [e.phone for e in self.entries if e.name == name]
            """,
            None,
            """
            import solution
            book = solution.PhoneBook()
            assert book.add("ann", "+4915112345678") == 1
            assert book.add("bob", "5551234") == 2
            assert book.lookup("ann") == ["+4915112345678"]
            try:
                book.add("eve", "555-1234")
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            assert len(book.entries) == 2
            """,
            "Phone book validating numbers against a pattern.",
            fix=(("regex=", "pattern="),),
        ),
        _p(
            "parse_version_tag",
            """
            from pydantic import BaseModel, Field

            class Tag(BaseModel):
                value: str = Field(regex=r"^v\\d+\\.\\d+\\.\\d+$")

            def parse_version_tag(tag):
                checked = Tag(value=tag).value
                return tuple(int(part) for part in checked[1:].split("."))
            """,
            None,
            """
            import solution
            assert solution.parse_version_tag("v1.2.3") == (1, 2, 3)
            assert solution.parse_version_tag("v10.0.1") == (10, 0, 1)
            try:
                solution.parse_version_tag("1.2.3")
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Parse a vMAJOR.MINOR.PATCH tag into a tuple.",
            fix=(("regex=", "pattern="),),
        ),
    ),
    siblings=(
        _s(
            "length_bounds",
            """
            from pydantic import BaseModel, Field

            class Handle(BaseModel):
                name: str = Field(min_length=3, max_length=8)

            def length_bounds(name):
                try:
                    return Handle(name=name).name
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.length_bounds("alice") == "alice"
            assert solution.length_bounds("al") is None
            assert solution.length_bounds("averyveryverylong") is None
            """,
            "Field(min_length=, max_length=) works on both pins.",
        ),
        _s(
            "numeric_bounds",
            """
            from pydantic import BaseModel, Field

            class Score(BaseModel):
                value: int = Field(ge=0, le=100)

            def numeric_bounds(value):
                try:
                    return Score(value=value).value
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.numeric_bounds(50) == 50
            assert solution.numeric_bounds("7") == 7
            assert solution.numeric_bounds(101) is None
            """,
            "Field(ge=, le=) works on both pins.",
        ),
    ),
)

_PYD_CONST = ApiChange(
    package="pydantic",
    api_family="pydantic.Field(const=)",
    old_api="Field(default, const=True)",
    new_api="typing.Literal[default]",
    old_pin=_PYD_OLD,
    new_pin=_PYD_NEW,
    change_date=_PYD_DATE,
    change_url=_PYD_URL,
    summary="pydantic 2 removed Field(const=True) (raises PydanticUserError); use a Literal type instead.",
    programs=(
        _p(
            "make_event",
            """
            from pydantic import BaseModel, Field

            class ClickEvent(BaseModel):
                kind: str = Field("click", const=True)
                x: int
                y: int

            def make_event(data):
                ev = ClickEvent(**data)
                return ev.kind, (ev.x, ev.y)
            """,
            """
            from typing import Literal
            from pydantic import BaseModel, Field

            class ClickEvent(BaseModel):
                kind: Literal["click"] = "click"
                x: int
                y: int

            def make_event(data):
                ev = ClickEvent(**data)
                return ev.kind, (ev.x, ev.y)
            """,
            """
            import solution
            assert solution.make_event({"x": 1, "y": 2}) == ("click", (1, 2))
            assert solution.make_event({"kind": "click", "x": "3", "y": 4}) == ("click", (3, 4))
            try:
                solution.make_event({"kind": "tap", "x": 1, "y": 2})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Event payload with a constant discriminator.",
        ),
        _p(
            "api_request",
            """
            from pydantic import BaseModel, Field

            class ApiRequest(BaseModel):
                version: int = Field(2, const=True)
                action: str

            def parse_request(payload):
                req = ApiRequest(**payload)
                return "v%d:%s" % (req.version, req.action)
            """,
            """
            from typing import Literal
            from pydantic import BaseModel, Field

            class ApiRequest(BaseModel):
                version: Literal[2] = 2
                action: str

            def parse_request(payload):
                req = ApiRequest(**payload)
                return "v%d:%s" % (req.version, req.action)
            """,
            """
            import solution
            assert solution.parse_request({"action": "list"}) == "v2:list"
            assert solution.parse_request({"version": 2, "action": "get"}) == "v2:get"
            try:
                solution.parse_request({"version": 1, "action": "get"})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Request envelope pinned to one protocol version.",
        ),
        _p(
            "circle_area",
            """
            import math
            from pydantic import BaseModel, Field

            class Circle(BaseModel):
                shape: str = Field("circle", const=True)
                radius: float

                def area(self):
                    return round(math.pi * self.radius ** 2, 3)

            def circle_area(data):
                return Circle(**data).area()
            """,
            """
            import math
            from typing import Literal
            from pydantic import BaseModel, Field

            class Circle(BaseModel):
                shape: Literal["circle"] = "circle"
                radius: float

                def area(self):
                    return round(math.pi * self.radius ** 2, 3)

            def circle_area(data):
                return Circle(**data).area()
            """,
            """
            import solution
            assert solution.circle_area({"radius": 1}) == 3.142
            assert solution.circle_area({"shape": "circle", "radius": 2}) == 12.566
            try:
                solution.circle_area({"shape": "square", "radius": 2})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Shape model with a constant kind and an area method.",
        ),
        _p(
            "unpack_envelope",
            """
            from pydantic import BaseModel, Field

            class Envelope(BaseModel):
                schema_id: str = Field("v1.envelope", const=True)
                body: dict

            def unpack_envelope(raw):
                env = Envelope(**raw)
                return env.schema_id, sorted(env.body)
            """,
            """
            from typing import Literal
            from pydantic import BaseModel, Field

            class Envelope(BaseModel):
                schema_id: Literal["v1.envelope"] = "v1.envelope"
                body: dict

            def unpack_envelope(raw):
                env = Envelope(**raw)
                return env.schema_id, sorted(env.body)
            """,
            """
            import solution
            assert solution.unpack_envelope({"body": {"b": 1, "a": 2}}) == ("v1.envelope", ["a", "b"])
            try:
                solution.unpack_envelope({"schema_id": "v2.envelope", "body": {}})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Message envelope with a fixed schema identifier.",
        ),
        _p(
            "plugin_registry",
            """
            from pydantic import BaseModel, Field, ValidationError

            class PluginSpec(BaseModel):
                type: str = Field("plugin", const=True)
                name: str

            class Registry:
                def __init__(self):
                    self.names = []

                def register(self, payload):
                    try:
                        spec = PluginSpec(**payload)
                    except ValidationError:
                        return False
                    self.names.append(spec.name)
                    return True
            """,
            """
            from typing import Literal
            from pydantic import BaseModel, Field, ValidationError

            class PluginSpec(BaseModel):
                type: Literal["plugin"] = "plugin"
                name: str

            class Registry:
                def __init__(self):
                    self.names = []

                def register(self, payload):
                    try:
                        spec = PluginSpec(**payload)
                    except ValidationError:
                        return False
                    self.names.append(spec.name)
                    return True
            """,
            """
            import solution
            reg = solution.Registry()
            assert reg.register({"name": "a"}) is True
            assert reg.register({"type": "theme", "name": "b"}) is False
            assert reg.register({"type": "plugin", "name": "c"}) is True
            assert reg.names == ["a", "c"]
            """,
            "Registry accepting only payloads of a fixed type.",
        ),
        _p(
            "point3d",
            """
            from pydantic import BaseModel, Field

            class Point3D(BaseModel):
                dims: int = Field(3, const=True)
                coords: list

            def point_norm_sq(data):
                p = Point3D(**data)
                return p.dims, sum(c * c for c in p.coords[: p.dims])
            """,
            """
            from typing import Literal
            from pydantic import BaseModel, Field

            class Point3D(BaseModel):
                dims: Literal[3] = 3
                coords: list

            def point_norm_sq(data):
                p = Point3D(**data)
                return p.dims, sum(c * c for c in p.coords[: p.dims])
            """,
            """
            import solution
            assert solution.point_norm_sq({"coords": [1, 2, 2]}) == (3, 9)
            assert solution.point_norm_sq({"dims": 3, "coords": [0, 0, 1, 9]}) == (3, 1)
            try:
                solution.point_norm_sq({"dims": 2, "coords": [1, 1]})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Three-dimensional point with a constant dimension count.",
        ),
    ),
    siblings=(
        _s(
            "literal_kind",
            """
            from typing import Literal
            from pydantic import BaseModel

            class Shape(BaseModel):
                kind: Literal["square", "circle"]
                size: float

            def literal_kind(data):
                try:
                    s = Shape(**data)
                except ValueError:
                    return None
                return s.kind, s.size
            """,
            """
            import solution
            assert solution.literal_kind({"kind": "square", "size": 2}) == ("square", 2.0)
            assert solution.literal_kind({"kind": "hexagon", "size": 2}) is None
            """,
            "Literal-typed fields work on both pins.",
        ),
        _s(
            "default_factory_field",
            """
            from pydantic import BaseModel, Field

            class Basket(BaseModel):
                items: list = Field(default_factory=list)
                owner: str = Field(default="anon")

            def default_factory_field(data):
                b = Basket(**data)
                return b.owner, len(b.items)
            """,
            """
            import solution
            assert solution.default_factory_field({}) == ("anon", 0)
            assert solution.default_factory_field({"items": [1, 2], "owner": "me"}) == ("me", 2)
            """,
            "Field(default_factory=) works on both pins.",
        ),
    ),
)

_PYD_UNIQUE = ApiChange(
    package="pydantic",
    api_family="pydantic.Field(unique_items=)",
    old_api="Field(unique_items=True)",
    new_api="@field_validator rejecting duplicates (or a set type)",
    old_pin=_PYD_OLD,
    new_pin=_PYD_NEW,
    change_date=_PYD_DATE,
    change_url=_PYD_URL,
    summary="pydantic 2 removed Field(unique_items=True) (raises PydanticUserError); enforce uniqueness with a validator or a set.",
    programs=(
        _p(
            "parse_tags",
            """
            from typing import List
            from pydantic import BaseModel, Field

            class Post(BaseModel):
                title: str
                tags: List[str] = Field(unique_items=True)

            def parse_tags(data):
                post = Post(**data)
                return post.title, post.tags
            """,
            """
            from typing import List
            from pydantic import BaseModel, Field, field_validator

            class Post(BaseModel):
                title: str
                tags: List[str] = Field()

                @field_validator("tags")
                @classmethod
                def _unique(cls, value):
                    if len(set(value)) != len(value):
                        raise ValueError("tags must be unique")
                    return value

            def parse_tags(data):
                post = Post(**data)
                return post.title, post.tags
            """,
            """
            import solution
            assert solution.parse_tags({"title": "t", "tags": ["a", "b"]}) == ("t", ["a", "b"])
            assert solution.parse_tags({"title": "t", "tags": []}) == ("t", [])
            try:
                solution.parse_tags({"title": "t", "tags": ["a", "a"]})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Post with unique tags.",
        ),
        _p(
            "roster_size",
            """
            from typing import List
            from pydantic import BaseModel, Field

            class Roster(BaseModel):
                members: List[int] = Field(..., unique_items=True)

            def roster_size(ids):
                return len(Roster(members=ids).members)
            """,
            """
            from typing import List
            from pydantic import BaseModel, Field, field_validator

            class Roster(BaseModel):
                members: List[int] = Field(...)

                @field_validator("members")
                @classmethod
                def _unique(cls, value):
                    if len(set(value)) != len(value):
                        raise ValueError("members must be unique")
                    return value

            def roster_size(ids):
                return len(Roster(members=ids).members)
            """,
            """
            import solution
            assert solution.roster_size([3, 1, 2]) == 3
            assert solution.roster_size(["7", 8]) == 2
            try:
                solution.roster_size([1, 2, 1])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Team roster rejecting duplicate member ids.",
        ),
        _p(
            "poll_options",
            """
            from typing import List
            from pydantic import BaseModel, Field, ValidationError

            class Poll(BaseModel):
                question: str
                options: List[str] = Field(min_items=2, unique_items=True)

            def poll_options(question, options):
                try:
                    poll = Poll(question=question, options=options)
                except ValidationError:
                    return None
                return [o.strip() for o in poll.options]
            """,
            """
            from typing import List
            from pydantic import BaseModel, Field, ValidationError, field_validator

            class Poll(BaseModel):
                question: str
                options: List[str] = Field(min_length=2)

                @field_validator("options")
                @classmethod
                def _unique(cls, value):
                    if len(set(value)) != len(value):
                        raise ValueError("options must be unique")
                    return value

            def poll_options(question, options):
                try:
                    poll = Poll(question=question, options=options)
                except ValidationError:
                    return None
                return [o.strip() for o in poll.options]
            """,
            """
            import solution
            assert solution.poll_options("q", ["yes", " no "]) == ["yes", "no"]
            assert solution.poll_options("q", ["yes"]) is None
            assert solution.poll_options("q", ["yes", "yes"]) is None
            """,
            "Poll with at least two distinct options.",
        ),
        _p(
            "route_legs",
            """
            from typing import List
            from pydantic import BaseModel, Field

            class Route(BaseModel):
                stops: List[str] = Field(unique_items=True)

                def legs(self):
                    return list(zip(self.stops, self.stops[1:]))

            def route_legs(stops):
                return Route(stops=stops).legs()
            """,
            """
            from typing import List
            from pydantic import BaseModel, Field, field_validator

            class Route(BaseModel):
                stops: List[str] = Field()

                @field_validator("stops")
                @classmethod
                def _unique(cls, value):
                    if len(set(value)) != len(value):
                        raise ValueError("stops must be unique")
                    return value

                def legs(self):
                    return list(zip(self.stops, self.stops[1:]))

            def route_legs(stops):
                return Route(stops=stops).legs()
            """,
            """
            import solution
            assert solution.route_legs(["a", "b", "c"]) == [("a", "b"), ("b", "c")]
            assert solution.route_legs(["x"]) == []
            try:
                solution.route_legs(["a", "b", "a"])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Route legs between distinct stops.",
        ),
        _p(
            "permission_set",
            """
            from typing import List
            from pydantic import BaseModel, Field, ValidationError

            class Grant(BaseModel):
                scopes: List[str] = Field(default_factory=list, unique_items=True)

            class PermissionSet:
                def __init__(self):
                    self.scopes = []

                def grant(self, scopes):
                    try:
                        grant = Grant(scopes=scopes)
                    except ValidationError:
                        return False
                    self.scopes = sorted(set(self.scopes) | set(grant.scopes))
                    return True
            """,
            """
            from typing import List
            from pydantic import BaseModel, Field, ValidationError, field_validator

            class Grant(BaseModel):
                scopes: List[str] = Field(default_factory=list)

                @field_validator("scopes")
                @classmethod
                def _unique(cls, value):
                    if len(set(value)) != len(value):
                        raise ValueError("scopes must be unique")
                    return value

            class PermissionSet:
                def __init__(self):
                    self.scopes = []

                def grant(self, scopes):
                    try:
                        grant = Grant(scopes=scopes)
                    except ValidationError:
                        return False
                    self.scopes = sorted(set(self.scopes) | set(grant.scopes))
                    return True
            """,
            """
            import solution
            ps = solution.PermissionSet()
            assert ps.grant(["read", "write"]) is True
            assert ps.grant(["read", "read"]) is False
            assert ps.grant(["admin"]) is True
            assert ps.scopes == ["admin", "read", "write"]
            """,
            "Accumulate permission scopes from validated grants.",
        ),
        _p(
            "palette",
            """
            from typing import List
            from pydantic import BaseModel, Field

            class Palette(BaseModel):
                name: str
                colors: List[str] = Field(unique_items=True)

            def palette_summary(name, colors):
                p = Palette(name=name, colors=colors)
                return "%s(%d): %s" % (p.name, len(p.colors), ",".join(p.colors))
            """,
            """
            from typing import List
            from pydantic import BaseModel, Field, field_validator

            class Palette(BaseModel):
                name: str
                colors: List[str] = Field()

                @field_validator("colors")
                @classmethod
                def _unique(cls, value):
                    if len(set(value)) != len(value):
                        raise ValueError("colors must be unique")
                    return value

            def palette_summary(name, colors):
                p = Palette(name=name, colors=colors)
                return "%s(%d): %s" % (p.name, len(p.colors), ",".join(p.colors))
            """,
            """
            import solution
            assert solution.palette_summary("warm", ["#f00", "#fa0"]) == "warm(2): #f00,#fa0"
            try:
                solution.palette_summary("dup", ["#f00", "#f00"])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Named colour palette with distinct entries.",
        ),
    ),
    siblings=(
        _s(
            "min_items_field",
            """
            from typing import List
            from pydantic import BaseModel, Field

            class Batch(BaseModel):
                ids: List[int] = Field(min_items=1)

            def min_items_field(ids):
                try:
                    return len(Batch(ids=ids).ids)
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.min_items_field([1, 2]) == 2
            assert solution.min_items_field([]) is None
            """,
            "Field(min_items=) still works on both pins (deprecated on v2).",
        ),
        _s(
            "validator_unique",
            """
            from typing import List
            from pydantic import BaseModel, validator

            class Tags(BaseModel):
                values: List[str]

                @validator("values")
                def _unique(cls, v):
                    if len(set(v)) != len(v):
                        raise ValueError("dup")
                    return v

            def validator_unique(values):
                try:
                    return Tags(values=values).values
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.validator_unique(["a", "b"]) == ["a", "b"]
            assert solution.validator_unique(["a", "a"]) is None
            """,
            "The v1-style @validator still works on both pins.",
        ),
    ),
)

_PYD_ROOT_VALIDATOR = ApiChange(
    package="pydantic",
    api_family="pydantic.root_validator",
    old_api="@root_validator (pre=False, no skip_on_failure)",
    new_api='@model_validator(mode="after")',
    old_pin=_PYD_OLD,
    new_pin=_PYD_NEW,
    change_date=_PYD_DATE,
    change_url=_PYD_URL,
    summary="pydantic 2 raises PydanticUserError for a bare @root_validator (pre=False without skip_on_failure=True); use @model_validator(mode='after').",
    programs=(
        _p(
            "date_range",
            """
            from pydantic import BaseModel, root_validator

            class DateRange(BaseModel):
                start: int
                end: int

                @root_validator
                def check_order(cls, values):
                    if values["start"] > values["end"]:
                        raise ValueError("start after end")
                    return values

            def span(start, end):
                r = DateRange(start=start, end=end)
                return r.end - r.start
            """,
            """
            from pydantic import BaseModel, model_validator

            class DateRange(BaseModel):
                start: int
                end: int

                @model_validator(mode="after")
                def check_order(self):
                    if self.start > self.end:
                        raise ValueError("start after end")
                    return self

            def span(start, end):
                r = DateRange(start=start, end=end)
                return r.end - r.start
            """,
            """
            import solution
            assert solution.span(3, 10) == 7
            assert solution.span("5", 5) == 0
            try:
                solution.span(10, 3)
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Date range whose start must not exceed its end.",
        ),
        _p(
            "order_total",
            """
            from pydantic import BaseModel, root_validator

            class Order(BaseModel):
                quantity: int
                unit_price: float
                total: float

                @root_validator
                def check_total(cls, values):
                    expected = values["quantity"] * values["unit_price"]
                    if abs(values["total"] - expected) > 1e-6:
                        raise ValueError("total mismatch")
                    return values

            def confirm_order(data):
                order = Order(**data)
                return round(order.total, 2)
            """,
            """
            from pydantic import BaseModel, model_validator

            class Order(BaseModel):
                quantity: int
                unit_price: float
                total: float

                @model_validator(mode="after")
                def check_total(self):
                    expected = self.quantity * self.unit_price
                    if abs(self.total - expected) > 1e-6:
                        raise ValueError("total mismatch")
                    return self

            def confirm_order(data):
                order = Order(**data)
                return round(order.total, 2)
            """,
            """
            import solution
            assert solution.confirm_order({"quantity": 3, "unit_price": 2.5, "total": 7.5}) == 7.5
            try:
                solution.confirm_order({"quantity": 3, "unit_price": 2.5, "total": 8})
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Order whose total must equal quantity times price.",
        ),
        _p(
            "password_change",
            """
            from pydantic import BaseModel, root_validator

            class PasswordChange(BaseModel):
                old: str
                new: str
                confirm: str

                @root_validator
                def check(cls, values):
                    if values["new"] != values["confirm"]:
                        raise ValueError("confirmation mismatch")
                    if values["new"] == values["old"]:
                        raise ValueError("unchanged")
                    return values

            def change_password(old, new, confirm):
                return len(PasswordChange(old=old, new=new, confirm=confirm).new)
            """,
            """
            from pydantic import BaseModel, model_validator

            class PasswordChange(BaseModel):
                old: str
                new: str
                confirm: str

                @model_validator(mode="after")
                def check(self):
                    if self.new != self.confirm:
                        raise ValueError("confirmation mismatch")
                    if self.new == self.old:
                        raise ValueError("unchanged")
                    return self

            def change_password(old, new, confirm):
                return len(PasswordChange(old=old, new=new, confirm=confirm).new)
            """,
            """
            import solution
            assert solution.change_password("a", "bcd", "bcd") == 3
            for args in (("a", "b", "c"), ("a", "a", "a")):
                try:
                    solution.change_password(*args)
                    raise AssertionError("expected rejection")
                except ValueError:
                    pass
            """,
            "Password change with confirmation and reuse checks.",
        ),
        _p(
            "normalized_rect",
            """
            from pydantic import BaseModel, root_validator

            class Rect(BaseModel):
                w: float
                h: float

                @root_validator
                def normalize(cls, values):
                    if values["w"] < values["h"]:
                        values["w"], values["h"] = values["h"], values["w"]
                    return values

            def landscape(w, h):
                r = Rect(w=w, h=h)
                return r.w, r.h, r.w * r.h
            """,
            """
            from pydantic import BaseModel, model_validator

            class Rect(BaseModel):
                w: float
                h: float

                @model_validator(mode="after")
                def normalize(self):
                    if self.w < self.h:
                        self.w, self.h = self.h, self.w
                    return self

            def landscape(w, h):
                r = Rect(w=w, h=h)
                return r.w, r.h, r.w * r.h
            """,
            """
            import solution
            assert solution.landscape(2, 5) == (5.0, 2.0, 10.0)
            assert solution.landscape(4, 3) == (4.0, 3.0, 12.0)
            """,
            "Rectangle normalised so width is the longer side.",
        ),
        _p(
            "interval_contains",
            """
            from pydantic import BaseModel, root_validator

            class Interval(BaseModel):
                low: float
                high: float

                @root_validator
                def check_bounds(cls, values):
                    if values["low"] >= values["high"]:
                        raise ValueError("empty interval")
                    return values

                def contains(self, x):
                    return self.low <= x <= self.high

            def interval_contains(low, high, points):
                iv = Interval(low=low, high=high)
                return [p for p in points if iv.contains(p)]
            """,
            """
            from pydantic import BaseModel, model_validator

            class Interval(BaseModel):
                low: float
                high: float

                @model_validator(mode="after")
                def check_bounds(self):
                    if self.low >= self.high:
                        raise ValueError("empty interval")
                    return self

                def contains(self, x):
                    return self.low <= x <= self.high

            def interval_contains(low, high, points):
                iv = Interval(low=low, high=high)
                return [p for p in points if iv.contains(p)]
            """,
            """
            import solution
            assert solution.interval_contains(0, 10, [-1, 0, 5, 10, 11]) == [0, 5, 10]
            try:
                solution.interval_contains(5, 5, [5])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Non-empty interval with a membership method.",
        ),
        _p(
            "discount_rule",
            """
            from typing import Optional
            from pydantic import BaseModel, root_validator

            class Discount(BaseModel):
                percent: Optional[float] = None
                amount: Optional[float] = None

                @root_validator
                def exactly_one(cls, values):
                    if (values.get("percent") is None) == (values.get("amount") is None):
                        raise ValueError("set exactly one of percent/amount")
                    return values

            def apply_discount(price, **spec):
                d = Discount(**spec)
                if d.percent is not None:
                    return round(price * (1 - d.percent / 100), 2)
                return round(price - d.amount, 2)
            """,
            """
            from typing import Optional
            from pydantic import BaseModel, model_validator

            class Discount(BaseModel):
                percent: Optional[float] = None
                amount: Optional[float] = None

                @model_validator(mode="after")
                def exactly_one(self):
                    if (self.percent is None) == (self.amount is None):
                        raise ValueError("set exactly one of percent/amount")
                    return self

            def apply_discount(price, **spec):
                d = Discount(**spec)
                if d.percent is not None:
                    return round(price * (1 - d.percent / 100), 2)
                return round(price - d.amount, 2)
            """,
            """
            import solution
            assert solution.apply_discount(100, percent=15) == 85.0
            assert solution.apply_discount(100, amount=7.5) == 92.5
            for spec in ({}, {"percent": 10, "amount": 5}):
                try:
                    solution.apply_discount(100, **spec)
                    raise AssertionError("expected rejection")
                except ValueError:
                    pass
            """,
            "Discount that must be given as either a percent or an amount.",
        ),
    ),
    siblings=(
        _s(
            "root_validator_pre",
            """
            from pydantic import BaseModel, root_validator

            class Pair(BaseModel):
                a: int
                b: int

                @root_validator(pre=True)
                def fill_b(cls, values):
                    values.setdefault("b", int(values["a"]) * 2)
                    return values

            def root_validator_pre(a, b=None):
                data = {"a": a} if b is None else {"a": a, "b": b}
                p = Pair(**data)
                return p.a, p.b
            """,
            """
            import solution
            assert solution.root_validator_pre(3) == (3, 6)
            assert solution.root_validator_pre(3, 1) == (3, 1)
            """,
            "@root_validator(pre=True) still works on both pins.",
        ),
        _s(
            "field_validator_v1_style",
            """
            from pydantic import BaseModel, validator

            class Temp(BaseModel):
                celsius: float

                @validator("celsius")
                def not_below_zero_kelvin(cls, v):
                    if v < -273.15:
                        raise ValueError("too cold")
                    return v

            def field_validator_v1_style(c):
                try:
                    return Temp(celsius=c).celsius * 9 / 5 + 32
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.field_validator_v1_style(100) == 212.0
            assert solution.field_validator_v1_style(-300) is None
            """,
            "The v1-style @validator still works on both pins.",
        ),
    ),
)

_PYD_BASESETTINGS = ApiChange(
    package="pydantic",
    api_family="pydantic.BaseSettings",
    old_api="from pydantic import BaseSettings",
    new_api="BaseModel populated from os.environ (or the pydantic-settings package)",
    old_pin=_PYD_OLD,
    new_pin=_PYD_NEW,
    change_date=_PYD_DATE,
    change_url=_PYD_URL,
    summary="pydantic 2 moved BaseSettings to the separate pydantic-settings package; importing it from pydantic raises PydanticImportError.",
    programs=(
        _p(
            "server_settings",
            """
            from pydantic import BaseSettings

            class ServerSettings(BaseSettings):
                host: str = "127.0.0.1"
                port: int = 8080

                class Config:
                    env_prefix = "APP_"

            def load_server_settings():
                s = ServerSettings()
                return s.host, s.port
            """,
            """
            import os
            from pydantic import BaseModel

            class ServerSettings(BaseModel):
                host: str = "127.0.0.1"
                port: int = 8080

            def load_server_settings():
                prefix = "APP_"
                data = {name: os.environ[prefix + name.upper()] for name in ServerSettings.model_fields if prefix + name.upper() in os.environ}
                s = ServerSettings(**data)
                return s.host, s.port
            """,
            """
            import os
            import solution
            os.environ.pop("APP_HOST", None)
            os.environ.pop("APP_PORT", None)
            assert solution.load_server_settings() == ("127.0.0.1", 8080)
            os.environ["APP_PORT"] = "9000"
            os.environ["APP_HOST"] = "0.0.0.0"
            assert solution.load_server_settings() == ("0.0.0.0", 9000)
            """,
            "Server host/port read from prefixed environment variables.",
        ),
        _p(
            "db_config",
            """
            from pydantic import BaseSettings

            class DbConfig(BaseSettings):
                url: str = "sqlite://"
                pool: int = 5

                class Config:
                    env_prefix = "DB_"

            def connection_summary():
                cfg = DbConfig()
                return "%s (pool=%d)" % (cfg.url, cfg.pool)
            """,
            """
            import os
            from pydantic import BaseModel

            class DbConfig(BaseModel):
                url: str = "sqlite://"
                pool: int = 5

            def connection_summary():
                data = {}
                for name in DbConfig.model_fields:
                    key = "DB_" + name.upper()
                    if key in os.environ:
                        data[name] = os.environ[key]
                cfg = DbConfig(**data)
                return "%s (pool=%d)" % (cfg.url, cfg.pool)
            """,
            """
            import os
            import solution
            os.environ.pop("DB_URL", None)
            os.environ.pop("DB_POOL", None)
            assert solution.connection_summary() == "sqlite:// (pool=5)"
            os.environ["DB_URL"] = "postgres://h/db"
            os.environ["DB_POOL"] = "12"
            assert solution.connection_summary() == "postgres://h/db (pool=12)"
            """,
            "Database URL and pool size from the environment.",
        ),
        _p(
            "feature_flags",
            """
            from pydantic import BaseSettings

            class FeatureFlags(BaseSettings):
                beta: bool = False
                dark_mode: bool = False

                class Config:
                    env_prefix = "FF_"

            def enabled_flags():
                flags = FeatureFlags()
                return sorted(name for name, on in flags.dict().items() if on)
            """,
            """
            import os
            from pydantic import BaseModel

            class FeatureFlags(BaseModel):
                beta: bool = False
                dark_mode: bool = False

            def enabled_flags():
                data = {name: os.environ["FF_" + name.upper()] for name in FeatureFlags.model_fields if "FF_" + name.upper() in os.environ}
                flags = FeatureFlags(**data)
                return sorted(name for name, on in flags.model_dump().items() if on)
            """,
            """
            import os
            import solution
            os.environ.pop("FF_BETA", None)
            os.environ.pop("FF_DARK_MODE", None)
            assert solution.enabled_flags() == []
            os.environ["FF_BETA"] = "true"
            os.environ["FF_DARK_MODE"] = "0"
            assert solution.enabled_flags() == ["beta"]
            os.environ["FF_DARK_MODE"] = "yes"
            assert solution.enabled_flags() == ["beta", "dark_mode"]
            """,
            "Boolean feature flags parsed from environment strings.",
        ),
        _p(
            "limits",
            """
            from pydantic import BaseSettings, Field

            class Limits(BaseSettings):
                max_items: int = Field(10, env="MAX_ITEMS")
                timeout: float = Field(2.5, env="REQUEST_TIMEOUT")

            def limits():
                lim = Limits()
                return lim.max_items, lim.timeout
            """,
            """
            import os
            from pydantic import BaseModel, Field

            class Limits(BaseModel):
                max_items: int = Field(10)
                timeout: float = Field(2.5)

            def limits():
                env_names = {"max_items": "MAX_ITEMS", "timeout": "REQUEST_TIMEOUT"}
                data = {field: os.environ[var] for field, var in env_names.items() if var in os.environ}
                lim = Limits(**data)
                return lim.max_items, lim.timeout
            """,
            """
            import os
            import solution
            os.environ.pop("MAX_ITEMS", None)
            os.environ.pop("REQUEST_TIMEOUT", None)
            assert solution.limits() == (10, 2.5)
            os.environ["MAX_ITEMS"] = "25"
            os.environ["REQUEST_TIMEOUT"] = "0.5"
            assert solution.limits() == (25, 0.5)
            """,
            "Limits with explicitly named environment variables.",
        ),
        _p(
            "paths_settings",
            """
            from pydantic import BaseSettings

            class Paths(BaseSettings):
                data_dir: str = "data"
                cache_dir: str = "cache"

                class Config:
                    env_prefix = "SVC_"

            def resolved_paths(root):
                p = Paths()
                return root + "/" + p.data_dir, root + "/" + p.cache_dir
            """,
            """
            import os
            from pydantic import BaseModel

            class Paths(BaseModel):
                data_dir: str = "data"
                cache_dir: str = "cache"

            def resolved_paths(root):
                data = {name: os.environ["SVC_" + name.upper()] for name in Paths.model_fields if "SVC_" + name.upper() in os.environ}
                p = Paths(**data)
                return root + "/" + p.data_dir, root + "/" + p.cache_dir
            """,
            """
            import os
            import solution
            os.environ.pop("SVC_DATA_DIR", None)
            os.environ.pop("SVC_CACHE_DIR", None)
            assert solution.resolved_paths("/srv") == ("/srv/data", "/srv/cache")
            os.environ["SVC_CACHE_DIR"] = "tmp"
            assert solution.resolved_paths("/srv") == ("/srv/data", "/srv/tmp")
            """,
            "Directory settings joined onto a root path.",
        ),
        _p(
            "mail_settings",
            """
            from pydantic import BaseSettings

            class MailSettings(BaseSettings):
                smtp_host: str = "localhost"
                smtp_port: int = 25
                use_tls: bool = False

                class Config:
                    env_prefix = "MAIL_"

                def summary(self):
                    scheme = "smtps" if self.use_tls else "smtp"
                    return "%s://%s:%d" % (scheme, self.smtp_host, self.smtp_port)

            def mail_summary():
                return MailSettings().summary()
            """,
            """
            import os
            from pydantic import BaseModel

            class MailSettings(BaseModel):
                smtp_host: str = "localhost"
                smtp_port: int = 25
                use_tls: bool = False

                def summary(self):
                    scheme = "smtps" if self.use_tls else "smtp"
                    return "%s://%s:%d" % (scheme, self.smtp_host, self.smtp_port)

            def mail_summary():
                data = {name: os.environ["MAIL_" + name.upper()] for name in MailSettings.model_fields if "MAIL_" + name.upper() in os.environ}
                return MailSettings(**data).summary()
            """,
            """
            import os
            import solution
            for key in ("MAIL_SMTP_HOST", "MAIL_SMTP_PORT", "MAIL_USE_TLS"):
                os.environ.pop(key, None)
            assert solution.mail_summary() == "smtp://localhost:25"
            os.environ["MAIL_SMTP_HOST"] = "mail.example.org"
            os.environ["MAIL_SMTP_PORT"] = "465"
            os.environ["MAIL_USE_TLS"] = "true"
            assert solution.mail_summary() == "smtps://mail.example.org:465"
            """,
            "Mail settings with a summary method.",
        ),
    ),
    siblings=(
        _s(
            "model_from_environ",
            """
            import os
            from pydantic import BaseModel

            class Env(BaseModel):
                workers: int = 1
                debug: bool = False

            def model_from_environ():
                data = {k[4:].lower(): v for k, v in os.environ.items() if k.startswith("ENV_")}
                e = Env(**{k: v for k, v in data.items() if k in ("workers", "debug")})
                return e.workers, e.debug
            """,
            """
            import os
            import solution
            os.environ.pop("ENV_WORKERS", None)
            os.environ.pop("ENV_DEBUG", None)
            assert solution.model_from_environ() == (1, False)
            os.environ["ENV_WORKERS"] = "4"
            os.environ["ENV_DEBUG"] = "1"
            assert solution.model_from_environ() == (4, True)
            """,
            "BaseModel coercion of environment strings works on both pins.",
        ),
        _s(
            "model_defaults_dict",
            """
            from pydantic import BaseModel

            class Options(BaseModel):
                retries: int = 3
                verbose: bool = False

            def model_defaults_dict(**overrides):
                return Options(**overrides).dict()
            """,
            """
            import solution
            assert solution.model_defaults_dict() == {"retries": 3, "verbose": False}
            assert solution.model_defaults_dict(retries="5") == {"retries": 5, "verbose": False}
            """,
            "BaseModel.dict() still works on both pins.",
        ),
    ),
)

_PYD_CONLIST = ApiChange(
    package="pydantic",
    api_family="pydantic.conlist(min_items=)",
    old_api="conlist(T, min_items=, max_items=)",
    new_api="conlist(T, min_length=, max_length=)",
    old_pin=_PYD_OLD,
    new_pin=_PYD_NEW,
    change_date=_PYD_DATE,
    change_url=_PYD_URL,
    summary="pydantic 2 renamed conlist's min_items/max_items to min_length/max_length (old names raise TypeError).",
    programs=(
        _p(
            "batch_sum",
            """
            from pydantic import BaseModel, conlist

            class Batch(BaseModel):
                items: conlist(int, min_items=1)

            def batch_sum(items):
                return sum(Batch(items=items).items)
            """,
            None,
            """
            import solution
            assert solution.batch_sum([1, 2, 3]) == 6
            assert solution.batch_sum(["4"]) == 4
            try:
                solution.batch_sum([])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Sum of a non-empty integer batch.",
            fix=(("min_items=", "min_length="),),
        ),
        _p(
            "polygon_perimeter",
            """
            from typing import Tuple
            from pydantic import BaseModel, conlist

            class Polygon(BaseModel):
                points: conlist(Tuple[float, float], min_items=3)

                def perimeter(self):
                    pts = self.points
                    total = 0.0
                    for i, (x1, y1) in enumerate(pts):
                        x2, y2 = pts[(i + 1) % len(pts)]
                        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    return round(total, 3)

            def polygon_perimeter(points):
                return Polygon(points=points).perimeter()
            """,
            None,
            """
            import solution
            assert solution.polygon_perimeter([(0, 0), (3, 0), (3, 4)]) == 12.0
            assert solution.polygon_perimeter([(0, 0), (1, 0), (1, 1), (0, 1)]) == 4.0
            try:
                solution.polygon_perimeter([(0, 0), (1, 1)])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Perimeter of a polygon with at least three vertices.",
            fix=(("min_items=", "min_length="),),
        ),
        _p(
            "survey_answers",
            """
            from pydantic import BaseModel, conlist, ValidationError

            class Survey(BaseModel):
                answers: conlist(str, min_items=1, max_items=3)

            def survey_answers(answers):
                try:
                    return [a.lower() for a in Survey(answers=answers).answers]
                except ValidationError:
                    return None
            """,
            None,
            """
            import solution
            assert solution.survey_answers(["Yes", "No"]) == ["yes", "no"]
            assert solution.survey_answers([]) is None
            assert solution.survey_answers(["a", "b", "c", "d"]) is None
            """,
            "Survey with one to three answers.",
            fix=(("min_items=", "min_length="), ("max_items=", "max_length=")),
        ),
        _p(
            "matrix_shape",
            """
            from pydantic import BaseModel, conlist

            class Matrix(BaseModel):
                rows: conlist(conlist(float, min_items=2), min_items=2)

                def shape(self):
                    return len(self.rows), len(self.rows[0])

            def matrix_shape(rows):
                return Matrix(rows=rows).shape()
            """,
            None,
            """
            import solution
            assert solution.matrix_shape([[1, 2], [3, 4], [5, 6]]) == (3, 2)
            for bad in ([[1, 2]], [[1], [2]]):
                try:
                    solution.matrix_shape(bad)
                    raise AssertionError("expected rejection")
                except ValueError:
                    pass
            """,
            "Shape of a matrix with at least two rows and columns.",
            fix=(("min_items=2), min_items=2)", "min_length=2), min_length=2)"),),
        ),
        _p(
            "playlist",
            """
            from pydantic import BaseModel, conlist, ValidationError

            class PlaylistModel(BaseModel):
                tracks: conlist(str, max_items=3)

            class Playlist:
                def __init__(self):
                    self.tracks = []

                def add(self, track):
                    try:
                        PlaylistModel(tracks=self.tracks + [track])
                    except ValidationError:
                        return False
                    self.tracks.append(track)
                    return True
            """,
            None,
            """
            import solution
            pl = solution.Playlist()
            assert [pl.add(t) for t in ("a", "b", "c", "d")] == [True, True, True, False]
            assert pl.tracks == ["a", "b", "c"]
            """,
            "Playlist capped at three tracks.",
            fix=(("max_items=", "max_length="),),
        ),
        _p(
            "waypoint_legs",
            """
            from pydantic import BaseModel, conlist

            class Trip(BaseModel):
                waypoints: conlist(str, min_items=2)

            def waypoint_legs(waypoints):
                trip = Trip(waypoints=waypoints)
                return ["%s->%s" % pair for pair in zip(trip.waypoints, trip.waypoints[1:])]
            """,
            None,
            """
            import solution
            assert solution.waypoint_legs(["a", "b", "c"]) == ["a->b", "b->c"]
            try:
                solution.waypoint_legs(["a"])
                raise AssertionError("expected rejection")
            except ValueError:
                pass
            """,
            "Legs of a trip with at least two waypoints.",
            fix=(("min_items=", "min_length="),),
        ),
    ),
    siblings=(
        _s(
            "conint_bounds",
            """
            from pydantic import BaseModel, conint

            class Age(BaseModel):
                years: conint(ge=0, le=150)

            def conint_bounds(years):
                try:
                    return Age(years=years).years
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.conint_bounds(30) == 30
            assert solution.conint_bounds(200) is None
            """,
            "conint(ge=, le=) works on both pins.",
        ),
        _s(
            "constr_length",
            """
            from pydantic import BaseModel, constr

            class Code(BaseModel):
                value: constr(min_length=2, max_length=4, to_lower=True)

            def constr_length(value):
                try:
                    return Code(value=value).value
                except ValueError:
                    return None
            """,
            """
            import solution
            assert solution.constr_length("ABC") == "abc"
            assert solution.constr_length("a") is None
            """,
            "constr(min_length=, max_length=, to_lower=) works on both pins.",
        ),
    ),
)
