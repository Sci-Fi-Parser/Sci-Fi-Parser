import cv2
import numpy as np
from lines import detect_line_segments 
from bars import detect_bars
from pathlib import Path

def draw_debug_overlay(
    image: np.ndarray,
    bars: list,
    lines: list,
    *,
    show_labels: bool = True,
) -> np.ndarray:
    out = image.copy()

    # Ensure BGR image for OpenCV drawing
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Draw bar candidates
    for bar in bars:
        b = bar.bbox

        # Adjust these field names to your BoundingBox class
        x1, y1 = int(b.x), int(b.y)
        x2, y2 = int(b.x + b.width), int(b.y + b.height)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_labels:
            cv2.putText(
                out,
                "bar",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # Draw line segments
    for line in lines:
        p1 = tuple(map(int, line.p1))
        p2 = tuple(map(int, line.p2))

        color = (255, 0, 0) if line.orientation == "horizontal" else (0, 0, 255)

        cv2.line(out, p1, p2, color, 2)
        cv2.circle(out, p1, 3, color, -1)
        cv2.circle(out, p2, 3, color, -1)

        if show_labels:
            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)

            cv2.putText(
                out,
                f"{line.orientation[:1]}",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    return out

if __name__ == "__main__":
    path = input("path: ")
    img = cv2.imread(path)
    lines = detect_line_segments(img)
    bars = detect_bars(img)
    print(len(bars))

    overlay = draw_debug_overlay(img, bars=bars, lines=lines)

    output_path = Path(__file__).resolve().parent / f"{path}-barline.png"

    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise RuntimeError(f"Failed to write debug image to {output_path}")

    print(f"Saved debug overlay to: {output_path}")

