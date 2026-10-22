# Object detection

The OCR/CV stage adds structured evidence to bar and line chart images before they are sent to the VLM. `start_ocr` in `detection_pipeline.py` is the entry point. PaddleOCR output is normalized into typed tokens, bar charts receive initial rectangle candidates, and `axis_analysis.py` infers axis-label roles and a linear Y-axis calibration. The raw result is stored for debugging, while a compact JSON representation is appended to the VLM prompt. OCR/CV remains supporting evidence: the VLM is responsible for the final chart data.

## Weaknesses and future work

- A numeric token at the bottom-left can initially belong to both X- and Y-axis candidates. The current resolver assigns a shared endpoint to the Y-axis and removes it from the X-axis evidence. This is usually suitable for bar charts, but can be wrong for line charts when the first X value or point is close to the Y-axis. Line-point geometry should eventually participate in this decision.
- Bar detection selects either a saturation mask or a grayscale mask for the whole image. It cannot combine candidates from both, so low-saturation, light, or low-contrast bars can be treated as background, especially when a chart mixes chromatic and achromatic bars.
- Main line-series detection is not implemented. The remaining line detector only finds near-horizontal and near-vertical segments and is not part of the extraction pipeline. Once line points are detected, the existing linear calibration (`value = slope * pixel_y + intercept`) can be applied to both line points and validated bar endpoints, including within its bounded extrapolation range.
- Initial bar candidates are not yet plot-filtered or interpreted as grouped, stacked, or touching bars. The detected bar count and confidence-gated calibrated values could later be compared with the VLM result to flag likely extraction errors. Currently OCR/CV is only provided before VLM inference; there is no post-extraction consistency check.
- The content and representation of OCR/CV evidence in the VLM context need further A/B benchmarking. Comparisons should include no context, selected evidence, and the full structured JSON, and should measure both improvements and cases where incorrect OCR/CV evidence harms the result.
- Calibration currently assumes an ordinary linear Y-axis. Logarithmic and reversed axes are unsupported.

Changes to object detection should be made carefully. The masks, rejected candidates, role assignments, calibration, and combined debug overlays should be reviewed visually, and OCR/CV component benchmarks plus the end-to-end VLM benchmark should be compared before accepting a change.
