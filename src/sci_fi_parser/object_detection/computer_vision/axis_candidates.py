"""Vertical line candidate extraction and cross-detector fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sci_fi_parser.object_detection.computer_vision.config import CvConfig
from sci_fi_parser.object_detection.computer_vision.geometry import interval_overlap
from sci_fi_parser.object_detection.computer_vision.lines import (
    MergedLine,
    detect_directional_lines,
    detect_merged_lines,
)

Source = Literal["hough", "morphology"]


@dataclass(frozen=True, slots=True)
class VerticalObservation:
    source: Source
    x: float
    top: float
    bottom: float
    supported_intervals: tuple[tuple[float, float], ...]
    line: MergedLine

    @property
    def length(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class VerticalCandidate:
    id: str
    x: float
    top: float
    bottom: float
    supported_intervals: tuple[tuple[float, float], ...]
    sources: Literal["hough", "morphology", "both"]
    hough_observations: tuple[VerticalObservation, ...]
    morphology_observations: tuple[VerticalObservation, ...]

    @property
    def p1(self) -> tuple[int, int]:
        return round(self.x), round(self.top)

    @property
    def p2(self) -> tuple[int, int]:
        return round(self.x), round(self.bottom)

    @property
    def orientation(self) -> Literal["vertical"]:
        return "vertical"

    @property
    def length(self) -> float:
        return self.bottom - self.top

    @property
    def supported_length(self) -> float:
        return sum(end - start for start, end in self.supported_intervals)


@dataclass(frozen=True, slots=True)
class CandidateDetection:
    hough_lines: tuple[MergedLine, ...]
    morphology_lines: tuple[MergedLine, ...]
    hough_vertical: tuple[VerticalObservation, ...]
    morphology_vertical: tuple[VerticalObservation, ...]
    candidates: tuple[VerticalCandidate, ...]


def _observation(line: MergedLine, source: Source) -> VerticalObservation:
    if source == "hough" and line.source_segments:
        intervals = tuple(
            (float(min(segment.p1[1], segment.p2[1])), float(max(segment.p1[1], segment.p2[1]) + 1))
            for segment in line.source_segments
        )
    else:
        intervals = ((float(min(line.p1[1], line.p2[1])), float(max(line.p1[1], line.p2[1]) + 1)),)
    return VerticalObservation(
        source=source,
        x=(line.p1[0] + line.p2[0]) / 2,
        top=float(min(line.p1[1], line.p2[1])),
        bottom=float(max(line.p1[1], line.p2[1]) + 1),
        supported_intervals=intervals,
        line=line,
    )


def _merged_intervals(observations: tuple[VerticalObservation, ...]) -> tuple[tuple[float, float], ...]:
    intervals: list[list[float]] = []
    raw_intervals = [interval for observation in observations for interval in observation.supported_intervals]
    for start, end in sorted(raw_intervals):
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return tuple((start, end) for start, end in intervals)


def fuse_vertical_observations(
    hough: list[VerticalObservation],
    morphology: list[VerticalObservation],
    image_width: int,
) -> list[VerticalCandidate]:
    """Greedily pair observations without cross-detector transitivity."""
    max_distance = max(2.0, image_width * 0.003)
    possible: list[tuple[float, float, float, int, int]] = []
    for h_index, hough_item in enumerate(hough):
        for m_index, morphology_item in enumerate(morphology):
            overlap = interval_overlap(
                hough_item.top,
                hough_item.bottom,
                morphology_item.top,
                morphology_item.bottom,
            )
            shorter = min(hough_item.length, morphology_item.length)
            distance = abs(hough_item.x - morphology_item.x)
            if distance <= max_distance and shorter > 0 and overlap / shorter >= 0.6:
                possible.append(
                    (distance, -overlap, -max(hough_item.length, morphology_item.length), h_index, m_index)
                )

    matched_hough: set[int] = set()
    matched_morphology: set[int] = set()
    groups: list[tuple[VerticalObservation, ...]] = []
    for _, _, _, h_index, m_index in sorted(possible):
        if h_index in matched_hough or m_index in matched_morphology:
            continue
        matched_hough.add(h_index)
        matched_morphology.add(m_index)
        groups.append((hough[h_index], morphology[m_index]))
    groups.extend((item,) for index, item in enumerate(hough) if index not in matched_hough)
    groups.extend((item,) for index, item in enumerate(morphology) if index not in matched_morphology)

    candidates = []
    for observations in groups:
        weights = [item.length for item in observations]
        x = float(np.average([item.x for item in observations], weights=weights))
        hough_items = tuple(item for item in observations if item.source == "hough")
        morphology_items = tuple(item for item in observations if item.source == "morphology")
        source = "both" if hough_items and morphology_items else observations[0].source
        candidates.append(
            VerticalCandidate(
                id="",
                x=x,
                top=min(item.top for item in observations),
                bottom=max(item.bottom for item in observations),
                supported_intervals=_merged_intervals(observations),
                sources=source,
                hough_observations=hough_items,
                morphology_observations=morphology_items,
            )
        )
    candidates.sort(key=lambda item: (item.x, item.top, item.bottom, item.sources))
    return [
        VerticalCandidate(
            id=f"C{index}",
            x=item.x,
            top=item.top,
            bottom=item.bottom,
            supported_intervals=item.supported_intervals,
            sources=item.sources,
            hough_observations=item.hough_observations,
            morphology_observations=item.morphology_observations,
        )
        for index, item in enumerate(candidates)
    ]


def detect_vertical_candidates(image: np.ndarray, config: CvConfig | None = None) -> CandidateDetection:
    """Run unchanged raw detectors and return their fused vertical union."""
    config = config or CvConfig()
    hough_lines = tuple(detect_merged_lines(image, config))
    morphology_lines = tuple(detect_directional_lines(image, config))
    hough = [_observation(line, "hough") for line in hough_lines if line.orientation == "vertical"]
    morphology = [
        _observation(line, "morphology") for line in morphology_lines if line.orientation == "vertical"
    ]
    candidates = fuse_vertical_observations(hough, morphology, image.shape[1])
    return CandidateDetection(
        hough_lines, morphology_lines, tuple(hough), tuple(morphology), tuple(candidates)
    )
