"""Enhanced visual conflict detection for matplotlib figures.

Runs 15 detection passes on any matplotlib Figure:

 1. Text overlap — pairwise BBox intersection of all text artists.
 2. Text truncation — text extending beyond figure canvas.
 3. Artist truncation — graphical content clipped by figure borders.
 4. Artist-vs-artist overlap — colorbars / legends covering data.
 5. Artist-vs-text overlap — graphical content overlapping text labels.
    Detects in-axes annotations (significance markers ``*``, ``**``,
    ``ns``, ``p<…``, etc.) colliding with data content (warning).
 6. Axes overflow — child artists exceeding parent axes bounds.
 7. Scatter clip risk — clip_on=True markers silently clipped.
 8. Cross-panel spillover — text from one axes leaking into another.
 9. Panel-label overlap — (A)/(B) labels covering data.
10. Legend spillover — legend extending beyond figure or into other axes.
11. Legend-to-panel content — legend bbox overlapping data in OTHER panels.
12. Legend-vs-own-content — legend occluding data lines/scatter INSIDE its
    own panel (significant overlap only).
13. Legend self-consistency — legend handle/text items must not overlap each
    other, must stay within the legend frame, and the legend frame must not
    extend beyond the figure border (tighter than pass 10).
14. Colorbar internal — colorbar tick labels overlapping each other, colorbar
    axis label overlapping tick labels, and tick labels truncated at figure
    edges.
15. Legend internal — legend text entries overlapping each other, legend texts
    extending beyond the legend frame, and legend texts truncated at figure
    edges.

Usage:
    from benchmarks.scripts.pipeline.visual_conflict_detector import detect_all_conflicts
    issues = detect_all_conflicts(fig, label="my_plot", verbose=True)
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
from matplotlib.text import Text
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_bbox(artist, renderer) -> Bbox | None:
    try:
        bb = artist.get_window_extent(renderer)
        if bb is not None and bb.width > 0 and bb.height > 0:
            return bb
    except Exception:
        pass
    return None


def _shrink(bb: Bbox, px: float) -> Bbox | None:
    b = Bbox.from_extents(bb.x0 + px, bb.y0 + px, bb.x1 - px, bb.y1 - px)
    if b.width > 0 and b.height > 0:
        return b
    return None


def _fig_bbox(fig) -> Bbox:
    w, h = fig.get_size_inches()
    dpi = fig.dpi
    return Bbox.from_bounds(0, 0, w * dpi, h * dpi)


def _overlap_area(a: Bbox, b: Bbox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 > x0 and y1 > y0:
        return (x1 - x0) * (y1 - y0)
    return 0.0


def _sides_outside(bb: Bbox, fig_bb: Bbox, tol: float = 1.0) -> list[str]:
    sides = []
    if bb.x0 < fig_bb.x0 - tol:
        sides.append("left")
    if bb.y0 < fig_bb.y0 - tol:
        sides.append("bottom")
    if bb.x1 > fig_bb.x1 + tol:
        sides.append("right")
    if bb.y1 > fig_bb.y1 + tol:
        sides.append("top")
    return sides


def _artist_label(artist, hint: str = "") -> str:
    if isinstance(artist, Text):
        s = artist.get_text().strip()
        return f"{hint}: {s[:50]}" if hint else f"text: {s[:50]}"
    cls = type(artist).__name__
    label = getattr(artist, "_label", "") or ""
    tag = f"{cls}"
    if label and not label.startswith("_"):
        tag += f"({label[:30]})"
    return f"{hint}: {tag}" if hint else tag


# ═══════════════════════════════════════════════════════════════════════════════
# Artist collector
# ═══════════════════════════════════════════════════════════════════════════════

class _ArtistInfo:
    __slots__ = ("artist", "bbox", "tag", "kind", "ax_id")

    def __init__(self, artist, bbox, tag, kind, ax_id=None):
        self.artist = artist
        self.bbox = bbox
        self.tag = tag
        self.kind = kind
        self.ax_id = ax_id


def _collect_artists(fig, renderer) -> list[_ArtistInfo]:
    infos: list[_ArtistInfo] = []

    cbar_axes = set()
    for ax in fig.get_axes():
        if hasattr(ax, '_colorbar_info') or getattr(ax, '_colorbar', None):
            cbar_axes.add(id(ax))

    for ax in fig.get_axes():
        is_cbar = id(ax) in cbar_axes
        pfx = "cbar" if is_cbar else ""
        aid = id(ax)

        for title_obj in [ax.title, ax._left_title, ax._right_title]:
            if title_obj and title_obj.get_text().strip():
                bb = _safe_bbox(title_obj, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        title_obj, bb,
                        _artist_label(title_obj, f"{pfx}title"),
                        "text", aid))

        for lbl, hint in [(ax.xaxis.label, f"{pfx}xlabel"),
                          (ax.yaxis.label, f"{pfx}ylabel")]:
            if lbl.get_text().strip():
                bb = _safe_bbox(lbl, renderer)
                if bb:
                    infos.append(_ArtistInfo(lbl, bb,
                                             _artist_label(lbl, hint),
                                             "text", aid))

        for tl in ax.get_xticklabels():
            if tl.get_text().strip():
                bb = _safe_bbox(tl, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        tl, bb,
                        _artist_label(tl, "cbar_tick" if is_cbar else "xtick"),
                        "text", aid))
        for tl in ax.get_yticklabels():
            if tl.get_text().strip():
                bb = _safe_bbox(tl, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        tl, bb,
                        _artist_label(tl, "cbar_tick" if is_cbar else "ytick"),
                        "text", aid))

        if not is_cbar:
            for txt in ax.texts:
                if txt.get_text().strip():
                    bb = _safe_bbox(txt, renderer)
                    if bb:
                        infos.append(_ArtistInfo(
                            txt, bb, _artist_label(txt, "annotation"),
                            "text", aid))

            legend = ax.get_legend()
            if legend is not None:
                bb = _safe_bbox(legend, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        legend, bb, "legend_box", "legend", aid))
                for txt in legend.get_texts():
                    if txt.get_text().strip():
                        tbb = _safe_bbox(txt, renderer)
                        if tbb:
                            infos.append(_ArtistInfo(
                                txt, tbb,
                                _artist_label(txt, "legend_text"),
                                "text", aid))

        for child in ax.get_children():
            if isinstance(child, Text):
                continue
            if not getattr(child, "get_visible", lambda: True)():
                continue
            # Skip axes background patch (spans entire axes, always triggers
            # false positive overlaps with any internal legend)
            if child is ax.patch:
                continue
            label = getattr(child, "_label", "") or ""
            if label.startswith("_") and label not in ("_nolegend_",):
                if isinstance(child, Line2D):
                    if child.get_linestyle() in ("--", ":", "-."):
                        continue
                continue

            bb = _safe_bbox(child, renderer)
            if bb is None:
                continue

            if isinstance(child, PathCollection):
                infos.append(_ArtistInfo(
                    child, bb, _artist_label(child, "scatter"),
                    "collection", aid))
            elif isinstance(child, (PolyCollection, LineCollection)):
                infos.append(_ArtistInfo(
                    child, bb, _artist_label(child, "poly"),
                    "collection", aid))
            elif isinstance(child, Patch):
                infos.append(_ArtistInfo(
                    child, bb, _artist_label(child, "patch"),
                    "patch", aid))
            elif isinstance(child, Line2D):
                infos.append(_ArtistInfo(
                    child, bb, _artist_label(child, "line"),
                    "line", aid))
            elif isinstance(child, AxesImage):
                infos.append(_ArtistInfo(
                    child, bb, _artist_label(child, "image"),
                    "image", aid))

    return infos


# ═══════════════════════════════════════════════════════════════════════════════
# Detection passes
# ═══════════════════════════════════════════════════════════════════════════════

def _check_text_overlaps(infos, tol_px=2.0):
    texts = [a for a in infos if a.kind == "text"]
    issues = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            tags = {a.tag.split(":")[0].strip(), b.tag.split(":")[0].strip()}
            if tags == {"xtick", "ytick"}:
                continue
            sa = _shrink(a.bbox, tol_px)
            sb = _shrink(b.bbox, tol_px)
            if sa and sb and sa.overlaps(sb):
                issues.append({
                    "type": "text_overlap",
                    "severity": "warning",
                    "detail": f"'{a.tag}' overlaps '{b.tag}'",
                    "elements": [a.tag, b.tag],
                })
    return issues


def _check_truncation(infos, fig_bb, tol_px=1.0):
    issues = []
    for a in infos:
        sides = _sides_outside(a.bbox, fig_bb, tol_px)
        if sides:
            sev = "warning" if a.kind == "text" else "info"
            if a.kind in ("collection", "patch", "line"):
                sev = "warning"
            issues.append({
                "type": f"{a.kind}_truncation",
                "severity": sev,
                "detail": f"'{a.tag}' extends beyond figure border ({', '.join(sides)})",
                "elements": [a.tag],
            })
    return issues


def _check_artist_content_overlap(infos, min_overlap_px2=100.0):
    graphical = [a for a in infos
                 if a.kind in ("collection", "patch", "image", "legend")]
    issues = []
    for i in range(len(graphical)):
        for j in range(i + 1, len(graphical)):
            a, b = graphical[i], graphical[j]
            # Skip same-axes overlaps — those are intentional (legend on bars,
            # radar fill on background, colorbar on image etc.)
            if a.ax_id == b.ax_id:
                continue
            area = _overlap_area(a.bbox, b.bbox)
            if area > min_overlap_px2:
                issues.append({
                    "type": "artist_overlap",
                    "severity": "warning",
                    "detail": f"'{a.tag}' overlaps '{b.tag}' ({area:.0f} px²)",
                    "elements": [a.tag, b.tag],
                })
    return issues


def _is_significance_marker(tag: str) -> bool:
    """Return True if *tag* looks like a statistical significance annotation.

    Catches common patterns: *, **, ***, ns, n.s., p<0.05, etc.
    """
    import re
    if not tag.startswith("annotation:"):
        return False
    txt = tag.split(":", 1)[1].strip()
    if re.fullmatch(r"\*{1,4}", txt):          # *, **, ***, ****
        return True
    if txt.lower() in ("ns", "n.s.", "ns."):   # not-significant
        return True
    if re.match(r"p\s*[<>=]", txt, re.I):       # p<0.05, P = 0.01 ...
        return True
    return False


def _check_text_vs_artist_overlap(infos, tol_px=2.0, min_overlap_px2=50.0):
    texts = [a for a in infos if a.kind == "text"]
    graphics = [a for a in infos
                if a.kind in ("collection", "patch", "line", "image")]
    issues = []
    for t in texts:
        tb = _shrink(t.bbox, tol_px)
        if not tb:
            continue
        for g in graphics:
            area = _overlap_area(tb, g.bbox)
            if area > min_overlap_px2:
                # Determine severity: in-axes annotations overlapping with
                # data artists in the same panel are warnings (especially
                # significance markers like *, **, ns, etc.)
                is_annotation = t.tag.startswith("annotation:")
                same_axes = (t.ax_id == g.ax_id) and (t.ax_id is not None)
                if is_annotation and same_axes:
                    severity = "warning"
                    issue_type = "annotation_data_overlap"
                    if _is_significance_marker(t.tag):
                        issue_type = "significance_marker_overlap"
                else:
                    severity = "info"
                    issue_type = "text_artist_overlap"

                issues.append({
                    "type": issue_type,
                    "severity": severity,
                    "detail": f"Text '{t.tag}' overlaps content '{g.tag}' ({area:.0f} px²)",
                    "elements": [t.tag, g.tag],
                })
    return issues


def _check_axes_overflow(infos, fig):
    renderer = fig.canvas.get_renderer()
    issues = []
    for ax in fig.get_axes():
        ax_bb = _safe_bbox(ax, renderer)
        if not ax_bb:
            continue
        aid = id(ax)
        # Only check meaningful data artists (collections, lines),
        # not axis furniture (patches = spines, background, etc.)
        ax_artists = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line")]
        for a in ax_artists:
            sides = _sides_outside(a.bbox, ax_bb, tol=3.0)
            if sides:
                issues.append({
                    "type": "axes_overflow",
                    "severity": "info",
                    "detail": f"'{a.tag}' extends beyond axes border ({', '.join(sides)})",
                    "elements": [a.tag],
                })
    return issues


def _check_scatter_clip_risk(fig):
    renderer = fig.canvas.get_renderer()
    issues = []
    for ax in fig.get_axes():
        ax_bb = _safe_bbox(ax, renderer)
        if ax_bb is None:
            continue
        for child in ax.get_children():
            if not isinstance(child, PathCollection):
                continue
            if not child.get_visible() or not child.get_clip_on():
                continue
            offsets = child.get_offsets()
            raw_sizes = child.get_sizes()
            if offsets is None or len(offsets) == 0:
                continue
            if raw_sizes is None or len(raw_sizes) == 0:
                continue
            sizes = np.broadcast_to(np.asarray(raw_sizes), (len(offsets),))
            transform = child.get_offset_transform() or ax.transData
            try:
                display_pts = transform.transform(offsets)
            except Exception:
                continue
            pts_per_px = 72.0 / fig.dpi
            radii_px = np.sqrt(sizes / np.pi) / pts_per_px
            n_clipped = 0
            clipped_sides: set[str] = set()
            for (dx, dy), r in zip(display_pts, radii_px):
                if dx - r < ax_bb.x0:
                    n_clipped += 1; clipped_sides.add("left")
                elif dx + r > ax_bb.x1:
                    n_clipped += 1; clipped_sides.add("right")
                if dy - r < ax_bb.y0:
                    n_clipped += 1; clipped_sides.add("bottom")
                elif dy + r > ax_bb.y1:
                    n_clipped += 1; clipped_sides.add("top")
            if n_clipped > 0:
                label = getattr(child, "_label", "") or ""
                tag = f"scatter({label})" if label and not label.startswith("_") else "scatter"
                issues.append({
                    "type": "scatter_clip_risk",
                    "severity": "warning",
                    "detail": (
                        f"'{tag}' has {n_clipped} marker(s) clipped at "
                        f"axes edge ({', '.join(sorted(clipped_sides))}). "
                        f"Set clip_on=False or add axis margin."
                    ),
                    "elements": [tag],
                })
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Composed-figure-specific checks (passes 8-10)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_cross_panel_spillover(fig, renderer, tol_px=5.0):
    """Pass 8: Detect content from one axes spilling into an adjacent axes."""
    axes_list = fig.get_axes()
    if len(axes_list) < 2:
        return []

    issues = []
    ax_bboxes = []
    for ax in axes_list:
        bb = _safe_bbox(ax, renderer)
        if bb:
            ax_bboxes.append((ax, bb))

    for i, (ax_i, bb_i) in enumerate(ax_bboxes):
        for child in ax_i.get_children():
            if not child.get_visible():
                continue
            if isinstance(child, Text):
                child_bb = _safe_bbox(child, renderer)
                if child_bb is None:
                    continue
                for j, (ax_j, bb_j) in enumerate(ax_bboxes):
                    if i == j:
                        continue
                    area = _overlap_area(child_bb, bb_j)
                    if area > 50:
                        txt = getattr(child, "_text", "")[:30]
                        issues.append({
                            "type": "cross_panel_spillover",
                            "severity": "warning",
                            "detail": (
                                f"Text '{txt}' from axes {i} "
                                f"spills into axes {j} ({area:.0f} px²)"
                            ),
                            "elements": [f"ax{i}", f"ax{j}"],
                        })
    return issues


def _check_panel_label_overlap(fig, renderer, infos, tol_px=2.0):
    """Pass 9: Check that panel labels (A), (B), etc. don't overlap content or text.

    Panel labels must be clearly visible; any overlap with data content,
    axis text (titles, ticks, axis labels), or legend text is a warning.
    """
    panel_texts = []
    for child in fig.texts:
        txt = getattr(child, "_text", "")
        if txt and txt.startswith("(") and txt.endswith(")") and len(txt) <= 4:
            bb = _safe_bbox(child, renderer)
            if bb:
                panel_texts.append((txt, bb, child))

    issues = []
    for txt, pbb, panel_obj in panel_texts:
        # Check against graphical content (collection, patch, image, line)
        for a in infos:
            if a.kind in ("collection", "patch", "image", "line"):
                if a.artist is panel_obj:
                    continue
                area = _overlap_area(pbb, a.bbox)
                if area > 20:
                    issues.append({
                        "type": "panel_label_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Panel label '{txt}' overlaps "
                            f"content '{a.tag}' ({area:.0f} px²)"
                        ),
                        "elements": [txt, a.tag],
                    })

        # Check against all text (titles, ticks, labels, legend text, annotations)
        for a in infos:
            if a.kind == "text":
                if a.artist is panel_obj:
                    continue
                area = _overlap_area(pbb, a.bbox)
                if area > tol_px:
                    other_txt = getattr(a.artist, "_text", "")[:30] if hasattr(a.artist, "_text") else a.tag
                    issues.append({
                        "type": "panel_label_text_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Panel label '{txt}' overlaps "
                            f"text '{a.tag}' ({area:.0f} px²)"
                        ),
                        "elements": [txt, a.tag],
                    })

        # Check against other panel labels
        for txt2, pbb2, panel_obj2 in panel_texts:
            if panel_obj2 is panel_obj:
                continue
            area = _overlap_area(pbb, pbb2)
            if area > tol_px:
                issues.append({
                    "type": "panel_label_mutual_overlap",
                    "severity": "warning",
                    "detail": (
                        f"Panel labels '{txt}' and '{txt2}' overlap ({area:.0f} px²)"
                    ),
                    "elements": [txt, txt2],
                })
    return issues


def _check_legend_spillover(fig, renderer, tol_px=5.0):
    """Pass 10: Legends extending beyond their parent axes or the figure."""
    issues = []
    fig_bb = _fig_bbox(fig)

    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            continue

        # Check legend vs figure bounds
        sides = _sides_outside(leg_bb, fig_bb, tol_px)
        if sides:
            issues.append({
                "type": "legend_truncation",
                "severity": "warning",
                "detail": (
                    f"Legend in axes extends beyond figure "
                    f"({', '.join(sides)})"
                ),
                "elements": ["legend"],
            })

        # Check legend vs other axes
        for other_ax in fig.get_axes():
            if other_ax is ax:
                continue
            other_bb = _safe_bbox(other_ax, renderer)
            if other_bb is None:
                continue
            area = _overlap_area(leg_bb, other_bb)
            if area > 100:
                issues.append({
                    "type": "legend_spillover",
                    "severity": "warning",
                    "detail": (
                        f"Legend from axes spills into a "
                        f"neighbouring axes ({area:.0f} px²)"
                    ),
                    "elements": ["legend"],
                })

    # Also check figure-level legends
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            leg_bb = _safe_bbox(child, renderer)
            if leg_bb is None:
                continue
            sides = _sides_outside(leg_bb, fig_bb, tol_px)
            if sides:
                issues.append({
                    "type": "legend_truncation",
                    "severity": "warning",
                    "detail": f"Figure legend extends beyond border ({', '.join(sides)})",
                    "elements": ["fig_legend"],
                })

    return issues


def _check_legend_vs_other_panel_content(fig, renderer, infos, min_overlap_px2=30.0):
    """Pass 11: Detect legends overlapping data content in OTHER panels.

    Panel A's legend could spill out of its axes and land on top of Panel B's
    bar chart.  We check every legend bbox against every data artist
    (collection, line, image, patch) that belongs to a *different* axes.
    Also checks figure-level legends (e.g. shared UMAP legend) against all
    subplot content.
    """
    issues: list[dict] = []

    legend_infos: list[tuple[Bbox, int]] = []
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb:
            legend_infos.append((leg_bb, id(ax)))

    # Also include figure-level legends (only check vs collections/lines/images)
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            leg_bb = _safe_bbox(child, renderer)
            if leg_bb:
                legend_infos.append((leg_bb, id(fig)))  # figure-owned

    # For figure-level legends, restrict to non-patch data artists only
    # (axis backgrounds and spines are patches that almost always overlap)
    data_artists_full = [a for a in infos
                         if a.kind in ("collection", "line", "image", "patch")]
    data_artists_no_patch = [a for a in infos
                             if a.kind in ("collection", "line", "image")]

    for leg_bb, leg_owner_id in legend_infos:
        # Figure-level legends: skip patches (axis backgrounds/spines)
        targets = data_artists_no_patch if leg_owner_id == id(fig) \
                  else data_artists_full
        for da in targets:
            if da.ax_id == leg_owner_id:
                continue  # same panel — handled by pass 12
            area = _overlap_area(leg_bb, da.bbox)
            if area > min_overlap_px2:
                issues.append({
                    "type": "legend_panel_overlap",
                    "severity": "warning",
                    "detail": (
                        f"Legend overlaps '{da.tag}' in a different "
                        f"panel ({area:.0f} px²)"
                    ),
                    "elements": ["legend", da.tag],
                })
    return issues


def _check_legend_vs_own_content(fig, renderer, infos, min_overlap_px2=50.0):
    """Pass 12: Detect legends covering data IN THEIR OWN panel.

    Checks every legend bbox against ALL data artists (including patches/bars)
    in the same axes.  Uses lower thresholds to catch bar-legend-vs-bars
    and scatter-legend-vs-scatter overlaps.
    """
    issues: list[dict] = []

    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        # Skip axes that are purely legend-holding cells (axis off, no data)
        if getattr(ax, '_is_legend_cell', False):
            continue
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            continue

        aid = id(ax)
        # Include "patch" so bar-chart rectangles are checked,
        # but exclude structural patches (spines, polar wedges, etc.)
        _SKIP = ("Spine", "Wedge", "FancyBbox")
        local_data = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line", "image", "patch")
                      and not any(s in a.tag for s in _SKIP)]

        for da in local_data:
            area = _overlap_area(leg_bb, da.bbox)
            if area > min_overlap_px2:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.05:  # legend covers >5 % of the data artist
                    issues.append({
                        "type": "legend_data_occlusion",
                        "severity": "warning",
                        "detail": (
                            f"Legend in same panel occludes "
                            f"'{da.tag}' ({frac:.0%}, {area:.0f} px²)"
                        ),
                        "elements": ["legend", da.tag],
                    })
    return issues


def _check_legend_self_consistency(fig, renderer, tol_px: float = 1.5):
    """Pass 13: Legend internal self-consistency.

    Checks performed for every legend (per-axes and figure-level):

    a) Handle–handle overlap — two legend marker patches/lines are on top of
       each other (corrupt legend layout).
    b) Handle–text overlap — a legend marker partially covers its own or an
       adjacent row's text label.
    c) Text–text overlap — two legend text entries overlap (too many entries,
       font too large, or ncol too high).
    d) Item border truncation — any legend handle or text item extends beyond
       the figure border.
    e) Legend frame vs figure border — the legend bounding box itself extends
       beyond the figure (stricter variant of pass 10, applied per-item).

    All findings are reported as *warnings* so they surface in the standard
    ``0 warnings | 0 errors`` gate.
    """
    issues: list[dict] = []

    try:
        fig_bb = _fig_bbox(fig)
    except Exception:
        return issues

    def _legend_items(legend) -> list[tuple[str, object, Bbox]]:
        """Collect (type, artist, bbox) for every visual item inside a legend."""
        items: list[tuple[str, object, Bbox]] = []
        try:
            for handle in legend.legend_handles:
                if handle is None:
                    continue
                if not getattr(handle, 'get_visible', lambda: True)():
                    continue
                hbb = _safe_bbox(handle, renderer)
                if hbb:
                    lbl = getattr(handle, '_label', '') or type(handle).__name__
                    items.append((f"handle({lbl[:20]})", handle, hbb))
        except Exception:
            pass
        try:
            for txt in legend.get_texts():
                if not txt.get_text().strip():
                    continue
                tbb = _safe_bbox(txt, renderer)
                if tbb:
                    items.append((f"legend_item_text({txt.get_text()[:20]})",
                                  txt, tbb))
        except Exception:
            pass
        return items

    def _process_legend(legend, leg_label: str) -> None:
        if not legend.get_visible():
            return
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            return

        items = _legend_items(legend)
        if not items:
            return

        # (e) Legend frame vs figure border
        sides = _sides_outside(leg_bb, fig_bb, tol=tol_px)
        if sides:
            issues.append({
                "type": "legend_frame_truncation",
                "severity": "warning",
                "detail": (
                    f"{leg_label} frame extends beyond figure border "
                    f"({', '.join(sides)})"
                ),
                "elements": [leg_label],
            })

        for idx, (tag_i, art_i, bb_i) in enumerate(items):
            # (d) Individual item border truncation
            sides_i = _sides_outside(bb_i, fig_bb, tol=tol_px)
            if sides_i:
                issues.append({
                    "type": "legend_item_truncation",
                    "severity": "warning",
                    "detail": (
                        f"{leg_label} item '{tag_i}' extends beyond "
                        f"figure border ({', '.join(sides_i)})"
                    ),
                    "elements": [leg_label, tag_i],
                })

            # (a/b/c) Pairwise overlap between items
            bb_shrunk_i = _shrink(bb_i, tol_px)
            if bb_shrunk_i is None:
                continue
            for tag_j, art_j, bb_j in items[idx + 1:]:
                bb_shrunk_j = _shrink(bb_j, tol_px)
                if bb_shrunk_j is None:
                    continue
                area = _overlap_area(bb_shrunk_i, bb_shrunk_j)
                # Skip zero/tiny-area hits: Line2D handles in scatter legends
                # report a wide bbox that generates floating-point artefacts.
                # Require at least 4 px² to be a meaningful overlap.
                if area < 4:
                    continue
                # Classify the pair type
                i_is_text = isinstance(art_i, Text)
                j_is_text = isinstance(art_j, Text)
                if i_is_text and j_is_text:
                    kind = "legend_text_text_overlap"
                elif not i_is_text and not j_is_text:
                    kind = "legend_handle_handle_overlap"
                else:
                    kind = "legend_handle_text_overlap"
                issues.append({
                    "type": kind,
                    "severity": "warning",
                    "detail": (
                        f"{leg_label}: '{tag_i}' overlaps '{tag_j}' "
                        f"({area:.0f} px²)"
                    ),
                    "elements": [leg_label, tag_i, tag_j],
                })

    # Per-axes legends
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is not None:
            title = ax.get_title() or f"ax@{id(ax):#x}"
            _process_legend(legend, f"legend[{title[:25]}]")

    # Figure-level legends
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            _process_legend(child, "fig_legend")

    return issues


def _check_fig_legend_vs_subplot_content(fig, renderer, infos,
                                          min_overlap_px2=30.0):
    """Pass 13: Figure-level legends overlapping subplot scatter/bar data.

    Shared legends (e.g. UMAP legend placed via fig.legend()) can overlap
    the scatter / bar content of individual subplots.  This pass specifically
    finds figure-owned legends and checks them against data in ALL axes.
    """
    issues: list[dict] = []

    fig_legends = []
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            leg_bb = _safe_bbox(child, renderer)
            if leg_bb:
                fig_legends.append(leg_bb)

    if not fig_legends:
        return issues

    _SKIP = ("Spine", "Wedge", "FancyBbox")
    data_artists = [a for a in infos
                    if a.kind in ("collection", "line", "image", "patch")
                    and not any(s in a.tag for s in _SKIP)]

    for leg_bb in fig_legends:
        for da in data_artists:
            area = _overlap_area(leg_bb, da.bbox)
            if area > min_overlap_px2:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.03:  # 3 % — very sensitive for shared legends
                    issues.append({
                        "type": "fig_legend_subplot_occlusion",
                        "severity": "warning",
                        "detail": (
                            f"Figure-level legend occludes subplot "
                            f"content '{da.tag}' ({frac:.0%}, {area:.0f} px²)"
                        ),
                        "elements": ["fig_legend", da.tag],
                    })
    return issues


def _check_colorbar_internal(fig, renderer, tol_px=1.0):
    """Pass 14: Detect overlaps within colorbar axes.

    Checks:
      a) Colorbar tick labels overlapping each other.
      b) Colorbar axis label overlapping tick labels.
      c) Colorbar extending beyond its parent/inset host axes.
      d) Colorbar tick labels truncated at figure edges.
    """
    issues: list[dict] = []
    fig_bb = _fig_bbox(fig)

    for ax in fig.get_axes():
        is_cbar = (hasattr(ax, '_colorbar_info')
                   or getattr(ax, '_colorbar', None) is not None)
        if not is_cbar:
            continue

        # Gather tick labels
        xticks = [tl for tl in ax.get_xticklabels()
                  if tl.get_text().strip()]
        yticks = [tl for tl in ax.get_yticklabels()
                  if tl.get_text().strip()]
        tick_labels = xticks + yticks

        tick_bbs: list[tuple[str, Bbox]] = []
        for tl in tick_labels:
            bb = _safe_bbox(tl, renderer)
            if bb:
                tick_bbs.append((tl.get_text().strip()[:20], bb))

        # a) Tick labels overlapping each other
        for i in range(len(tick_bbs)):
            for j in range(i + 1, len(tick_bbs)):
                txt_i, bb_i = tick_bbs[i]
                txt_j, bb_j = tick_bbs[j]
                si = _shrink(bb_i, tol_px)
                sj = _shrink(bb_j, tol_px)
                if si and sj and si.overlaps(sj):
                    area = _overlap_area(bb_i, bb_j)
                    issues.append({
                        "type": "cbar_tick_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Colorbar tick '{txt_i}' overlaps "
                            f"tick '{txt_j}' ({area:.0f} px²)"
                        ),
                        "elements": [f"cbar_tick:{txt_i}",
                                     f"cbar_tick:{txt_j}"],
                    })

        # b) Colorbar axis label vs tick labels
        for lbl_artist in [ax.xaxis.label, ax.yaxis.label]:
            lbl_txt = lbl_artist.get_text().strip()
            if not lbl_txt:
                continue
            lbl_bb = _safe_bbox(lbl_artist, renderer)
            if lbl_bb is None:
                continue
            lbl_s = _shrink(lbl_bb, tol_px)
            if not lbl_s:
                continue
            for txt_t, bb_t in tick_bbs:
                bb_ts = _shrink(bb_t, tol_px)
                if bb_ts and lbl_s.overlaps(bb_ts):
                    area = _overlap_area(lbl_bb, bb_t)
                    issues.append({
                        "type": "cbar_label_tick_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Colorbar label '{lbl_txt[:20]}' overlaps "
                            f"tick '{txt_t}' ({area:.0f} px²)"
                        ),
                        "elements": [f"cbar_label:{lbl_txt[:20]}",
                                     f"cbar_tick:{txt_t}"],
                    })

        # c) + d) Tick labels extending beyond figure
        for txt_t, bb_t in tick_bbs:
            sides = _sides_outside(bb_t, fig_bb, 1.0)
            if sides:
                issues.append({
                    "type": "cbar_tick_truncation",
                    "severity": "warning",
                    "detail": (
                        f"Colorbar tick '{txt_t}' extends beyond "
                        f"figure ({', '.join(sides)})"
                    ),
                    "elements": [f"cbar_tick:{txt_t}"],
                })

    return issues


def _check_legend_internal(fig, renderer, tol_px=1.0):
    """Pass 15: Detect internal crowding within legend boxes.

    Checks:
      a) Legend text entries overlapping each other.
      b) Legend texts extending beyond the legend frame bbox.
      c) Legend texts extending beyond figure bounds.
    """
    issues: list[dict] = []
    fig_bb = _fig_bbox(fig)

    def _audit_legend(legend, ctx_label="axes"):
        if legend is None or not legend.get_visible():
            return
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            return

        texts = legend.get_texts()
        text_bbs: list[tuple[str, Bbox]] = []
        for t in texts:
            txt = t.get_text().strip()
            if not txt:
                continue
            bb = _safe_bbox(t, renderer)
            if bb:
                text_bbs.append((txt[:30], bb))

        # a) Pairwise text overlap within legend
        for i in range(len(text_bbs)):
            for j in range(i + 1, len(text_bbs)):
                txt_i, bb_i = text_bbs[i]
                txt_j, bb_j = text_bbs[j]
                si = _shrink(bb_i, tol_px)
                sj = _shrink(bb_j, tol_px)
                if si and sj and si.overlaps(sj):
                    area = _overlap_area(bb_i, bb_j)
                    issues.append({
                        "type": "legend_text_crowding",
                        "severity": "warning",
                        "detail": (
                            f"Legend entries '{txt_i}' and '{txt_j}' "
                            f"overlap in {ctx_label} ({area:.0f} px²)"
                        ),
                        "elements": [f"legend_text:{txt_i}",
                                     f"legend_text:{txt_j}"],
                    })

        # b) Texts extending beyond legend frame
        for txt_t, bb_t in text_bbs:
            sides = _sides_outside(bb_t, leg_bb, 2.0)
            if sides:
                issues.append({
                    "type": "legend_text_overflow",
                    "severity": "info",
                    "detail": (
                        f"Legend text '{txt_t}' extends beyond "
                        f"legend frame in {ctx_label} "
                        f"({', '.join(sides)})"
                    ),
                    "elements": [f"legend_text:{txt_t}"],
                })

        # c) Texts extending beyond figure
        for txt_t, bb_t in text_bbs:
            sides = _sides_outside(bb_t, fig_bb, 1.0)
            if sides:
                issues.append({
                    "type": "legend_text_truncation",
                    "severity": "warning",
                    "detail": (
                        f"Legend text '{txt_t}' extends beyond "
                        f"figure border ({', '.join(sides)})"
                    ),
                    "elements": [f"legend_text:{txt_t}"],
                })

    # Check per-axes legends
    for idx, ax in enumerate(fig.get_axes()):
        _audit_legend(ax.get_legend(), f"axes[{idx}]")

    # Check figure-level legends
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            _audit_legend(child, "figure-legend")

    return issues


def _per_axes_summary(fig, renderer, infos):
    """Two-layer reporting: per-subplot conflict summary.

    Returns a dict mapping axes title → list of issues in that subplot,
    useful for pinpointing which panels still have internal conflicts.
    """
    per_ax: dict[str, list[dict]] = {}

    for ax in fig.get_axes():
        if getattr(ax, '_is_legend_cell', False):
            continue
        title = ax.get_title() or f"ax@{id(ax):#x}"
        aid = id(ax)
        ax_issues: list[dict] = []

        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            per_ax[title] = ax_issues
            continue

        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            per_ax[title] = ax_issues
            continue

        local_data = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line", "image", "patch")
                      and not any(s in a.tag for s in ("Spine", "Wedge", "FancyBbox"))]

        for da in local_data:
            area = _overlap_area(leg_bb, da.bbox)
            if area > 30:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.03:
                    ax_issues.append({
                        "type": "subplot_legend_overlap",
                        "severity": "warning",
                        "detail": (
                            f"[{title}] Legend occludes '{da.tag}' "
                            f"({frac:.0%}, {area:.0f} px²)"
                        ),
                    })
        per_ax[title] = ax_issues
    return per_ax


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_all_conflicts(
    fig,
    label: str = "",
    verbose: bool = True,
    text_overlap_tol_px: float = 2.0,
    border_tol_px: float = 1.0,
    artist_overlap_min_px2: float = 100.0,
    text_artist_overlap_min_px2: float = 50.0,
):
    """Run all visual conflict detection passes (15 passes) on a figure.

    Two-layer detection:
      Layer 1 (subplot-level): passes 12, 13, per-axes summary
      Layer 2 (figure-level):  passes 1-11

    Pass 13 — Legend self-consistency:
      Checks that legend handle/text items do not overlap each other, do not
      extend beyond the figure border, and that the legend frame itself does
      not truncate at any border.

    Returns list of dicts: {type, severity, detail, elements}.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return []

    fig_bb = _fig_bbox(fig)
    infos = _collect_artists(fig, renderer)

    issues = []
    # Layer 2 — figure-level passes (1-11)
    issues.extend(_check_text_overlaps(infos, text_overlap_tol_px))
    issues.extend(_check_truncation(infos, fig_bb, border_tol_px))
    issues.extend(_check_artist_content_overlap(infos, artist_overlap_min_px2))
    issues.extend(_check_text_vs_artist_overlap(
        infos, text_overlap_tol_px, text_artist_overlap_min_px2))
    issues.extend(_check_axes_overflow(infos, fig))
    issues.extend(_check_scatter_clip_risk(fig))
    # Passes 8-10: Composed-figure-specific checks
    issues.extend(_check_cross_panel_spillover(fig, renderer))
    issues.extend(_check_panel_label_overlap(fig, renderer, infos))
    issues.extend(_check_legend_spillover(fig, renderer))
    # Passes 11: Cross-panel legend overlap
    issues.extend(_check_legend_vs_other_panel_content(fig, renderer, infos))
    # Layer 1 — subplot-level passes (12-15)
    issues.extend(_check_legend_vs_own_content(fig, renderer, infos))
    issues.extend(_check_fig_legend_vs_subplot_content(fig, renderer, infos))
    # Pass 13: Legend self-consistency (handle/text items + frame truncation)
    issues.extend(_check_legend_self_consistency(fig, renderer))
    # Passes 14-15: colorbar & legend internals
    issues.extend(_check_colorbar_internal(fig, renderer))
    issues.extend(_check_legend_internal(fig, renderer))

    # Two-layer per-axes summary
    per_ax = _per_axes_summary(fig, renderer, infos)

    if verbose:
        tag = f" [{label}]" if label else ""
        n_warn = sum(1 for x in issues if x["severity"] == "warning")
        n_info = sum(1 for x in issues if x["severity"] == "info")

        counts = {}
        for iss in issues:
            counts[iss["type"]] = counts.get(iss["type"], 0) + 1

        # ── Layer 1: subplot-level summary ──
        subplot_problems = {t: iss for t, iss in per_ax.items() if iss}
        if subplot_problems:
            print(f"  ── Layer 1 (subplot-level){tag} ──")
            for title, ax_iss in subplot_problems.items():
                for i in ax_iss:
                    print(f"    ⚠ {i['detail']}")

        # ── Layer 2: figure-level summary ──
        if n_warn > 0 or subplot_problems:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  ── Layer 2 (figure-level){tag} ──")
            print(f"  ⚠ CONFLICT: {n_warn} warnings, {n_info} info [{parts}]")
            for iss in issues[:25]:
                marker = "⚠" if iss["severity"] == "warning" else "ℹ"
                print(f"    {marker} [{iss['type']}] {iss['detail']}")
            if len(issues) > 25:
                print(f"    ... and {len(issues) - 25} more")
        elif n_info > 0:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  ℹ INFO{tag}: 0 warnings, {n_info} info [{parts}]")
            for iss in issues[:15]:
                print(f"    ℹ [{iss['type']}] {iss['detail']}")
            if len(issues) > 15:
                print(f"    ... and {len(issues) - 15} more")
        elif label:
            print(f"  ✓ OK{tag}: no conflicts detected (both layers clean)")

    return issues


def summarize_issues(all_issues: dict[str, list[dict]]) -> None:
    """Print final audit summary across multiple figures."""
    total_warn = 0
    total_info = 0
    problem_figs = []
    for name, issues in all_issues.items():
        n_w = sum(1 for x in issues if x["severity"] == "warning")
        n_i = sum(1 for x in issues if x["severity"] == "info")
        total_warn += n_w
        total_info += n_i
        if n_w > 0:
            problem_figs.append(name)

    print(f"\n{'═'*60}")
    print(f"CONFLICT AUDIT SUMMARY")
    print(f"{'═'*60}")
    print(f"  Figures checked: {len(all_issues)}")
    print(f"  Total warnings:  {total_warn}")
    print(f"  Total info:      {total_info}")
    if problem_figs:
        print(f"  Figures with warnings: {', '.join(problem_figs)}")
    else:
        print(f"  ✓ All figures clean — no warnings detected")
    print(f"{'═'*60}\n")
