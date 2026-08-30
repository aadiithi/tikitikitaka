"""aigcdet.data - dataset manifests, shortcut audits, and canonicalisation.

Three responsibilities, kept in three separate modules so each has one job:

* `manifest`   - turn a folder of images (or explicit real/fake dirs) into a
                 CSV of (image_path, label, source, family), and split it so
                 a fake generator family never appears on both sides of
                 train/test.
* `shortcuts`  - the metadata-only leak probe: can the label be guessed from
                 file size, dimensions, format, or JPEG history alone, with
                 no pixels examined at all?
* `normalize`  - canonicalisation: force every image through the same
                 size/aspect/colour-mode pipeline before it reaches a
                 backbone, so a class-correlated size or format difference
                 can't leak into the model as a shortcut.
"""

from __future__ import annotations

__all__ = ["manifest", "shortcuts", "normalize"]
