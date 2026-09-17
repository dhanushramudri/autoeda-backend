# jman_template.docx

The real JMAN Word template ("JMAN Word Template v1.2 (Cover Page, No
Appendix)"), converted from `.dotx` to `.docx`.

`.dotx` files use the OOXML content type
`.../wordprocessingml.template.main+xml`, which `python-docx` refuses to
open (`Document()` only accepts `.../wordprocessingml.document.main+xml`).
This file is the same template with only that one string patched inside
`[Content_Types].xml` — no visual or structural change, it's a byte-for-byte
copy of the original cover page, styles, header/footer, and theme.

`app/eda/report_builder.py` opens this file fresh for every report, fills
in the cover's title/subtitle/date placeholders, strips the template's
demo/showcase body content, inserts the run's real content, and saves a
new file — this source file itself is never modified.

To refresh from a newer template version, redo the same conversion:

```python
import zipfile

src = "JMAN Word Template vX.X (...).dotx"
dst = "app/eda/assets/jman_template.docx"

with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "[Content_Types].xml":
            data = data.replace(
                b"application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
                b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
            )
        zout.writestr(item, data)
```
