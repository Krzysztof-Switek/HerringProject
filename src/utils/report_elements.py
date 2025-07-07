from pathlib import Path
from PIL import Image as PILImage
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle, Image, PageBreak)
from .report_constants import USABLE_WIDTH
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import unicodedata
def usun_polskie_znaki(text):
    if isinstance(text, str):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    return text

class ReportElement:
    def get_flowables(self):
        raise NotImplementedError

class ReportText(ReportElement):
    def __init__(self, text, style=None, spacer=6):
        self.text = usun_polskie_znaki(text)
        self.style = style or getSampleStyleSheet()['Normal']
        self.spacer = spacer
    def get_flowables(self):
        fs = []
        if self.text:
            fs.append(Paragraph(self.text, self.style))
        if self.spacer:
            fs.append(Spacer(1, self.spacer))
        return fs

class ReportTable(ReportElement):
    def __init__(
        self,
        data,
        table_width=USABLE_WIDTH,
        min_col_width=35,
        font_size=8,
        max_cols_in_table=6
    ):
        self.data = [[usun_polskie_znaki(str(cell)) for cell in row] for row in data]
        self.table_width = table_width
        self.min_col_width = min_col_width
        self.font_size = font_size
        self.max_cols_in_table = max_cols_in_table

    def get_flowables(self):
        if not self.data or len(self.data) < 2:
            return [
                Paragraph('<font color="red"><b>Brak danych do tabeli</b></font>', getSampleStyleSheet()['Normal']),
                Spacer(1, 8)
            ]
        num_cols = len(self.data[0])
        need_split = (self.table_width / num_cols) < self.min_col_width or num_cols > self.max_cols_in_table

        cell_style = ParagraphStyle(
            'cell',
            fontSize=self.font_size,
            alignment=1,
            leading=self.font_size + 2,
            spaceAfter=0,
            spaceBefore=0,
        )

        flowables = []
        block_size = self.max_cols_in_table if need_split else num_cols
        for start in range(0, num_cols, block_size):
            end = min(start + block_size, num_cols)
            sub_data = []
            for row in self.data:
                sub_data.append([
                    Paragraph(str(cell), cell_style)
                    for cell in row[start:end]
                ])
            col_width = max(self.table_width / (end - start), self.min_col_width)
            sub_tbl = Table(sub_data, colWidths=[col_width] * (end - start), repeatRows=1)
            sub_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, 0), self.font_size + 1),
                ('FONTSIZE', (0, 1), (-1, -1), self.font_size),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.grey)
            ]))
            flowables.append(sub_tbl)
            flowables.append(Spacer(1, 10))
        return flowables

class ReportImage(ReportElement):
    def __init__(self, path, width=USABLE_WIDTH, height=None, caption=None):
        self.path = str(path)
        self.width = width
        self.height = height
        self.caption = caption
    def get_flowables(self):
        flows = []
        p = Path(self.path)
        if not p.exists():
            flows.append(Paragraph(
                f"<font color='red'><b>Brak pliku wykresu: {p.name}</b></font>", getSampleStyleSheet()['Normal']))
        else:
            height = self.height
            if height is None:
                with PILImage.open(self.path) as img:
                    w, h = img.size
                    height = self.width * h / w
            flows.append(Image(self.path, width=self.width, height=height))
        if self.caption:
            flows.append(Spacer(1, 4))
            flows.append(Paragraph(self.caption, getSampleStyleSheet()['Italic']))
        flows.append(Spacer(1, 8))
        return flows

# --- DODAJ PONIŻEJ ReportImage ---
class ReportImageRow(ReportElement):  # <<< DODANE
    """Umieszcza dwa obrazy w jednym wierszu, szerokość dzielona proporcjonalnie."""
    def __init__(self, paths, width=USABLE_WIDTH, height=None, captions=None):
        assert 1 <= len(paths) <= 2, "Możesz podać 1 lub 2 obrazy!"
        self.paths = [str(p) for p in paths]
        self.width = width
        self.height = height
        self.captions = captions or [""] * len(paths)

    def get_flowables(self):
        from reportlab.platypus import Table, TableStyle
        flowables = []
        n = len(self.paths)
        cell_width = self.width / n
        cells = []
        caption_cells = []
        for i, path in enumerate(self.paths):
            p = Path(path)
            if not p.exists():
                img_flow = Paragraph(f"<font color='red'><b>Brak pliku wykresu: {p.name}</b></font>", getSampleStyleSheet()['Normal'])
            else:
                h = self.height
                if h is None:
                    with PILImage.open(path) as img:
                        w, h_img = img.size
                        h = cell_width * h_img / w
                img_flow = Image(path, width=cell_width, height=h)
            cells.append(img_flow)
            caption = self.captions[i] if self.captions else ""
            caption_cells.append(Paragraph(caption, getSampleStyleSheet()['Italic']) if caption else Spacer(1, 2))
        # Obrazki w wierszu
        tbl = Table([cells], colWidths=[cell_width]*n)
        tbl.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        flowables.append(tbl)
        # Opisy pod spodem
        tbl2 = Table([caption_cells], colWidths=[cell_width]*n)
        tbl2.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        flowables.append(tbl2)
        flowables.append(Spacer(1, 8))
        return flowables


class ReportPageBreak(ReportElement):
    def get_flowables(self):
        return [PageBreak()]
