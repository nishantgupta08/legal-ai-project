import os
import glob
import statistics
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict


class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor."""
        pass

    def extract_word_bboxes(
        self, ocr_data: Dict
    ) -> List[Tuple[int, int, int, int, str]]:
        """Extract word-level bounding boxes and text from OCR data."""
        return sorted(
            [
                (
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                    ocr_data["text"][i].strip(),
                )
                for i in range(len(ocr_data["text"]))
                if int(ocr_data["conf"][i]) > 0 and ocr_data["text"][i].strip()
            ],
            key=lambda b: (b[1], b[0]),
        )

    def group_words_into_lines(
        self, word_bboxes: List[Tuple[int, int, int, int, str]]
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """Group words into lines based on Y-coordinates."""
        line_bboxes, line_texts, current_line = [], [], []

        def _store_line():
            nonlocal current_line
            current_line.sort(key=lambda wb: wb[0])
            line_texts.append(" ".join(wb[4] for wb in current_line))
            min_x, max_x = min(wb[0] for wb in current_line), max(
                wb[0] + wb[2] for wb in current_line
            )
            min_y, max_y = min(wb[1] for wb in current_line), max(
                wb[1] + wb[3] for wb in current_line
            )
            line_bboxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
            current_line = []

        for x, y, w, h, text in word_bboxes:
            if not current_line or abs(y - current_line[-1][1]) < h * 0.6:
                current_line.append((x, y, w, h, text))
            else:
                _store_line()
                current_line.append((x, y, w, h, text))

        if current_line:
            _store_line()

        return line_texts, line_bboxes

    def group_lines_into_paragraphs(
        self,
        line_texts: List[str],
        line_bboxes: List[Tuple[int, int, int, int]],
        spacing_multiplier: float = 0.5,
    ) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """Merge lines into paragraphs based on spacing."""
        paragraph_bboxes, paragraph_texts, current_paragraph = [], [], []

        line_y_positions = [y for (_, y, _, _) in line_bboxes]
        line_gaps = [
            line_y_positions[i + 1] - line_y_positions[i]
            for i in range(len(line_y_positions) - 1)
        ]
        median_line_spacing = (
            statistics.median(line_gaps) * spacing_multiplier if line_gaps else 10
        )

        def _store_paragraph():
            nonlocal current_paragraph
            min_x, max_x = min(pb[0] for pb in current_paragraph), max(
                pb[0] + pb[2] for pb in current_paragraph
            )
            min_y, max_y = min(pb[1] for pb in current_paragraph), max(
                pb[1] + pb[3] for pb in current_paragraph
            )
            paragraph_bboxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
            paragraph_texts.append(" ".join(pb[4] for pb in current_paragraph))
            current_paragraph = []

        for i, (x, y, w, h) in enumerate(line_bboxes):
            if (
                not current_paragraph
                or (y - (current_paragraph[-1][1] + current_paragraph[-1][3]))
                < median_line_spacing
            ):
                current_paragraph.append((x, y, w, h, line_texts[i]))
            else:
                _store_paragraph()
                current_paragraph.append((x, y, w, h, line_texts[i]))

        if current_paragraph:
            _store_paragraph()

        return paragraph_texts, paragraph_bboxes

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = "output",
        page_range: Tuple[int, int] = None,
    ) -> List[str]:
        """Convert PDF to images, extract text, and store results."""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_dir = os.path.join(output_dir, pdf_name)

        if os.path.exists(pdf_output_dir) and os.listdir(pdf_output_dir):
            all_paragraph_texts = []
            for file_name in sorted(
                glob.glob(os.path.join(pdf_output_dir, "page_*/paragraph_texts.txt")),
                key=lambda x: int(x.split("page_")[1].split("\\")[0]),
            ):
                with open(file_name, "r", encoding="utf-8") as f:
                    all_paragraph_texts.extend(f.read().splitlines())
            return all_paragraph_texts

        os.makedirs(pdf_output_dir, exist_ok=True)
        images = convert_from_path(pdf_path)
        all_paragraph_texts = []

        start_page = page_range[0] - 1 if page_range else 0
        end_page = page_range[1] if page_range else len(images)

        for i in range(start_page, end_page):
            image = images[i]
            page_folder = os.path.join(pdf_output_dir, f"page_{i+1}")
            os.makedirs(page_folder, exist_ok=True)

            image_path = os.path.join(page_folder, f"page_{i+1}.jpg")
            image.save(image_path, "JPEG")

            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
            word_bboxes = self.extract_word_bboxes(ocr_data)
            line_texts, line_bboxes = self.group_words_into_lines(word_bboxes)
            paragraph_texts, _ = self.group_lines_into_paragraphs(
                line_texts, line_bboxes
            )

            all_paragraph_texts.extend(paragraph_texts)

            with open(
                os.path.join(page_folder, "paragraph_texts.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n\n".join(paragraph_texts))

        return all_paragraph_texts
