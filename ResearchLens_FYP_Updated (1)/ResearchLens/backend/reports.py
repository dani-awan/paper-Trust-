"""Export readable research notes; no invented references or report content."""
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from backend.config import APP_NAME, CREDITS


def sources_text(sources):
    lines = []
    for s in sources:
        label = f"[{s['id']}] {s['title']}"
        if s.get('page'):
            label += f" - page {s['page']}"
        if s.get('url'):
            label += f" - {s['url']}"
        lines.append(label)
    return '\n'.join(lines)


def markdown_report(title, turns):
    parts = [f'# {title}', f'{APP_NAME} | Customized by {CREDITS}',
             f"Exported {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             'AI-assisted research notes. Check primary sources. Web search is not an exhaustive literature review.']
    for turn in turns:
        parts.extend([f"## Question\n{turn['question']}", f"### Answer\n{turn['answer']}"])
        if turn.get('notices'):
            parts.append('Notes: ' + ' '.join(turn['notices']))
        if turn.get('sources'):
            parts.append('### References\n' + sources_text(turn['sources']))
    return '\n\n'.join(parts)


def pdf_report(title, turns):
    font = 'Helvetica'
    candidates = [Path('assets/DejaVuSans.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('C:/Windows/Fonts/arial.ttf')]
    for path in candidates:
        if path.exists():
            if 'ResearchFont' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('ResearchFont', str(path)))
            font = 'ResearchFont'
            break
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = font
    styles['Title'].textColor = colors.HexColor('#15304a')
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 15
    styles['BodyText'].wordWrap = 'CJK'
    styles['BodyText'].spaceAfter = 6
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, title=title, author=CREDITS, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48)
    story = []
    def add(text, style='BodyText'):
        # No markup or URLs from uploaded documents are interpreted as PDF XML.
        for line in str(text).splitlines():
            line = re.sub(r'^#{1,6}\s*', '', line).replace('**', '').replace('`', '')
            story.append(Paragraph(escape(line) or ' ', styles[style]))
    add(title, 'Title')
    add(f'{APP_NAME} | Customized by {CREDITS}')
    add(f"Exported {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    add('AI-assisted research notes. Verify claims against primary sources. Web search is not an exhaustive literature review.')
    for n, turn in enumerate(turns, 1):
        story.append(Spacer(1, 12))
        add(f'Question {n}', 'Heading2')
        add(turn['question'])
        add('Answer', 'Heading3')
        add(turn['answer'])
        for notice in turn.get('notices', []):
            add('Note: ' + notice)
        if turn.get('sources'):
            add('References', 'Heading3')
            add(sources_text(turn['sources']))
    def footer(canvas, document):
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(48, 26, APP_NAME + ' - Research notes')
        canvas.drawRightString(document.pagesize[0] - 48, 26, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
