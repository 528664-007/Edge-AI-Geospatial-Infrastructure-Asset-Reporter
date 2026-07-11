"""
reporter.py - Automated PDF summary report generator (and optional email
dispatch) for the Edge-AI Geospatial Infrastructure & Asset Reporter.

Builds a print-ready PDF containing:
  * A title block with generation timestamp and reporting window
  * Headline KPI stats (total detections, average confidence, date range)
  * A breakdown table of detections by anomaly type
  * A detailed table of the most recent anomalies

Uses only ReportLab's Platypus layout API - no external charting
dependency required, so it stays lightweight enough to run on the same
edge box doing inference.
"""

from __future__ import annotations

import datetime as dt
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

import config
import database

_styles = getSampleStyleSheet()

_TITLE_STYLE = ParagraphStyle(
    "ReportTitle",
    parent=_styles["Title"],
    fontSize=22,
    textColor=colors.HexColor("#0B3D91"),
    spaceAfter=4,
)
_SUBTITLE_STYLE = ParagraphStyle(
    "ReportSubtitle",
    parent=_styles["Normal"],
    fontSize=10,
    textColor=colors.HexColor("#555555"),
    spaceAfter=16,
)
_SECTION_STYLE = ParagraphStyle(
    "SectionHeader",
    parent=_styles["Heading2"],
    fontSize=14,
    textColor=colors.HexColor("#0B3D91"),
    spaceBefore=18,
    spaceAfter=8,
)
_BODY_STYLE = _styles["Normal"]


def _kpi_table(stats: dict) -> Table:
    first_seen = stats["first_seen"].strftime("%Y-%m-%d %H:%M") if stats["first_seen"] else "N/A"
    last_seen = stats["last_seen"].strftime("%Y-%m-%d %H:%M") if stats["last_seen"] else "N/A"

    data = [
        ["Total Detections", "Average Confidence", "First Seen", "Last Seen"],
        [
            str(stats["total_detections"]),
            f"{stats['average_confidence'] * 100:.1f}%",
            first_seen,
            last_seen,
        ],
    ]
    table = Table(data, colWidths=[42 * mm, 42 * mm, 50 * mm, 50 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F2F5FB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ]
        )
    )
    return table


def _breakdown_table(stats: dict) -> Table:
    header = ["Anomaly Type", "Count", "% of Total"]
    total = stats["total_detections"] or 1
    rows = [header]
    for anomaly_type, count in stats["counts_by_type"].items():
        label = config.ANOMALY_CLASSES.get(anomaly_type, anomaly_type)
        pct = f"{(count / total) * 100:.1f}%"
        rows.append([label, str(count), pct])

    if len(rows) == 1:
        rows.append(["No detections recorded", "-", "-"])

    table = Table(rows, colWidths=[90 * mm, 30 * mm, 30 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#193A6B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FB")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _recent_anomalies_table(anomalies: list) -> Table:
    header = ["Timestamp", "Type", "Confidence", "Latitude", "Longitude"]
    rows = [header]
    for record in anomalies:
        label = config.ANOMALY_CLASSES.get(record["anomaly_type"], record["anomaly_type"])
        timestamp = record["timestamp"]
        ts_str = (
            timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(timestamp, dt.datetime)
            else str(timestamp)
        )
        rows.append(
            [
                ts_str,
                label,
                f"{record['confidence'] * 100:.1f}%",
                f"{record['latitude']:.5f}",
                f"{record['longitude']:.5f}",
            ]
        )

    if len(rows) == 1:
        rows.append(["-", "No anomalies recorded", "-", "-", "-"])

    table = Table(rows, colWidths=[35 * mm, 48 * mm, 25 * mm, 25 * mm, 25 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#193A6B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FB")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def generate_report(output_path: Optional[Path] = None, top_n: Optional[int] = None) -> Path:
    """Builds the PDF summary report and returns the path it was written to."""
    output_path = output_path or (
        config.REPORTS_DIR / f"asset_report_{dt.datetime.now():%Y%m%d_%H%M%S}.pdf"
    )
    top_n = top_n or config.REPORT_TOP_N_ANOMALIES

    stats = database.get_summary_stats()
    recent = database.get_recent_anomalies(limit=top_n)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        title=config.REPORT_TITLE,
    )

    story = []
    story.append(Paragraph(config.REPORT_TITLE, _TITLE_STYLE))
    story.append(
        Paragraph(
            f"Generated {dt.datetime.now():%Y-%m-%d %H:%M:%S} &middot; "
            f"Device: {config.DEVICE.upper()} &middot; "
            f"Reporting window: last {top_n} recorded anomalies",
            _SUBTITLE_STYLE,
        )
    )
    story.append(HRFlowable(width="100%", color=colors.HexColor("#0B3D91"), thickness=1.2))

    story.append(Paragraph("Summary", _SECTION_STYLE))
    story.append(_kpi_table(stats))

    story.append(Paragraph("Breakdown by Anomaly Type", _SECTION_STYLE))
    story.append(_breakdown_table(stats))

    story.append(Paragraph(f"Top {top_n} Most Recent Anomalies", _SECTION_STYLE))
    story.append(_recent_anomalies_table(recent))

    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "This report was generated automatically by the Edge-AI Geospatial "
            "Infrastructure & Asset Reporter. Coordinates are derived from "
            "on-device geolocation captured at inference time.",
            _BODY_STYLE,
        )
    )

    doc.build(story)
    return output_path


def email_report(pdf_path: Path, subject: Optional[str] = None, body: Optional[str] = None) -> bool:
    """Emails the generated PDF as an attachment using the SMTP settings in
    config.py. Returns True on success, False if SMTP is not configured or
    sending fails (never raises, so this can't crash a dashboard button)."""
    if not config.SMTP_HOST or not config.SMTP_USERNAME or not config.REPORT_EMAIL_TO:
        return False

    subject = subject or f"{config.REPORT_TITLE} - {dt.date.today():%Y-%m-%d}"
    body = body or (
        "Attached is the latest automated infrastructure and asset detection "
        "summary report from the Edge-AI Geospatial monitoring pipeline."
    )

    message = MIMEMultipart()
    message["From"] = config.REPORT_EMAIL_FROM
    message["To"] = ", ".join(config.REPORT_EMAIL_TO)
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with open(pdf_path, "rb") as pdf_file:
        attachment = MIMEApplication(pdf_file.read(), _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename=pdf_path.name)
        message.attach(attachment)

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as server:
            if config.SMTP_USE_TLS:
                server.starttls()
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            server.sendmail(config.REPORT_EMAIL_FROM, config.REPORT_EMAIL_TO, message.as_string())
        return True
    except Exception as exc:
        print(f"[reporter] Failed to send email report: {exc}")
        return False


def generate_and_send_report() -> dict:
    """Convenience wrapper used by the Streamlit 'Generate Report' button:
    builds the PDF, attempts email delivery, and returns a status dict the
    UI can render directly."""
    pdf_path = generate_report()
    emailed = email_report(pdf_path)
    return {
        "pdf_path": str(pdf_path),
        "emailed": emailed,
        "email_configured": bool(
            config.SMTP_HOST and config.SMTP_USERNAME and config.REPORT_EMAIL_TO
        ),
    }


if __name__ == "__main__":
    result = generate_and_send_report()
    print(result)
