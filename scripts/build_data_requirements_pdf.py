from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "ao_european_elo_data_requirements.pdf"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    build_pdf()
    print(f"PDF written: {OUTPUT_PDF}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("AORegular", FONT_REGULAR))


def build_pdf() -> None:
    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.35 * cm,
        title="AO European Elo Veri İhtiyaçları ve Alan Sözlüğü",
        author="Akıl Oyunu",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(
        [
            PageTemplate(
                id="normal",
                frames=[frame],
                onPage=draw_footer,
            )
        ]
    )
    doc.build(story())


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("AORegular", 7.5)
    canvas.setFillColor(colors.HexColor("#5F6B7A"))
    canvas.drawString(
        doc.leftMargin,
        0.72 * cm,
        "AO European Elo - Veri İhtiyaçları ve Alan Sözlüğü",
    )
    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        0.72 * cm,
        f"Sayfa {doc.page}",
    )
    canvas.restoreState()


def story() -> list:
    styles = make_styles()
    flowables: list = []

    flowables.extend(
        [
            Paragraph("AO European Elo", styles["Title"]),
            Paragraph("Veri İhtiyaçları ve Alan Sözlüğü", styles["Subtitle"]),
            Spacer(1, 0.35 * cm),
            Paragraph(
                "Bu doküman, UEFA kulüp turnuvaları için geliştirilen AO European "
                "Elo başlangıç puanı modelinde kullanılacak veri setlerini ve her "
                "alanın ne anlama geldiğini açıklar. Sistem ve model mantığını "
                "anlatan mevcut iki PDF'in yanına, veri toplama ve CSV hazırlama "
                "kılavuzu olarak tasarlanmıştır.",
                styles["Body"],
            ),
            Spacer(1, 0.2 * cm),
            note_box(
                "Ana fikir",
                "Ülke puanları lig/ülke gücünü ölçmek için kullanılır. Kulüp "
                "puanları ise takımın kendi Avrupa geçmişini ölçmek için kullanılır. "
                "played ve matches alanları performansı değil, bu Avrupa sinyaline "
                "ne kadar güvenileceğini yani exposure'i temsil eder.",
                styles,
            ),
            Spacer(1, 0.28 * cm),
            Paragraph("1. Veri Seti Özeti", styles["H1"]),
            simple_table(
                [
                    ["CSV", "Ne işe yarar?", "Modeldeki rol"],
                    [
                        "teams.csv",
                        "Takım kimliği, ülke ve lig bilgisi",
                        "Diğer tablolarla join için ana takım tablosu",
                    ],
                    [
                        "country_coefficients.csv",
                        "Son 5 sezon UEFA ülke puanları",
                        "League Strength ve Domestic Prior",
                    ],
                    [
                        "domestic_context.csv",
                        "Lig sırası, lig büyüklüğü, kupa ve Avrupa giriş bilgisi",
                        "Domestic Achievement Score",
                    ],
                    [
                        "club_european_points.csv",
                        "Kulübün son 5 sezon Avrupa puanı, oynama ve maç bilgisi",
                        "European Prior ve European Exposure",
                    ],
                ],
                styles,
                col_widths=[3.6 * cm, 6.3 * cm, 6.6 * cm],
            ),
            Spacer(1, 0.25 * cm),
            Paragraph("2. Sezon Notasyonu", styles["H1"]),
            Paragraph(
                "Her satır bir hedef sezon için hazırlanır. Örneğin hedef sezon "
                "2025/26 ise t son tamamlanmış veya modele dahil edilen en güncel "
                "Avrupa sezonunu, t_minus_1 bir önceki sezonu, t_minus_4 ise beş "
                "yıllık pencerenin en eski sezonunu temsil eder. Aynı notasyon hem "
                "ülke puanları hem de kulüp puanları için kullanılır.",
                styles["Body"],
            ),
            simple_table(
                [
                    ["Alan eki", "Anlam"],
                    ["t", "En güncel sezon"],
                    ["t_minus_1", "1 sezon önce"],
                    ["t_minus_2", "2 sezon önce"],
                    ["t_minus_3", "3 sezon önce"],
                    ["t_minus_4", "4 sezon önce"],
                ],
                styles,
                col_widths=[4.0 * cm, 12.5 * cm],
            ),
            PageBreak(),
            Paragraph("3. teams.csv", styles["H1"]),
            Paragraph(
                "Bu tablo takımlar için tekil kimlik ve temel metadata tablosudur. "
                "team_id tüm diğer CSV'lerde aynı takımı temsil edecek şekilde sabit "
                "kalmalıdır.",
                styles["Body"],
            ),
            field_table(
                [
                    ["team_id", "Evet", "Bizim sistemde benzersiz takım kimliği", "1001, 1002"],
                    ["team_name", "Evet", "Takımın standart adı", "Galatasaray"],
                    ["country", "Evet", "Takımın ülkesi", "Turkey"],
                    ["country_code", "Evet", "Ülke kodu", "TUR"],
                    ["domestic_league", "Evet", "Takımın yerel ligi", "Super Lig"],
                ],
                styles,
            ),
            Paragraph("4. country_coefficients.csv", styles["H1"]),
            Paragraph(
                "Bu tablo lig/ülke gücünü ölçmek için kullanılır. UEFA tarafından "
                "açıklanan ülke katsayıları esas alınır. Model, son 5 sezonu "
                "ağırlıklandırarak Weighted Country Score üretir.",
                styles["Body"],
            ),
            field_table(
                [
                    ["season", "Evet", "Hedef sezon", "2025/26"],
                    ["country", "Evet", "Ülke adı", "Turkey"],
                    ["country_code", "Evet", "Ülke kodu", "TUR"],
                    ["points_t_minus_4", "Evet", "4 sezon önceki UEFA ülke puanı", "6.700"],
                    ["points_t_minus_3", "Evet", "3 sezon önceki UEFA ülke puanı", "11.800"],
                    ["points_t_minus_2", "Evet", "2 sezon önceki UEFA ülke puanı", "12.000"],
                    ["points_t_minus_1", "Evet", "1 sezon önceki UEFA ülke puanı", "8.600"],
                    ["points_t", "Evet", "En güncel sezon UEFA ülke puanı", "9.900"],
                    ["official_five_year_total", "Evet", "UEFA resmi 5 yıllık toplam", "49.000"],
                    ["official_country_rank", "Evet", "UEFA resmi ülke sıralaması", "9"],
                ],
                styles,
            ),
            note_box(
                "Modeldeki anlam",
                "Ülke puanı takımın kendi Avrupa geçmişi değildir. Bu veri, takımın "
                "geldiği ligin Avrupa seviyesini temsil eder ve Domestic Prior "
                "içindeki league strength bileşenini besler.",
                styles,
            ),
            PageBreak(),
            Paragraph("5. domestic_context.csv", styles["H1"]),
            Paragraph(
                "Bu tablo takımın Avrupa sezonuna hangi yerel başarı seviyesinden "
                "geldiğini anlatır. Lig sırası mümkünse normal sezon değil, resmi "
                "sezon sonu final standings olmalıdır.",
                styles["Body"],
            ),
            field_table(
                [
                    ["season", "Evet", "Hedef sezon", "2025/26"],
                    ["team_id", "Evet", "teams.csv ile aynı takım kimliği", "1001"],
                    ["domestic_position", "Koşullu", "Final lig sırası. Bilinmiyorsa boş kalabilir", "1, 2, 6"],
                    ["league_team_count", "Koşullu", "O ligdeki takım sayısı. Bilinmiyorsa boş kalabilir", "18, 20"],
                    ["is_league_champion", "Evet", "Lig şampiyonu mu?", "true / false"],
                    ["is_cup_winner", "Evet", "Ulusal kupa şampiyonu mu?", "true / false"],
                    ["european_entry_type", "Evet", "Avrupa biletinin kazanılma yolu", "League Champion, Cup Winner"],
                    ["competition", "Önerilir", "Katıldığı UEFA turnuvası", "UCL, UEL, UECL"],
                    ["entry_round", "Önerilir", "Avrupa'ya başladığı tur", "League Phase, Q2, Playoff"],
                ],
                styles,
            ),
            note_box(
                "Modeldeki anlam",
                "domestic_position ve league_team_count birlikte percentile üretir. "
                "is_league_champion şampiyonluk tabanını, is_cup_winner kupa tabanını "
                "etkiler. Lig pozisyonu bilinmiyor ama kupa kazanıldıysa double bonus "
                "verilmez.",
                styles,
            ),
            Paragraph("6. club_european_points.csv", styles["H1"]),
            Paragraph(
                "Bu tablo kulübün kendi Avrupa geçmişini ve bu geçmişin kaç sezon/maç "
                "veriye dayandığını toplar. Burada mümkünse official club coefficient "
                "toplamını değil, kulübün sezon sezon kendi Avrupa puanlarını "
                "kullanmak gerekir.",
                styles["Body"],
            ),
            PageBreak(),
            field_table(
                [
                    ["season", "Evet", "Hedef sezon", "2025/26"],
                    ["team_id", "Evet", "teams.csv ile aynı takım kimliği", "1001"],
                    ["team_name_source", "Evet", "Veri kaynağındaki takım adı", "Galatasaray A.S."],
                    ["country_code", "Evet", "Ülke kodu", "TUR"],
                    ["club_points_t_minus_4..t", "Evet", "Kulübün ilgili sezonda kendi Avrupa puanı", "0, 4, 8.5"],
                    ["played_t_minus_4..t", "Evet", "O sezon Avrupa maçı oynadı mı?", "0 / 1"],
                    ["matches_t_minus_4..t", "Evet", "O sezon oynanan toplam Avrupa maç sayısı", "0, 6, 8, 12"],
                    ["match_cap_t_minus_4..t", "Evet", "Exposure için yeterli sezonluk maç eşiği", "6 veya 8"],
                    ["official_club_coefficient", "Önerilir", "UEFA resmi kulüp katsayısı toplam kontrol alanı", "31.500"],
                    ["country_part", "Önerilir", "Ülke payından gelen minimum katsayı kontrol alanı", "9.800"],
                ],
                styles,
            ),
            note_box(
                "Kritik ayrim",
                "club_points performans sinyalidir. played ve matches veri miktarı "
                "sinyalidir. official_club_coefficient ve country_part ana formülde "
                "kullanılmaz; veri kontrolü ve olası fallback analizi için tutulur.",
                styles,
            ),
            Paragraph("7. Match Cap Kurali", styles["H1"]),
            simple_table(
                [
                    ["Dönem", "UCL", "UEL", "UECL", "Not"],
                    ["2023/24 ve öncesi", "6", "6", "6", "Grup formatı dönemi"],
                    ["2024/25 ve sonrası", "8", "8", "6", "Yeni lig aşaması formatı"],
                ],
                styles,
                col_widths=[4.8 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 5.4 * cm],
            ),
            Paragraph(
                "match_cap maksimum sezon maçı değildir. Exposure açısından yeterli "
                "sezonluk örneklem eşiğidir. Takım cap'in üzerinde maç oynarsa ilgili "
                "sezonun match exposure değeri 1.0'da sınırlanır.",
                styles["Body"],
            ),
            PageBreak(),
            Paragraph("8. Veri Kaynağı Önceliği", styles["H1"]),
            simple_table(
                [
                    ["Veri grubu", "Birincil kaynak", "Kontrol kaynağı"],
                    ["Ülke katsayıları", "UEFA country rankings", "Kassiesa veya resmi UEFA raporları"],
                    ["Kulüp sezon puanları", "UEFA club coefficients / club rankings", "Kassiesa club coefficients"],
                    ["Maç sayıları", "UEFA maç merkezi / turnuva kayıtları", "Football-data kaynakları, manuel kontrol"],
                    ["Lig sıralaması", "Lig resmi sitesi", "FBref, Transfermarkt, Soccerway"],
                    ["Kupa şampiyonluğu", "Federasyon veya kupa resmi kayıtları", "UEFA entry list"],
                    ["Giriş turu", "UEFA access list / entry list", "Turnuva kura ve fikstür kayıtları"],
                ],
                styles,
                col_widths=[4.2 * cm, 6.0 * cm, 6.3 * cm],
            ),
            Paragraph("9. Validation ve Hazırlık Kontrol Listesi", styles["H1"]),
            *bullet_list(
                [
                    "Country_Strength_Benchmark ve European_History_Benchmark > 0 olmadan model çalıştırılmaz.",
                    "played alanları sadece 0 veya 1 olmalıdır.",
                    "played = 0 ise matches = 0 ve club_points = 0 olmalıdır.",
                    "played = 1 ise matches en az 1 olmalıdır.",
                    "matches > 0 ise match_cap > 0 olmalıdır.",
                    "domestic_position >= 1 ve domestic_position <= league_team_count olmalıdır.",
                    "is_league_champion = true ise domestic_position normalde 1 olmalıdır; değilse manuel kontrol gerekir.",
                    "Avrupa geçmişi olmayan takımlar için club_points, played ve matches alanları 0 yazılmalıdır.",
                    "Official club coefficient, kulübün kendi sezon puanlarıyla karıştırılmamalıdır.",
                ],
                styles,
            ),
            note_box(
                "Sonuc",
                "Bu veri setleri tamamlandığında model her takım için Domestic Prior, "
                "European Prior, European Exposure ve final AO First Elo puanını "
                "üretebilir. Bu doküman veri toplama kapsamıdır; model matematik "
                "detayı için mevcut teknik spesifikasyon PDF'leri referans alınmalıdır.",
                styles,
            ),
        ]
    )

    return flowables


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="AORegular",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#162033"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="AORegular",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#4A5568"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="AORegular",
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#1B365D"),
            spaceBefore=9,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=9.2,
            leading=13.2,
            textColor=colors.HexColor("#1F2933"),
            spaceAfter=7,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=7.4,
            leading=9.4,
            textColor=colors.HexColor("#1F2933"),
        ),
        "NoteTitle": ParagraphStyle(
            "NoteTitle",
            parent=base["BodyText"],
            fontName="AORegular",
            fontSize=8.6,
            leading=10.4,
            textColor=colors.HexColor("#1B365D"),
            spaceAfter=3,
        ),
    }


def simple_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], col_widths: list[float]) -> Table:
    return make_table(rows, styles, col_widths)


def field_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    return make_table(
        [["Alan", "Zorunlu mu?", "Anlam", "Örnek"], *rows],
        styles,
        [4.2 * cm, 2.3 * cm, 7.5 * cm, 2.5 * cm],
    )


def make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], col_widths: list[float]) -> Table:
    wrapped = [[Paragraph(str(cell), styles["Small"]) for cell in row] for row in rows]
    table = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#162033")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def note_box(title: str, body: str, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [
            [Paragraph(title, styles["NoteTitle"])],
            [Paragraph(body, styles["Small"])],
        ],
        colWidths=[16.5 * cm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FF")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#A9BEDA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> list:
    flowables: list = []
    for item in items:
        flowables.append(Paragraph(f"- {item}", styles["Body"]))
    return flowables


if __name__ == "__main__":
    main()
