from __future__ import annotations

from reportlab.lib.units import cm

from pdf_common import (
    PdfSpec,
    body,
    build_pdf,
    bullets,
    callout,
    cover,
    formula,
    h1,
    h2,
    page_break,
    styles,
    table,
)


SPEC = PdfSpec(
    filename="AkilOyunu_VeriAnlamlandirma.pdf",
    title="AO European Elo",
    subtitle="Veri İhtiyaçları ve Anlamlandırma Kılavuzu",
    version="AO European Elo v2 | Veri sözleşmesi revizyonu 2026-08-13",
    document_date="13 Ağustos 2026",
    subject="AO First Elo, AO Live Elo ve production 1X2 için veri alanları ve kalite kuralları",
)


def main() -> None:
    output_path, docs_path = build_pdf(SPEC, story())
    print(f"PDF written: {output_path}")
    print(f"PDF synced:  {docs_path}")


def story() -> list[object]:
    s = styles()
    out: list[object] = cover(
        SPEC,
        [
            ["Kapsam", "AO First + AO Live + production 1X2"],
            ["Kimlik anahtarı", "Kalıcı AO team_id / club_id"],
            ["Zaman standardı", "Timezone-aware exact UTC"],
            ["xG davranışı", "Opsiyonel; iki taraflı ve scope doğrulanmış"],
            ["Eksik veri", "Sessiz sıfır yok; açık fallback veya validation"],
            ["Ana referans", "docs/ai/DATA_CONTRACTS.md"],
        ],
        s,
        summary_title="Veri felsefesi",
        summary=(
            "Her alan ya performans, ya kanıt miktarı, ya maç bağlamı ya da audit bilgisi taşır. "
            "Eksik veri performansın sıfır olduğu anlamına gelmez. Model yalnız açıkça tanımlanmış "
            "zero-history satırını sıfır kabul eder; diğer eksikler hata veya fallback üretir."
        ),
    )

    out += [
        h1("1. Veri Katmanları", s),
        table(
            [
                ["Katman", "Ne zaman gerekir?", "Ürettiği sonuç"],
                ["Static season data", "Sezon başlamadan önce", "AO First Elo"],
                ["Fixture metadata", "Kickoff'tan önce", "Locked AO ve production 1X2"],
                ["Match settlement", "Maç bittikten sonra", "Power Elo güncellemesi"],
                ["xG", "Maç sonrası, uygunsa", "Bounded performans düzeltmesi"],
                ["Tie/progression", "Eleme eşleşmesi bitince", "Sezonluk progression bonusu"],
                ["Domestic feature state", "Prediction servisinde", "Domestic Poisson 1X2 bileşeni"],
                ["Prediction audit", "Her tahminde", "Monitoring ve yeniden üretilebilirlik"],
            ],
            [4.1 * cm, 5.4 * cm, 6.95 * cm],
            s,
        ),
        callout(
            "Rating ve prediction verisi farklıdır",
            "Structural ML ve Domestic Poisson feature'ları AO First veya AO Live ratingine "
            "yazılmaz. Bu veriler yalnız 1X2 tahmin servisinde kullanılır.",
            s,
            tone="green",
        ),
        h1("2. Ortak Kimlik ve Zaman Kuralları", s),
        *bullets(
            [
                "team_id/club_id AO tarafından yönetilen kalıcı kimliktir; ad değişikliği kimliği değiştirmez.",
                "UEFA, TheSportsDB, FotMob veya başka provider ID'leri ayrı mapping alanıdır.",
                "season hedef sezonu ifade eder; 2026/27 gibi kararlı formatta tutulur.",
                "kickoff_utc timezone-aware olmalı; yerel saat metni tek başına kabul edilmez.",
                "Maç sırası kickoff_utc ve match_id ile deterministiktir.",
                "Aynı takım aynı kickoff anında iki farklı maçta bulunamaz.",
            ],
            s,
        ),
        h2("2.1 Beş sezon notasyonu", s),
        table(
            [
                ["Alan eki", "Static puan tablolarında anlamı"],
                ["t", "Hedef sezondan önce tamamlanan en son sezon"],
                ["t_minus_1", "Bir önceki tamamlanmış sezon"],
                ["t_minus_2", "İki sezon önce"],
                ["t_minus_3", "Üç sezon önce"],
                ["t_minus_4", "Beş yıllık pencerenin en eski sezonu"],
            ],
            [5.0 * cm, 11.45 * cm],
            s,
        ),
        formula("Ağırlıklar: 0.07 / 0.13 / 0.20 / 0.27 / 0.33", s),
        page_break(),
        h1("3. teams.csv", s),
        body(
            "Takım evreninin ana kimlik tablosudur. Bir team_id yalnız bir kez bulunur ve diğer "
            "static tablolar bu kimliğe bağlanır.",
            s,
        ),
        table(
            [
                ["Alan", "Zorunlu", "Anlam", "Örnek"],
                ["team_id", "Evet", "Kalıcı AO kulüp kimliği", "AO-UEFA-50067"],
                ["team_name", "Evet", "Standart görünen takım adı", "Bayern München"],
                ["country", "Evet", "Ülke adı", "Germany"],
                ["country_code", "Evet", "Kararlı ülke kodu", "GER"],
                ["domestic_league", "Evet", "Yerel lig adı", "Bundesliga"],
            ],
            [3.3 * cm, 2.2 * cm, 7.0 * cm, 3.95 * cm],
            s,
        ),
        h1("4. country_coefficients.csv", s),
        body(
            "UEFA ülke puanları takımın kendi geçmişini değil, takımın geldiği ligin Avrupa "
            "gücünü besler. Anahtar season + country_code'dur.",
            s,
        ),
        table(
            [
                ["Alan", "Zorunlu", "Anlam"],
                ["season", "Evet", "Ratingin üretildiği hedef sezon"],
                ["country / country_code", "Evet", "Ülke adı ve join anahtarı"],
                ["points_t_minus_4 ... points_t", "Evet", "Beş tamamlanmış sezonun UEFA ülke puanı"],
                ["official_five_year_total", "Hayır", "Resmi toplamı kontrol eden audit alanı"],
                ["official_country_rank", "Hayır", "Resmi sıra audit alanı"],
            ],
            [5.1 * cm, 2.5 * cm, 8.85 * cm],
            s,
        ),
        callout(
            "Yorum",
            "Bu puan yüksekse takım otomatik olarak Avrupa'da başarılı sayılmaz. Yalnız Domestic "
            "Prior içindeki League Strength yükselir; kulübün kendi kanıtı ayrı dosyadan gelir.",
            s,
            tone="blue",
        ),
        h1("5. domestic_context.csv", s),
        body(
            "Takımın hedef Avrupa sezonuna hangi yerel başarıyla geldiğini ve Domestic Surprise "
            "için son beş lig performansını taşır. Anahtar season + team_id'dir.",
            s,
        ),
        table(
            [
                ["Alan", "Zorunlu", "Anlam"],
                ["season / team_id", "Evet", "Hedef sezon ve kalıcı kulüp anahtarı"],
                ["domestic_position", "Koşullu", "Resmi final lig sırası"],
                ["league_team_count", "Koşullu", "O sezon ligdeki takım sayısı; position ile birlikte"],
                ["is_league_champion", "Evet", "Yalnız true/false veya 0/1"],
                ["is_cup_winner", "Evet", "Ulusal ana kupa şampiyonu"],
                ["european_entry_type", "Evet", "Avrupa biletinin yolu; metadata"],
                ["competition / entry_round", "Önerilir", "UCL/UEL/UECL ve giriş aşaması metadata'sı"],
                ["history_position_t_minus_5 ... t_minus_1", "Surprise için", "Beş tamamlanmış yerel sezon sırası"],
                ["history_team_count_t_minus_5 ... t_minus_1", "Surprise için", "Aynı sezonların lig büyüklüğü"],
            ],
            [5.4 * cm, 2.6 * cm, 8.45 * cm],
            s,
        ),
        *bullets(
            [
                "Position verilmişse 1..league_team_count aralığında tam sayı olmalıdır.",
                "Champion=true ve bilinen position varsa position tam olarak 1 olmalıdır.",
                "Kupa şampiyonu tek başına duble bonusu alamaz.",
                "History position ve team count her sezon için birlikte dolu veya birlikte boş olmalıdır.",
                "Beş tam history sezonu yoksa Domestic Surprise adjustment sıfır olur.",
            ],
            s,
        ),
        page_break(),
        h1("6. club_european_points.csv", s),
        body(
            "Kulübün kendi Avrupa performansı ile bu sinyalin veri miktarını aynı satırda fakat "
            "ayrı alanlarda taşır. Anahtar season + team_id + country_code'dur.",
            s,
        ),
        table(
            [
                ["Alan", "Zorunlu", "Anlam"],
                ["season / team_id / country_code", "Evet", "Tam join anahtarı"],
                ["team_name_source", "Evet", "Kaynak sistemdeki ad; identity audit"],
                ["club_points_t_minus_4 ... t", "Evet", "Kulübün sezon bazlı kendi Avrupa puanı"],
                ["played_t_minus_4 ... t", "Evet", "O sezon Avrupa maçı oynandı mı?"],
                ["matches_t_minus_4 ... t", "Evet", "O sezon tamamlanan Avrupa maç sayısı"],
                ["match_cap_t_minus_4 ... t", "Evet", "Exposure için pozitif yeterli örneklem eşiği"],
                ["official_club_coefficient", "Hayır", "UEFA resmi coefficient audit alanı"],
                ["country_part", "Hayır", "Ülke payı audit alanı; ana formüle girmez"],
            ],
            [5.3 * cm, 2.6 * cm, 8.55 * cm],
            s,
        ),
        callout(
            "Performans ile güveni karıştırmayın",
            "club_points European Prior'ı, played ve matches European Exposure'ı besler. Çok maç "
            "oynamak tek başına güçlü performans değildir; yalnız puan sinyaline güveni artırır.",
            s,
            tone="green",
        ),
        h2("6.1 Avrupa geçmişi olmayan takım", s),
        formula(
            [
                "club_points_* = 0",
                "played_* = 0",
                "matches_* = 0",
                "match_cap_* > 0 olmalı",
            ],
            s,
        ),
        body(
            "Satır atlanmaz. Eksik satır sistem tarafından 'Avrupa geçmişi yok' diye yorumlanmaz; "
            "validation hatası verir. Bu ayrım sessiz join hatasını önler.",
            s,
        ),
        h1("7. Static Verinin Modeldeki Anlamı", s),
        table(
            [
                ["Sinyal", "Ne ölçer?", "Ne ölçmez?"],
                ["Country points", "Ligin Avrupa seviyesi", "Kulübün kendi performansı"],
                ["Domestic position", "Güncel yerel başarı", "Avrupa maç gücü"],
                ["Historical positions", "Takımın yerel normali ve oynaklığı", "Güncel form"],
                ["Club points", "Kulübün Avrupa performansı", "Veri güveni"],
                ["Played/matches", "Avrupa kanıt miktarı", "Performans kalitesi"],
                ["Official totals", "Kaynak doğrulama", "Ana rating formülü"],
            ],
            [4.1 * cm, 6.3 * cm, 6.05 * cm],
            s,
        ),
        page_break(),
        h1("8. Pre-Match Fixture Verisi", s),
        body(
            "Sonuç içermeyen fixture satırı hem Current AO hem production ensemble tahmini için "
            "kickoff'tan önce doğrulanır ve kilitlenir.",
            s,
        ),
        table(
            [
                ["Alan grubu", "Alanlar", "Neden gerekir?"],
                ["Kimlik", "match_id, season", "Duplicate ve sezon state kontrolü"],
                ["Zaman", "kickoff_utc", "Pre-match lock ve chronology"],
                ["Turnuva", "competition, round, stage", "Format ve progression sözleşmesi"],
                ["Eşleşme", "tie_id, is_knockout, is_tie_decider", "Tek/çift maç state'i"],
                ["Format", "is_single_match_tie / format_type", "Draw 0.12 veya 0.24 seçimi"],
                ["Takımlar", "home_team_id, away_team_id", "AO Live state lookup"],
                ["Saha", "is_neutral", "H=148.544 veya 0"],
            ],
            [3.5 * cm, 7.0 * cm, 5.95 * cm],
            s,
        ),
        *bullets(
            [
                "generated_at_utc kickoff_utc değerinden küçük olmalıdır.",
                "is_single_match_tie maç sonucu veya uzatmadan türetilemez; format bilgisidir.",
                "Tahmin girdisinde home_goals, away_goals, xG veya advanced_team_id bulunmaz.",
                "Tie/stage alanları turnuva formatıyla tutarlı olmalıdır.",
            ],
            s,
        ),
        h1("9. Match Settlement Verisi", s),
        table(
            [
                ["Alan", "Anlam / kural"],
                ["home_goals / away_goals", "90 dakika veya uzatma oynandıysa 120 dakika saha skoru"],
                ["decided_on_penalties", "Shootout oldu mu? Shootout golleri skora eklenmez"],
                ["xg_home / xg_away", "Birlikte bulunur veya birlikte boş kalır"],
                ["xg_analysis_eligible", "Scope ve kalite doğrulanmışsa true"],
                ["advanced_team_id", "Tie decider sonunda turu geçen takım"],
                ["stage / tie_id", "Progression bonusunun tek ve doğru aşamada uygulanması"],
            ],
            [5.2 * cm, 11.25 * cm],
            s,
        ),
        callout(
            "Penaltı örneği",
            "Bir takım rövanşı sahada 2-0 kazanıp aggregate eşitlik sonrası penaltıyla elense bile "
            "o maç Power Elo için 2-0 galibiyettir. Shootout yalnız GD/xG ek sinyalini kapatır; "
            "progression bonusunu advanced_team_id belirler.",
            s,
            tone="amber",
        ),
        page_break(),
        h1("10. xG Veri Sözleşmesi", s),
        body(
            "xG maçın skorunu değiştirmez; skorla kazanılan Elo'yu performans üretimine göre "
            "kontrollü biçimde artırır veya azaltır.",
            s,
        ),
        table(
            [
                ["Kalite alanı", "Kabul koşulu"],
                ["Provider", "Açık ve aynı metodoloji; maç bazında source ID/URL saklanır"],
                ["İki taraf", "Home ve away xG birlikte finite ve >=0"],
                ["Zaman kapsamı", "Field score 90 ise 90; uzatma varsa uyumlu 120 dakika"],
                ["Maç içi penaltı", "Provider kapsamı belgelenir ve iki taraf için tutarlı olur"],
                ["Shootout", "Kesinlikle xG toplamına eklenmez"],
                ["Eksik/şüpheli", "xg_analysis_eligible=false ve iki değer boş"],
            ],
            [5.0 * cm, 11.45 * cm],
            s,
        ),
        *bullets(
            [
                "Eksik xG sıfırla, lig ortalamasıyla veya tahminle doldurulmaz.",
                "Farklı provider xG değerleri tek ana kolonda sessizce karıştırılmaz.",
                "Aynı maç için kaynak snapshot zamanı ve checksum saklanmalıdır.",
                "xG maçtan sonra gelir; pre-match ML feature olarak hedef maça sızamaz.",
            ],
            s,
        ),
        h1("11. Structural ML Feature Verisi", s),
        body(
            "Production ML modeli yalnız kickoff'tan önce üretilebilen feature'ları okur. Feature "
            "store satırı match_id ile tektir ve training/inference şeması artifact içinde dondurulur.",
            s,
        ),
        table(
            [
                ["Feature grubu", "Örnek alanlar"],
                ["AO temel", "AO log-odds, expected score, Live/First farkı, exposure farkı"],
                ["Format", "UCL/UEL/UECL, stage, round, format_type, neutral, leg"],
                ["Avrupa formu", "Son 3/8 residual, gol atma/yeme, home/away residual"],
                ["Yoğunluk", "Dinlenme günü, son 14/30 gündeki maç sayısı"],
                ["Tie durumu", "Önceki leg sayısı ve kickoff öncesi aggregate lead"],
                ["Eksik veri", "Train medyanı/UNKNOWN ve açık missing davranışı"],
            ],
            [4.2 * cm, 12.25 * cm],
            s,
        ),
        callout(
            "Yasak feature'lar",
            "Hedef maçın sonucu, golü, xG'si, tur sonucu, post-match Elo'su veya gelecekteki bir "
            "maçtan gelen bilgi hiçbir pre-match feature'a giremez.",
            s,
            tone="red",
        ),
        page_break(),
        h1("12. Domestic Poisson State Verisi", s),
        body(
            "Yerel maçlar causal sırayla işlenerek takım hücum, savunma ve reliability state'i "
            "oluşturur. Eğitim evreninde 45.423 maç ve 19 lig vardır; 171 kulüp AO kimliğiyle "
            "production artifact'ına eşlenmiştir.",
            s,
        ),
        table(
            [
                ["Girdi", "Kural"],
                ["Domestic match ID", "Lig içinde benzersiz"],
                ["Kickoff UTC", "Sonuçtan önce state snapshot üretmeye uygun"],
                ["Home/away source team ID", "AO mapping'den ayrı provider identity"],
                ["Home/away goals", "Tamamlanmış saha skoru, non-negative integer"],
                ["League + season", "Aynı lig içi sıfır merkezleme ve season carry"],
                ["AO mapping", "Yalnız güvenli eşleşmeler Avrupa feature'ına aktarılır"],
            ],
            [5.4 * cm, 11.05 * cm],
            s,
        ),
        body(
            "AO mapping'i olmayan yerel takımlar model state'ini eğitmeye devam eder fakat Avrupa "
            "fixture'ı için feature üretmez. Yerel geçmiş yoksa Poisson bileşeni AO fallback kullanır.",
            s,
        ),
        h1("13. Production Prediction Input ve Log", s),
        h2("13.1 Asgari base input", s),
        formula(
            [
                "match_id, season, kickoff_utc, competition, round, stage, format_type",
                "is_neutral, home_club_id, away_club_id",
                "ao_home_probability, ao_draw_probability, ao_away_probability",
                "+ frozen Structural Logistic feature schema",
            ],
            s,
        ),
        h2("13.2 Her tahminde loglanan audit", s),
        *bullets(
            [
                "AO, ham ML, Current ML, ham Poisson, AO Poisson ve final H/D/A.",
                "Domestic Poisson coverage: BOTH / ONE / NONE ve component fallback.",
                "Prediction status, fallback reason ve rating_feedback_applied=false.",
                "Model/config/contract/manifest/ML/state SHA-256 kimlikleri.",
                "generated_at_utc ve kickoff_utc; tahminin gerçekten pre-match olduğunun kanıtı.",
            ],
            s,
        ),
        h1("14. Ana Output Dosyaları", s),
        table(
            [
                ["Dosya", "İçerik", "Kullanım"],
                ["ao_first_elo.csv", "Static bileşenler, exposure, surprise, final sıra", "Sezon seed'i ve audit"],
                ["ratings_state.csv", "First, Power, progression, Live, son event", "Güncel state snapshot"],
                ["state_checkpoint.json", "Processed maç/tie ve checksum", "Güvenli resume"],
                ["match_updates.csv", "Pre/post rating, E, GD, xG, delta, progression", "Tam settlement audit"],
                ["pre_match_prediction_log.csv", "AO/ML/Poisson/final 1X2 ve hash'ler", "Prospective monitoring"],
            ],
            [4.4 * cm, 7.4 * cm, 4.65 * cm],
            s,
        ),
        page_break(),
        h1("15. Veri Kaynağı Önceliği", s),
        table(
            [
                ["Veri", "Birincil tercih", "Kontrol"],
                ["UEFA ülke/kulüp puanı", "UEFA resmi ranking/coefficient", "Dondurulmuş ikinci kaynak"],
                ["UEFA fixture ve sonuç", "UEFA resmi match endpoint/merkezi", "Bağımsız skor kaynağı"],
                ["Lig sırası/kupa", "Lig veya federasyon resmi kaydı", "Güvenilir tarihsel veri sağlayıcı"],
                ["xG", "Tek ve belgelenmiş sağlayıcı", "Ortak maç provider karşılaştırması"],
                ["Yerel lig maçları", "Lisanslı/API schedule sonucu", "Final tablo maç sayısı uyumu"],
                ["Takım kimliği", "AO identity registry", "UEFA/provider ID audit"],
            ],
            [4.2 * cm, 6.2 * cm, 6.05 * cm],
            s,
        ),
        h1("16. Veri Kalitesi Kabul Listesi", s),
        table(
            [
                ["Kontrol", "Beklenen"],
                ["Anahtar benzersizliği", "Takım, ülke-sezon, takım-sezon, maç ve tie duplicate yok"],
                ["Sayısal kalite", "Finite; izin verilmedikçe non-negative"],
                ["Boolean kalite", "Tanımlı true/false veya 0/1"],
                ["Kimlik coverage", "Tüm maç takımları AO state/mapping içinde"],
                ["Chronology", "Exact UTC monoton; aynı kickoff batch leakage yok"],
                ["Skor", "Field score ile shootout ayrılmış"],
                ["xG", "İki taraflı, scope uyumlu veya tamamen boş"],
                ["Olasılık", "Finite, >=0, toplam 1"],
                ["Rating", "Power zero-sum; exposure/cap/sign invariantları"],
                ["Artifact", "Contract ve bütün production SHA-256 değerleri uyumlu"],
            ],
            [5.3 * cm, 11.15 * cm],
            s,
        ),
        h1("17. Toplama ve Güncelleme Takvimi", s),
        table(
            [
                ["Zaman", "İşlem"],
                ["Sezon öncesi", "Static dört CSV, AO First snapshot ve initial rank audit"],
                ["Fixture yayımlandığında", "Kimlik, kickoff, format, neutral ve tie metadata"],
                ["Kickoff öncesi", "Feature snapshot, Current AO ve production prediction lock"],
                ["Maç sonrası", "90/120 skor, xG scope, Power settlement"],
                ["Tie tamamlanınca", "advanced_team_id ve tek progression event"],
                ["Her batch sonrası", "State checkpoint, checksum ve monitoring log"],
                ["Aylık/sezon sonu", "Coverage, fallback, loss, calibration ve data-quality raporu"],
            ],
            [5.0 * cm, 11.45 * cm],
            s,
        ),
        callout(
            "Analize hazır veri tanımı",
            "Bir veri seti yalnız satır sayısı tamamlandığında değil; kimlik, zaman, skor, xG scope, "
            "duplicate, coverage ve leakage kontrolleri geçtiğinde analize hazırdır.",
            s,
            tone="green",
        ),
    ]
    return out


if __name__ == "__main__":
    main()
