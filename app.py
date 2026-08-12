import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import warnings
import io
import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')

print("--- MENYIAPKAN SMART DEPLOYMENT GRADIO (GRID LAYOUT & KONTROL KANAN) ---")

# 1. Load Preprocessor dan Model
preprocessor = joblib.load('preprocessor_all.pkl')
knn_model    = joblib.load('knn_model_all.pkl')
dt_model     = joblib.load('dt_model_all.pkl')

# 2. Baca Struktur Data
df_ref = pd.read_csv('diabetic_data.csv', na_values='?', nrows=100)
kolom_sampah = ['encounter_id', 'patient_nbr', 'weight', 'max_glu_serum', 'payer_code', 'medical_specialty', 'readmitted']
df_ref = df_ref.drop(columns=[col for col in kolom_sampah if col in df_ref.columns])
nama_semua_kolom = df_ref.columns.tolist()

# 2b. HITUNG METRIK PERFORMA MODEL (pakai sample test set)
print("--- MENGHITUNG METRIK PERFORMA MODEL ---")
try:
    from sklearn.model_selection import train_test_split
    df_full = pd.read_csv('diabetic_data.csv', na_values='?', low_memory=False)
    df_full['target'] = df_full['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df_full = df_full.drop(columns=[c for c in kolom_sampah if c in df_full.columns]).dropna()
    X_full = df_full.drop(columns=['target'])
    y_full = df_full['target']
    _, X_test_raw, _, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    # Gunakan sample 2000 agar KNN tetap cepat
    sample_idx = X_test_raw.sample(min(2000, len(X_test_raw)), random_state=42).index
    X_samp = X_test_raw.loc[sample_idx]
    y_samp = y_test.loc[sample_idx]
    X_samp_proc = preprocessor.transform(X_samp)
    pred_dt_eval  = dt_model.predict(X_samp_proc)
    pred_knn_eval = knn_model.predict(X_samp_proc)
    def hitung_metrik(y_true, y_pred):
        return {
            'accuracy':  round(accuracy_score(y_true, y_pred) * 100, 2),
            'precision': round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            'recall':    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
            'f1':        round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
            'cm':        confusion_matrix(y_true, y_pred).tolist()
        }
    metrik_dt  = hitung_metrik(y_samp, pred_dt_eval)
    metrik_knn = hitung_metrik(y_samp, pred_knn_eval)
    print(f"✅ Metrik berhasil dihitung (n_sample={len(y_samp)})")
except Exception as e:
    print(f"⚠️ Gagal menghitung metrik: {e}")
    metrik_dt  = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'cm': [[0,0],[0,0]]}
    metrik_knn = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'cm': [[0,0],[0,0]]}


# 3. KAMUS TERJEMAHAN (Label Kolom)
kamus_kolom = {
    'race': ('Ras / Etnis', 'Race / Ethnicity'),
    'gender': ('Jenis Kelamin', 'Gender'),
    'age': ('Kelompok Usia', 'Age Group'),
    'time_in_hospital': ('Lama Rawat Inap (Hari)', 'Time in Hospital (Days)'),
    'num_lab_procedures': ('Jumlah Tes Laboratorium', 'Number of Lab Procedures'),
    'num_procedures': ('Jumlah Prosedur Medis', 'Number of Medical Procedures'),
    'num_medications': ('Jumlah Obat yang Diberikan', 'Number of Medications'),
    'number_diagnoses': ('Total Diagnosis', 'Total Diagnoses'),
    'admission_type_id': ('Tipe Masuk RS (ID)', 'Admission Type ID'),
    'discharge_disposition_id': ('Status Keluar RS (ID)', 'Discharge Disposition ID'),
    'admission_source_id': ('Sumber Masuk RS (ID)', 'Admission Source ID'),
    'number_outpatient': ('Kunjungan Rawat Jalan', 'Number of Outpatient Visits'),
    'number_emergency': ('Kunjungan Darurat', 'Number of Emergency Visits'),
    'number_inpatient': ('Kunjungan Rawat Inap', 'Number of Inpatient Visits'),
    'diag_1': ('Diagnosis Utama (Diag 1)', 'Primary Diagnosis (Diag 1)'),
    'diag_2': ('Diagnosis Kedua (Diag 2)', 'Secondary Diagnosis (Diag 2)'),
    'diag_3': ('Diagnosis Ketiga (Diag 3)', 'Additional Diagnosis (Diag 3)'),
    'max_glu_serum': ('Glukosa Serum Maks', 'Max Glu Serum'),
    'A1Cresult': ('Hasil Tes A1C', 'A1C Result'),
    'metformin': ('Metformin', 'Metformin'),
    'repaglinide': ('Repaglinide', 'Repaglinide'),
    'nateglinide': ('Nateglinide', 'Nateglinide'),
    'chlorpropamide': ('Chlorpropamide', 'Chlorpropamide'),
    'glimepiride': ('Glimepiride', 'Glimepiride'),
    'acetohexamide': ('Acetohexamide', 'Acetohexamide'),
    'glipizide': ('Glipizide', 'Glipizide'),
    'glyburide': ('Glyburide', 'Glyburide'),
    'tolbutamide': ('Tolbutamide', 'Tolbutamide'),
    'pioglitazone': ('Pioglitazone', 'Pioglitazone'),
    'rosiglitazone': ('Rosiglitazone', 'Rosiglitazone'),
    'acarbose': ('Acarbose', 'Acarbose'),
    'miglitol': ('Miglitol', 'Miglitol'),
    'troglitazone': ('Troglitazone', 'Troglitazone'),
    'tolazamide': ('Tolazamide', 'Tolazamide'),
    'examide': ('Examide', 'Examide'),
    'citoglipton': ('Citoglipton', 'Citoglipton'),
    'insulin': ('Insulin', 'Insulin'),
    'glyburide-metformin': ('Glyburide-Metformin', 'Glyburide-Metformin'),
    'glipizide-metformin': ('Glipizide-Metformin', 'Glipizide-Metformin'),
    'glimepiride-pioglitazone': ('Glimepiride-Pioglitazone', 'Glimepiride-Pioglitazone'),
    'metformin-rosiglitazone': ('Metformin-Rosiglitazone', 'Metformin-Rosiglitazone'),
    'metformin-pioglitazone': ('Metformin-Pioglitazone', 'Metformin-Pioglitazone'),
    'change': ('Perubahan Resep Obat', 'Medication Change'),
    'diabetesMed': ('Diberikan Obat Diabetes', 'Diabetes Medication Prescribed'),
}

# 4. TEKS STATIS & REKOMENDASI (BILINGUAL)
teks_ui = {
    "ID": {
        "title": "# 🏥 Sistem Prediksi Diabetes: All-Features (Bilingual)",
        "algo_label": "Pilih Algoritma Klasifikasi",
        "btn_predict": "🔍 Jalankan Analisis",
        "status_high": "🚨 BERISIKO TINGGI (Rawat Ulang <30 Hari)",
        "status_low": "✅ AMAN (Risiko Rendah)",
        "result_title": "### 📋 Papan Hasil Prediksi",
        "status_text": "Status Pasien",
        "chart_title": "Tingkat Keyakinan Algoritma",
        "chart_safe": "Aman",
        "chart_risk": "Berisiko",
        "rec_title": "### 💡 Rekomendasi Medis Otomatis",
        "rec_high": "* 🛑 **Tunda Pemulangan:** Lakukan observasi klinis lanjutan.\n* 💊 **Evaluasi Terapi:** Tinjau ulang kesesuaian obat yang diberikan.\n* 📅 **Jadwal Kontrol Ketat:** Wajib jadwalkan kontrol ulang < 3 hari.",
        "rec_low": "* 🏠 **Pemulangan Normal:** Pasien dapat dipulangkan sesuai jadwal.\n* 🥗 **Edukasi Gaya Hidup:** Berikan edukasi pola makan diabetes.\n* 📅 **Jadwal Kontrol:** Jadwalkan rawat jalan rutin bulanan."
    },
    "EN": {
        "title": "# 🏥 Diabetes Prediction System: All-Features (Bilingual)",
        "algo_label": "Select Classification Algorithm",
        "btn_predict": "🔍 Run Analysis",
        "status_high": "🚨 HIGH RISK (Early Readmission <30 Days)",
        "status_low": "✅ SAFE (Low Risk)",
        "result_title": "### 📋 Prediction Results Board",
        "status_text": "Patient Status",
        "chart_title": "Algorithm Confidence Level",
        "chart_safe": "Safe",
        "chart_risk": "At Risk",
        "rec_title": "### 💡 Automated Medical Recommendations",
        "rec_high": "* 🛑 **Delay Discharge:** Conduct further clinical observation.\n* 💊 **Evaluate Therapy:** Review the suitability of medications.\n* 📅 **Strict Follow-up:** Mandatory check-up in < 3 days.",
        "rec_low": "* 🏠 **Standard Discharge:** Patient is cleared for discharge.\n* 🥗 **Lifestyle Education:** Provide diabetic diet education.\n* 📅 **Routine Follow-up:** Schedule standard monthly visits."
    }
}

def terjemahkan_label(kolom, bahasa):
    if kolom in kamus_kolom:
        return kamus_kolom[kolom][0] if bahasa == "ID" else kamus_kolom[kolom][1]
    return kolom.replace('_', ' ').title()

# 5. FUNGSI PREDIKSI DINAMIS & PEMBUATAN GRAFIK
def prediksi_dinamis(*args):
    bahasa    = args[-1]
    algoritma = args[-2]
    input_values = args[:-2]

    data_dict = {col: [val] for col, val in zip(nama_semua_kolom, input_values)}
    input_df  = pd.DataFrame(data_dict)
    input_processed = preprocessor.transform(input_df)

    teks = teks_ui[bahasa]

    # ================================================================
    # MODE BANDINGKAN KEDUANYA
    # ================================================================
    is_compare = algoritma in ["⚔️ Bandingkan Keduanya", "⚔️ Compare Both"]

    if is_compare:
        # Jalankan kedua model
        pred_dt  = dt_model.predict(input_processed)[0]
        prob_dt  = dt_model.predict_proba(input_processed)[0]
        pred_knn = knn_model.predict(input_processed)[0]
        prob_knn = knn_model.predict_proba(input_processed)[0]

        pa_dt,  pr_dt  = prob_dt[0],  prob_dt[1]  if len(prob_dt)  > 1 else (1 - prob_dt[0])
        pa_knn, pr_knn = prob_knn[0], prob_knn[1] if len(prob_knn) > 1 else (1 - prob_knn[0])

        st_dt  = teks["status_high"] if pred_dt  == 1 else teks["status_low"]
        st_knn = teks["status_high"] if pred_knn == 1 else teks["status_low"]

        def card_color(pred):
            if pred == 1: return "#ffe6e6", "#e74c3c", "#c0392b"
            return "#e8f8f5", "#2ecc71", "#27ae60"

        bg_dt,  bd_dt,  tc_dt  = card_color(pred_dt)
        bg_knn, bd_knn, tc_knn = card_color(pred_knn)

        # Vonis kesepakatan
        if pred_dt == pred_knn:
            if pred_dt == 1:
                verdict_bg, verdict_bd, verdict_icon = "#ffebee", "#c0392b", "🚨"
                verdict_text = ("KEDUA ALGORITMA SEPAKAT: PASIEN BERISIKO TINGGI" if bahasa == "ID"
                                else "BOTH ALGORITHMS AGREE: HIGH RISK PATIENT")
            else:
                verdict_bg, verdict_bd, verdict_icon = "#e8f8f5", "#27ae60", "✅"
                verdict_text = ("KEDUA ALGORITMA SEPAKAT: PASIEN AMAN" if bahasa == "ID"
                                else "BOTH ALGORITHMS AGREE: PATIENT IS SAFE")
            verdict_sub = ("Tingkat kepercayaan tinggi — kedua model memberikan hasil yang sama."
                           if bahasa == "ID" else
                           "High confidence — both models produce the same result.")
        else:
            verdict_bg, verdict_bd, verdict_icon = "#fff8e1", "#f39c12", "⚠️"
            verdict_text = ("ALGORITMA TIDAK SEPAKAT — PERLU EVALUASI LANJUT" if bahasa == "ID"
                            else "ALGORITHMS DISAGREE — FURTHER EVALUATION NEEDED")
            verdict_sub  = ("Kedua model memberi hasil berbeda. Disarankan konsultasi klinis tambahan."
                            if bahasa == "ID" else
                            "Models give conflicting results. Additional clinical consultation is advised.")

        judul_str = "📋 Papan Hasil Prediksi" if bahasa == "ID" else "📋 Prediction Results Board"
        dt_label  = "Decision Tree"
        knn_label = "K-Nearest Neighbors"
        safe_lbl  = teks['status_text']
        algo_lbl  = "Algoritma"
        safe_prob = "Prob. Aman" if bahasa == "ID" else "Safe Prob."
        risk_prob = "Prob. Berisiko" if bahasa == "ID" else "Risk Prob."

        hasil_teks = f"""
        <div style='margin-bottom:8px;'>
          <h3 style='color:#2c3e50; margin:0 0 10px 0; font-size:1.05rem;'>{judul_str}</h3>

          <!-- VERDICT BANNER -->
          <div style='background:{verdict_bg}; border:2px solid {verdict_bd}; border-radius:10px; padding:12px 16px; margin-bottom:12px; text-align:center;'>
            <span style='font-size:1.6rem;'>{verdict_icon}</span>
            <p style='color:{verdict_bd}; font-weight:700; font-size:0.95rem; margin:4px 0 2px 0;'>{verdict_text}</p>
            <p style='color:#555; font-size:0.78rem; margin:0;'>{verdict_sub}</p>
          </div>

          <!-- SIDE BY SIDE CARDS -->
          <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
            <div style='background:{bg_dt}; border-left:4px solid {bd_dt}; border-radius:8px; padding:12px;'>
              <p style='font-size:0.75rem; font-weight:700; color:{tc_dt}; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.5px;'>🌳 {dt_label}</p>
              <p style='font-size:0.85rem; font-weight:700; color:{tc_dt}; margin:0 0 4px 0;'>{st_dt}</p>
              <p style='font-size:0.75rem; color:#555; margin:0;'>{safe_prob}: <b>{pa_dt:.1%}</b></p>
              <p style='font-size:0.75rem; color:#555; margin:0;'>{risk_prob}: <b style="color:{bd_dt}">{pr_dt:.1%}</b></p>
            </div>
            <div style='background:{bg_knn}; border-left:4px solid {bd_knn}; border-radius:8px; padding:12px;'>
              <p style='font-size:0.75rem; font-weight:700; color:{tc_knn}; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.5px;'>🎯 {knn_label}</p>
              <p style='font-size:0.85rem; font-weight:700; color:{tc_knn}; margin:0 0 4px 0;'>{st_knn}</p>
              <p style='font-size:0.75rem; color:#555; margin:0;'>{safe_prob}: <b>{pa_knn:.1%}</b></p>
              <p style='font-size:0.75rem; color:#555; margin:0;'>{risk_prob}: <b style="color:{bd_knn}">{pr_knn:.1%}</b></p>
            </div>
          </div>
        </div>
        """

        # Rekomendasi mayoritas
        pred_final  = pred_dt  # DT sebagai referensi jika tidak sepakat
        rek_final   = teks["rec_high"] if pred_final == 1 else teks["rec_low"]
        teks_rekomendasi = f"{teks['rec_title']}\n{rek_final}"

        # Grafik perbandingan probabilitas
        fig, axes = plt.subplots(1, 2, figsize=(6, 1.8))
        for ax, lbl, pa, pr, tc in [
            (axes[0], f"🌳 DT",  pa_dt,  pr_dt,  bd_dt),
            (axes[1], f"🎯 KNN", pa_knn, pr_knn, bd_knn)
        ]:
            cats = [teks["chart_safe"], teks["chart_risk"]]
            vals = [pa, pr]
            cols = ['#2ecc71', '#e74c3c']
            ax.barh(cats, vals, color=cols, alpha=0.8)
            ax.set_xlim(0, 1)
            ax.set_title(lbl, fontsize=9, fontweight='bold', color=tc)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            for i, v in enumerate(vals):
                ax.text(min(v + 0.02, 0.88), i, f"{v:.1%}", va='center', fontsize=7, fontweight='bold')
        plt.suptitle(teks["chart_title"], fontsize=9, y=1.02)
        plt.tight_layout()

        state = {
            "bahasa": bahasa, "algoritma": algoritma,
            "status": f"DT: {st_dt} | KNN: {st_knn}",
            "pred": int(pred_dt), "prob_aman": float(pa_dt), "prob_risiko": float(pr_dt),
            "rekomendasi": rek_final, "input_values": list(input_values)
        }
        return hasil_teks, fig, teks_rekomendasi, state

    # ================================================================
    # MODE SATU ALGORITMA (perilaku lama)
    # ================================================================
    if algoritma == "Decision Tree":
        pred = dt_model.predict(input_processed)[0]
        prob = dt_model.predict_proba(input_processed)[0]
    else:
        pred = knn_model.predict(input_processed)[0]
        prob = knn_model.predict_proba(input_processed)[0]

    prob_aman   = prob[0]
    prob_risiko = prob[1] if len(prob) > 1 else (1 - prob_aman)

    status      = teks["status_high"] if pred == 1 else teks["status_low"]
    rekomendasi = teks["rec_high"]    if pred == 1 else teks["rec_low"]

    if pred == 1:
        bg_color, border_color, text_color = "#ffe6e6", "#e74c3c", "#c0392b"
    else:
        bg_color, border_color, text_color = "#e8f8f5", "#2ecc71", "#27ae60"

    judul_bersih = teks['result_title'].replace('### ', '')
    hasil_teks = f"""
    <div style='background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 10px 15px; border-radius: 5px; margin-bottom: 6px;'>
        <h3 style='color: {text_color}; margin: 0 0 6px 0; font-size: 1.1rem;'>{judul_bersih}</h3>
        <p style='font-size: 14px; margin: 0;'><b>{teks['status_text']}</b>: <span style='color: {text_color}; font-weight: bold;'>{status}</span></p>
        <p style='font-size: 12px; margin: 3px 0 0 0; color: #555;'><b>Algoritma</b>: {algoritma}</p>
    </div>
    """
    teks_rekomendasi = f"{teks['rec_title']}\n{rekomendasi}"

    fig, ax = plt.subplots(figsize=(5, 1.6))
    kategori   = [teks["chart_safe"], teks["chart_risk"]]
    nilai_prob = [prob_aman, prob_risiko]
    warna      = ['#2ecc71', '#e74c3c']
    ax.barh(kategori, nilai_prob, color=warna, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_title(teks["chart_title"], fontsize=10, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(nilai_prob):
        ax.text(v + 0.02, i, f"{v:.1%}", color='black', va='center', fontweight='bold')
    plt.tight_layout()

    return hasil_teks, fig, teks_rekomendasi, {
        "bahasa": bahasa, "algoritma": algoritma, "status": status,
        "pred": int(pred), "prob_aman": float(prob_aman), "prob_risiko": float(prob_risiko),
        "rekomendasi": rekomendasi, "input_values": list(input_values)
    }

# 6. FUNGSI GENERATE LAPORAN PDF
def generate_pdf(state_data):
    if not state_data:
        return None

    bahasa = state_data["bahasa"]
    algoritma = state_data["algoritma"]
    status = state_data["status"]
    pred = state_data["pred"]
    prob_aman = state_data["prob_aman"]
    prob_risiko = state_data["prob_risiko"]
    rekomendasi_raw = state_data["rekomendasi"]
    input_values = state_data["input_values"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(os.path.dirname(__file__), f"laporan_prediksi_{timestamp}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Warna berdasarkan hasil
    risk_color = colors.HexColor('#c0392b') if pred == 1 else colors.HexColor('#27ae60')
    risk_bg = colors.HexColor('#ffe6e6') if pred == 1 else colors.HexColor('#e8f8f5')

    # Style custom
    judul_style = ParagraphStyle('judul', parent=styles['Title'], fontSize=16,
                                  textColor=colors.HexColor('#2c3e50'), spaceAfter=4, alignment=TA_CENTER)
    sub_style = ParagraphStyle('sub', parent=styles['Normal'], fontSize=10,
                                textColor=colors.HexColor('#7f8c8d'), alignment=TA_CENTER, spaceAfter=12)
    section_style = ParagraphStyle('section', parent=styles['Heading2'], fontSize=12,
                                    textColor=colors.HexColor('#2980b9'), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle('body', parent=styles['Normal'], fontSize=10,
                                 textColor=colors.HexColor('#2c3e50'), leading=16)

    # -- HEADER --
    judul = "Laporan Prediksi Pasien Diabetes" if bahasa == "ID" else "Diabetes Patient Prediction Report"
    story.append(Paragraph("🏥 " + judul, judul_style))
    generated_text = f"Dibuat: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')} WIB" if bahasa == "ID" else f"Generated: {datetime.datetime.now().strftime('%B %d, %Y, %H:%M')}"
    story.append(Paragraph(generated_text, sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2980b9')))
    story.append(Spacer(1, 0.4*cm))

    # -- HASIL PREDIKSI --
    section_title = "Hasil Prediksi" if bahasa == "ID" else "Prediction Result"
    story.append(Paragraph(section_title, section_style))

    result_data = [
        [Paragraph(f"<b>{'Status Pasien' if bahasa == 'ID' else 'Patient Status'}</b>", body_style),
         Paragraph(f"<font color='{'#c0392b' if pred == 1 else '#27ae60'}'><b>{status}</b></font>", body_style)],
        [Paragraph(f"<b>Algoritma</b>", body_style), Paragraph(algoritma, body_style)],
        [Paragraph(f"<b>{'Probabilitas Aman' if bahasa == 'ID' else 'Safe Probability'}</b>", body_style),
         Paragraph(f"{prob_aman:.1%}", body_style)],
        [Paragraph(f"<b>{'Probabilitas Berisiko' if bahasa == 'ID' else 'Risk Probability'}</b>", body_style),
         Paragraph(f"<font color='#c0392b'>{prob_risiko:.1%}</font>", body_style)],
    ]
    result_table = Table(result_data, colWidths=[7*cm, 9*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), risk_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, risk_color),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [risk_bg, colors.white]),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.5*cm))

    # -- GRAFIK PROBABILITAS --
    fig_buf = io.BytesIO()
    fig_pdf, ax_pdf = plt.subplots(figsize=(5, 2))
    teks = teks_ui[bahasa]
    kategori = [teks["chart_safe"], teks["chart_risk"]]
    nilai_prob = [prob_aman, prob_risiko]
    ax_pdf.barh(kategori, nilai_prob, color=['#2ecc71', '#e74c3c'], alpha=0.85)
    ax_pdf.set_xlim(0, 1)
    ax_pdf.set_title(teks["chart_title"], fontsize=10)
    ax_pdf.spines['top'].set_visible(False)
    ax_pdf.spines['right'].set_visible(False)
    for i, v in enumerate(nilai_prob):
        ax_pdf.text(v + 0.02, i, f"{v:.1%}", va='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    fig_pdf.savefig(fig_buf, format='PNG', dpi=120, bbox_inches='tight')
    plt.close(fig_pdf)
    fig_buf.seek(0)
    story.append(RLImage(fig_buf, width=14*cm, height=5.5*cm))
    story.append(Spacer(1, 0.4*cm))

    # -- DATA INPUT PASIEN --
    input_section = "Data Input Pasien" if bahasa == "ID" else "Patient Input Data"
    story.append(Paragraph(input_section, section_style))
    input_table_data = [[Paragraph("<b>Fitur</b>", body_style), Paragraph("<b>Nilai</b>", body_style)]]
    for col, val in zip(nama_semua_kolom, input_values):
        label = terjemahkan_label(col, bahasa)
        input_table_data.append([Paragraph(label, body_style), Paragraph(str(val), body_style)])
    input_table = Table(input_table_data, colWidths=[9*cm, 7*cm], repeatRows=1)
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f8ff'), colors.white]),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#dfe6e9')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.5*cm))

    # -- REKOMENDASI MEDIS --
    rec_section = "Rekomendasi Medis" if bahasa == "ID" else "Medical Recommendations"
    story.append(Paragraph(rec_section, section_style))
    rec_lines = [l.strip().lstrip('* ').replace('**', '') for l in rekomendasi_raw.strip().split('\n') if l.strip()]
    for line in rec_lines:
        story.append(Paragraph(f"• {line}", body_style))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#bdc3c7')))
    disclaimer = "⚠️ Laporan ini hanya sebagai alat bantu klinis. Keputusan medis tetap merupakan wewenang tenaga kesehatan profesional." if bahasa == "ID" else "⚠️ This report is a clinical decision support tool only. Final medical decisions remain the responsibility of qualified healthcare professionals."
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(disclaimer, ParagraphStyle('disc', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#7f8c8d'), alignment=TA_CENTER)))

    doc.build(story)
    return pdf_path

# 6. MEMBANGUN UI & LAYOUT PAPAN ANALISIS (DENGAN GRID)
komponen_dict = {}

fitur_demografi = [col for col in nama_semua_kolom if 'gl' not in col and 'ide' not in col and col not in ['metformin', 'insulin', 'change', 'diabetesMed']]
fitur_obat = [col for col in nama_semua_kolom if col not in fitur_demografi]

# 7. CUSTOM CSS PREMIUM
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== GLOBAL ===== */
body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #f0f4f8 0%, #e8edf5 100%) !important;
}

/* ===== HEADER BANNER ===== */
#app-header {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 40%, #0277bd 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 8px;
    box-shadow: 0 8px 32px rgba(21, 101, 192, 0.35);
    position: relative;
    overflow: hidden;
}
#app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
#app-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 280px; height: 160px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
#app-header h1 {
    color: white !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.3px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
#app-header p {
    color: rgba(255,255,255,0.82) !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
    font-weight: 400;
}

/* ===== LANGUAGE SELECTOR ===== */
#lang-selector {
    background: white;
    border-radius: 12px;
    padding: 8px 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 4px !important;
}

/* ===== INPUT CARDS ===== */
.gradio-tabs {
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    overflow: hidden !important;
    background: white !important;
}
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 12px 20px !important;
    color: #5c6bc0 !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.25s ease !important;
}
.tab-nav button.selected {
    color: #1565c0 !important;
    border-bottom: 3px solid #1565c0 !important;
    background: #f0f7ff !important;
}
.tab-nav button:hover:not(.selected) {
    background: #f8f9ff !important;
    color: #1565c0 !important;
}

/* ===== INPUTS ===== */
.gradio-container input, .gradio-container select {
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    border: 1.5px solid #e0e7ff !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.gradio-container input:focus, .gradio-container select:focus {
    border-color: #1565c0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.12) !important;
}
label span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #37474f !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ===== RIGHT PANEL (Analysis Card) ===== */
#analysis-panel {
    background: white;
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.09);
    padding: 20px;
    min-height: 200px;
}

/* ===== ALGO SELECTOR ===== */
#algo-group {
    background: white;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    padding: 18px 20px;
    margin-bottom: 12px;
}

/* ===== CTA BUTTON ===== */
#btn-predict {
    background: linear-gradient(135deg, #1565c0, #0277bd) !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    padding: 16px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(21, 101, 192, 0.4) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    margin-top: 10px !important;
}
#btn-predict:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(21, 101, 192, 0.5) !important;
    background: linear-gradient(135deg, #1976d2, #0288d1) !important;
}
#btn-predict:active {
    transform: translateY(0px) !important;
}

/* ===== DOWNLOAD BUTTON ===== */
#btn-download {
    background: linear-gradient(135deg, #2e7d32, #388e3c) !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px !important;
    box-shadow: 0 4px 14px rgba(46,125,50,0.35) !important;
    transition: all 0.25s ease !important;
    margin-top: 8px !important;
}
#btn-download:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(46,125,50,0.45) !important;
}

/* ===== RADIO BUTTONS ===== */
.gradio-radio label {
    border-radius: 8px !important;
    padding: 8px 14px !important;
    border: 1.5px solid #e0e7ff !important;
    transition: all 0.2s !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
.gradio-radio label:has(input:checked) {
    background: #e3f2fd !important;
    border-color: #1565c0 !important;
    color: #1565c0 !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #90a4ae; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #607d8b; }

/* ===== PLOT AREA ===== */
.gradio-plot {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as aplikasi:

    with gr.Row(elem_id="lang-selector"):
        lang_input = gr.Radio(choices=["ID", "EN"], value="ID", label="🌐 Bahasa / Language")

    ui_title = gr.HTML("""
    <div id="app-header">
        <h1>🏥 Sistem Prediksi Rawat Ulang Pasien Diabetes</h1>
        <p>Masukkan data rekam medis pasien, lalu pilih algoritma dan klik Jalankan Analisis</p>
    </div>
    """)

    with gr.Row():
        # --- KOLOM KIRI (INPUT DATA DENGAN GRID) ---
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.TabItem("📊 Demografi & Riwayat / Demographics"):
                    setengah_demo = len(fitur_demografi) // 2
                    with gr.Row():
                        with gr.Column():
                            for col in fitur_demografi[:setengah_demo]:
                                label_awal = terjemahkan_label(col, "ID")
                                if pd.api.types.is_object_dtype(df_ref[col]) or pd.api.types.is_string_dtype(df_ref[col]):
                                    opsi = df_ref[col].dropna().unique().tolist()
                                    komponen_dict[col] = gr.Dropdown(choices=opsi, label=label_awal, value=opsi[0] if opsi else None)
                                else:
                                    komponen_dict[col] = gr.Number(label=label_awal, value=0, minimum=0)

                        with gr.Column():
                            for col in fitur_demografi[setengah_demo:]:
                                label_awal = terjemahkan_label(col, "ID")
                                if pd.api.types.is_object_dtype(df_ref[col]) or pd.api.types.is_string_dtype(df_ref[col]):
                                    opsi = df_ref[col].dropna().unique().tolist()
                                    komponen_dict[col] = gr.Dropdown(choices=opsi, label=label_awal, value=opsi[0] if opsi else None)
                                else:
                                    komponen_dict[col] = gr.Number(label=label_awal, value=0, minimum=0)

                with gr.TabItem("💊 Data Obat / Medications"):
                    setengah_obat = len(fitur_obat) // 2
                    with gr.Row():
                        with gr.Column():
                            for col in fitur_obat[:setengah_obat]:
                                label_awal = terjemahkan_label(col, "ID")
                                if pd.api.types.is_object_dtype(df_ref[col]) or pd.api.types.is_string_dtype(df_ref[col]):
                                    opsi = ['No', 'Steady', 'Up', 'Down'] if col not in ['change', 'diabetesMed'] else ['No', 'Ch', 'Yes']
                                    komponen_dict[col] = gr.Dropdown(choices=opsi, label=label_awal, value='No')
                                else:
                                    komponen_dict[col] = gr.Number(label=label_awal, value=0, minimum=0)

                        with gr.Column():
                            for col in fitur_obat[setengah_obat:]:
                                label_awal = terjemahkan_label(col, "ID")
                                if pd.api.types.is_object_dtype(df_ref[col]) or pd.api.types.is_string_dtype(df_ref[col]):
                                    opsi = ['No', 'Steady', 'Up', 'Down'] if col not in ['change', 'diabetesMed'] else ['No', 'Ch', 'Yes']
                                    komponen_dict[col] = gr.Dropdown(choices=opsi, label=label_awal, value='No')
                                else:
                                    komponen_dict[col] = gr.Number(label=label_awal, value=0, minimum=0)

        # --- KOLOM KANAN (KONTROL PREDIKSI & PAPAN ANALISIS) ---
        with gr.Column(scale=4):
            # 1. Kontrol Algoritma dan Tombol Analisis
            with gr.Group(elem_id="algo-group"):
                algo_input = gr.Radio(
                    ["Decision Tree", "K-Nearest Neighbors (KNN)", "⚔️ Bandingkan Keduanya"],
                    label=teks_ui["ID"]["algo_label"],
                    value="Decision Tree"
                )
                btn_predict = gr.Button(teks_ui["ID"]["btn_predict"], variant="primary", elem_id="btn-predict")

            # 2. Papan Analisis
            with gr.Group(elem_id="analysis-panel"):
                placeholder_html = gr.HTML("""
                <div style="text-align:center; padding: 40px 20px; color: #90a4ae;">
                    <div style="font-size: 3rem; margin-bottom: 12px;">🔬</div>
                    <p style="font-size: 1rem; font-weight: 600; color: #78909c; margin:0;">Hasil Analisis Akan Muncul Di Sini</p>
                    <p style="font-size: 0.82rem; color: #b0bec5; margin-top:6px;">Analysis Results Will Appear Here</p>
                </div>
                """)
                out_result = gr.HTML()
                out_plot = gr.Plot(label="Grafik Analisis", show_label=False)
                out_rekomendasi = gr.Markdown()
                btn_download = gr.Button("📥 Unduh Laporan PDF", variant="secondary", visible=False, elem_id="btn-download")
                out_pdf = gr.File(label="File PDF Siap Diunduh", visible=False)

    # Menyusun kembali urutan input agar sesuai dengan dataset
    komponen_input = [komponen_dict[col] for col in nama_semua_kolom]
    semua_input_prediksi = komponen_input + [algo_input, lang_input]

    def ubah_bahasa(bahasa):
        teks = teks_ui[bahasa]
        
        # HTML untuk Judul Utama
        if bahasa == "ID":
            html_judul = """
            <div id="app-header">
                <h1>🏥 Sistem Prediksi Rawat Ulang Pasien Diabetes</h1>
                <p>Masukkan data rekam medis pasien, lalu pilih algoritma dan klik Jalankan Analisis</p>
            </div>
            """
        else:
            html_judul = """
            <div id="app-header">
                <h1>🏥 Diabetes Patient Readmission Prediction System</h1>
                <p>Enter patient medical data, select an algorithm, then click Run Analysis</p>
            </div>
            """
            
        updates = [
            gr.update(value=html_judul),  # ui_title
            gr.update(value=teks["btn_predict"]),
            gr.update(label=teks["algo_label"])
        ]
        for col in nama_semua_kolom:
            updates.append(gr.update(label=terjemahkan_label(col, bahasa)))
        return updates

    semua_komponen_ui = [ui_title, btn_predict, algo_input] + komponen_input

    # State untuk menyimpan data prediksi terakhir
    state_prediksi = gr.State()

    lang_input.change(fn=ubah_bahasa, inputs=lang_input, outputs=semua_komponen_ui)
    btn_predict.click(
        fn=prediksi_dinamis,
        inputs=semua_input_prediksi,
        outputs=[out_result, out_plot, out_rekomendasi, state_prediksi]
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=None)),
        outputs=[placeholder_html, btn_download, out_pdf]
    )
    btn_download.click(
        fn=generate_pdf,
        inputs=[state_prediksi],
        outputs=[out_pdf]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[out_pdf]
    )

    # ============================================================
    # DASHBOARD PERFORMA MODEL
    # ============================================================
    with gr.Accordion("📊 Dashboard Performa Model / Model Performance Dashboard", open=False, elem_id="dashboard-accordion"):
        def build_dashboard_html():
            def bar(value, color):
                return f"<div style='background:#eef2f7;border-radius:6px;overflow:hidden;height:10px;margin-top:4px;'><div style='width:{value}%;background:{color};height:100%;border-radius:6px;transition:width 0.6s;'></div></div>"

            def metric_row(label, val_dt, val_knn, color):
                better = "dt" if val_dt >= val_knn else "knn"
                star_dt  = " ⭐" if better == "dt"  else ""
                star_knn = " ⭐" if better == "knn" else ""
                return f"""
                <tr>
                  <td style='padding:8px 12px;font-size:0.82rem;font-weight:600;color:#546e7a;'>{label}</td>
                  <td style='padding:8px 12px;text-align:center;'>
                    <span style='font-size:1rem;font-weight:700;color:{color};'>{val_dt:.1f}%</span>{star_dt}
                    {bar(val_dt, color)}
                  </td>
                  <td style='padding:8px 12px;text-align:center;'>
                    <span style='font-size:1rem;font-weight:700;color:{color};'>{val_knn:.1f}%</span>{star_knn}
                    {bar(val_knn, color)}
                  </td>
                </tr>"""

            html = f"""
            <div style='font-family:Inter,sans-serif; padding:4px;'>
              <p style='font-size:0.78rem;color:#90a4ae;margin:0 0 14px 0;'>
                ℹ️ Dihitung dari <b>sampel 2.000 data uji</b> (test set 20%, random_state=42).
                Hasil KNN diambil dari sampel untuk menjaga kecepatan aplikasi.
              </p>
              <table style='width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.07);'>
                <thead>
                  <tr style='background:linear-gradient(135deg,#1a237e,#1565c0);'>
                    <th style='padding:12px 16px;text-align:left;color:white;font-size:0.85rem;'>Metrik / Metric</th>
                    <th style='padding:12px 16px;text-align:center;color:white;font-size:0.85rem;'>🌳 Decision Tree</th>
                    <th style='padding:12px 16px;text-align:center;color:white;font-size:0.85rem;'>🎯 K-Nearest Neighbors</th>
                  </tr>
                </thead>
                <tbody>
                  {metric_row('Akurasi / Accuracy', metrik_dt['accuracy'], metrik_knn['accuracy'], '#1565c0')}
                  {metric_row('Presisi / Precision', metrik_dt['precision'], metrik_knn['precision'], '#6a1b9a')}
                  {metric_row('Recall / Sensitivity', metrik_dt['recall'], metrik_knn['recall'], '#00838f')}
                  {metric_row('F1-Score', metrik_dt['f1'], metrik_knn['f1'], '#e65100')}
                </tbody>
              </table>

              <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px;'>
                {''.join([
                    f"<div style='background:white;border-radius:10px;padding:12px 16px;box-shadow:0 2px 10px rgba(0,0,0,0.06);'>"
                    f"<p style='font-weight:700;color:#37474f;font-size:0.85rem;margin:0 0 8px 0;'>{'Confusion Matrix'} &mdash; {lbl}</p>"
                    f"<table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>"
                    f"<tr><td style='padding:4px;background:#e8f5e9;text-align:center;border-radius:4px 0 0 0;'><b style='color:#2e7d32;'>TP: {cm[1][1]}</b></td>"
                    f"<td style='padding:4px;background:#ffebee;text-align:center;border-radius:0 4px 0 0;'><b style='color:#c62828;'>FP: {cm[0][1]}</b></td></tr>"
                    f"<tr><td style='padding:4px;background:#ffebee;text-align:center;border-radius:0 0 0 4px;'><b style='color:#c62828;'>FN: {cm[1][0]}</b></td>"
                    f"<td style='padding:4px;background:#e8f5e9;text-align:center;border-radius:0 0 4px 0;'><b style='color:#2e7d32;'>TN: {cm[0][0]}</b></td></tr>"
                    f"</table></div>"
                    for lbl, cm in [('🌳 Decision Tree', metrik_dt['cm']), ('🎯 KNN', metrik_knn['cm'])]
                ])}
              </div>

              <p style='font-size:0.72rem;color:#b0bec5;margin-top:10px;text-align:center;'>
                TP=True Positive &nbsp;|&nbsp; TN=True Negative &nbsp;|&nbsp; FP=False Positive &nbsp;|&nbsp; FN=False Negative
              </p>
            </div>
            """
            return html

        dashboard_html = gr.HTML(value=build_dashboard_html())

aplikasi.launch(share=True)