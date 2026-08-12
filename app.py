import streamlit as st
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

# ============================================================
# UI STREAMLIT
# ============================================================

st.set_page_config(page_title="Sistem Prediksi Diabetes", page_icon="🏥", layout="wide")

st.markdown('''
<style>
    div[data-testid="stMarkdownContainer"] h1 {
        background: linear-gradient(135deg, #1a237e 0%, #1565c0 40%, #0277bd 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
</style>
''', unsafe_allow_html=True)

if 'bahasa' not in st.session_state:
    st.session_state['bahasa'] = 'ID'
if 'state_prediksi' not in st.session_state:
    st.session_state['state_prediksi'] = None

col_lang_1, col_lang_2 = st.columns([8, 2])
with col_lang_2:
    st.session_state['bahasa'] = st.radio("🌐 Bahasa / Language", ["ID", "EN"], index=0 if st.session_state['bahasa'] == 'ID' else 1, horizontal=True)

bahasa = st.session_state['bahasa']
teks = teks_ui[bahasa]

if bahasa == 'ID':
    st.markdown("# 🏥 Sistem Prediksi Rawat Ulang Pasien Diabetes\nMasukkan data rekam medis pasien, lalu pilih algoritma dan klik Jalankan Analisis")
else:
    st.markdown("# 🏥 Diabetes Patient Readmission Prediction System\nEnter patient medical data, select an algorithm, then click Run Analysis")

fitur_demografi = [col for col in nama_semua_kolom if 'gl' not in col and 'ide' not in col and col not in ['metformin', 'insulin', 'change', 'diabetesMed']]
fitur_obat = [col for col in nama_semua_kolom if col not in fitur_demografi]

input_dict = {}
col_main_left, col_main_right = st.columns([6, 4])

with col_main_left:
    tab1, tab2 = st.tabs(["📊 Demografi & Riwayat" if bahasa == 'ID' else "📊 Demographics & History", "💊 Data Obat" if bahasa == 'ID' else "💊 Medications"])
    
    with tab1:
        col_demo_1, col_demo_2 = st.columns(2)
        setengah_demo = len(fitur_demografi) // 2
        for i, col in enumerate(fitur_demografi):
            target_col = col_demo_1 if i < setengah_demo else col_demo_2
            with target_col:
                label_awal = terjemahkan_label(col, bahasa)
                if pd.api.types.is_object_dtype(df_ref[col]) or pd.api.types.is_string_dtype(df_ref[col]):
                    opsi = df_ref[col].dropna().unique().tolist()
                    input_dict[col] = st.selectbox(label_awal, opsi, key=f"input_{col}")
                else:
                    input_dict[col] = st.number_input(label_awal, value=0, min_value=0, key=f"input_{col}")

    with tab2:
        col_obat_1, col_obat_2 = st.columns(2)
        setengah_obat = len(fitur_obat) // 2
        for i, col in enumerate(fitur_obat):
            target_col = col_obat_1 if i < setengah_obat else col_obat_2
            with target_col:
                label_awal = terjemahkan_label(col, bahasa)
                if pd.api.types.is_object_dtype(df_ref[col]) or pd.api.types.is_string_dtype(df_ref[col]):
                    opsi = ['No', 'Steady', 'Up', 'Down'] if col not in ['change', 'diabetesMed'] else ['No', 'Ch', 'Yes']
                    input_dict[col] = st.selectbox(label_awal, opsi, key=f"input_{col}")
                else:
                    input_dict[col] = st.number_input(label_awal, value=0, min_value=0, key=f"input_{col}")

with col_main_right:
    st.markdown("### 🎛️ " + teks['algo_label'])
    algoritma = st.radio("", ["Decision Tree", "K-Nearest Neighbors (KNN)", "⚔️ Bandingkan Keduanya" if bahasa == 'ID' else "⚔️ Compare Both"])
    
    if st.button(teks['btn_predict'], use_container_width=True, type='primary'):
        args_input = [input_dict[col] for col in nama_semua_kolom]
        args_input.append(algoritma)
        args_input.append(bahasa)
        
        with st.spinner("Memproses analisis..." if bahasa == 'ID' else "Processing analysis..."):
            hasil_teks, fig, teks_rekomendasi, state = prediksi_dinamis(*args_input)
            st.session_state['state_prediksi'] = state
            
    st.markdown("---")
    
    if st.session_state['state_prediksi'] is not None:
        state = st.session_state['state_prediksi']
        args_input = [state['input_values'][i] for i in range(len(nama_semua_kolom))]
        args_input.append(state['algoritma'])
        args_input.append(bahasa)
        hasil_teks, fig, teks_rekomendasi, state_terbaru = prediksi_dinamis(*args_input)
        
        st.markdown(hasil_teks, unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown(teks_rekomendasi)
        
        pdf_path = generate_pdf(state_terbaru)
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as pdf_file:
                st.download_button(
                    label="📥 Unduh Laporan PDF" if bahasa == 'ID' else "📥 Download PDF Report",
                    data=pdf_file,
                    file_name="Laporan_Prediksi_Diabetes.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    else:
        st.info("🔬 Hasil Analisis Akan Muncul Di Sini" if bahasa == 'ID' else "🔬 Analysis Results Will Appear Here")
