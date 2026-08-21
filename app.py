"""
app.py — Giao diện phần mềm TDM Aminoglycosid (Streamlit)
Đã chuyển toàn bộ dữ liệu bệnh nhân lên Supabase Cloud, tích hợp phiên bản và bản quyền.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import io

import database as db
from pk_calculations import (
    PatientInfo, InitialDoseInput, MeasuredLevels, DoseAdjustment,
    compute_bmi, compute_ibw, compute_dosing_weight, compute_crcl,
    compute_ke_population, compute_t_half, compute_vd_population,
    compute_total_dose, compute_suggested_tau,
    compute_predicted_cp_population, compute_predicted_ctrough_population,
    compute_ke_individual, compute_t_half_individual, compute_true_peak,
    compute_true_trough, compute_vd_individual, compute_predicted_cp_adjusted,
    compute_predicted_ctrough_adjusted, simulate_dosing_curve,
)

# =============================================================================
# CẤU HÌNH TRANG & SESSION STATE ĐĂNG NHẬP
# =============================================================================
st.set_page_config(page_title="TDM Aminoglycosid", layout="wide")
db.init_db()

# Khởi tạo trạng thái đăng nhập
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "fullname" not in st.session_state:
    st.session_state.fullname = ""

# =============================================================================
# MÀN HÌNH ĐĂNG NHẬP (NẾU CHƯA ĐĂNG NHẬP)
# =============================================================================
if not st.session_state.logged_in:
    st.title("🔐 Đăng nhập hệ thống TDM Aminoglycosid")
    st.write("Vui lòng đăng nhập bằng tài khoản được cấp trên hệ thống Cloud.")
    
    col_login1, col_login2 = st.columns([1, 2])
    with col_login1:
        with st.form("login_form"):
            input_user = st.text_input("Tên đăng nhập")
            input_pass = st.text_input("Mật khẩu", type="password")
            submit_btn = st.form_submit_button("Đăng nhập")
            
            if submit_btn:
                user_data = db.check_login(input_user, input_pass)
                if user_data:
                    fullname, role = user_data
                    st.session_state.logged_in = True
                    st.session_state.username = input_user
                    st.session_state.fullname = fullname
                    st.session_state.role = role
                    st.success(f"Đăng nhập thành công! Xin chào bác sĩ/dược sĩ {fullname}.")
                    st.rerun()
                else:
                    st.error("Tên đăng nhập, mật khẩu không đúng hoặc chưa kết nối được Cloud!")
    st.stop()

# =============================================================================
# GIAO DIỆN CHÍNH (SAU KHI ĐÃ ĐĂNG NHẬP)
# =============================================================================

# Sidebar hiển thị thông tin tài khoản, bản quyền và phiên bản
with st.sidebar:
    st.markdown("### 👤 Thông tin tài khoản")
    st.write(f"**Họ tên:** {st.session_state.fullname}")
    st.write(f"**Tài khoản:** `{st.session_state.username}`")
    st.write(f"**Vai trò:** `{st.session_state.role}`")
    
    if st.session_state.role == "admin":
        st.info("⚙️ Quyền Quản trị viên (Admin).")

    if st.button("🚪 Đăng xuất", type="primary"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.fullname = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Thông tin phần mềm")
    st.markdown("**Phần mềm TDM Aminoglycosid**")
    st.markdown("Phiên bản: `V.21.8.2026`")
    st.markdown("Bản quyền sở hữu:")
    st.markdown("📧 `thienphankh28@gmail.com`")

# Session State Initialization cho phần tính toán
if "loaded_patient" not in st.session_state:
    st.session_state.loaded_patient = None
if "loaded_tdm" not in st.session_state:
    st.session_state.loaded_tdm = None
if "sec3_calcs" not in st.session_state:
    st.session_state.sec3_calcs = {"ke": 0.0, "thalf": 0.0, "vd": 0.0, "cp": 0.0, "ctr": 0.0, "calculated": False}
if "sec4_calcs" not in st.session_state:
    st.session_state.sec4_calcs = {"cp_pred": 0.0, "ctr_pred": 0.0}

st.title("💊 Phần mềm TDM Aminoglycosid")

tab1, tab2 = st.tabs(["🧮 Tính toán & TDM", "🗂 CSDL Bệnh nhân (Cloud)"])

with tab1:
    st.subheader("🔍 Truy xuất bệnh nhân từ Cloud")
    lookup_msyt = st.text_input("Nhập MSYT để tự động điền dữ liệu", key="lookup")
    if st.button("Tải dữ liệu bệnh nhân"):
        p_data, t_data = db.get_latest_tdm(lookup_msyt)
        if p_data:
            st.session_state.loaded_patient = p_data
            st.session_state.loaded_tdm = t_data
            st.success(f"Đã tải thành công bệnh nhân: {lookup_msyt}")
        else:
            st.error("Không tìm thấy MSYT trong Cloud CSDL.")
            st.session_state.loaded_patient = None
            st.session_state.loaded_tdm = None

    st.divider()

    p_def = st.session_state.loaded_patient or {}
    t_def = st.session_state.loaded_tdm or {}
    
    # Ép kiểu rõ ràng sang float để tránh lỗi lệch kiểu dữ liệu với min_value
    default_weight = float(p_def.get("weight", 81.0))
    default_height = float(p_def.get("height", 180.34))
    default_age = float(p_def.get("age", 67.0))
    
    default_dose_mg_kg = 5.3
    if t_def.get("new_dose") and default_weight > 0:
        default_dose_mg_kg = float(round(float(t_def.get("new_dose")) / default_weight, 2))

    st.header("1. Thông tin bệnh nhân")
    c1, c2, c3 = st.columns(3)
    with c1:
        msyt_input = st.text_input("MSYT (Bắt buộc để lưu)", value=p_def.get("msyt", lookup_msyt))
        gender_index = 0 if p_def.get("gender", "nam") == "nam" else 1
        gender = st.selectbox("Giới tính", ["nam", "nữ"], index=gender_index)
        weight_kg = st.number_input("Cân nặng (kg)", value=default_weight, min_value=0.1)
        height_cm = st.number_input("Chiều cao (cm)", value=default_height, min_value=1.0)
    with c2:
        age = st.number_input("Tuổi", value=default_age, min_value=0.0)
        is_cf = st.checkbox("Đối tượng: Xơ nang (Cystic Fibrosis)", value=bool(p_def.get("is_cf", True)))
    with c3:
        dose_mg_per_kg = st.number_input("Liều AG (mg/kg)", value=default_dose_mg_kg)
        infusion_time_h = st.number_input("Thời gian truyền ban đầu (h, t')", value=t_def.get("new_t_inf", 1.0), min_value=0.1)
        tau_h = st.number_input("Khoảng đưa liều hiện tại (h)", value=t_def.get("new_tau", 24.0), min_value=1.0)

    c4, c5 = st.columns(2)
    with c4:
        target_cp = st.number_input("Cp kỳ vọng (μg/mL)", value=t_def.get("pred_cp", 20.0))
    with c5:
        target_ctrough = st.number_input("Ctr kỳ vọng (μg/mL)", value=t_def.get("pred_ctrough", 1.0))

    patient = PatientInfo(gender, height_cm, weight_kg, 100.0, age, is_cf)
    bmi = compute_bmi(patient)
    ibw = compute_ibw(patient)
    dosing_weight = compute_dosing_weight(patient, ibw)
    
    st.header("2. Thông số quần thể tham khảo")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("BMI", f"{bmi:.2f}")
    r1.metric("IBW (kg)", f"{ibw:.2f}")
    r2.metric("Cân nặng tính liều (kg)", f"{dosing_weight:.2f}")
    total_dose = compute_total_dose(dose_mg_per_kg, dosing_weight)
    r3.metric("Tổng liều lý thuyết (mg)", f"{total_dose:.1f}")

    st.divider()

    st.header("3. Cá thể hoá theo kết quả TDM")
    t1, t2_col = st.columns(2)
    with t1:
        tdm_date = st.date_input("Ngày thực hiện TDM", value=datetime.date.today())
    with t2_col:
        current_scr = st.number_input("Scr ngày TDM (μmol/L)", value=t_def.get("scr", 80.0), min_value=0.1)
    
    is_first_dose = st.checkbox("Liều ĐẦU TIÊN (chưa tích luỹ)", value=False)
    dose_given_mg = st.number_input("Tổng liều đã dùng khi lấy mẫu (mg)", value=float(round(total_dose, 2)))
    tau_at_sampling_h = st.number_input("Khoảng đưa liều (τ) lúc lấy mẫu (h)", value=float(tau_h))

    m1, m2, m3, m4 = st.columns(4)
    with m1: t1_h = st.number_input("T1 (h, từ lúc truyền)", value=float(t_def.get("t1", 3.0)))
    with m2: c1_val = st.number_input("C1 (μg/mL)", value=float(t_def.get("c1", 17.5)))
    with m3: t2_h = st.number_input("T2 (h, từ lúc truyền)", value=float(t_def.get("t2", 17.0)))
    with m4: c2_val = st.number_input("C2 (μg/mL)", value=float(t_def.get("c2", 4.8)))

    if st.button("🧮 TÍNH TOÁN CÁ THỂ HÓA"):
        if t2_h <= t1_h or c1_val <= 0 or c2_val <= 0 or c1_val <= c2_val:
            st.error("⚠️ Dữ liệu T/C không hợp lệ.")
            st.session_state.sec3_calcs["calculated"] = False
        else:
            measured = MeasuredLevels(is_first_dose, t1_h, c1_val, t2_h, c2_val, infusion_time_h, tau_at_sampling_h, dose_given_mg)
            ke_ind = compute_ke_individual(measured)
            t_half_ind = compute_t_half_individual(ke_ind)
            true_peak = compute_true_peak(measured, ke_ind)
            true_trough = compute_true_trough(true_peak, ke_ind, infusion_time_h, tau_at_sampling_h)
            vd_ind = compute_vd_individual(dose_given_mg, ke_ind, true_peak, infusion_time_h, tau_at_sampling_h, is_first_dose)
            
            st.session_state.sec3_calcs = {
                "ke": ke_ind, "thalf": t_half_ind, "vd": vd_ind, 
                "cp": true_peak, "ctr": true_trough, "calculated": True
            }

    c_res = st.session_state.sec3_calcs
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("Ke cá thể (h⁻¹)", f"{c_res['ke']:.4f}")
    i2.metric("T1/2 cá thể (h)", f"{c_res['thalf']:.2f}")
    i3.metric("Vd cá thể (L)", f"{c_res['vd']:.2f}")
    i4.metric("Cp thật (μg/mL)", f"{c_res['cp']:.2f}")
    i5.metric("Ctrough thật (μg/mL)", f"{c_res['ctr']:.3f}")

    if st.button("💾 Xác nhận & Lưu lên Cloud (Mục 3)"):
        if not msyt_input:
            st.error("Vui lòng nhập MSYT ở Mục 1.")
        elif not c_res["calculated"]:
            st.warning("Vui lòng nhấn nút 'Tính toán' trước khi lưu.")
        else:
            date_str = tdm_date.strftime("%Y-%m-%d")
            db.save_patient_info(msyt_input, gender, weight_kg, height_cm, age, int(is_cf))

            if db.check_tdm_exists(msyt_input, date_str):
                st.session_state.confirm_overwrite_3 = True
            else:
                db.save_sec3_data(msyt_input, date_str, current_scr, t1_h, c1_val, t2_h, c2_val, c_res['ke'], c_res['thalf'], c_res['vd'], c_res['cp'], c_res['ctr'])
                st.success(f"Đã lưu block TDM lên Cloud cho ngày {date_str}.")
    
    if st.session_state.get('confirm_overwrite_3', False):
        st.warning(f"⚠️ Kết quả TDM ngày {tdm_date.strftime('%d/%m/%Y')} đã tồn tại trên Cloud. Ghi đè?")
        if st.button("Đồng ý ghi đè (Mục 3)"):
            db.save_sec3_data(msyt_input, tdm_date.strftime("%Y-%m-%d"), current_scr, t1_h, c1_val, t2_h, c2_val, c_res['ke'], c_res['thalf'], c_res['vd'], c_res['cp'], c_res['ctr'])
            st.success("Đã ghi đè thành công.")
            st.session_state.confirm_overwrite_3 = False

    st.divider()

    st.header("4. Hiệu chỉnh liều theo TDM")
    a1, a2, a3 = st.columns(3)
    with a1: new_dose_mg = st.number_input("Liều mới - Dose (mg)", value=400.0)
    with a2: new_tau_h = st.number_input("Khoảng đưa liều mới - τ (h)", value=36.0)
    with a3: new_t_inf_h = st.number_input("Thời gian truyền mới - t' (h)", value=float(infusion_time_h), min_value=0.1)

    if st.button("🧮 TÍNH TOÁN LIỀU MỚI"):
        if not c_res["calculated"]:
            st.error("Bạn phải thực hiện Tính toán Mục 3 trước!")
        else:
            cp_new = compute_predicted_cp_adjusted(new_dose_mg, c_res['ke'], c_res['vd'], new_t_inf_h, new_tau_h)
            ctrough_new = compute_predicted_ctrough_adjusted(cp_new, c_res['ke'], new_t_inf_h, new_tau_h)
            st.session_state.sec4_calcs = {"cp_pred": cp_new, "ctr_pred": ctrough_new}

    s4 = st.session_state.sec4_calcs
    n1, n2 = st.columns(2)
    n1.metric("C'p dự đoán SS (μg/mL)", f"{s4['cp_pred']:.2f}")
    n2.metric("C'tr dự đoán SS (μg/mL)", f"{s4['ctr_pred']:.3f}")

    if st.button("💾 Xác nhận & Lưu lên Cloud (Mục 4)"):
        if not msyt_input:
            st.error("Vui lòng nhập MSYT.")
        elif s4['cp_pred'] == 0:
            st.warning("Vui lòng nhấn nút 'Tính toán liều mới' trước.")
        else:
            date_str = tdm_date.strftime("%Y-%m-%d")
            existing_block = db.get_specific_tdm_block(msyt_input, date_str)

            if existing_block and existing_block.get('new_dose') is not None:
                st.session_state.confirm_overwrite_4 = True
            else:
                db.save_sec4_data(msyt_input, date_str, new_dose_mg, new_tau_h, new_t_inf_h, s4['cp_pred'], s4['ctr_pred'])
                st.success(f"Đã lưu phác đồ mới lên Cloud block ngày {date_str}.")

    if st.session_state.get('confirm_overwrite_4', False):
        st.warning(f"⚠️ Block ngày {tdm_date.strftime('%d/%m/%Y')} đã có phác đồ. Ghi đè?")
        if st.button("Đồng ý ghi đè (Mục 4)"):
            db.save_sec4_data(msyt_input, tdm_date.strftime("%Y-%m-%d"), new_dose_mg, new_tau_h, new_t_inf_h, s4['cp_pred'], s4['ctr_pred'])
            st.success("Đã ghi đè thành công.")
            st.session_state.confirm_overwrite_4 = False

    st.divider()

    st.header("5. Đồ thị nồng độ qua 10 chu kỳ liều")
    num_cycles = st.slider("Số chu kỳ", 2, 20, 10)
    if c_res["calculated"] and s4["cp_pred"] > 0:
        sim_times, sim_concs = simulate_dosing_curve(
            ke=c_res['ke'], vd=c_res['vd'], t_inf_old=infusion_time_h, tau_old=tau_at_sampling_h,
            peak_1=c_res['cp'], trough_1=c_res['ctr'], dose_new=new_dose_mg, tau_new=new_tau_h,
            t_inf_new=new_t_inf_h, num_cycles=num_cycles
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_times, y=sim_concs, mode="lines", name="Nồng độ dự đoán C(t)", line=dict(color="#1f77b4", width=2)))
        fig.add_hline(y=target_cp, line_dash="dash", line_color="green", annotation_text=f"Cp kỳ vọng ({target_cp})", annotation_position="top left")
        fig.add_hline(y=target_ctrough, line_dash="dash", line_color="orange", annotation_text=f"Ctr kỳ vọng ({target_ctrough})", annotation_position="bottom left")
        fig.update_layout(xaxis_title="Thời gian (giờ)", yaxis_title="Nồng độ (μg/mL)", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Biểu đồ sẽ xuất hiện sau khi bạn hoàn thành Mục 3 và Mục 4.")

# =============================================================================
# TAB 2 — QUẢN LÝ BỆNH NHÂN TRÊN CLOUD
# =============================================================================
with tab2:
    st.header("Tra cứu & Quản lý CSDL Bệnh nhân trên Cloud")

    lookup_tab2 = st.text_input("Nhập MSYT để tra cứu thông tin và quản lý lịch sử TDM", key="tab2_lookup")
    
    if lookup_tab2:
        df_patient = db.get_patient_by_msyt(lookup_tab2)
        df_history = db.get_history_by_msyt(lookup_tab2)

        if not df_patient.empty:
            st.subheader("Thông tin hành chính bệnh nhân")
            st.dataframe(df_patient)
            
            st.subheader("Lịch sử các Block TDM")
            if not df_history.empty:
                st.dataframe(df_history)
                
                with st.expander("🗑️ Tùy chọn xóa dữ liệu trên Cloud"):
                    dates_list = df_history['tdm_date'].tolist()
                    selected_date_to_delete = st.selectbox("Chọn ngày TDM cần xóa", dates_list, key="select_date_del")
                    if st.button("Xóa Block TDM ngày này", type="secondary"):
                        db.delete_tdm_block(lookup_tab2, selected_date_to_delete)
                        st.success(f"Đã xóa thành công block TDM ngày {selected_date_to_delete}!")
                        st.rerun()

                    st.markdown("---")
                    st.warning("⚠️ Thao tác này sẽ xóa vĩnh viễn thông tin và toàn bộ lịch sử TDM của bệnh nhân này trên Cloud!")
                    if st.button("Xóa toàn bộ Bệnh nhân này", type="primary"):
                        db.delete_patient(lookup_tab2)
                        st.success(f"Đã xóa bệnh nhân {lookup_tab2} khỏi Cloud!")
                        st.rerun()
            else:
                st.info("Bệnh nhân này chưa có lịch sử TDM trên Cloud.")
        else:
            st.error("Không tìm thấy bệnh nhân với MSYT vừa nhập trên Cloud.")
