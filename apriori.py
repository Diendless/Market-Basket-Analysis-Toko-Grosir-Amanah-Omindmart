import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

#load dataset
df = pd.read_csv("dataset_grosir.csv")

# Ubah kolom tanggal ke format datetime
df['date_time'] = pd.to_datetime(df['date_time'], format="%d/%m/%Y")

# Tambahkan kolom bulan dari kolom tanggal
df["month"] = df['date_time'].dt.month

# Ubah format angka bulan menjadi nama bulan
df["month"] = df["month"].replace(
    [i for i in range(1, 12 + 1)], 
    ["January", "February", "March", "April", "May", "June", 
     "July", "August", "September", "October", "November", "December"]
)

st.title("🛒 Market Basket Analysis Toko Grosir Amanah Omindmart")

# Fungsi untuk mengambil data berdasarkan filter weekday/weekend dan bulan
def get_data(weekday_weekend = '', month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

# Input dari pengguna via dropdown
def user_input_features():
    item = st.selectbox("Item", ['Sunco 2L', 'Sunco 1L', 'Hemart 1L','Hemart 1/2 L Refill', 'Fortune 1L', 'Indo Goreng', 'Indo G Geprek', 'Indo Rendang', 'Indo Aceh', 'Indo AB', 'Indo Soto', 'Sedap Soto', 'Fortune 2 L', 'Hemart 2 L refill', 'Gas 3kg', 'Sedap Goreng', 'Aqua Galon', 'Amanah Galon', 'Minyak Kita 1L', 'Gula Pasir', 'Amanah 330ml', 'Amanah 500ml', 'Sunlight', 'Aqua 600ml', 'Bango', 'Garam', 'Le Mineral 600ml', 'Sedap AB', 'Wall Stapler', 'Sawit', 'Le Mineral Galon', 'Lencana Merah', 'Teh Pucuk', 'Kara'])
    weekday_weekend = st.selectbox("Weekday / Weekend", ['Weekday', 'Weekend'])
    month = st.selectbox("Month", ['July', 'August', 'September'])

    return weekday_weekend, month, item

weekday_weekend, month, item = user_input_features() # Ambil input dari user

data = get_data(weekday_weekend.lower(), month) # Filter data sesuai input user

# Fungsi encode: jika jumlah ≤0 maka 0, jika ≥1 maka 1
def encode(x):
    if x <=0:
        return 0
    elif x >= 1:
        return 1

# Proses jika data ditemukan (bukan "No Result")    
if type(data) != type ("No Result"):
    item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.02
    frequent_items = apriori(item_count_pivot, min_support= support, use_colnames=True)
    
    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

# Fungsi untuk mengubah tipe set ke string dengan koma
def parse_list(x):
    x = list(x)
    return ", ".join(x) if len(x) > 0 else None

# Fungsi untuk mencari item yang sering dibeli bersama dengan item yang dipilih user
def return_item_df(item_antecedents):
    if data is None:
        return None
    
    # Ambil kolom yang diperlukan
    data_rules = rules[["antecedents", "consequents"]].copy()

    # Ubah format set ke string agar bisa dicari
    data_rules["antecedents"] = data_rules["antecedents"].apply(parse_list)
    data_rules["consequents"] = data_rules["consequents"].apply(parse_list)
    
    # Kembalikan hasil rekomendasi (jika ada)
    result = data_rules.loc[data_rules["antecedents"] == item_antecedents]
    return list(result.iloc[0, :]) if not result.empty else None

# Tampilkan hasil rekomendasi ke user
if data is not None:
    st.markdown("Hasil Rekomendasi :")
    recommendation = return_item_df(item)
    if recommendation is None:
        st.warning(f"Tidak ditemukan rekomendasi untuk item **{item}**")
    else:
        st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{recommendation[1]}** secara bersamaan")