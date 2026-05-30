"""
============================================================
PySpark Shell Initializer — Auto-load Fragmented Data
============================================================
Chạy bằng:
  pyspark --master spark://spark-master:7077 -i /app/init_fragments.py

Sau khi chạy xong, bạn sẽ có sẵn các SQL Views:
  - global_relation         (bảng gốc)
  - h_site_1, h_site_2      (phân mảnh ngang)
  - v_site_1, v_site_2      (phân mảnh dọc)

Gõ SQL trực tiếp:
  spark.sql("SELECT ... FROM h_site_1 ...").show()
============================================================
"""

import os
from pyspark.sql import functions as F

print('\n' + '=' * 60)
print('  KHỞI TẠO DỮ LIỆU PHÂN MẢNH')
print('=' * 60)

# ── Paths ──
DATA_DIR = '/app'
OUTPUT_DIR = os.path.join(DATA_DIR, 'spark_output')
CSV_PATH = os.path.join(DATA_DIR, 'ECommerce_consumer behaviour.csv')
H_DIR = os.path.join(OUTPUT_DIR, 'fragments_horizontal')
V_DIR = os.path.join(OUTPUT_DIR, 'fragments_vertical')

# ── Check if fragments exist, if not → create them ──
has_h = os.path.exists(os.path.join(H_DIR, 'site_1'))
has_v = os.path.exists(os.path.join(V_DIR, 'site_1_order_info'))

if has_h and has_v:
    print('\n  📂 Fragments đã có sẵn → load từ Parquet...')

    # Load bảng gốc
    cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_data.parquet')
    if os.path.exists(cleaned_path):
        df = spark.read.parquet(cleaned_path)
    else:
        df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

    df.createOrReplaceTempView('global_relation')
    print(f'  ✅ global_relation: {df.count():,} rows')

    # Load phân mảnh ngang
    h1 = spark.read.parquet(os.path.join(H_DIR, 'site_1'))
    h2 = spark.read.parquet(os.path.join(H_DIR, 'site_2'))
    h1.createOrReplaceTempView('h_site_1')
    h2.createOrReplaceTempView('h_site_2')
    print(f'  ✅ h_site_1: {h1.count():,} rows')
    print(f'  ✅ h_site_2: {h2.count():,} rows')

    # Load phân mảnh dọc
    v1 = spark.read.parquet(os.path.join(V_DIR, 'site_1_order_info'))
    v2 = spark.read.parquet(os.path.join(V_DIR, 'site_2_product_info'))
    v1.createOrReplaceTempView('v_site_1')
    v2.createOrReplaceTempView('v_site_2')
    print(f'  ✅ v_site_1 (Order Info): {v1.count():,} rows × {len(v1.columns)} cols → {v1.columns}')
    print(f'  ✅ v_site_2 (Product Info): {v2.count():,} rows × {len(v2.columns)} cols → {v2.columns}')

else:
    print('\n  ⚠️ Fragments chưa tồn tại → tạo mới từ CSV...')

    # Load & Clean
    df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
    median_days = df.approxQuantile('days_since_prior_order', [0.5], 0.01)[0]
    df = df.fillna({'days_since_prior_order': median_days})
    df = df.fillna({'department': 'unknown', 'product_name': 'unknown'})
    df = df.dropna(subset=['order_id', 'user_id', 'product_id'])
    for c in ['order_id', 'user_id', 'product_id', 'department_id']:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast('int'))

    df.createOrReplaceTempView('global_relation')
    total = df.count()
    print(f'  ✅ global_relation: {total:,} rows')

    # ── Phân mảnh ngang ──
    site1_depts = ['produce', 'dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry']
    site2_depts = ['bakery', 'canned goods', 'deli', 'dry goods pasta',
                   'meat seafood', 'breakfast', 'household', 'personal care',
                   'babies', 'international', 'pets', 'alcohol',
                   'missing', 'other', 'bulk', 'unknown']

    h1 = df.filter(F.col('department').isin(site1_depts))
    h2 = df.filter(F.col('department').isin(site2_depts))
    h1.createOrReplaceTempView('h_site_1')
    h2.createOrReplaceTempView('h_site_2')

    # Save
    os.makedirs(H_DIR, exist_ok=True)
    h1.write.mode('overwrite').parquet(os.path.join(H_DIR, 'site_1'))
    h2.write.mode('overwrite').parquet(os.path.join(H_DIR, 'site_2'))
    print(f'  ✅ h_site_1: {h1.count():,} rows (saved)')
    print(f'  ✅ h_site_2: {h2.count():,} rows (saved)')

    # ── Phân mảnh dọc ──
    df_id = df.withColumn('_row_id', F.monotonically_increasing_id())

    v1 = df_id.select('_row_id', 'order_id', 'user_id', 'product_id',
                       'order_number', 'order_dow', 'order_hour_of_day',
                       'days_since_prior_order')
    v2 = df_id.select('_row_id', 'order_id', 'user_id', 'product_id',
                       'add_to_cart_order', 'reordered',
                       'department_id', 'department', 'product_name')

    v1.createOrReplaceTempView('v_site_1')
    v2.createOrReplaceTempView('v_site_2')

    os.makedirs(V_DIR, exist_ok=True)
    v1.write.mode('overwrite').parquet(os.path.join(V_DIR, 'site_1_order_info'))
    v2.write.mode('overwrite').parquet(os.path.join(V_DIR, 'site_2_product_info'))
    print(f'  ✅ v_site_1 (Order Info): {v1.count():,} rows × {len(v1.columns)} cols (saved)')
    print(f'  ✅ v_site_2 (Product Info): {v2.count():,} rows × {len(v2.columns)} cols (saved)')


# ══════════════════════════════════════════════════════
# Hướng dẫn sử dụng
# ══════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  SẴN SÀNG! Gõ SQL queries bên dưới:')
print('=' * 60)
print('''
  📋 Các bảng đã đăng ký:
    global_relation         — Bảng gốc đầy đủ
    h_site_1, h_site_2      — Phân mảnh NGANG (chia theo department)
    v_site_1, v_site_2      — Phân mảnh DỌC  (chia theo nhóm cột)

  💡 Ví dụ phân mảnh NGANG:
    spark.sql("SELECT product_name, COUNT(*) AS cnt FROM h_site_1 GROUP BY product_name ORDER BY cnt DESC LIMIT 5").show()

  💡 Ví dụ phân mảnh DỌC (JOIN 2 site):
    spark.sql("SELECT s1.order_dow, s2.product_name FROM v_site_1 s1 JOIN v_site_2 s2 ON s1._row_id = s2._row_id LIMIT 5").show()

  🌐 Spark UI: http://localhost:4040
''')
