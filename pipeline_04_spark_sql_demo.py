"""
============================================================
Pipeline 04: Spark SQL Interactive Demo
============================================================
Mục đích: Đăng ký các mảnh dữ liệu thành SQL Tables/Views
để có thể truy vấn trực tiếp trong Spark SQL UI (Thymeleaf)
tại http://localhost:4040/sqlserver

Kiến trúc Docker:
  - spark-master  (coordinator)
  - spark-worker-1 (Machine 1 / Site 1)
  - spark-worker-2 (Machine 2 / Site 2)

Chạy:
  docker compose run --rm -p 4040:4040 spark-submit \
    /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
    /app/pipeline_04_spark_sql_demo.py
============================================================
"""

import sys, os, time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F
from config.spark_config import get_spark, CSV_PATH, OUTPUT_DIR
from modules.data_loader import load_csv, save_parquet
from modules.data_cleaner import clean_data
from modules.fragmentation import (
    horizontal_fragment, vertical_fragment,
    HORIZONTAL_PREDICATES, VERTICAL_SCHEMA, ROW_ID_COL,
    PRIMARY_KEYS
)


def main():
    spark = get_spark('Pipeline04_SQL_Demo')
    print(f'Spark {spark.version} | {spark.sparkContext.master}')
    print(f'Spark UI: {spark.sparkContext.uiWebUrl}')

    # ── Load & Clean ──────────────────────────────────────
    cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_data.parquet')
    if os.path.exists(cleaned_path):
        from modules.data_loader import load_parquet
        df = load_parquet(spark, cleaned_path)
    else:
        df = load_csv(spark, CSV_PATH)
        df = clean_data(df)

    total = df.count()
    print(f'\nGlobal Relation: {total:,} rows x {len(df.columns)} cols')

    # ══════════════════════════════════════════════════════
    # ĐĂNG KÝ BẢNG GỐC
    # ══════════════════════════════════════════════════════
    df.createOrReplaceTempView('global_relation')
    print('\n✅ Registered: global_relation')

    # ══════════════════════════════════════════════════════
    # PHẦN 1: PHÂN MẢNH NGANG → SQL Tables
    # ══════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('PHÂN MẢNH NGANG → SQL Tables')
    print('=' * 60)

    h_frags = horizontal_fragment(df)
    for site_name, frag in h_frags.items():
        view_name = f'h_{site_name}'
        frag.createOrReplaceTempView(view_name)
        print(f'  ✅ Registered: {view_name} ({frag.count():,} rows)')

    # ── Demo queries (horizontal) ─────────────────────────
    print('\n' + '-' * 60)
    print('DEMO: Truy vấn phân tán trên mảnh ngang')
    print('-' * 60)

    # Q1: Top sản phẩm mỗi site
    print('\n📊 Q1: Top 5 sản phẩm mua nhiều nhất — TỪNG SITE')
    q1 = """
    SELECT 'site_1' AS site, product_name, COUNT(*) AS purchase_count
    FROM h_site_1
    GROUP BY product_name
    ORDER BY purchase_count DESC
    LIMIT 5
    """
    spark.sql(q1).show(truncate=40)
    print('  → Site 1 thực hiện query CỤC BỘ trên dữ liệu của mình')

    q1b = """
    SELECT 'site_2' AS site, product_name, COUNT(*) AS purchase_count
    FROM h_site_2
    GROUP BY product_name
    ORDER BY purchase_count DESC
    LIMIT 5
    """
    spark.sql(q1b).show(truncate=40)
    print('  → Site 2 thực hiện query CỤC BỘ trên dữ liệu của mình')

    # Q1 merged: UNION rồi re-aggregate
    print('\n📊 Q1 MERGED: Tổng hợp kết quả từ 2 site (UNION ALL)')
    q1_merged = """
    SELECT product_name, SUM(purchase_count) AS total_purchases
    FROM (
        SELECT product_name, COUNT(*) AS purchase_count FROM h_site_1 GROUP BY product_name
        UNION ALL
        SELECT product_name, COUNT(*) AS purchase_count FROM h_site_2 GROUP BY product_name
    ) combined
    GROUP BY product_name
    ORDER BY total_purchases DESC
    LIMIT 5
    """
    spark.sql(q1_merged).show(truncate=40)

    # Q2: Avg reorder rate phân tán
    print('\n📊 Q2: Trung bình Reorder Rate theo department (phân tán)')
    q2 = """
    SELECT department,
           SUM(local_sum) / SUM(local_count) AS avg_reorder_rate
    FROM (
        SELECT department,
               SUM(reordered) AS local_sum,
               COUNT(*) AS local_count
        FROM h_site_1
        GROUP BY department

        UNION ALL

        SELECT department,
               SUM(reordered) AS local_sum,
               COUNT(*) AS local_count
        FROM h_site_2
        GROUP BY department
    ) site_results
    GROUP BY department
    ORDER BY avg_reorder_rate DESC
    """
    spark.sql(q2).show(10, truncate=False)
    print('  → Mỗi site tính SUM + COUNT cục bộ → coordinator tổng hợp weighted avg')

    # Q3: Reconstruction = UNION ALL
    print('\n📊 Q3: RECONSTRUCTION — Khôi phục bảng gốc (UNION ALL)')
    q3 = """
    SELECT COUNT(*) AS total_rows FROM (
        SELECT * FROM h_site_1
        UNION ALL
        SELECT * FROM h_site_2
    ) reconstructed
    """
    result = spark.sql(q3)
    result.show()
    reconstructed_count = result.collect()[0][0]
    print(f'  Bảng gốc: {total:,} | Reconstructed: {reconstructed_count:,} '
          f'| Match: {"✅" if reconstructed_count == total else "❌"}')

    # ══════════════════════════════════════════════════════
    # PHẦN 2: PHÂN MẢNH DỌC → SQL Tables
    # ══════════════════════════════════════════════════════
    print('\n\n' + '=' * 60)
    print('PHÂN MẢNH DỌC → SQL Tables')
    print('=' * 60)

    v_frags = vertical_fragment(df)
    for site_name, frag in v_frags.items():
        view_name = f'v_{site_name}'
        frag.createOrReplaceTempView(view_name)
        print(f'  ✅ Registered: {view_name}')
        print(f'     Columns: {frag.columns}')
        print(f'     Rows: {frag.count():,}')

    # ── Demo queries (vertical) ───────────────────────────
    print('\n' + '-' * 60)
    print('DEMO: Truy vấn phân tán trên mảnh dọc')
    print('-' * 60)

    # Q4: Query cần JOIN 2 site
    print('\n📊 Q4: order_dow + product_name (cần JOIN 2 site)')
    q4 = """
    SELECT s1.order_id, s1.order_dow, s2.product_name
    FROM v_site_1_order_info s1
    INNER JOIN v_site_2_product_info s2
        ON s1._row_id = s2._row_id
    LIMIT 10
    """
    spark.sql(q4).show(truncate=30)
    print('  → Cột order_dow ở Site 1, cột product_name ở Site 2')
    print('  → Coordinator gửi sub-query đến cả 2 site rồi JOIN trên _row_id')

    # Q5: Filter site 1 → lấy data site 2
    print('\n📊 Q5: Đơn hàng buổi sáng (hour < 10) → tên sản phẩm + department')
    q5 = """
    SELECT s2.department, COUNT(*) AS morning_purchases
    FROM v_site_1_order_info s1
    INNER JOIN v_site_2_product_info s2
        ON s1._row_id = s2._row_id
    WHERE s1.order_hour_of_day < 10
    GROUP BY s2.department
    ORDER BY morning_purchases DESC
    """
    spark.sql(q5).show(10, truncate=False)
    print('  → Filter trên cột order_hour_of_day (Site 1)')
    print('  → Lấy department, product_name (Site 2)')
    print('  → Cần semi-join: Site 1 filter → gửi _row_id → Site 2 lookup')

    # Q6: Query chỉ cần 1 site (không JOIN)
    print('\n📊 Q6: Thống kê reorder rate (chỉ cần Site 2 — không cần JOIN)')
    q6 = """
    SELECT department,
           AVG(reordered) AS reorder_rate,
           COUNT(*) AS total_items
    FROM v_site_2_product_info
    GROUP BY department
    ORDER BY reorder_rate DESC
    """
    spark.sql(q6).show(10, truncate=False)
    print('  → Tất cả cột đều thuộc Site 2 → query cục bộ, KHÔNG cần JOIN')

    # Q7: Reconstruction = JOIN
    print('\n📊 Q7: RECONSTRUCTION — Khôi phục bảng gốc (JOIN trên _row_id)')
    q7 = """
    SELECT COUNT(*) AS total_rows,
           COUNT(DISTINCT s1._row_id) AS unique_keys
    FROM v_site_1_order_info s1
    INNER JOIN v_site_2_product_info s2
        ON s1._row_id = s2._row_id
    """
    spark.sql(q7).show()

    # ══════════════════════════════════════════════════════
    # Q8: So sánh hiệu suất: Centralized vs Distributed
    # ══════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('Q8: SO SÁNH — Truy vấn tập trung vs Phân tán')
    print('=' * 60)

    # Centralized
    print('\n  [Centralized] Query trên bảng gốc:')
    q8a = """
    SELECT department, COUNT(*) AS cnt, AVG(reordered) AS avg_reorder
    FROM global_relation
    GROUP BY department
    ORDER BY cnt DESC
    LIMIT 5
    """
    spark.sql(q8a).show(truncate=False)

    # Distributed horizontal
    print('  [Distributed - Ngang] UNION ALL từ 2 site:')
    q8b = """
    SELECT department, SUM(cnt) AS cnt, SUM(s) / SUM(cnt) AS avg_reorder
    FROM (
        SELECT department, COUNT(*) AS cnt, SUM(reordered) AS s FROM h_site_1 GROUP BY department
        UNION ALL
        SELECT department, COUNT(*) AS cnt, SUM(reordered) AS s FROM h_site_2 GROUP BY department
    ) t
    GROUP BY department
    ORDER BY cnt DESC
    LIMIT 5
    """
    spark.sql(q8b).show(truncate=False)

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('TẤT CẢ SQL VIEWS ĐÃ ĐĂNG KÝ')
    print('=' * 60)
    print('\n  Bảng gốc:')
    print('    global_relation')
    print('\n  Phân mảnh ngang:')
    print('    h_site_1  (Machine 1 — produce, dairy eggs, snacks, ...)')
    print('    h_site_2  (Machine 2 — bakery, canned goods, deli, ...)')
    print('\n  Phân mảnh dọc:')
    print('    v_site_1_order_info    (Machine 1 — Order Info columns)')
    print('    v_site_2_product_info  (Machine 2 — Product Info columns)')

    print('\n' + '=' * 60)
    print(f'Spark UI: {spark.sparkContext.uiWebUrl}')
    print('Xem tab "SQL / DataFrame" để thấy execution plan')
    print('Xem tab "Stages" để thấy task phân bố trên 2 workers')
    print('=' * 60)

    # Keep alive for Spark UI inspection
    mode = os.environ.get('SPARK_MODE', 'local').lower()
    if mode == 'docker':
        print('\n⏳ Keeping Spark alive for UI inspection...')
        print('   Press Ctrl+C to stop.')
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
    else:
        print(f'\nSpark UI: {spark.sparkContext.uiWebUrl}')
        input('Press Enter to stop Spark...')

    spark.stop()


if __name__ == '__main__':
    main()
