import sys, os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F
from config.spark_config import get_spark, CSV_PATH, OUTPUT_DIR
from modules.data_loader import load_csv, save_parquet
from modules.data_cleaner import clean_data
from modules.fragmentation import (
    horizontal_fragment, reconstruct_horizontal, save_fragments,
    vertical_fragment, reconstruct_vertical,
    distributed_query_horizontal, distributed_query_vertical,
    PRIMARY_KEYS
)


def main():
    spark = get_spark('Pipeline03_Fragmentation')
    print(f'Spark {spark.version} | {spark.sparkContext.master}')
    print(f'UI: {spark.sparkContext.uiWebUrl}')

    # ── Load & Clean Data ─────────────────────────────────
    print('\n' + '=' * 60)
    print('LOADING DATA')
    print('=' * 60)

    # Kiểm tra xem đã có cleaned data chưa
    cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_data.parquet')
    if os.path.exists(cleaned_path):
        from modules.data_loader import load_parquet
        df = load_parquet(spark, cleaned_path)
    else:
        df = load_csv(spark, CSV_PATH)
        df = clean_data(df)

    total_rows = df.count()
    total_cols = len(df.columns)
    print(f'\nGlobal Relation: {total_rows:,} rows × {total_cols} columns')
    print(f'Columns: {df.columns}')
    df.show(5, truncate=False)

    # ══════════════════════════════════════════════════════
    # PART 1: PHÂN MẢNH NGANG (Horizontal Fragmentation)
    # ══════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('PART 1: PHÂN MẢNH NGANG (Horizontal Fragmentation)')
    print('Chia dữ liệu theo HÀNG — mỗi site lưu các department khác nhau')
    print('=' * 60)

    h_fragments = horizontal_fragment(df)

    # Xem sample data từ mỗi site
    for site_name, frag in h_fragments.items():
        print(f'\n  📋 Sample từ {site_name}:')
        frag.select('order_id', 'user_id', 'department', 'product_name') \
            .show(3, truncate=30)

    # ── Truy vấn phân tán trên mảnh ngang ──
    print('\n' + '-' * 60)
    print('TRUY VẤN PHÂN TÁN TRÊN MẢNH NGANG')
    print('-' * 60)

    # Query 1: Top 5 sản phẩm được mua nhiều nhất
    print('\n📊 Query 1: Top 5 sản phẩm mua nhiều nhất (phân tán)')
    def q1_top_products(frag):
        return frag.groupBy('product_name').agg(
            F.count('*').alias('purchase_count')
        ).orderBy(F.desc('purchase_count')).limit(5)

    merged_q1, site_results = distributed_query_horizontal(
        h_fragments, q1_top_products,
        description='Top 5 products per site → merge'
    )
    print('\n  Kết quả từng site:')
    for site, result in site_results.items():
        print(f'\n  [{site}]')
        result.show(5, truncate=40)

    print('\n  Kết quả tổng hợp (sau merge + re-aggregate):')
    # Re-aggregate vì mỗi site trả top 5 riêng
    final_q1 = merged_q1.groupBy('product_name').agg(
        F.sum('purchase_count').alias('total_purchases')
    ).orderBy(F.desc('total_purchases')).limit(5)
    final_q1.show(5, truncate=40)

    # Query 2: Trung bình reorder rate theo department
    print('\n📊 Query 2: Avg reorder rate (phân tán)')
    def q2_avg_reorder(frag):
        return frag.groupBy('department').agg(
            F.mean('reordered').alias('avg_reorder_rate'),
            F.count('*').alias('count')
        )

    merged_q2, _ = distributed_query_horizontal(
        h_fragments, q2_avg_reorder,
        description='Avg reorder rate per department per site'
    )
    # Weighted average cho kết quả chính xác
    final_q2 = merged_q2.groupBy('department').agg(
        (F.sum(F.col('avg_reorder_rate') * F.col('count')) / F.sum('count')).alias('avg_reorder_rate')
    ).orderBy(F.desc('avg_reorder_rate'))
    print('\n  Kết quả tổng hợp (weighted average):')
    final_q2.show(10, truncate=False)

    # ── Reconstruction ──
    print('\n' + '-' * 60)
    print('RECONSTRUCTION (Khôi phục bảng gốc từ mảnh ngang)')
    print('-' * 60)
    reconstructed_h = reconstruct_horizontal(h_fragments)

    # Verify: so sánh bảng gốc vs reconstructed
    print(f'\n  Bảng gốc:        {total_rows:,} rows')
    print(f'  Reconstructed:   {reconstructed_h.count():,} rows')
    print(f'  Match: {"✅ PASS" if reconstructed_h.count() == total_rows else "❌ FAIL"}')

    # Save mảnh ngang
    save_fragments(h_fragments, OUTPUT_DIR, 'horizontal')

    # ══════════════════════════════════════════════════════
    # PART 2: PHÂN MẢNH DỌC (Vertical Fragmentation)
    # ══════════════════════════════════════════════════════
    print('\n\n' + '=' * 60)
    print('PART 2: PHÂN MẢNH DỌC (Vertical Fragmentation)')
    print('Chia dữ liệu theo CỘT — mỗi site lưu nhóm cột khác nhau')
    print('=' * 60)

    v_fragments = vertical_fragment(df)

    # Xem schema từ mỗi site
    for site_name, frag in v_fragments.items():
        print(f'\n  📋 Schema {site_name}: {frag.columns}')
        frag.show(3, truncate=30)

    # ── Truy vấn phân tán trên mảnh dọc ──
    print('\n' + '-' * 60)
    print('TRUY VẤN PHÂN TÁN TRÊN MẢNH DỌC')
    print('-' * 60)

    # Query 3: Lấy order_dow + product_name (cần JOIN 2 site)
    print('\n📊 Query 3: Order day + Product name (cần JOIN 2 site)')
    result_q3 = distributed_query_vertical(
        v_fragments,
        query_cols=['order_id', 'order_dow', 'product_name'],
        description='SELECT order_dow, product_name (cross-site JOIN)'
    )
    if result_q3 is not None:
        result_q3.select('order_id', 'order_dow', 'product_name').show(5, truncate=30)

    # Query 4: Filter trên cột từ site 1, lấy data từ site 2
    print('\n📊 Query 4: Đơn hàng buổi sáng (hour < 10) → tên sản phẩm')
    result_q4 = distributed_query_vertical(
        v_fragments,
        query_cols=['order_hour_of_day', 'product_name', 'department'],
        filter_expr=F.col('order_hour_of_day') < 10,
        description='WHERE hour < 10 → product_name, department (cross-site)'
    )
    if result_q4 is not None:
        result_q4.groupBy('department').agg(
            F.count('*').alias('morning_purchases')
        ).orderBy(F.desc('morning_purchases')).show(10, truncate=False)

    # Query 5: Chỉ cần cột từ 1 site (không cần JOIN)
    print('\n📊 Query 5: Thống kê reorder rate (chỉ cần Site 2 - Product Info)')
    result_q5 = distributed_query_vertical(
        v_fragments,
        query_cols=['department', 'reordered'],
        description='SELECT department, reordered (single-site, no JOIN needed)'
    )
    if result_q5 is not None:
        result_q5.groupBy('department').agg(
            F.mean('reordered').alias('reorder_rate')
        ).orderBy(F.desc('reorder_rate')).show(10, truncate=False)

    # ── Reconstruction ──
    print('\n' + '-' * 60)
    print('RECONSTRUCTION (Khôi phục bảng gốc từ mảnh dọc)')
    print('-' * 60)
    reconstructed_v = reconstruct_vertical(v_fragments)

    # Verify
    print(f'\n  Bảng gốc:        {total_rows:,} rows × {total_cols} columns')
    print(f'  Reconstructed:   {reconstructed_v.count():,} rows × {len(reconstructed_v.columns)} columns')
    print(f'  Rows match:      {"✅" if reconstructed_v.count() == total_rows else "❌"}')
    print(f'  Cols match:      {"✅" if set(reconstructed_v.columns) == set(df.columns) else "❌"}'
          f' {sorted(reconstructed_v.columns)}')

    # Save mảnh dọc
    save_fragments(v_fragments, OUTPUT_DIR, 'vertical')

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print('\n\n' + '=' * 60)
    print('TỔNG KẾT PHÂN MẢNH')
    print('=' * 60)

    print(f'''
┌─────────────────────────────────────────────────────────────┐
│                    BẢNG GỐC (Global)                        │
│              {total_rows:>10,} rows × {total_cols:>2} columns                  │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
    ┌─────────────▼──────────┐  ┌────────▼────────────────┐
    │  PHÂN MẢNH NGANG       │  │  PHÂN MẢNH DỌC          │
    │  (Chia theo HÀNG)      │  │  (Chia theo CỘT)        │
    └──────┬─────────┬───────┘  └──────┬──────────┬───────┘
           │         │                 │          │
      ┌────▼───┐ ┌───▼────┐      ┌────▼───┐ ┌───▼────────┐
      │ Site 1 │ │ Site 2 │      │ Site 1 │ │  Site 2    │
      │{h_fragments['site_1'].count():>7,} │ │{h_fragments['site_2'].count():>7,} │      │Order   │ │ Product    │
      │ rows   │ │ rows   │      │Info    │ │ Info       │
      │        │ │        │      │{len(v_fragments['site_1_order_info'].columns)} cols  │ │ {len(v_fragments['site_2_product_info'].columns)} cols     │
      └────────┘ └────────┘      └────────┘ └────────────┘
    ''')

    print(f'  Phân mảnh ngang:')
    for site, frag in h_fragments.items():
        print(f'    {site}: {frag.count():>10,} rows')
    print(f'    Reconstruction: ✅ {reconstructed_h.count():,} rows')

    print(f'\n  Phân mảnh dọc:')
    for site, frag in v_fragments.items():
        print(f'    {site}: {frag.count():>10,} rows × {len(frag.columns)} cols → {frag.columns}')
    print(f'    Reconstruction: ✅ {reconstructed_v.count():,} rows × {len(reconstructed_v.columns)} cols')

    print('\n  Output saved to:', OUTPUT_DIR)
    print('=' * 60)

    # Only prompt in local/interactive mode
    mode = os.environ.get('SPARK_MODE', 'local').lower()
    if mode == 'local':
        print(f'\nSpark UI: {spark.sparkContext.uiWebUrl}')
        input('Press Enter to stop Spark...')

    spark.stop()


if __name__ == '__main__':
    main()
