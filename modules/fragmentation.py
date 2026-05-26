# ============================================================
# Module: Phan Manh Du Lieu Phan Tan
# (Horizontal & Vertical Fragmentation)
#
# Trong CSDL phan tan, bang goc (Global Relation) duoc chia
# thanh cac manh (fragment) luu tai cac site khac nhau.
#
# 1. Phan manh ngang (Horizontal): chia theo HANG
#    - Moi site luu 1 tap con cac dong du lieu
#    - Dua tren dieu kien loc (predicate)
#    - VD: Site1 = department IN (produce, dairy eggs, ...)
#          Site2 = department IN (frozen, beverages, ...)
#
# 2. Phan manh doc (Vertical): chia theo COT
#    - Moi site luu 1 tap con cac cot
#    - Phai giu lai cot khoa (PK) o tat ca manh de JOIN lai
#    - VD: Site1 = (order_id, user_id, order_number, ...)  [Order Info]
#          Site2 = (order_id, product_id, product_name, ...) [Product Info]
#
# Tinh chat dam bao:
#   - Completeness: UNION tat ca manh = bang goc
#   - Reconstruction: co the khoi phuc lai bang goc
#   - Disjointness: cac manh khong chong lan (tru PK o vertical)
# ============================================================

import sys
# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import os


# ════════════════════════════════════════════════════════════
#  HORIZONTAL FRAGMENTATION (Phân mảnh ngang)
# ════════════════════════════════════════════════════════════

# Định nghĩa điều kiện phân mảnh ngang theo department
# Mô phỏng: 2 site (2 máy), mỗi site quản lý các department khác nhau
HORIZONTAL_PREDICATES = {
    'site_1': [
        'produce', 'dairy eggs', 'snacks', 'beverages',
        'frozen', 'pantry'
    ],
    'site_2': [
        'bakery', 'canned goods', 'deli', 'dry goods pasta',
        'meat seafood', 'breakfast', 'household', 'personal care',
        'babies', 'international', 'pets', 'alcohol',
        'missing', 'other', 'bulk', 'unknown'
    ]
}


def horizontal_fragment(df, predicates=None):
    """
    Phân mảnh ngang: chia DataFrame theo điều kiện trên cột 'department'.

    Mỗi site nhận các dòng có department thuộc danh sách tương ứng.
    Đảm bảo tính Completeness & Disjointness.

    Args:
        df: Spark DataFrame gốc (Global Relation)
        predicates: dict {site_name: [list_of_departments]}

    Returns:
        dict {site_name: DataFrame} — các mảnh ngang
    """
    if predicates is None:
        predicates = HORIZONTAL_PREDICATES

    fragments = {}
    total_original = df.count()

    print('\n' + '─' * 60)
    print('PHÂN MẢNH NGANG (Horizontal Fragmentation)')
    print('─' * 60)
    print(f'Bảng gốc (Global Relation): {total_original:,} rows')
    print(f'Số site: {len(predicates)}')
    print(f'Tiêu chí phân mảnh: cột "department"')
    print()

    for site_name, departments in predicates.items():
        fragment = df.filter(F.col('department').isin(departments))
        frag_count = fragment.count()
        fragments[site_name] = fragment
        print(f'  📦 {site_name}: {frag_count:,} rows ({frag_count/total_original*100:.1f}%)')
        print(f'       Departments: {departments[:4]}{"..." if len(departments) > 4 else ""}')

    # Kiểm tra tính đúng đắn
    _verify_horizontal(df, fragments)

    return fragments


def _verify_horizontal(original_df, fragments):
    """Kiểm tra Completeness & Disjointness của phân mảnh ngang."""
    total_original = original_df.count()

    # Completeness: UNION ALL = bảng gốc
    union_df = None
    for frag in fragments.values():
        union_df = frag if union_df is None else union_df.unionAll(frag)
    total_union = union_df.count()

    # Disjointness: không có dòng nào thuộc >= 2 mảnh
    frag_list = list(fragments.values())
    is_disjoint = True
    for i in range(len(frag_list)):
        for j in range(i + 1, len(frag_list)):
            overlap = frag_list[i].intersect(frag_list[j]).count()
            if overlap > 0:
                is_disjoint = False

    print()
    print(f'  ✅ Completeness: {"PASS" if total_union == total_original else "FAIL"}'
          f' (UNION={total_union:,}, Original={total_original:,})')
    print(f'  ✅ Disjointness: {"PASS" if is_disjoint else "FAIL"}')


def reconstruct_horizontal(fragments):
    """
    Khôi phục bảng gốc từ các mảnh ngang: UNION ALL.

    Reconstruction: R = σ_p1(R) ∪ σ_p2(R) ∪ ... ∪ σ_pn(R)
    """
    print('\n  🔄 Reconstruction (UNION ALL)...')
    union_df = None
    for site_name, frag in fragments.items():
        union_df = frag if union_df is None else union_df.unionAll(frag)
    total = union_df.count()
    print(f'  ✅ Reconstructed: {total:,} rows')
    return union_df


# ════════════════════════════════════════════════════════════
#  VERTICAL FRAGMENTATION (Phân mảnh dọc)
# ════════════════════════════════════════════════════════════

# row_id: surrogate key duy nhat cho moi dong (vi composite key co the trung)
# row_id duoc tao tu dong truoc khi phan manh va phai co mat o TAT CA manh
ROW_ID_COL = '_row_id'
PRIMARY_KEYS = [ROW_ID_COL]

VERTICAL_SCHEMA = {
    'site_1_order_info': {
        'description': 'Thong tin don hang (Order Info)',
        'columns': [ROW_ID_COL,
            'order_id', 'user_id', 'product_id',
            'order_number', 'order_dow', 'order_hour_of_day',
            'days_since_prior_order'
        ]
    },
    'site_2_product_info': {
        'description': 'Thong tin san pham (Product Info)',
        'columns': [ROW_ID_COL,
            'order_id', 'user_id', 'product_id',
            'add_to_cart_order', 'reordered',
            'department_id', 'department', 'product_name'
        ]
    }
}


def vertical_fragment(df, schema=None):
    """
    Phan manh doc: chia DataFrame theo COT.

    Them row_id lam surrogate key duy nhat truoc khi chia.
    Moi site nhan cac cot khac nhau nhung cung giu row_id.
    Dam bao co the JOIN lai bang goc (Lossless Join).

    Args:
        df: Spark DataFrame goc (Global Relation)
        schema: dict {site_name: {'columns': [...], 'description': '...'}}

    Returns:
        dict {site_name: DataFrame} — cac manh doc
    """
    if schema is None:
        schema = VERTICAL_SCHEMA

    # Them row_id lam unique key truoc khi phan manh
    df_with_id = df.withColumn(ROW_ID_COL, F.monotonically_increasing_id())

    fragments = {}
    total_original = df_with_id.count()
    all_cols = set(df_with_id.columns)

    print('\n' + '=' * 60)
    print('PHAN MANH DOC (Vertical Fragmentation)')
    print('=' * 60)
    print(f'Bang goc (Global Relation): {total_original:,} rows, {len(all_cols) - 1} columns (+row_id)')
    print(f'Surrogate Key: {ROW_ID_COL} (unique per row)')
    print(f'So site: {len(schema)}')
    print()

    for site_name, config in schema.items():
        cols = [c for c in config['columns'] if c in all_cols]
        fragment = df_with_id.select(*cols)
        frag_count = fragment.count()
        fragments[site_name] = fragment

        non_pk_cols = [c for c in cols if c not in PRIMARY_KEYS]
        print(f'  [>] {site_name} -- {config["description"]}')
        print(f'       {frag_count:,} rows x {len(cols)} columns')
        print(f'       Key: {PRIMARY_KEYS}')
        print(f'       Data: {non_pk_cols}')

    # Kiem tra tinh dung dan
    _verify_vertical(df_with_id, fragments, schema)

    return fragments


def _verify_vertical(original_df, fragments, schema):
    """Kiểm tra Completeness & Reconstruction của phân mảnh dọc."""
    all_original_cols = set(original_df.columns)

    # Completeness: tất cả cột gốc phải có mặt trong ít nhất 1 mảnh
    all_frag_cols = set()
    for config in schema.values():
        all_frag_cols.update(config['columns'])

    missing_cols = all_original_cols - all_frag_cols
    is_complete = len(missing_cols) == 0

    # Lossless Join: kiểm tra có thể JOIN lại không
    can_join = all(
        set(PRIMARY_KEYS).issubset(set(config['columns']))
        for config in schema.values()
    )

    print()
    print(f'  ✅ Completeness (all columns covered): {"PASS" if is_complete else "FAIL"}'
          + (f' (missing: {missing_cols})' if not is_complete else ''))
    print(f'  ✅ Lossless Join (PKs in all fragments): {"PASS" if can_join else "FAIL"}')


def reconstruct_vertical(fragments):
    """
    Khoi phuc bang goc tu cac manh doc: JOIN tren row_id (surrogate key).

    Reconstruction: R = pi_L1(R) |><| pi_L2(R) |><| ... |><| pi_Ln(R)
    Sau do drop row_id de tra ve schema goc.
    """
    print('\n  [~] Reconstruction (JOIN on row_id)...')
    result = None
    for site_name, frag in fragments.items():
        if result is None:
            result = frag
        else:
            # Tranh trung cot (order_id, user_id, product_id,...) khi join.
            # Chi lay _row_id va cac cot chua co trong result.
            select_cols = [c for c in frag.columns if c == ROW_ID_COL or c not in result.columns]
            sub = frag.select(*select_cols)
            result = result.join(sub, on=PRIMARY_KEYS, how='inner')

    # Drop surrogate key to restore original schema
    if ROW_ID_COL in result.columns:
        result = result.drop(ROW_ID_COL)

    total = result.count()
    print(f'  [OK] Reconstructed: {total:,} rows x {len(result.columns)} columns')
    return result


# ════════════════════════════════════════════════════════════
#  UTILITY: Lưu / Đọc các mảnh từ Parquet
# ════════════════════════════════════════════════════════════

def save_fragments(fragments, output_dir, frag_type='horizontal'):
    """Lưu tất cả mảnh ra Parquet files."""
    base_dir = os.path.join(output_dir, f'fragments_{frag_type}')
    os.makedirs(base_dir, exist_ok=True)

    print(f'\n  💾 Saving {frag_type} fragments to {base_dir}')
    for site_name, frag in fragments.items():
        path = os.path.join(base_dir, site_name)
        frag.write.mode('overwrite').parquet(path)
        print(f'     Saved: {site_name} → {path}')


def load_fragments(spark, output_dir, frag_type='horizontal', site_names=None):
    """Đọc các mảnh từ Parquet files."""
    base_dir = os.path.join(output_dir, f'fragments_{frag_type}')
    fragments = {}

    if site_names is None:
        # Auto-detect from directory listing
        import glob
        site_names = [os.path.basename(d) for d in glob.glob(os.path.join(base_dir, '*'))
                      if os.path.isdir(d)]

    for site_name in site_names:
        path = os.path.join(base_dir, site_name)
        if os.path.exists(path):
            fragments[site_name] = spark.read.parquet(path)

    print(f'  📂 Loaded {len(fragments)} {frag_type} fragments')
    return fragments


# ════════════════════════════════════════════════════════════
#  DISTRIBUTED QUERY trên các mảnh
# ════════════════════════════════════════════════════════════

def distributed_query_horizontal(fragments, query_func, description=''):
    """
    Thực hiện truy vấn phân tán trên các mảnh ngang.

    Chiến lược: chạy query trên TỪNG mảnh → UNION kết quả.
    (Mô phỏng: mỗi site thực hiện query cục bộ, rồi gửi kết quả về coordinator)

    Args:
        fragments: dict {site: DataFrame}
        query_func: function(df) -> DataFrame — hàm truy vấn
        description: mô tả query
    """
    print(f'\n  🔍 Distributed Query (Horizontal): {description}')
    results = {}
    for site_name, frag in fragments.items():
        local_result = query_func(frag)
        local_count = local_result.count()
        results[site_name] = local_result
        print(f'     {site_name}: {local_count:,} results')

    # Merge kết quả từ tất cả sites
    merged = None
    for r in results.values():
        merged = r if merged is None else merged.unionAll(r)

    total = merged.count()
    print(f'     📊 Total merged: {total:,} rows')
    return merged, results


def distributed_query_vertical(fragments, query_cols, filter_expr=None, description=''):
    """
    Thực hiện truy vấn phân tán trên các mảnh dọc.

    Chiến lược: xác định mảnh chứa cột cần thiết → JOIN → filter.
    (Mô phỏng: coordinator gửi sub-query đến site có cột, rồi join kết quả)

    Args:
        fragments: dict {site: DataFrame}
        query_cols: list of column names needed
        filter_expr: optional Spark Column expression for filtering
        description: mô tả query
    """
    print(f'\n  🔍 Distributed Query (Vertical): {description}')

    # Tìm xem cần dữ liệu từ site nào
    needed_sites = {}
    for site_name, frag in fragments.items():
        frag_cols = set(frag.columns)
        relevant = frag_cols.intersection(set(query_cols + PRIMARY_KEYS))
        if len(relevant) > len(set(PRIMARY_KEYS)):  # Has more than just PKs
            needed_sites[site_name] = frag
            print(f'     {site_name}: provides {list(relevant - set(PRIMARY_KEYS))}')

    if not needed_sites:
        print('     ⚠️ No site contains the required columns!')
        return None

    # JOIN các mảnh cần thiết
    result = None
    for site_name, frag in needed_sites.items():
        available_cols = [c for c in query_cols if c in frag.columns]
        if result is not None:
            # Chi select cot chua co trong result de tranh duplicate column names
            available_cols = [c for c in available_cols if c not in result.columns]
        select_cols = list(set(PRIMARY_KEYS + available_cols))
        sub = frag.select(*select_cols)

        if result is None:
            result = sub
        else:
            result = result.join(sub, on=PRIMARY_KEYS, how='inner')

    # Apply filter if provided
    if filter_expr is not None:
        result = result.filter(filter_expr)

    # Drop surrogate row_id to present clean query results
    if ROW_ID_COL in result.columns:
        result = result.drop(ROW_ID_COL)

    total = result.count()
    print(f'     📊 Query result: {total:,} rows x {len(result.columns)} columns')
    return result
