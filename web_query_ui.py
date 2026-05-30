"""Flask Web UI for Spark Distributed Query Demo"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from pyspark.sql import SparkSession, functions as F

app = Flask(__name__)
spark = None
views_ready = False

# ── Worker 1 preset queries (h_site_1 / v_site_1) ───────────────────────────
PRESET_W1 = {
    "w1_h_top5": {
        "title": "🏆 Top 5 sản phẩm",
        "sql": "SELECT product_name, COUNT(*) AS purchase_count\nFROM h_site_1\nGROUP BY product_name\nORDER BY purchase_count DESC\nLIMIT 5",
        "desc": "[Ngang] Query cục bộ h_site_1. Đếm số lần mua theo tên sản phẩm. Không cần network I/O."
    },
    "w1_h_dept_stats": {
        "title": "📦 Thống kê theo Department",
        "sql": "SELECT department,\n  COUNT(*) AS total_orders,\n  ROUND(AVG(reordered), 4) AS reorder_rate,\n  SUM(reordered) AS total_reorders\nFROM h_site_1\nGROUP BY department\nORDER BY total_orders DESC",
        "desc": "[Ngang] Thống kê h_site_1: tổng đơn, tỉ lệ mua lại, tổng mua lại — theo department."
    },
    "w1_h_hourly": {
        "title": "🕐 Đơn hàng theo giờ",
        "sql": "SELECT order_hour_of_day AS hour,\n  COUNT(*) AS order_count,\n  ROUND(AVG(reordered), 3) AS reorder_rate\nFROM h_site_1\nGROUP BY order_hour_of_day\nORDER BY hour",
        "desc": "[Ngang] Phân bố đơn hàng theo giờ trong ngày trên h_site_1 (Worker 1)."
    },
    "w1_h_dow": {
        "title": "📅 Đơn hàng theo ngày tuần",
        "sql": "SELECT order_dow AS day_of_week,\n  COUNT(*) AS orders,\n  COUNT(DISTINCT user_id) AS unique_users\nFROM h_site_1\nGROUP BY order_dow\nORDER BY day_of_week",
        "desc": "[Ngang] Số đơn và người dùng unique theo ngày trong tuần — h_site_1."
    },
    "w1_h_reorder_top": {
        "title": "🔁 Top sản phẩm mua lại nhiều",
        "sql": "SELECT product_name,\n  SUM(reordered) AS reorder_count,\n  COUNT(*) AS total,\n  ROUND(SUM(reordered)/COUNT(*)*100, 1) AS reorder_pct\nFROM h_site_1\nGROUP BY product_name\nHAVING COUNT(*) > 50\nORDER BY reorder_pct DESC\nLIMIT 10",
        "desc": "[Ngang] Sản phẩm có tỉ lệ mua lại cao nhất trên h_site_1 (chỉ tính sp >50 lần mua)."
    },
    "w1_v_schema": {
        "title": "🔍 Xem cấu trúc v_site_1",
        "sql": "SELECT * FROM v_site_1 LIMIT 10",
        "desc": "[Dọc] Preview 10 dòng đầu của v_site_1 — chứa order_id, user_id, order_dow, order_hour_of_day, days_since_prior_order."
    },
    "w1_v_active_users": {
        "title": "👤 Top user đặt nhiều đơn nhất",
        "sql": "SELECT user_id,\n  COUNT(DISTINCT order_id) AS order_count,\n  ROUND(AVG(days_since_prior_order), 1) AS avg_days_between_orders\nFROM v_site_1\nWHERE days_since_prior_order IS NOT NULL\nGROUP BY user_id\nORDER BY order_count DESC\nLIMIT 10",
        "desc": "[Dọc] Top 10 user có nhiều đơn hàng nhất, kèm tần suất mua bình quân — v_site_1."
    },
    "w1_v_order_pattern": {
        "title": "📊 Pattern đặt hàng theo giờ/ngày",
        "sql": "SELECT order_dow AS day,\n  order_hour_of_day AS hour,\n  COUNT(*) AS orders\nFROM v_site_1\nGROUP BY order_dow, order_hour_of_day\nORDER BY orders DESC\nLIMIT 15",
        "desc": "[Dọc] Tổ hợp ngày+giờ có nhiều đơn nhất — phát hiện peak order time trên v_site_1."
    },
    "w1_cross_union": {
        "title": "🔗 UNION kiểm chứng completeness",
        "sql": "SELECT 'h_site_1' AS fragment, COUNT(*) AS rows FROM h_site_1\nUNION ALL\nSELECT 'h_site_2', COUNT(*) FROM h_site_2\nUNION ALL\nSELECT 'global_relation', COUNT(*) FROM global_relation",
        "desc": "[Cross] Kiểm chứng: site1 + site2 = global. Chạy từ Worker 1 nhưng query cả 2 mảnh."
    },
}

# ── Worker 2 preset queries (h_site_2 / v_site_2) ───────────────────────────
PRESET_W2 = {
    "w2_h_top5": {
        "title": "🏆 Top 5 sản phẩm",
        "sql": "SELECT product_name, COUNT(*) AS purchase_count\nFROM h_site_2\nGROUP BY product_name\nORDER BY purchase_count DESC\nLIMIT 5",
        "desc": "[Ngang] Query cục bộ h_site_2. Đếm số lần mua theo tên sản phẩm. Không cần network I/O."
    },
    "w2_h_dept_stats": {
        "title": "📦 Thống kê theo Department",
        "sql": "SELECT department,\n  COUNT(*) AS total_orders,\n  ROUND(AVG(reordered), 4) AS reorder_rate,\n  SUM(reordered) AS total_reorders\nFROM h_site_2\nGROUP BY department\nORDER BY total_orders DESC",
        "desc": "[Ngang] Thống kê h_site_2: tổng đơn, tỉ lệ mua lại, tổng mua lại — theo department."
    },
    "w2_h_top_add_cart": {
        "title": "🛒 Vị trí thêm vào giỏ",
        "sql": "SELECT add_to_cart_order AS cart_position,\n  COUNT(*) AS frequency,\n  ROUND(AVG(reordered), 3) AS reorder_rate\nFROM h_site_2\nWHERE add_to_cart_order IS NOT NULL\nGROUP BY add_to_cart_order\nORDER BY cart_position\nLIMIT 15",
        "desc": "[Ngang] Phân tích vị trí thêm vào giỏ hàng — h_site_2 có cột add_to_cart_order."
    },
    "w2_h_reorder_dept": {
        "title": "🔁 Reorder rate theo Department",
        "sql": "SELECT department,\n  COUNT(*) AS total,\n  SUM(reordered) AS reorders,\n  ROUND(SUM(reordered)/COUNT(*)*100, 2) AS reorder_pct\nFROM h_site_2\nGROUP BY department\nORDER BY reorder_pct DESC",
        "desc": "[Ngang] Tỉ lệ % mua lại theo từng department — h_site_2 (Worker 2)."
    },
    "w2_h_product_variety": {
        "title": "🌈 Đa dạng sản phẩm theo Dept",
        "sql": "SELECT department,\n  COUNT(DISTINCT product_name) AS unique_products,\n  COUNT(*) AS total_orders,\n  ROUND(COUNT(*)/COUNT(DISTINCT product_name), 1) AS orders_per_product\nFROM h_site_2\nGROUP BY department\nORDER BY unique_products DESC",
        "desc": "[Ngang] Số sản phẩm unique và mức độ bán trung bình mỗi sản phẩm theo dept — h_site_2."
    },
    "w2_v_schema": {
        "title": "🔍 Xem cấu trúc v_site_2",
        "sql": "SELECT * FROM v_site_2 LIMIT 10",
        "desc": "[Dọc] Preview 10 dòng đầu của v_site_2 — chứa product_name, department, reordered, add_to_cart_order."
    },
    "w2_v_product_rank": {
        "title": "🥇 Rank sản phẩm theo reorder",
        "sql": "SELECT product_name, department,\n  COUNT(*) AS total_purchases,\n  SUM(reordered) AS reorders,\n  ROUND(SUM(reordered)/COUNT(*)*100, 1) AS reorder_pct,\n  RANK() OVER (PARTITION BY department ORDER BY SUM(reordered) DESC) AS dept_rank\nFROM v_site_2\nGROUP BY product_name, department\nHAVING COUNT(*) > 30\nORDER BY department, dept_rank\nLIMIT 20",
        "desc": "[Dọc] Rank sản phẩm mua lại nhiều nhất trong mỗi department — dùng WINDOW FUNCTION trên v_site_2."
    },
    "w2_v_dept_local": {
        "title": "📊 Avg reorder rate theo Dept",
        "sql": "SELECT department,\n  ROUND(AVG(reordered), 4) AS reorder_rate,\n  COUNT(*) AS total\nFROM v_site_2\nGROUP BY department\nORDER BY reorder_rate DESC\nLIMIT 10",
        "desc": "[Dọc] Query cục bộ hoàn toàn trên v_site_2 — không cần JOIN. Hiệu suất tốt nhất."
    },
    "w2_cross_join": {
        "title": "🔗 JOIN cross-site (morning)",
        "sql": "SELECT s2.department,\n  COUNT(*) AS morning_purchases,\n  ROUND(AVG(s2.reordered), 3) AS reorder_rate\nFROM v_site_1 s1\nJOIN v_site_2 s2 ON s1._row_id = s2._row_id\nWHERE s1.order_hour_of_day BETWEEN 6 AND 10\nGROUP BY s2.department\nORDER BY morning_purchases DESC\nLIMIT 10",
        "desc": "[Cross] Filter hour từ v_site_1 → JOIN lấy department từ v_site_2. Semi-join optimization."
    },
}

# Combined for /query endpoint reference
ALL_PRESETS = {**PRESET_W1, **PRESET_W2}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spark Distributed Query UI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --blue:#1a73e8;--blue-light:#e8f0fe;--blue-mid:#4285f4;--blue-dark:#1558b0;
  --white:#ffffff;--bg:#f0f4ff;--border:#d2e3fc;--text:#1d2b4f;--muted:#5f7cb0;
  --card-shadow:0 2px 12px rgba(26,115,232,0.10);
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}

/* HEADER */
.header{background:var(--white);border-bottom:2px solid var(--border);padding:14px 32px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.header-left{display:flex;align-items:center;gap:14px}
.logo{width:38px;height:38px;background:linear-gradient(135deg,var(--blue-mid),var(--blue-dark));border-radius:10px;display:flex;align-items:center;justify-content:center}
.logo svg{width:20px;height:20px;fill:#fff}
.header-title{font-size:17px;font-weight:700;color:var(--text)}
.header-sub{font-size:12px;color:var(--muted);margin-top:1px}
.nav-links{display:flex;gap:8px;flex-wrap:wrap}
.nav-links a{font-size:11px;font-weight:500;color:var(--blue);text-decoration:none;padding:5px 12px;border:1.5px solid var(--border);border-radius:20px;background:var(--white);transition:.2s;white-space:nowrap}
.nav-links a:hover{background:var(--blue-light);border-color:var(--blue)}

/* MAIN */
.workspace{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:20px 24px;min-height:0}

/* PANEL */
.panel{background:var(--white);border-radius:14px;border:1.5px solid var(--border);box-shadow:var(--card-shadow);display:flex;flex-direction:column;overflow:hidden}
.panel-header{padding:14px 18px 12px;border-bottom:1px solid var(--border);background:var(--white)}
.panel-title-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.docker-badge{display:flex;align-items:center;gap:7px}
.dot{width:9px;height:9px;border-radius:50%;background:#34a853;box-shadow:0 0 0 2px rgba(52,168,83,.25);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 2px rgba(52,168,83,.25)}50%{box-shadow:0 0 0 5px rgba(52,168,83,.1)}}
.docker-label{font-size:13px;font-weight:700;color:var(--text)}
.docker-sub{font-size:11px;color:var(--muted)}
.preset-row{display:flex;gap:6px;flex-wrap:wrap}
.preset-btn{font-size:10.5px;font-weight:500;padding:4px 10px;border-radius:14px;border:1.5px solid var(--border);background:var(--white);color:var(--muted);cursor:pointer;transition:.15s;white-space:nowrap}
.preset-btn:hover{background:var(--blue-light);color:var(--blue);border-color:var(--blue)}
.preset-btn.active{background:var(--blue);color:#fff;border-color:var(--blue)}

.panel-body{flex:1;display:flex;flex-direction:column;padding:14px 18px;gap:10px;overflow-y:auto}
.desc-box{background:var(--blue-light);border-left:3px solid var(--blue-mid);border-radius:0 6px 6px 0;padding:8px 12px;font-size:11.5px;color:var(--blue-dark);line-height:1.5;display:none}

.sql-editor{flex:1;min-height:110px;max-height:200px;background:var(--bg);border:1.5px solid var(--border);border-radius:8px;color:var(--text);padding:10px 12px;font-family:'Courier New',Courier,monospace;font-size:12.5px;resize:vertical;line-height:1.6;transition:.2s}
.sql-editor:focus{outline:none;border-color:var(--blue);background:#fff;box-shadow:0 0 0 3px rgba(26,115,232,.12)}

.action-row{display:flex;align-items:center;gap:10px}
.run-btn{background:linear-gradient(135deg,var(--blue-mid),var(--blue-dark));color:#fff;border:none;padding:9px 22px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:.2s;display:flex;align-items:center;gap:7px;letter-spacing:.2px}
.run-btn:hover{filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 4px 12px rgba(26,115,232,.3)}
.run-btn:disabled{opacity:.6;cursor:not-allowed;transform:none;box-shadow:none}
.spinner{display:none;width:13px;height:13px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite}
.run-btn.loading .spinner{display:block}
.run-btn.loading .label{display:none}
@keyframes spin{to{transform:rotate(360deg)}}
.timing{font-size:11px;color:var(--muted);flex:1;text-align:right}

/* RESULT */
.result-section{border-top:1px solid var(--border);padding-top:10px}
.meta-chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:8px}
.chip{background:var(--blue-light);color:var(--blue-dark);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:600;border:1px solid var(--border)}
.result-wrap{border:1px solid var(--border);border-radius:8px;overflow:auto;max-height:280px}
table{width:100%;border-collapse:collapse;font-size:11.5px}
th{background:var(--blue-light);color:var(--blue-dark);font-weight:700;text-align:left;padding:8px 12px;border-bottom:2px solid var(--border);position:sticky;top:0}
td{padding:7px 12px;border-bottom:1px solid #ecf2ff;font-family:'Courier New',monospace;color:var(--text)}
tr:hover td{background:#f5f9ff}
.error-box{background:#fff0f0;border:1.5px solid #ffc0c0;border-radius:8px;padding:12px;color:#c0392b;font-size:11.5px;font-family:monospace;white-space:pre-wrap;line-height:1.5}
.empty-hint{text-align:center;padding:28px;color:var(--muted);font-size:12px;line-height:1.8}
.empty-hint strong{display:block;font-size:14px;color:var(--text);margin-bottom:4px}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">
      <svg viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
    </div>
    <div>
      <div class="header-title">Spark Distributed Query UI</div>
      <div class="header-sub">Truy vấn song song trên 2 Docker Workers — IS211 Distributed Data</div>
    </div>
  </div>
  <div class="nav-links">
    <a href="http://localhost:8080" target="_blank">⚡ Master UI :8080</a>
    <a href="http://localhost:4041" target="_blank">🔍 Spark App :4041</a>
    <a href="http://localhost:8081" target="_blank">🐳 Worker 1 :8081</a>
    <a href="http://localhost:8082" target="_blank">🐳 Worker 2 :8082</a>
  </div>
</div>

<div class="workspace">

  <!-- DOCKER 1 PANEL -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title-row">
        <div class="docker-badge">
          <div class="dot"></div>
          <div>
            <div class="docker-label">🐳 Docker Worker 1</div>
            <div class="docker-sub">spark-worker-1 · port 8081</div>
          </div>
        </div>
      </div>
      <div class="preset-row" id="presets1">
        {% for key, q in queries_w1.items() %}
        <button class="preset-btn" onclick="selectPreset(1,'{{key}}')" id="p1_{{key}}">{{q.title}}</button>
        {% endfor %}
      </div>
    </div>
    <div class="panel-body">
      <div id="desc1" class="desc-box"></div>
      <textarea id="sql1" class="sql-editor" placeholder="Nhập SQL query hoặc chọn preset bên trên...&#10;&#10;Ví dụ: SELECT * FROM h_site_1 LIMIT 10&#10;&#10;Ctrl+Enter để chạy"></textarea>
      <div class="action-row">
        <button class="run-btn" id="btn1" onclick="runQuery(1)">
          <div class="spinner"></div>
          <span class="label">▶ Chạy Query</span>
        </button>
        <span class="timing" id="timing1"></span>
      </div>
      <div id="result1"></div>
    </div>
  </div>

  <!-- DOCKER 2 PANEL -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title-row">
        <div class="docker-badge">
          <div class="dot"></div>
          <div>
            <div class="docker-label">🐳 Docker Worker 2</div>
            <div class="docker-sub">spark-worker-2 · port 8082</div>
          </div>
        </div>
      </div>
      <div class="preset-row" id="presets2">
        {% for key, q in queries_w2.items() %}
        <button class="preset-btn" onclick="selectPreset(2,'{{key}}')" id="p2_{{key}}">{{q.title}}</button>
        {% endfor %}
      </div>
    </div>
    <div class="panel-body">
      <div id="desc2" class="desc-box"></div>
      <textarea id="sql2" class="sql-editor" placeholder="Nhập SQL query hoặc chọn preset bên trên...&#10;&#10;Ví dụ: SELECT * FROM v_site_2 LIMIT 10&#10;&#10;Ctrl+Enter để chạy"></textarea>
      <div class="action-row">
        <button class="run-btn" id="btn2" onclick="runQuery(2)">
          <div class="spinner"></div>
          <span class="label">▶ Chạy Query</span>
        </button>
        <span class="timing" id="timing2"></span>
      </div>
      <div id="result2"></div>
    </div>
  </div>

</div>

<script>
const queriesW1 = {{ w1_json|safe }};
const queriesW2 = {{ w2_json|safe }};

function selectPreset(panel, key) {
  const bank = panel === 1 ? queriesW1 : queriesW2;
  const q = bank[key];
  document.getElementById('sql'+panel).value = q.sql;
  const desc = document.getElementById('desc'+panel);
  desc.style.display = 'block';
  desc.textContent = q.desc;
  document.querySelectorAll('#presets'+panel+' .preset-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('p'+panel+'_'+key).classList.add('active');
}

async function runQuery(panel) {
  const sql = document.getElementById('sql'+panel).value.trim();
  if (!sql) return;
  const btn = document.getElementById('btn'+panel);
  const timing = document.getElementById('timing'+panel);
  const resultDiv = document.getElementById('result'+panel);
  btn.classList.add('loading'); btn.disabled = true;
  timing.textContent = 'Đang thực thi...';
  resultDiv.innerHTML = '<div class="empty-hint">⏳ Đang chạy trên cluster...</div>';
  try {
    const res = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({sql})});
    const data = await res.json();
    if (data.error) {
      resultDiv.innerHTML = '<div class="result-section"><div class="error-box">'+data.error+'</div></div>';
      timing.textContent = '';
    } else {
      let html = '<div class="result-section"><div class="meta-chips">';
      html += '<span class="chip">'+data.row_count+' rows</span>';
      html += '<span class="chip">'+data.columns.length+' cols</span>';
      html += '<span class="chip">'+data.duration_ms+' ms</span>';
      html += '</div><div class="result-wrap"><table><thead><tr>';
      data.columns.forEach(c => html += '<th>'+c+'</th>');
      html += '</tr></thead><tbody>';
      data.rows.forEach(r => { html += '<tr>'; r.forEach(v => html += '<td>'+(v===null?'<em style="color:#aaa">NULL</em>':v)+'</td>'); html += '</tr>'; });
      html += '</tbody></table></div></div>';
      resultDiv.innerHTML = html;
      timing.innerHTML = '✅ Xong &nbsp;·&nbsp; <a href="http://localhost:4041/SQL/" target="_blank" style="color:var(--blue)">Spark UI →</a>';
    }
  } catch(e) {
    resultDiv.innerHTML = '<div class="result-section"><div class="error-box">Network error: '+e.message+'</div></div>';
    timing.textContent = '';
  }
  btn.classList.remove('loading'); btn.disabled = false;
}

document.getElementById('sql1').addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') runQuery(1); });
document.getElementById('sql2').addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') runQuery(2); });
</script>
</body>
</html>
"""

def init_spark():
    global spark, views_ready
    spark = SparkSession.builder \
        .appName('WebQueryUI') \
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print(f'Spark UI: {spark.sparkContext.uiWebUrl}')

    OUTPUT_DIR = '/app/spark_output'
    H_DIR = os.path.join(OUTPUT_DIR, 'fragments_horizontal')
    V_DIR = os.path.join(OUTPUT_DIR, 'fragments_vertical')
    CSV_PATH = '/app/ECommerce_consumer behaviour.csv'

    cleaned = os.path.join(OUTPUT_DIR, 'cleaned_data.parquet')
    if os.path.exists(cleaned):
        df = spark.read.parquet(cleaned)
    else:
        df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
    df.createOrReplaceTempView('global_relation')

    if os.path.exists(os.path.join(H_DIR, 'site_1')):
        spark.read.parquet(os.path.join(H_DIR, 'site_1')).createOrReplaceTempView('h_site_1')
        spark.read.parquet(os.path.join(H_DIR, 'site_2')).createOrReplaceTempView('h_site_2')
    else:
        site1_depts = ['produce','dairy eggs','snacks','beverages','frozen','pantry']
        df.filter(F.col('department').isin(site1_depts)).createOrReplaceTempView('h_site_1')
        df.filter(~F.col('department').isin(site1_depts)).createOrReplaceTempView('h_site_2')

    if os.path.exists(os.path.join(V_DIR, 'site_1_order_info')):
        spark.read.parquet(os.path.join(V_DIR, 'site_1_order_info')).createOrReplaceTempView('v_site_1')
        spark.read.parquet(os.path.join(V_DIR, 'site_2_product_info')).createOrReplaceTempView('v_site_2')
    else:
        df_id = df.withColumn('_row_id', F.monotonically_increasing_id())
        df_id.select('_row_id','order_id','user_id','product_id','order_number','order_dow','order_hour_of_day','days_since_prior_order').createOrReplaceTempView('v_site_1')
        df_id.select('_row_id','order_id','user_id','product_id','add_to_cart_order','reordered','department_id','department','product_name').createOrReplaceTempView('v_site_2')

    views_ready = True
    print('All views registered!')


@app.route('/')
def index():
    w1_tmpl = {k: type('Q', (), v) for k, v in PRESET_W1.items()}
    w2_tmpl = {k: type('Q', (), v) for k, v in PRESET_W2.items()}
    return render_template_string(
        HTML_TEMPLATE,
        queries_w1=w1_tmpl,
        queries_w2=w2_tmpl,
        w1_json=json.dumps(PRESET_W1),
        w2_json=json.dumps(PRESET_W2),
    )


@app.route('/query', methods=['POST'])
def run_query():
    if not views_ready:
        return jsonify({'error': 'Spark is still initializing...'}), 503
    sql = request.json.get('sql', '').strip()
    if not sql:
        return jsonify({'error': 'Empty query'})
    try:
        t0 = time.time()
        result = spark.sql(sql)
        rows = result.limit(200).collect()
        duration = int((time.time() - t0) * 1000)
        return jsonify({
            'columns': result.columns,
            'rows': [[str(cell) for cell in row] for row in rows],
            'row_count': len(rows),
            'duration_ms': duration
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    init_spark()
    app.run(host='0.0.0.0', port=5000, debug=False)
