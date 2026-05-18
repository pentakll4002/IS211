import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import os

def run_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    _plot_top_departments(df, output_dir)
    _plot_shopping_days(df, output_dir)
    _plot_hour_of_day(df, output_dir)
    _plot_reorder_rate(df, output_dir)
    _plot_top_products(df, output_dir)

    print(f'EDA charts saved to {output_dir}')

def _plot_top_departments(df, output_dir):
    dept = df.groupBy('department').agg(
        F.count('product_id').alias('count')
    ).orderBy(F.desc('count')).limit(15).toPandas()

    plt.figure(figsize=(12, 6))
    plt.barh(dept['department'], dept['count'], color='steelblue')
    plt.xlabel('Count')
    plt.title('Top 15 Departments')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_top_departments.png'), dpi=100)
    plt.close()

def _plot_shopping_days(df, output_dir):
    dow = df.groupBy('order_dow').agg(
        F.countDistinct('order_id').alias('orders')
    ).orderBy('order_dow').toPandas()

    day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    dow['day'] = dow['order_dow'].map(lambda x: day_names[int(x) % 7])

    plt.figure(figsize=(10, 5))
    plt.bar(dow['day'], dow['orders'], color='coral')
    plt.title('Orders by Day of Week')
    plt.ylabel('Number of Orders')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_shopping_days.png'), dpi=100)
    plt.close()

def _plot_hour_of_day(df, output_dir):
    hour = df.groupBy('order_hour_of_day').agg(
        F.countDistinct('order_id').alias('orders')
    ).orderBy('order_hour_of_day').toPandas()

    plt.figure(figsize=(12, 5))
    plt.plot(hour['order_hour_of_day'], hour['orders'], 'o-', color='green')
    plt.title('Orders by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Orders')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_hour_of_day.png'), dpi=100)
    plt.close()

def _plot_reorder_rate(df, output_dir):
    dept_reorder = df.groupBy('department').agg(
        F.mean('reordered').alias('reorder_rate')
    ).orderBy(F.desc('reorder_rate')).toPandas()

    plt.figure(figsize=(12, 6))
    plt.barh(dept_reorder['department'], dept_reorder['reorder_rate'], color='mediumpurple')
    plt.xlabel('Reorder Rate')
    plt.title('Reorder Rate by Department')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_reorder_rate.png'), dpi=100)
    plt.close()

def _plot_top_products(df, output_dir):
    top = df.groupBy('product_name').agg(
        F.count('*').alias('count')
    ).orderBy(F.desc('count')).limit(20).toPandas()

    plt.figure(figsize=(12, 7))
    plt.barh(top['product_name'], top['count'], color='teal')
    plt.xlabel('Count')
    plt.title('Top 20 Products')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_top_products.png'), dpi=100)
    plt.close()
