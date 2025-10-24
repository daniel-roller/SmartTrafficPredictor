# -*- coding: utf-8 -*-
"""
交通流量預測系統 - 精簡版評估器 (路徑修正版)
只生成整體比較結果，不產生大量單獨圖片
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from config import config
from utils import print_subsection_header

# 修正中文字體設定
def setup_chinese_fonts():
    """設定中文字體"""
    print("🔤 設定中文字體...")
    
    font_candidates = [
        'Microsoft YaHei',      # 微軟雅黑
        'Microsoft JhengHei',   # 微軟正黑體
        'SimHei',               # 黑體
        'Microsoft YaHei UI',   # 微軟雅黑UI
        'Arial Unicode MS',     # Arial Unicode
        'DejaVu Sans'          # 備用字體
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
        print(f"✅ 使用字體: {selected_font}")
    else:
        matplotlib.rcParams['font.sans-serif'] = ['sans-serif']
        print("⚠️ 使用系統預設字體")
    
    matplotlib.rcParams.update({
        'axes.unicode_minus': False,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

# 設定中文字體
setup_chinese_fonts()
plt.ioff()
sns.set_style('whitegrid')

class ModelEvaluator:
    """精簡版模型評估器 - 路徑修正版"""
    
    def __init__(self):
        self.results = {}
        setup_chinese_fonts()
        
        # 確保輸出目錄存在
        os.makedirs(config.PLOTS_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        print(f"📁 評估器初始化完成")
        print(f"   📊 圖片輸出: {config.PLOTS_DIR}")
        print(f"   📄 報告輸出: {config.RESULTS_DIR}")
        
    def create_overall_comparison_dashboard(self, all_results: Dict) -> None:
        """建立整體比較儀表板 - 單一綜合圖表"""
        print_subsection_header("📊 生成整體模型比較儀表板")
        
        # 準備數據
        station_data = []
        for dataset_name, results in all_results.items():
            valid_results = {name: result for name, result in results.items() 
                           if result.get('test_metrics')}
            if valid_results:
                for model_name, result in valid_results.items():
                    metrics = result['test_metrics']
                    station_data.append({
                        'station': dataset_name,
                        'model': model_name,
                        'r2': metrics['R²'],
                        'rmse': metrics['RMSE'],
                        'mae': metrics['MAE'],
                        'mape': metrics['MAPE'],
                        'training_time': result['training_time'],
                        'features': result['feature_count']
                    })
        
        if not station_data:
            print("❌ 沒有有效數據")
            return
        
        df = pd.DataFrame(station_data)
        
        # 建立 2x3 綜合儀表板
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🚗 交通流量預測模型整體比較分析儀表板', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. 各測站最佳R²比較 (左上)
        best_r2_by_station = df.loc[df.groupby('station')['r2'].idxmax()]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(best_r2_by_station)]
        
        bars = axes[0,0].bar(range(len(best_r2_by_station)), best_r2_by_station['r2'], 
                            color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        axes[0,0].set_title('📊 各測站最佳R²表現', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('R² 分數', fontsize=12)
        axes[0,0].set_xticks(range(len(best_r2_by_station)))
        axes[0,0].set_xticklabels(best_r2_by_station['station'], rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤和最佳模型
        for i, (bar, row) in enumerate(zip(bars, best_r2_by_station.itertuples())):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{row.r2:.3f}\n({row.model})', ha='center', va='bottom', 
                          fontsize=9, fontweight='bold')
        
        # 2. 模型整體性能比較 (右上)
        model_stats = df.groupby('model').agg({
            'r2': ['mean', 'std'],
            'rmse': 'mean',
            'mae': 'mean'
        }).round(4)
        
        models = model_stats.index
        mean_r2 = model_stats[('r2', 'mean')].values
        std_r2 = model_stats[('r2', 'std')].values
        
        bars = axes[0,1].bar(models, mean_r2, yerr=std_r2, capsize=5,
                            color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)], 
                            alpha=0.8, edgecolor='white', linewidth=1)
        axes[0,1].set_title('🏆 模型平均性能對比', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('平均 R² ± 標準差', fontsize=12)
        axes[0,1].tick_params(axis='x', rotation=15)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, mean, std in zip(bars, mean_r2, std_r2):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                          f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                          fontsize=10, fontweight='bold')
        
        # 3. 效能分佈散點圖 (左下)
        model_colors = {'Ridge': '#3498db', 'RandomForest': '#2ecc71', 'XGBoost': '#e74c3c'}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            axes[1,0].scatter(model_data['training_time'], model_data['r2'], 
                             s=model_data['features']*3, alpha=0.7,
                             color=model_colors.get(model, '#7f8c8d'), 
                             label=model, edgecolors='white', linewidth=1)
        
        axes[1,0].set_title('⚡ 效能 vs 效率分析', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('訓練時間 (秒)', fontsize=12)
        axes[1,0].set_ylabel('R² 分數', fontsize=12)
        axes[1,0].legend(title='模型類型')
        axes[1,0].grid(True, alpha=0.3)
        
        # 添加理想區域標示
        axes[1,0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='優秀門檻')
        axes[1,0].text(0.02, 0.98, '💡 理想區域:\n左上角', 
                      transform=axes[1,0].transAxes, va='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        # 4. 誤差指標熱力圖 (右下)
        try:
            pivot_rmse = df.pivot(index='station', columns='model', values='rmse')
            im = axes[1,1].imshow(pivot_rmse.values, cmap='Reds_r', aspect='auto')
            
            axes[1,1].set_title('🔥 RMSE誤差熱力圖', fontsize=14, fontweight='bold')
            axes[1,1].set_xticks(range(len(pivot_rmse.columns)))
            axes[1,1].set_xticklabels(pivot_rmse.columns, rotation=45)
            axes[1,1].set_yticks(range(len(pivot_rmse.index)))
            axes[1,1].set_yticklabels(pivot_rmse.index)
            
            # 添加數值標籤
            for i in range(len(pivot_rmse.index)):
                for j in range(len(pivot_rmse.columns)):
                    text = axes[1,1].text(j, i, f'{pivot_rmse.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontweight='bold')
            
            # 添加顏色條
            plt.colorbar(im, ax=axes[1,1], shrink=0.8)
        except Exception as e:
            print(f"⚠️ 熱力圖生成失敗: {e}")
            axes[1,1].text(0.5, 0.5, '熱力圖生成失敗', ha='center', va='center', 
                          transform=axes[1,1].transAxes)
        
        # 5. 綜合雷達圖 (右中)
        categories = ['準確度', '速度', '穩定性', '簡潔性']
        
        # 計算每個模型的綜合指標
        radar_data = {}
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            max_r2 = df['r2'].max() if df['r2'].max() > 0 else 1
            max_time = df['training_time'].max() if df['training_time'].max() > 0 else 1
            max_features = df['features'].max() if df['features'].max() > 0 else 1
            
            radar_data[model] = [
                model_data['r2'].mean() / max_r2,  # 準確度
                1 - (model_data['training_time'].mean() / max_time),  # 速度
                1 - (model_data['rmse'].std() / model_data['rmse'].mean()) if model_data['rmse'].mean() > 0 else 0,  # 穩定性
                1 - (model_data['features'].mean() / max_features)  # 簡潔性
            ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 3, 5, projection='polar')
        
        for i, (model, values) in enumerate(radar_data.items()):
            values += values[:1]
            color = list(model_colors.values())[i] if i < len(model_colors) else '#7f8c8d'
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax_radar.fill(angles, values, alpha=0.25, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('🎯 綜合性能雷達圖', fontsize=14, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 6. 成功率統計 (右上)
        # 計算統計數據
        total_experiments = len(df)
        excellent_count = len(df[df['r2'] >= 0.8])
        good_count = len(df[(df['r2'] >= 0.6) & (df['r2'] < 0.8)])
        fair_count = len(df[(df['r2'] >= 0.3) & (df['r2'] < 0.6)])
        poor_count = len(df[df['r2'] < 0.3])
        
        # 圓餅圖
        sizes = [excellent_count, good_count, fair_count, poor_count]
        labels = ['優秀(≥0.8)', '良好(0.6-0.8)', '普通(0.3-0.6)', '待改善(<0.3)']
        colors_pie = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        
        # 過濾掉 0 值
        filtered_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors_pie) if size > 0]
        if filtered_sizes:
            sizes_filtered, labels_filtered, colors_filtered = zip(*filtered_sizes)
            wedges, texts, autotexts = axes[0,2].pie(sizes_filtered, labels=labels_filtered, 
                                                    colors=colors_filtered, autopct='%1.1f%%', startangle=90)
        axes[0,2].set_title('📈 模型效能分佈', fontsize=14, fontweight='bold')
        
        # 添加統計表格
        stats_text = f"""
📊 實驗統計摘要:
• 總實驗數: {total_experiments}
• 最佳R²: {df['r2'].max():.4f}
• 平均R²: {df['r2'].mean():.4f}
• 最低RMSE: {df['rmse'].min():.2f}
• 最快訓練: {df['training_time'].min():.2f}s
        """
        
        axes[1,2].text(0.1, 0.5, stats_text.strip(), transform=axes[1,2].transAxes,
                      fontsize=12, va='center',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
        axes[1,2].axis('off')
        
        plt.tight_layout()
        
        # 儲存儀表板 - 使用修正的路徑
        dashboard_path = os.path.join(config.PLOTS_DIR, "整體模型比較儀表板.png")
        
        try:
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 整體儀表板已儲存: {dashboard_path}")
            
            # 建立簡要說明
            self.create_dashboard_description(dashboard_path)
            
        except Exception as e:
            print(f"❌ 儀表板儲存失敗: {e}")
            print(f"   嘗試儲存到: {dashboard_path}")
        
        plt.close(fig)
    
    def create_dashboard_description(self, image_path: str) -> None:
        """建立儀表板說明文件"""
        desc_path = image_path.replace('.png', '_說明.txt')
        
        description = f"""
📊 交通流量預測模型整體比較儀表板說明
{'=' * 60}

📋 基本資訊:
- 圖表類型: 綜合比較儀表板
- 生成時間: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- 目的: 一頁總覽所有實驗結果

📊 儀表板內容說明:

1️⃣ 各測站最佳R²表現 (左上):
   - 顯示每個測站的最佳預測準確度
   - 柱狀圖標示使用的最佳模型名稱
   - 顏色區分不同測站

2️⃣ 模型平均性能對比 (右上):
   - 比較三種模型的整體表現
   - 誤差棒顯示性能穩定性
   - 數值標示平均值±標準差

3️⃣ 效能vs效率分析 (左下):
   - 散點圖展示訓練時間與準確度關係
   - 氣泡大小代表模型複雜度
   - 理想區域在左上角(高準確度+短時間)

4️⃣ RMSE誤差熱力圖 (中下):
   - 顏色深淺表示預測誤差大小
   - 淺色=低誤差(好), 深色=高誤差(差)
   - 可快速識別最佳模型-測站組合

5️⃣ 綜合性能雷達圖 (右中):
   - 四個維度綜合評估各模型
   - 面積越大表示綜合表現越好
   - 平衡考慮準確度、速度、穩定性、簡潔性

6️⃣ 模型效能分佈 (右上圓餅圖):
   - 統計各性能等級的實驗比例
   - 綠色=優秀, 橙色=良好, 紅色=需改善

📈 使用指南:
• 快速識別: 看左上角找出各測站最佳模型
• 模型選擇: 看右上角選擇整體最佳模型
• 效率評估: 看左下角平衡準確度與效率
• 細節分析: 看熱力圖找出具體數值

💡 決策建議:
- 整體最佳模型: 右上圖平均R²最高者
- 效率最佳模型: 左下圖左上區域模型
- 穩定最佳模型: 右上圖誤差棒最小者
- 綜合最佳模型: 雷達圖面積最大者

{'=' * 60}
※ 本儀表板包含所有關鍵資訊，無需查看其他圖表
"""
        
        try:
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description)
            print(f"📝 儀表板說明已儲存: {desc_path}")
        except Exception as e:
            print(f"⚠️ 說明文件儲存失敗: {e}")
    
    def generate_summary_report(self, all_results: Dict) -> str:
        """生成精簡摘要報告"""
        print_subsection_header("📝 生成精簡摘要報告")
        
        # 準備數據
        station_data = {}
        all_experiments = []
        
        for dataset_name, results in all_results.items():
            valid_results = {name: result for name, result in results.items() 
                           if result.get('test_metrics')}
            if valid_results:
                best_model = max(valid_results.keys(), 
                               key=lambda x: valid_results[x]['test_metrics']['R²'])
                best_result = valid_results[best_model]
                
                station_data[dataset_name] = {
                    'best_model': best_model,
                    'best_r2': best_result['test_metrics']['R²'],
                    'best_rmse': best_result['test_metrics']['RMSE'],
                    'best_mae': best_result['test_metrics']['MAE']
                }
                
                # 收集所有實驗數據
                for model_name, result in valid_results.items():
                    all_experiments.append({
                        'station': dataset_name,
                        'model': model_name,
                        'r2': result['test_metrics']['R²'],
                        'rmse': result['test_metrics']['RMSE'],
                        'time': result['training_time']
                    })
        
        if not all_experiments:
            print("❌ 沒有有效實驗數據")
            return ""
        
        # 計算整體統計
        df = pd.DataFrame(all_experiments)
        model_stats = df.groupby('model')['r2'].agg(['mean', 'std', 'count']).round(4)
        
        # 生成報告
        report_lines = [
            "# 🚗 交通流量預測模型比較研究報告 (精簡版)",
            "",
            "## 📋 研究摘要",
            "",
            f"本研究對 {len(station_data)} 個測站的交通流量進行預測模型比較，",
            f"總共完成 {len(all_experiments)} 個實驗，使用 3 種機器學習模型。",
            "",
            "### 🎯 主要發現",
            "",
            f"- **最佳整體模型**: {model_stats['mean'].idxmax()} (平均R² = {model_stats['mean'].max():.4f})",
            f"- **最穩定模型**: {model_stats['std'].idxmin()} (標準差 = {model_stats['std'].min():.4f})",
            f"- **優秀實驗比例**: {len(df[df['r2'] >= 0.8]) / len(df) * 100:.1f}% (R² ≥ 0.8)",
            "",
            "---",
            "",
            "## 📊 各測站最佳結果總覽",
            "",
            "| 測站名稱 | 最佳模型 | R² 分數 | RMSE | MAE | 效能等級 |",
            "|----------|----------|---------|------|-----|----------|"
        ]
        
        for station, data in station_data.items():
            r2 = data['best_r2']
            if r2 >= 0.8:
                grade = "🟢 優秀"
            elif r2 >= 0.6:
                grade = "🟡 良好"
            elif r2 >= 0.3:
                grade = "🟠 普通"
            else:
                grade = "🔴 待改善"
            
            report_lines.append(
                f"| {station} | {data['best_model']} | "
                f"{data['best_r2']:.4f} | {data['best_rmse']:.2f} | "
                f"{data['best_mae']:.2f} | {grade} |"
            )
        
        report_lines.extend([
            "",
            "## 🏆 模型整體性能統計",
            "",
            "| 模型名稱 | 平均R² | 標準差 | 實驗次數 | 整體評價 |",
            "|----------|--------|--------|----------|----------|"
        ])
        
        for model_name, stats in model_stats.iterrows():
            mean_r2 = stats['mean']
            std_r2 = stats['std']
            
            if mean_r2 >= 0.7 and std_r2 <= 0.1:
                evaluation = "🥇 優秀穩定"
            elif mean_r2 >= 0.6:
                evaluation = "🥈 表現良好"
            elif mean_r2 >= 0.4:
                evaluation = "🥉 尚可接受"
            else:
                evaluation = "❌ 需要改善"
            
            report_lines.append(
                f"| {model_name} | {mean_r2:.4f} | {std_r2:.4f} | "
                f"{int(stats['count'])} | {evaluation} |"
            )
        
        report_lines.extend([
            "",
            "## 💡 結論與建議",
            "",
            "### 🎯 實務應用建議",
            f"1. **推薦模型**: {model_stats['mean'].idxmax()} - 整體表現最佳",
            f"2. **穩定選擇**: {model_stats['std'].idxmin()} - 最穩定可靠",
            "",
            "### 📈 部署優先級",
        ])
        
        # 根據結果給出部署建議
        high_perf_stations = [name for name, data in station_data.items() if data['best_r2'] >= 0.7]
        medium_perf_stations = [name for name, data in station_data.items() if 0.4 <= data['best_r2'] < 0.7]
        
        report_lines.extend([
            f"- **優先部署** ({len(high_perf_stations)}個): {', '.join(high_perf_stations) if high_perf_stations else '無'}",
            f"- **次要部署** ({len(medium_perf_stations)}個): {', '.join(medium_perf_stations) if medium_perf_stations else '無'}",
            "",
            "### 📊 關鍵指標",
            f"- 整體成功率: {len(df[df['r2'] >= 0.6]) / len(df) * 100:.1f}%",
            f"- 平均準確度: R² = {df['r2'].mean():.4f}",
            f"- 最佳表現: R² = {df['r2'].max():.4f}",
            "",
            "---",
            "",
            f"📅 **報告生成時間**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
            f"📊 **視覺化圖表**: [整體模型比較儀表板.png]({os.path.join(config.PLOTS_DIR, '整體模型比較儀表板.png')})",
            "",
            "*本精簡報告包含所有關鍵發現，詳細數據請參考CSV檔案*"
        ])
        
        # 儲存報告 - 使用修正的路徑
        report_content = "\n".join(report_lines)
        report_path = os.path.join(config.RESULTS_DIR, "模型比較摘要報告.md")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 摘要報告已儲存: {report_path}")
        except Exception as e:
            print(f"❌ 報告儲存失敗: {e}")
            print(f"   嘗試儲存到: {report_path}")
        
        return report_content
    
    def save_detailed_csv(self, all_results: Dict) -> None:
        """儲存詳細CSV結果"""
        print("💾 儲存詳細CSV結果...")
        
        csv_data = []
        for dataset_name, results in all_results.items():
            for model_name, result in results.items():
                if result.get('test_metrics'):
                    metrics = result['test_metrics']
                    
                    # 效能等級判定
                    r2_score = metrics['R²']
                    if r2_score >= 0.8:
                        grade = "優秀"
                    elif r2_score >= 0.6:
                        grade = "良好"
                    elif r2_score >= 0.3:
                        grade = "普通"
                    else:
                        grade = "待改善"
                    
                    csv_data.append({
                        '測站名稱': dataset_name,
                        '模型名稱': model_name,
                        'R²分數': round(metrics['R²'], 6),
                        'RMSE': round(metrics['RMSE'], 4),
                        'MAE': round(metrics['MAE'], 4),
                        'MAPE': round(metrics['MAPE'], 4),
                        '訓練時間_秒': round(result['training_time'], 4),
                        '特徵數量': result['feature_count'],
                        '效能等級': grade,
                        '實驗時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        if csv_data:
            csv_path = os.path.join(config.RESULTS_DIR, "模型比較結果.csv")
            
            try:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ 詳細CSV已儲存: {csv_path}")
                print(f"   📊 包含 {len(csv_data)} 筆實驗記錄")
            except Exception as e:
                print(f"❌ CSV儲存失敗: {e}")
                print(f"   嘗試儲存到: {csv_path}")
    
    def evaluate_all_results(self, all_results: Dict) -> None:
        """執行完整評估 - 精簡版 (只生成必要檔案)"""
        print_subsection_header("📊 執行模型評估 (精簡版)")
        
        # 檢查數據
        valid_datasets = []
        for dataset_name, results in all_results.items():
            valid_results = {name: result for name, result in results.items() 
                           if result.get('test_metrics')}
            if valid_results:
                valid_datasets.append(dataset_name)
        
        if not valid_datasets:
            print("❌ 沒有有效的實驗結果")
            return
        
        print(f"✅ 找到 {len(valid_datasets)} 個有效測站結果")
        print(f"📁 輸出目錄: {config.EXPERIMENT_DIR}")
        
        # 強制重新設定字體
        setup_chinese_fonts()
        
        # 只生成一個綜合儀表板
        try:
            self.create_overall_comparison_dashboard(all_results)
        except Exception as e:
            print(f"⚠️ 儀表板生成失敗: {e}")
        
        # 儲存CSV數據
        try:
            self.save_detailed_csv(all_results)
        except Exception as e:
            print(f"⚠️ CSV儲存失敗: {e}")
        
        # 生成精簡報告
        try:
            self.generate_summary_report(all_results)
        except Exception as e:
            print(f"⚠️ 報告生成失敗: {e}")
        
        # 顯示結果摘要
        print("\n" + "="*70)
        print("🎉 精簡版評估完成！")
        print("="*70)
        print("📁 產出檔案 (精簡版):")
        print("   📊 整體模型比較儀表板.png - 一頁總覽所有結果")
        print("   📝 儀表板說明文件")
        print("   📋 模型比較摘要報告.md - 關鍵發現總結") 
        print("   💾 模型比較結果.csv - 詳細數據表格")
        
        print(f"\n📂 檔案位置:")
        print(f"   🖼️  圖片: {config.PLOTS_DIR}")
        print(f"   📄 報告: {config.RESULTS_DIR}")
        print(f"   📁 實驗: {config.EXPERIMENT_DIR}")
        
        # 快速摘要
        station_count = len(valid_datasets)
        total_experiments = sum(len([r for r in results.values() if r.get('test_metrics')]) 
                              for results in all_results.values())
        
        print(f"\n📈 實驗總結:")
        print(f"   🎯 測站數量: {station_count} 個")
        print(f"   🧪 總實驗數: {total_experiments} 個")
        print(f"   📊 平均每站: {total_experiments/station_count:.1f} 個實驗")
        
        print("\n✅ 精簡版評估完成！只需查看4個產出檔案即可獲得完整資訊。")
        print(f"🔍 快速檢查: 請前往 {config.EXPERIMENT_DIR} 查看所有產出檔案")