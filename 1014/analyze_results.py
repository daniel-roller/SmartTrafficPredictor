import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import Config

class ResultsAnalyzer:
    """結果分析器"""
    
    def __init__(self):
        self.config = Config()
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.style.use('seaborn-v0_8')
        
    def load_results(self):
        """載入訓練結果"""
        results_path = os.path.join(self.config.RESULTS_DIR, "training_results.csv")
        if not os.path.exists(results_path):
            raise FileNotFoundError("找不到訓練結果檔案，請先執行訓練")
        
        return pd.read_csv(results_path)
    
    def create_horizon_comparison_plot(self, df):
        """建立預測長度比較圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance vs Prediction Horizon', fontsize=16, fontweight='bold')
        
        # RMSE比較
        pivot_rmse = df.pivot_table(values='test_rmse', index='horizon', columns='model_type', aggfunc='mean')
        pivot_rmse.plot(kind='line', ax=axes[0,0], marker='o', linewidth=2)
        axes[0,0].set_title('RMSE vs Horizon')
        axes[0,0].set_xlabel('Prediction Horizon (hours)')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # MAPE比較
        pivot_mape = df.pivot_table(values='test_mape', index='horizon', columns='model_type', aggfunc='mean')
        pivot_mape.plot(kind='line', ax=axes[0,1], marker='s', linewidth=2)
        axes[0,1].set_title('MAPE vs Horizon')
        axes[0,1].set_xlabel('Prediction Horizon (hours)')
        axes[0,1].set_ylabel('MAPE (%)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # 訓練時間比較
        pivot_time = df.pivot_table(values='train_time', index='horizon', columns='model_type', aggfunc='mean')
        pivot_time.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Training Time vs Horizon')
        axes[1,0].set_xlabel('Prediction Horizon (hours)')
        axes[1,0].set_ylabel('Training Time (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        
        # 模型綜合排名 (RMSE + MAPE的綜合分數)
        df_rank = df.copy()
        df_rank['combined_score'] = (df_rank['test_rmse'] / df_rank['test_rmse'].max() + 
                                    df_rank['test_mape'] / df_rank['test_mape'].max()) / 2
        
        pivot_score = df_rank.pivot_table(values='combined_score', index='horizon', columns='model_type', aggfunc='mean')
        pivot_score.plot(kind='line', ax=axes[1,1], marker='^', linewidth=2)
        axes[1,1].set_title('Combined Performance Score vs Horizon')
        axes[1,1].set_xlabel('Prediction Horizon (hours)')
        axes[1,1].set_ylabel('Combined Score (lower is better)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_DIR, "horizon_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def create_model_comparison_boxplot(self, df):
        """建立模型比較箱形圖"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE箱形圖
        sns.boxplot(data=df, x='model_type', y='test_rmse', ax=axes[0])
        axes[0].set_title('RMSE Distribution by Model Type')
        axes[0].set_ylabel('RMSE')
        
        # MAPE箱形圖
        sns.boxplot(data=df, x='model_type', y='test_mape', ax=axes[1])
        axes[1].set_title('MAPE Distribution by Model Type')
        axes[1].set_ylabel('MAPE (%)')
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_DIR, "model_comparison_boxplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def create_segment_performance_heatmap(self, df):
        """建立路段表現熱力圖"""
        # 取平均RMSE建立熱力圖
        pivot_data = df.pivot_table(values='test_rmse', 
                                   index='segment_name', 
                                   columns='model_type', 
                                   aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Average RMSE'})
        plt.title('Average RMSE by Segment and Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Segment Name')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_DIR, "segment_performance_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def generate_summary_report(self, df):
        """生成總結報告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SmartTrafficPredictor - 多執行緒訓練結果報告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 基本統計
        report_lines.append("📊 基本統計:")
        report_lines.append(f"  - 總模型數量: {len(df)}")
        report_lines.append(f"  - 路段數量: {df['segment_name'].nunique()}")
        report_lines.append(f"  - 模型類型: {', '.join(df['model_type'].unique())}")
        report_lines.append(f"  - 預測長度範圍: {df['horizon'].min()} ~ {df['horizon'].max()} 小時")
        report_lines.append("")
        
        # 最佳表現
        best_overall = df.loc[df['test_rmse'].idxmin()]
        report_lines.append("🏆 最佳整體表現:")
        report_lines.append(f"  - 模型: {best_overall['model_type']}")
        report_lines.append(f"  - 路段: {best_overall['segment_name']}")
        report_lines.append(f"  - 預測長度: {best_overall['horizon']} 小時")
        report_lines.append(f"  - RMSE: {best_overall['test_rmse']:.3f}")
        report_lines.append(f"  - MAPE: {best_overall['test_mape']:.2f}%")
        report_lines.append("")
        
        # 各模型平均表現
        model_avg = df.groupby('model_type')[['test_rmse', 'test_mape', 'train_time']].mean()
        report_lines.append("📈 各模型平均表現:")
        for model_type in model_avg.index:
            row = model_avg.loc[model_type]
            report_lines.append(f"  - {model_type}:")
            report_lines.append(f"    * RMSE: {row['test_rmse']:.3f}")
            report_lines.append(f"    * MAPE: {row['test_mape']:.2f}%")
            report_lines.append(f"    * 平均訓練時間: {row['train_time']:.1f}秒")
        report_lines.append("")
        
        # 預測長度分析
        horizon_analysis = df.groupby('horizon')[['test_rmse', 'test_mape']].mean()
        report_lines.append("⏰ 預測長度分析:")
        for horizon in sorted(horizon_analysis.index):
            row = horizon_analysis.loc[horizon]
            horizon_name = self.get_horizon_name(horizon)
            report_lines.append(f"  - {horizon_name} ({horizon}h): RMSE={row['test_rmse']:.3f}, MAPE={row['test_mape']:.2f}%")
        report_lines.append("")
        
        # 建議
        report_lines.append("💡 建議:")
        best_short_term = df[df['horizon'] <= 72].groupby('model_type')['test_rmse'].mean().idxmin()
        best_long_term = df[df['horizon'] >= 720].groupby('model_type')['test_rmse'].mean().idxmin()
        fastest_training = df.groupby('model_type')['train_time'].mean().idxmin()
        
        report_lines.append(f"  - 短期預測 (≤3天): 推薦使用 {best_short_term}")
        report_lines.append(f"  - 長期預測 (≥1月): 推薦使用 {best_long_term}")
        report_lines.append(f"  - 快速訓練需求: 推薦使用 {fastest_training}")
        report_lines.append("")
        
        # 儲存報告
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.config.RESULTS_DIR, "summary_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(report_content)
        return report_path
    
    def get_horizon_name(self, hours):
        """將小時數轉換為易讀的名稱"""
        if hours == 24:
            return "1天"
        elif hours == 72:
            return "3天" 
        elif hours == 168:
            return "1週"
        elif hours == 720:
            return "1月"
        elif hours == 4320:
            return "半年"
        else:
            return f"{hours}小時"
    
    def run_full_analysis(self):
        """執行完整分析"""
        print("📊 開始結果分析...")
        
        # 載入結果
        df = self.load_results()
        print(f"📦 載入了 {len(df)} 筆訓練結果")
        
        # 建立各種圖表
        print("📈 建立預測長度比較圖...")
        horizon_plot = self.create_horizon_comparison_plot(df)
        
        print("📊 建立模型比較箱形圖...")
        boxplot = self.create_model_comparison_boxplot(df)
        
        print("🗺️ 建立路段表現熱力圖...")
        heatmap = self.create_segment_performance_heatmap(df)
        
        print("📝 生成總結報告...")
        report = self.generate_summary_report(df)
        
        print("\n✅ 分析完成！生成的檔案:")
        print(f"  - {horizon_plot}")
        print(f"  - {boxplot}")
        print(f"  - {heatmap}")
        print(f"  - {report}")

def main():
    analyzer = ResultsAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()