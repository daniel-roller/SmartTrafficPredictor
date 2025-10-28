import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 不顯示圖表，只儲存
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import Config

class ResultsAnalyzerNoDisplay:
    """結果分析器 - 不顯示圖表版本"""
    
    def __init__(self):
        self.config = Config()
        # 設定中文字體 (如果需要)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 路段名稱映射
        self.name_mapping = {
            'Highway1_North_WuJia_RuiLong': 'Highway 1 North',
            'Highway3_North_ZhongTou_WuRi': 'Highway 3 North',
            'Highway5_North_PengShan_Tunnel': 'Highway 5 North'
        }
        
        print("🔧 結果分析器初始化完成 (無顯示模式)")
    
    def load_results(self):
        """載入訓練結果"""
        results_file = os.path.join(self.config.RESULTS_DIR, "training_results.csv")
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"找不到結果檔案: {results_file}")
        
        df = pd.read_csv(results_file)
        
        # 簡化路段名稱顯示
        df['segment_display'] = df['segment_name'].map(self.name_mapping).fillna(df['segment_name'])
        
        print(f"📊 載入 {len(df)} 筆訓練結果")
        return df
    
    def create_performance_summary(self, df):
        """建立表現摘要"""
        print("📈 建立表現摘要...")
        
        # 按模型類型統計
        model_summary = df.groupby('model_type').agg({
            'test_rmse': ['mean', 'std', 'min', 'max'],
            'test_mape': ['mean', 'std', 'min', 'max'],
            'train_time': ['mean', 'sum']
        }).round(3)
        
        # 按路段統計
        segment_summary = df.groupby('segment_display').agg({
            'test_rmse': ['mean', 'std', 'min', 'max'],
            'test_mape': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # 儲存摘要
        summary_path = os.path.join(self.config.RESULTS_DIR, "performance_summary.csv")
        model_summary.to_csv(summary_path.replace('.csv', '_by_model.csv'))
        segment_summary.to_csv(summary_path.replace('.csv', '_by_segment.csv'))
        
        print(f"💾 表現摘要已儲存")
        return model_summary, segment_summary
    
    def create_model_comparison_plot(self, df):
        """建立模型比較圖 - 不顯示"""
        print("📊 建立模型比較圖...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE 比較
        sns.boxplot(data=df, x='model_type', y='test_rmse', ax=axes[0])
        axes[0].set_title('RMSE Distribution by Model Type')
        axes[0].set_ylabel('RMSE')
        
        # MAPE 比較
        sns.boxplot(data=df, x='model_type', y='test_mape', ax=axes[1])
        axes[1].set_title('MAPE Distribution by Model Type')
        axes[1].set_ylabel('MAPE (%)')
        
        plt.tight_layout()
        
        # 儲存但不顯示
        save_path = os.path.join(self.config.RESULTS_DIR, "model_comparison_boxplot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 關閉圖表，不顯示
        
        print(f"✅ 模型比較圖已儲存: {save_path}")
    
    def create_segment_heatmap(self, df):
        """建立路段表現熱力圖 - 不顯示"""
        print("🗺️ 建立路段表現熱力圖...")
        
        # 建立樞紐表
        pivot_rmse = df.pivot_table(
            values='test_rmse', 
            index='segment_display', 
            columns='model_type', 
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 建立熱力圖
        sns.heatmap(
            pivot_rmse, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlBu_r',
            ax=ax,
            cbar_kws={'label': 'Average RMSE'}
        )
        
        ax.set_title('Average RMSE by Segment and Model Type')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Segment Name')
        
        plt.tight_layout()
        
        # 儲存但不顯示
        save_path = os.path.join(self.config.RESULTS_DIR, "segment_performance_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 路段熱力圖已儲存: {save_path}")
    
    def create_training_time_analysis(self, df):
        """建立訓練時間分析 - 不顯示"""
        print("⏱️ 建立訓練時間分析...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 訓練時間 vs 表現
        scatter = axes[0].scatter(
            df['train_time'], 
            df['test_rmse'], 
            c=df['model_type'].astype('category').cat.codes,
            alpha=0.7
        )
        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_ylabel('Test RMSE')
        axes[0].set_title('Training Time vs Performance')
        
        # 模型訓練時間比較
        model_time = df.groupby('model_type')['train_time'].mean()
        axes[1].bar(model_time.index, model_time.values)
        axes[1].set_title('Average Training Time by Model')
        axes[1].set_ylabel('Training Time (seconds)')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config.RESULTS_DIR, "training_time_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 訓練時間分析已儲存: {save_path}")
    
    def generate_text_report(self, df, model_summary, segment_summary):
        """生成文字報告"""
        print("📝 生成綜合報告...")
        
        report_lines = [
            "🚦 SmartTrafficPredictor 分析報告",
            "=" * 50,
            f"📅 生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📊 總訓練結果: {len(df)} 個模型",
            "",
            "🏆 模型表現排名 (按平均RMSE):",
            "-" * 30
        ]
        
        # 模型排名
        model_ranking = model_summary['test_rmse']['mean'].sort_values()
        for rank, (model, rmse) in enumerate(model_ranking.items(), 1):
            mape = model_summary.loc[model, ('test_mape', 'mean')]
            time = model_summary.loc[model, ('train_time', 'mean')]
            report_lines.append(f"{rank}. {model}: RMSE={rmse:.2f}, MAPE={mape:.1f}%, 訓練時間={time:.0f}秒")
        
        report_lines.extend([
            "",
            "🛣️ 路段表現分析:",
            "-" * 20
        ])
        
        # 路段分析
        segment_ranking = segment_summary['test_rmse']['mean'].sort_values()
        for segment, rmse in segment_ranking.items():
            mape = segment_summary.loc[segment, ('test_mape', 'mean')]
            report_lines.append(f"• {segment}: RMSE={rmse:.2f}, MAPE={mape:.1f}%")
        
        # 找出表現最差的路段
        worst_segment = segment_ranking.index[-1]
        worst_rmse = segment_ranking.iloc[-1]
        
        report_lines.extend([
            "",
            "⚠️ 需要關注的問題:",
            "-" * 15,
            f"• {worst_segment} 表現較差 (RMSE: {worst_rmse:.2f})",
            "• 建議：檢查資料品質、增加特徵工程、調整模型參數",
            "",
            "💡 改進建議:",
            "-" * 10,
            "1. 對表現差的路段進行資料清理",
            "2. 增加更多時間特徵 (節假日、氣象等)",
            "3. 嘗試不同的窗口大小和預測長度",
            "4. 考慮集成學習方法",
            "",
            "📁 檔案位置:",
            f"• 模型檔案: {self.config.MODELS_DIR}",
            f"• 結果檔案: {self.config.RESULTS_DIR}",
            f"• 圖表檔案: {self.config.RESULTS_DIR}/*.png"
        ])
        
        # 儲存報告
        report_path = os.path.join(self.config.RESULTS_DIR, "analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 綜合報告已儲存: {report_path}")
        
        # 也印出重點摘要
        print("\n" + "=" * 50)
        print("📊 重點摘要:")
        print(f"🏆 最佳模型: {model_ranking.index[0]} (RMSE: {model_ranking.iloc[0]:.2f})")
        print(f"⚠️ 問題路段: {worst_segment} (RMSE: {worst_rmse:.2f})")
        print(f"📈 平均改善空間: {(worst_rmse - model_ranking.iloc[0]):.2f} RMSE points")
    
    def run_complete_analysis(self):
        """執行完整分析 - 不顯示圖表"""
        try:
            # 載入結果
            df = self.load_results()
            
            # 建立摘要
            model_summary, segment_summary = self.create_performance_summary(df)
            
            # 建立圖表 (不顯示)
            self.create_model_comparison_plot(df)
            self.create_segment_heatmap(df)
            self.create_training_time_analysis(df)
            
            # 生成報告
            self.generate_text_report(df, model_summary, segment_summary)
            
            print(f"\n🎉 分析完成！所有圖表已儲存至: {self.config.RESULTS_DIR}")
            
        except Exception as e:
            print(f"❌ 分析失敗: {e}")
            import traceback
            traceback.print_exc()

def main():
    analyzer = ResultsAnalyzerNoDisplay()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()