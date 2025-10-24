# -*- coding: utf-8 -*-
"""
檢查必要套件是否已安裝
"""

def check_packages():
    """檢查所有必要的套件"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'seaborn', 'joblib', 'psutil', 'holidays'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安裝")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少套件: {missing_packages}")
        print("請執行: pip install", " ".join(missing_packages))
        return False
    else:
        print("\n🎉 所有套件已安裝！")
        return True

if __name__ == "__main__":
    check_packages()