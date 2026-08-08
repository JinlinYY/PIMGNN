"""Implement the cignn fix_sklearn baseline module."""
import sys
import subprocess

def check_sklearn():
    """Run the check sklearn baseline operation."""
    try:
        import sklearn
        print(f"✓ sklearn已安装，版本: {sklearn.__version__}")
        
        # Evaluate the test subset.
        from sklearn.model_selection import train_test_split
        import numpy as np
        X = np.arange(100)
        train, test = train_test_split(X, test_size=0.2, random_state=42)
        print(f"✓ train_test_split功能正常")
        print(f"  训练集大小: {len(train)}, 测试集大小: {len(test)}")
        return True
    except ImportError as e:
        print(f"✗ sklearn未安装或导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ sklearn功能测试失败: {e}")
        return False

def install_sklearn():
    """Run the install sklearn baseline operation."""
    print("\n正在安装scikit-learn...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
        print("✓ 安装完成！")
        return True
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        print("\n请手动运行以下命令安装:")
        print("pip install scikit-learn")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("sklearn可用性检查")
    print("=" * 50)
    
    if not check_sklearn():
        print("\n" + "=" * 50)
        print("尝试安装sklearn...")
        print("=" * 50)
        if install_sklearn():
            print("\n重新检查...")
            check_sklearn()
    else:
        print("\n✓ sklearn完全可用！")

