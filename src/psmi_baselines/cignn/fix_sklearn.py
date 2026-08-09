import sys
import subprocess

def check_sklearn():
    try:
        import sklearn
        print(f"✓ sklearn Installed , version : {sklearn.__version__}")
        
        # Evaluate the test subset.
        from sklearn.model_selection import train_test_split
        import numpy as np
        X = np.arange(100)
        train, test = train_test_split(X, test_size=0.2, random_state=42)
        print(f"✓ train_test_split Features normal ")
        print(f" training set Size : {len(train)}, test set Size : {len(test)}")
        return True
    except ImportError as e:
        print(f"scikit-learn is unavailable: {e}")
        return False
    except Exception as e:
        print(f"scikit-learn validation failed: {e}")
        return False

def install_sklearn():
    print("\n True at Installation scikit-learn...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
        print("✓ Installation complete !")
        return True
    except Exception as e:
        print(f"Installation failed: {e}")
        print("\nInstall scikit-learn manually with:")
        print("pip install scikit-learn")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("sklearn available Sex check ")
    print("=" * 50)
    
    if not check_sklearn():
        print("\n" + "=" * 50)
        print(" attempt Installation sklearn...")
        print("=" * 50)
        if install_sklearn():
            print("\n Heavy new check ...")
            check_sklearn()
    else:
        print("\n✓ sklearn perfect available !")

