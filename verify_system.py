"""
Quick Verification Script - Tests all components
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*70)
    print("TESTING PACKAGE IMPORTS")
    print("="*70)
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'joblib': 'joblib',
        'scipy': 'scipy'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"[OK] {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    # Optional: Prophet
    try:
        import prophet
        print(f"[OK] prophet (optional)")
    except ImportError:
        print(f"⚠️  prophet - NOT INSTALLED (optional, will use Gradient Boosting only)")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All required packages installed!")
    return True


def test_data_files():
    """Test if required data files exist"""
    print("\n" + "="*70)
    print("TESTING DATA FILES")
    print("="*70)
    
    required_files = [
        '15-KV XLPE Cable.xlsx'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required data files present!")
    else:
        print("\n❌ Some data files missing. Ensure 15-KV XLPE Cable.xlsx is in root directory.")
    
    return all_exist


def test_code_files():
    """Test if all code files exist"""
    print("\n" + "="*70)
    print("TESTING CODE FILES")
    print("="*70)
    
    code_files = [
        'arabcab.py',
        'market_forecast_ml.py',
        'inventory_optimizer.py',
        'Dashboard.py'
    ]
    
    all_exist = True
    for file in code_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ All code files present!")
    else:
        print("\n❌ Some code files missing.")
    
    return all_exist


def test_directories():
    """Test if output directories exist or can be created"""
    print("\n" + "="*70)
    print("TESTING DIRECTORIES")
    print("="*70)
    
    dirs = ['models', 'outputs', 'data']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"✅ {dir_name}/ - CREATED")
            except Exception as e:
                print(f"❌ {dir_name}/ - FAILED: {e}")
        else:
            print(f"✅ {dir_name}/ - EXISTS")
    
    print("\n✅ All directories ready!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("CABLEFLOW-AI SYSTEM VERIFICATION")
    print("ARABCAB Competition 2026")
    print("="*70)
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Code Files", test_code_files()))
    results.append(("Directories", test_directories()))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - SYSTEM READY!")
        print("\nNext steps:")
        print("  1. python arabcab.py                # Train Model 1 (~30s)")
        print("  2. python market_forecast_ml.py     # Train Model 2 (~2min)")
        print("  3. python inventory_optimizer.py    # Optimize inventory (~10s)")
        print("  4. streamlit run Dashboard.py       # Launch dashboard")
    else:
        print("❌ SOME TESTS FAILED - FIX ISSUES ABOVE")
        print("\nCommon fixes:")
        print("  • Missing packages: pip install -r requirements.txt")
        print("  • Missing data: Ensure 15-KV XLPE Cable.xlsx is in root")
    
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
