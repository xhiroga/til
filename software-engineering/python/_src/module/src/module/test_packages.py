#!/usr/bin/env python3
"""
パッケージと名前空間パッケージの違いを確認するテストスクリプト

spam: 通常のパッケージ（__init__.pyあり）
ham: 名前空間パッケージ（__init__.pyなし）
"""

import sys
import os

def test_package_types():
    """パッケージタイプの確認"""
    print("=== パッケージタイプの確認 ===")
    
    # spamパッケージ（通常のパッケージ）
    try:
        import spam
        print(f"✓ spam パッケージをインポート成功")
        print(f"  spam.__file__: {spam.__file__}")
        print(f"  spam.__path__: {spam.__path__}")
        print(f"  spam.__version__: {spam.__version__}")
        print(f"  spam.spam_function(): {spam.spam_function()}")
        
        # spamパッケージ内のモジュール
        from spam import eggs
        print(f"  spam.eggs.eggs_function(): {eggs.eggs_function()}")
        
    except ImportError as e:
        print(f"✗ spam パッケージのインポートに失敗: {e}")
    
    print()
    
    # hamパッケージ（名前空間パッケージ）
    try:
        import ham
        print(f"✓ ham 名前空間パッケージをインポート成功")
        print(f"  ham.__file__: {getattr(ham, '__file__', 'None (名前空間パッケージ)')}")
        print(f"  ham.__path__: {ham.__path__}")
        
        # hamパッケージ内のモジュール
        from ham import bacon
        print(f"  ham.bacon.bacon_function(): {bacon.bacon_function()}")
        
    except ImportError as e:
        print(f"✗ ham パッケージのインポートに失敗: {e}")

def test_package_attributes():
    """パッケージ属性の詳細確認"""
    print("\n=== パッケージ属性の詳細確認 ===")
    
    try:
        import spam
        print("spam パッケージの属性:")
        print(f"  type(spam): {type(spam)}")
        print(f"  hasattr(spam, '__file__'): {hasattr(spam, '__file__')}")
        print(f"  hasattr(spam, '__path__'): {hasattr(spam, '__path__')}")
        print(f"  spam.__spec__.submodule_search_locations: {spam.__spec__.submodule_search_locations}")
        
    except Exception as e:
        print(f"spam パッケージの属性確認でエラー: {e}")
    
    print()
    
    try:
        import ham
        print("ham パッケージの属性:")
        print(f"  type(ham): {type(ham)}")
        print(f"  hasattr(ham, '__file__'): {hasattr(ham, '__file__')}")
        print(f"  hasattr(ham, '__path__'): {hasattr(ham, '__path__')}")
        print(f"  ham.__spec__.submodule_search_locations: {ham.__spec__.submodule_search_locations}")
        
    except Exception as e:
        print(f"ham パッケージの属性確認でエラー: {e}")

def check_directory_structure():
    """ディレクトリ構造の確認"""
    print("\n=== ディレクトリ構造の確認 ===")
    
    current_dir = os.path.dirname(__file__)
    
    spam_dir = os.path.join(current_dir, "spam")
    ham_dir = os.path.join(current_dir, "ham")
    
    print(f"spam ディレクトリ: {spam_dir}")
    if os.path.exists(spam_dir):
        spam_init = os.path.join(spam_dir, "__init__.py")
        print(f"  __init__.py 存在: {os.path.exists(spam_init)}")
        print(f"  ファイル一覧: {os.listdir(spam_dir)}")
    
    print(f"ham ディレクトリ: {ham_dir}")
    if os.path.exists(ham_dir):
        ham_init = os.path.join(ham_dir, "__init__.py")
        print(f"  __init__.py 存在: {os.path.exists(ham_init)}")
        print(f"  ファイル一覧: {os.listdir(ham_dir)}")

if __name__ == "__main__":
    print("Python パッケージと名前空間パッケージの比較テスト")
    print("=" * 50)
    
    check_directory_structure()
    test_package_types()
    test_package_attributes() 