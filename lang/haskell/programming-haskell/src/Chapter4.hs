module Chapter4 where

-- 関数定義

-- 4.5 ラムダ式

add :: Int -> (Int -> Int)
add_ :: Int -> (Int -> Int)
add x y = x + y -- 関数定義

add_ = \x -> (\y -> x + y) -- ラムダ式の代入

-- 注文: function や const のようなキーワードがないので、等式による関数定義とラムダ式による関数定義の見分けがつきづらいのをなんとかしてほしい。

-- 練習問題
halve :: [Int] -> ([Int], [Int])
halve ns = (take ((length ns) `div` 2) ns, drop ((length ns) `div` 2) ns)
