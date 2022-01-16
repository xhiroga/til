module Chapter3 where

-- 3 型と型クラス

bools :: [Bool] -- bools の型は [Bool] である、と読む
bools = [True, False]

ns :: [Int] -- リスト
ns = [1, 2, 3]

tuple :: (String, Int, Bool) -- 要素の型が異なっても構わない
tuple = ("One", 1, False)

-- Haskellにおける関数は、「ある型の引数を別の型に変換する」すなわち、複数の引数は定義上ありえない。
-- 複数の引数はタプルで表現する。（関数の型定義がシンプルになって賢い整理だと思う）
add :: (Int, Int) -> Int
add (x, y) = x + y

-- Haskellでは、複数の引数を取る関数を関数を返す関数という形でも定義できる（カリー化）
-- 部分適用ができるのでタプルを引数に取る関数よりオススメとのこと。
add_ :: Int -> (Int -> Int)
add_ = \x -> (x +)

-- このように括弧が増えるのを防ぐため、括弧は省略して書いていいし、その場合右から順に関数とする
mult5 :: Int -> (Int -> (Int -> (Int -> (Int -> Int))))
mult5 v w x y z = v * w * x * y * z

-- 注文: プログラマーがやりたいことは順番を問わない部分適用だと思われるので、カリー化を用いた実装（引数の順序が重要になる）は扱いづらい気がする。
-- JavaのBuilderパターンみたいなことがしたいのよ。

-- 3.7 多層型（ジェネリクス）

last3 :: [a] -> a
last3 ss = last ss

-- 注文: （クラスに予約されているようだが）型変数も大文字で表現するべきではないか？その点Typescriptはエラい。
-- もしくは基本型も小文字であるべきだ（と思っていたのだが、基本型も型クラス制約に指定できる...?）

-- 関数の型だけでなく、その型に対する型クラス制約までも定義している → 多重定義型
plus :: Num a => a -> (a -> a) -- Num型の制約のある型aを前提に、a型を受け取ってa型を返す関数
plus x y = x + y

-- 3.11 練習問題

second :: [a] -> a
second xs = head (tail xs)

swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

twice :: (a -> a) -> (a -> a)
twice f x = f (f x)
