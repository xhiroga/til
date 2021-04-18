module Chapter2 where

average :: Foldable t => t Int -> Int
average ns = sum ns `div` length ns -- Kotlinで見た中置記法(infix)が使える（多分こっちがオリジナル）

-- リスト操作をはじめとした組み込み関数をプレリュード(prelude)と呼ぶ。

last1 :: [a] -> a
last1 ns = ns !! (length ns - 1) -- 0番目から数えてn番目の要素を取り出す。

last2 :: [a] -> a
last2 ns = head (drop (length ns - 1) ns) -- drop の第一引数がリストでないことが気になったが、これはdropが他動詞だから？

init1 :: [a] -> [a]
init1 ns = take (length ns - 1) ns

init2 :: [a] -> [a]
init2 ns = reverse (tail (reverse ns))
