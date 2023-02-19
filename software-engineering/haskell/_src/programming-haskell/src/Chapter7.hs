module Chapter7 where

-- 高階関数

-- 7.5 関数合成演算子

(.) :: (b -> c) -> (a -> b) -> (a -> c)
f . g = \x -> f (g x) -- f composed with g, gと合成されたf
