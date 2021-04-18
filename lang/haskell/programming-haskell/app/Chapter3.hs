module Chapter3 where

bools = [True, False]

nums = [[1]]

-- Haskellでは、複数の引数を取る関数はカリー化された関数として定義される
add :: (Int, Int) -> Int
add x y = x+y

add_ :: Int -> (Int -> Int)
add__ x = {y = x + Y}
-- add__ x y = x + y

add__ :: Int -> Int -> Int -> Int
add__ x y z = x + y + z

copy a = (a, a)

apply fun a = fun a