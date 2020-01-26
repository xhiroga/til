module Chapter2 where

n = a `div` length ns
  where
    a = 10
    ns = [1, 2, 3, 4, 5]

last1 ns = ns !! (length ns - 1)

last2 ns = head (drop (length ns - 1) ns)

init1 ns = take (length ns - 1) ns

init2 ns = reverse (tail (reverse ns))