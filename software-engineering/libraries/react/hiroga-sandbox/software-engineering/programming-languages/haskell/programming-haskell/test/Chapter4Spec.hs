module Chapter4Spec where

import Chapter4
import Test.Hspec

ns = [1, 2, 3, 4, 5, 6]

spec :: Spec
spec = do
  describe "練習問題" $
    it "4.8.1" $
      halve(ns) `shouldBe` ([1,2,3], [4,5,6])
