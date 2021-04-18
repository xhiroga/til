module Chapter2Spec where

import Chapter2
import Test.Hspec

ns = [1, 2, 3, 4, 5]

spec :: Spec
spec = do
  describe "aveerage" $
    it "get average" $
      average ns `shouldBe` 3
  describe "練習問題" $
    it "2.7.1.1" $
      2 ^ 3 * 4 `shouldBe` 32
  describe "練習問題" $
    it "2.7.4" $
      last1 ns `shouldBe` 5
