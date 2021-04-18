module Chapter4Spec where

import Chapter7
import Test.Hspec

ns = [2, 3, 4, 5]

spec :: Spec
spec = do
  describe "練習問題" $
    it "7.9.4" $
      dec2int ns `shouldBe` 2345
