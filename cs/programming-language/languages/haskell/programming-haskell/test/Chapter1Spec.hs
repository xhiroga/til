module Chapter1Spec where

import Test.Hspec
import Chapter1 

spec :: Spec
spec = do
    describe "double" $
      it "2 x 2 = 4" $
        double 2 `shouldBe` 4
