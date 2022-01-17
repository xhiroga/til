variable=foo

echo ${variable}bar # foobar
echo {$variable}bar # {foo}bar

echo $"variable"bar # variablebar
echo "$variable"bar # foobar

# 以上からわかる優先度と効果
# 1. $が変数を参照するのがもっとも早い
# 2. {}は単体で使っても特に意味がないが、＄などの演算子の対象を限定する
# 3. エスケープした対象はおそらくtokenではなくなり、$演算子の影響を受けない。