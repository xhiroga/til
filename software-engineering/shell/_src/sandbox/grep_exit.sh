#仮説:  $()記法は期待した結果がないと止まるっぽい?
#結論: grepは結果が見つからないとexit 1を返す。 bashで-eオプションだとそこで落ちる
cfn_files=$(ls | grep -e "cfn-splited\(.*\).yaml")
echo foo
