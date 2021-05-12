var=foo
if [ var = foo ]; then # []の内側が前後にスペースを持たないと解釈されない。
    echo $var
else
    echo 'not found'
fi

for el1 in x y z # 半角スペースまたは改行で区切ったものがリストとして扱われる。
do
    echo $el1
done

for el2 in `ls`
do
    echo $el2
done

for el3 in a b
# echo 'Is this output?' # どうもfor分の後にはすぐにdo doneを続けないといけないらしい。
do
    echo $el3
done
