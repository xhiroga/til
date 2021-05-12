# Redux

# About
## 前提
* propsとstateの違い: stateが変るとcomponentが再描画される。それだとややこしいので、stateをpropsにマッピングするのがredux
* reactとreduxは本来全く関係ない

## Reducer
元々のstateとactionのpayloadをinputに、新たなstateをoutputする関数。
combineReducers, stateに対してreducerを紐つける

## React Redux
Stateの渡し方
Reduxが全てのstateを管理している。react-reduxを使うことで、stateをpropsにマッピングしてくれる。