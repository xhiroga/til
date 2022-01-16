(* 二つの自然数の和を返す *)
Definition plus (n : nat)(m : nat): nat := n + m .

Eval compute in plus 1 2.

(* 命題: 任意の命題 A に対して「A ならば A」。 *)
(* 読み方: prop0 型クラス制約Propのある型変数Aにおいて、Aを引数にAを返す *)
Definition prop0 : forall (A : Prop), A -> A :=
  (* プログラムは照明に相当する *)
  fun A x => x.

(* 命題1. 任意の命題 A B C に対して、「B ならば C」ならば「A ならば B」ならば 「A ならば C」。 *)
(* 読み方: prop1: 型クラス制約PropのあるA,B,Cにおいて、Bを引数にCを返し、Aを引数にBを返すなら、Aを引数にCを返す *)
Definition prop1 : forall (A B C : Prop), (B -> C) -> (A -> B) -> (A -> C) :=
  fun A B C f g x => f (g (x)).

Eval compute in prop1.

(* 練習問題 *)
Definition question0 : forall (A B : Prop), A -> (A -> B) -> B := 
  fun A B x f => f(x).

Eval compute in question0.

Definition question1 : forall (A B C: Prop), (A -> B -> C) -> B -> A -> C := 
  fun A B C f b a => f(a)(b).

Eval compute in question1.
