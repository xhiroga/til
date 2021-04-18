(* 二つの自然数の和を返す *)
Definition plus (n : nat)(m : nat): nat := n + m .

Eval compute in plus 1 2.

(* 命題: 任意の命題 A に対して「A ならば A」。 *)
Definition prop0 : forall (A : Prop), A -> A :=
  (* プログラムは照明に相当する *)
  fun A x => x.

(* 命題1. 任意の命題 A B C に対して、「B ならば C」ならば「A ならば B」ならば 「A ならば C」。 *)
Definition prop1 : forall (A B C : Prop), (B -> C) -> (A -> B) -> (A -> C) :=
  fun A B C f g x => f (g (x)).

Eval compute in prop1.
