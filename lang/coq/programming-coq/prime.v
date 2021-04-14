(* 二つの自然数の和を返す *)
Definition plus (n : nat)(m : nat): nat := n + m .

(* 命題: 任意の命題 A に対して「A ならば A」。 *)
Definition prop0 : forall (A : Prop), A -> A :=
  (* プログラムは照明に相当する *)
  fun A x => x.
