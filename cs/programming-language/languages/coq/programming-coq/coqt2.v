Definition prop0 : forall (A : Prop), A -> A.
Proof.
intros.
apply H.
Qed.
Print prop0.


(* 練習問題 *)
Goal forall (P Q : Prop), (forall P : Prop, (P -> Q) -> Q) -> ((P -> Q) -> P) -> P.

(* 現在のゴールに対する仮定を変数H(もしやHypothesis?),H0...に代入する *)
intros.

(* ゴールの一つ手前の仮定をサブゴールに設定するコマンド *)
(* -> P は (P -> Q) -> P を当てはめれば導けるから、 P -> Q をいうアプローチに切り替える *)
apply H0. 

(* Pを仮定に、Qをサブゴールに振り分ける *)
intro.

(* 
    P -> (P -> Q)
    ならば
    P -> Q
    ならば
    P -> (P -> Q) -> P

*)
apply (H (P -> Q)).

apply (H P).
