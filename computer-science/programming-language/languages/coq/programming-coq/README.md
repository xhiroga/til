# Coq

## prerequisite

```shell script
brew install coq
```

## Coq Commands

```shell script
coqc prime.v
# will produce prime.vo

coqtop
# see https://coq.inria.fr/refman/proof-engine/vernacular-commands.html
Coq < Require Import prime.

Coq < Print plus.
plus = fun n m : nat => n + m
     : nat -> nat -> nat

Arguments plus (_ _)%nat_scope
Coq < Eval compute in plus 1 2.
     = 3
     : nat
```

## references

- [プログラミング Coq](https://www.iij-ii.co.jp/activities/programming-coq/coqt1.html)
- [Introduction and Contents](https://coq.inria.fr/distrib/current/refman/)
