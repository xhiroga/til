interface Prop {
  (): {};
}
type A = Prop;
type B = Prop;
type C = Prop;

type IfBThenC = (b: B) => C;
type IfAThenB = (a: A) => B;

const Prop1: (IfBThenC, IfAThenB, A) => C = (ifBThenC, ifAThenB, a) => {
  return ifBThenC(ifAThenB(a));
};
