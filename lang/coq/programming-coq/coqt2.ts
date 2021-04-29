type P = { type: "P" };
type Q = { type: "Q" };

type IfPThenQ = (p: P) => Q;
type IfPThenQ_Then_Q = (ifPThenQ: IfPThenQ) => Q;
type IfPThenQ_Then_P = (ifPThenQ: IfPThenQ) => P;

// Goal
type Prop1 = (
  ifPThenQ_Then_Q: IfPThenQ_Then_Q,
  ifPThenQ_Then_P: IfPThenQ_Then_P
) => P;
const prop1: Prop1 = (ifPThenQ_Then_Q, ifPThenQ_Then_P) => {
  const some: IfPThenQ;
  return ifPThenQ_Then_P(some);
};
