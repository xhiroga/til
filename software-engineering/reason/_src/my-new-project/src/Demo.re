Js.log("Hello, BuckleScript and Reason!");

let listA = [1, 2, 3];
let listB = [0, ...listA];

List.map(it => {Js.log(it)}, listB);

let triple = ("seven", 8, '9');
// let (first, others) = triple; // it shows error
