// Generated by BUCKLESCRIPT, PLEASE EDIT WITH CARE
'use strict';

var List = require("bs-platform/lib/js/list.js");

console.log("Hello, BuckleScript and Reason!");

var listA = {
  hd: 1,
  tl: {
    hd: 2,
    tl: {
      hd: 3,
      tl: /* [] */0
    }
  }
};

var listB = {
  hd: 0,
  tl: listA
};

List.map((function (it) {
        console.log(it);
        
      }), listB);

var triple = [
  "seven",
  8,
  /* "9" */57
];

exports.listA = listA;
exports.listB = listB;
exports.triple = triple;
/*  Not a pure module */