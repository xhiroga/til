// function greeter_error(person){ // アノテーションなしだとチェックされない。
function greeter_error(person: string){
    return "Hello, " + person;
}

let user_invalid = 123;

console.log(greeter_error(user_invalid))