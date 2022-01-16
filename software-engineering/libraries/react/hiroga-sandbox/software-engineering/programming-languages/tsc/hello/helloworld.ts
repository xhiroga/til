// 型アノテーションは任意。 :string がなくても動く。
function greeter(person: string) {
	return "Hello, " + person;
}

let user = "Jane User";

console.log(greeter(user));
