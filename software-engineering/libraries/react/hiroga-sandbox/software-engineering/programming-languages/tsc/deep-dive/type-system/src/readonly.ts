
class Point {
    readonly x: number; // 宣言したプロパティはインスタンスの中でconstになる、と考えて概ねよい。ただし、緩い型にキャストするような方法で変更することができる。
    y: number;
    // readonly で引数を修飾できる。これはコンストラクタ限定で、普通のfunctionの引数は全てprivateらしい。
    // 別にコンストラクタ引数もデフォルト不変で良くないか？と思ったが、JavaScriptの制約だろうか。
    // https://stackoverflow.com/questions/54627366/a-parameter-property-in-only-allowed-in-constructor-implementation
    constructor(readonly arg_x: number, y: number) {
        this.x = arg_x;
        this.y = y;
    }
    add(point: Point) {
        return new Point(this.x + point.x, this.y + point.y);
    }
}

var p1 = new Point(0, 10);
var p2 = new Point(10, 20);
var p3 = p1.add(p2); // {x:10,y:30}
