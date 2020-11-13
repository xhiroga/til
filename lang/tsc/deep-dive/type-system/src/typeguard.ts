/**
 * Just some interfaces
 */
interface Foo {
    foo: number;
    common: string;
}

interface Bar {
    bar: number;
    common: string;
}

/**
 * User Defined Type Guard!
 */
// arg is Foo は Boolean らしい。この書き方はコンパイラにだけ伝わるメッセージってことでしょう。
function isFoo(arg: any): arg is Foo {
    return arg.foo !== undefined;
}

/**
 * Sample usage of the User Defined Type Guard
 */
function doStuff(arg: Foo | Bar) {
    if (isFoo(arg)) {
        console.log(arg.foo); // OK
        // console.log(arg.bar); // Error!
    }
    else {
        // console.log(arg.foo); // Error!
        console.log(arg.bar); // OK
    }
}

doStuff({ foo: 123, common: '123' });
doStuff({ bar: 123, common: '123' });

// in演算子を使えばプロパティの有無で型ガードが使える。更に、同名プロパティで型がリテラル型の場合、それも判別に使える。
interface Groudon {
    kind: "ground",
    earthQuake: 100
}

interface Kyogre {
    kind: "water",
    surf: 90
}

type HoennLegend = Groudon | Kyogre

function goHoennLegend(legend: HoennLegend) {
    if (legend.kind === "ground") {
        console.log(legend.earthQuake)
    } else if (legend.kind === "water") {
        console.log(legend.surf)
    } else {
        const _exhaustiveCheck: never = legend  // これまでの私は単にエラーを投げていたが、この書き方（想定外のケースではコンパイルエラーにする）だとコンパイル時に気付ける。
    }
}
