import { Flux } from 'reactor-core-js/flux';

Flux.range(1, 10)
    .take(5)
    .map(v => v * 2)
    .flatMap(v => Flux.range(v, 2))
    .subscribe(v => console.log(v));
