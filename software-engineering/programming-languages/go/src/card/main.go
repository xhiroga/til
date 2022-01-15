package main

import "fmt"

func main(){
  card := newCard() // := で変数宣言時に型推論をさせられる。
  fmt.Printf(card)
}

func newCard() string{
  return "Ace of Spades"
}
