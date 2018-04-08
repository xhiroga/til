// スイッチのON/OFFをデジタル入力から受け取り、それを元にライトを点灯させる

int switchState = 0;

void setup() {
  // 1つ目のデフォルト関数
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(2, INPUT);

}

void loop() {
  // 2つ目のデフォルト関数
  switchState = digitalRead(2);

  if (switchState == LOW){
    digitalWrite(3, HIGH);
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
  }
  else {
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);
    digitalWrite(5, HIGH);

    delay(250); // 250秒待つ
    digitalWrite(4, HIGH);
    digitalWrite(5, LOW);
    delay(250);
  }
}