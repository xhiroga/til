# ARDUINO

## 使い方
### Lチカ
ARDUINO本体のPOWERの5V/3VとGNDを両端として、間にLEDを挟むようにしてケーブルで繋いでいく。  
LEDが使用可能な電力は、オームの法則により5V/220ohm=2.3mA  

### Lチカ(デジタル)
Arduino IDEのTool > BoardとPortの設定をチェックし、Upload。  
```
void loop() {
    digitalWrite(3, HIGH) // デジタル出力3番をOUTの意
}
```

## 便利な関数
```
    delay(150); // ミリ秒待つ
    analogRead(A0); // 0~1023の範囲で入力を読み取る
    analogWrite(A0, 255) // 0~255の範囲で出力する
```
